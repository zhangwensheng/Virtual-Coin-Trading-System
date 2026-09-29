import pandas as pd
import pytest

from futures_strategy.boll_short_portfolio import BollPortfolio
from futures_strategy.boll_short_service import BollMarketService, symbol_rules


NOW = 1789002000.


class Market:
    def __init__(self):
        self.fail = False

    def metadata(self):
        if self.fail:
            raise TimeoutError('external timeout')
        return {'symbols':[dict(symbol='AAAUSDT',baseAsset='AAA',quoteAsset='USDT',
            status='TRADING',contractType='PERPETUAL',onboardDate=1500000000000,filters=[
            dict(filterType='PRICE_FILTER',tickSize='.01'),
            dict(filterType='LOT_SIZE',stepSize='.001',minQty='.001',maxQty='10000'),
            dict(filterType='MARKET_LOT_SIZE',stepSize='.002',minQty='.002',maxQty='5000'),
            dict(filterType='MIN_NOTIONAL',notional='5')])]}

    def tickers(self):
        return [dict(symbol='AAAUSDT',priceChangePercent='10',quoteVolume='30000000')]

    def clock(self):
        return NOW

    def bars(self,symbol,interval,now):
        index = pd.date_range(end=pd.Timestamp(now,unit='s',tz='UTC').floor('D' if interval=='1d' else 'min'),
                              periods=40, freq='1D' if interval=='1d' else '1min')
        return pd.DataFrame(dict(open=100.,high=101.,low=99.,close=100.,volume=100.),index=index)

    def quotes(self,symbols):
        return {s:dict(bid=100.,ask=100.02,mark=100.01,timestamp=NOW) for s in symbols}


def test_filter_rules_use_market_lot_constraints():
    rules = symbol_rules(Market().metadata()['symbols'][0])
    assert rules == dict(tick_size=.01,step_size=.002,min_qty=.002,max_qty=5000.,min_notional=5.)


def test_scan_records_rejected_daily_gate_without_opening(tmp_path):
    p = BollPortfolio(tmp_path/'s.db')
    try:
        s = BollMarketService(p,market=Market(),clock=lambda:NOW)
        s.scan_once(now=NOW)
        view = s.snapshot()
        assert view['last_scan'] == NOW
        assert len(view['candidates']) == 1
        assert view['candidates'][0]['stage'] == 'INELIGIBLE'
        assert p.snapshot(now=NOW)['positions'] == []
        assert p.snapshot(now=NOW)['events'][0]['kind'] == 'SCAN'
    finally:
        p.db.close()


def test_scan_failure_is_visible_and_does_not_claim_fresh_scan(tmp_path):
    p = BollPortfolio(tmp_path/'s.db')
    try:
        m = Market(); s = BollMarketService(p,market=m,clock=lambda:NOW)
        s.scan_once(now=NOW)
        m.fail = True
        s.scan_once(now=NOW+100)
        view = s.snapshot()
        assert view['connection'] == 'DEGRADED'
        assert view['last_scan'] == NOW
        assert view['errors'][-1]['type'] == 'TimeoutError'
        assert p.snapshot(now=NOW)['positions'] == []
    finally:
        p.db.close()


def test_bad_exchange_clock_disables_new_signals(tmp_path):
    p = BollPortfolio(tmp_path/'s.db')
    try:
        s = BollMarketService(p,market=Market(),clock=lambda:NOW+100)
        s.scan_once(now=NOW+100)
        assert s.snapshot()['connection'] == 'DEGRADED'
        assert s.snapshot()['candidates'] == []
    finally:
        p.db.close()


def test_small_exchange_clock_offset_is_calibrated_for_quotes_and_snapshot(tmp_path):
    class OffsetMarket(Market):
        def clock(self): return NOW+3
    p = BollPortfolio(tmp_path/'s.db')
    try:
        s = BollMarketService(p,market=OffsetMarket(),clock=lambda:NOW)
        s.scan_once(now=NOW)
        assert s.snapshot()['connection'] == 'CONNECTED'
        p.update_quotes({'AAAUSDT':dict(bid=100.,ask=100.02,mark=100.01,timestamp=NOW+3)})
        assert p.snapshot()['timestamp'] == NOW+3
    finally:
        p.db.close()


def test_real_signal_flows_to_paper_fill_once_after_startup(tmp_path):
    from test_boll_short_signals import _daily, _minute_rows
    t = pd.Timestamp('2026-09-10 00:38:00',tz='UTC').timestamp()
    current = [t-60]
    class SignalMarket(Market):
        def clock(self): return current[0]
        def bars(self,symbol,interval,now): return _daily() if interval=='1d' else _minute_rows()
        def quotes(self,symbols):
            return {s:dict(bid=100.,ask=100.02,mark=100.01,timestamp=current[0]) for s in symbols}
    p = BollPortfolio(tmp_path/'s.db')
    try:
        s = BollMarketService(p,market=SignalMarket(),clock=lambda:current[0])
        p.set_running(True,now=t-60)
        current[0] = t
        s.scan_once(now=t)
        assert len(p.snapshot(now=t)['positions']) == 1
        saved = p.snapshot(now=t)['positions'][0]['signal']
        assert saved['strategy_version'] == 'wr28-v4-top7-wick35'
        assert saved['entry_features']['daily_reference_price'] == 105.5
        assert saved['entry_features']['daily_reference_time'] == t
        assert saved['entry_features']['daily_wr'] == 27.5
        assert saved['entry_features']['breakdown_lower_wick_fraction'] == .25
        assert saved['entry_features']['return_15m_pct'] == pytest.approx(5.5)
        assert saved['rank'] == 1
        cash = p.snapshot(now=t)['account']['cash']
        s.scan_once(now=t)
        assert len(p.snapshot(now=t)['positions']) == 1
        assert p.snapshot(now=t)['account']['cash'] == cash
    finally:
        p.db.close()


def test_dropping_off_ranking_invalidates_old_pump_on_reentry(tmp_path):
    from test_boll_short_signals import _daily, _minute_rows
    t = pd.Timestamp('2026-09-10 00:38:00',tz='UTC').timestamp()
    current = [t-120]
    class RankingMarket(Market):
        def clock(self): return current[0]
        def tickers(self): return [] if current[0]==t-60 else super().tickers()
        def bars(self,symbol,interval,now): return _daily() if interval=='1d' else _minute_rows()
        def quotes(self,symbols):
            return {s:dict(bid=100.,ask=100.02,mark=100.01,timestamp=current[0]) for s in symbols}
    p = BollPortfolio(tmp_path/'s.db')
    try:
        s = BollMarketService(p,market=RankingMarket(),clock=lambda:current[0])
        p.set_running(True,now=current[0])
        s.scan_once(now=current[0])
        current[0] = t-60; s.scan_once(now=current[0])
        current[0] = t; s.scan_once(now=current[0])
        assert p.snapshot(now=t)['positions'] == []
        assert s.snapshot()['candidates'][0]['reason'] == 'RANKING_REENTRY_REQUIRES_NEW_PUMP'
    finally:
        p.db.close()


def test_transient_symbol_failure_retries_in_same_minute(tmp_path):
    class FlakyMarket(Market):
        attempts = 0
        def bars(self,symbol,interval,now):
            if interval=='1d':
                self.attempts += 1
                if self.attempts==1:
                    raise TimeoutError('one transient request')
            return super().bars(symbol,interval,now)
    class TwoIterations:
        waits = 0
        def is_set(self): return self.waits >= 2
        def wait(self,_): self.waits += 1
    p=BollPortfolio(tmp_path/'s.db')
    try:
        s=BollMarketService(p,market=FlakyMarket(),clock=lambda:NOW)
        s.stop_event=TwoIterations()
        s._scan_loop()
        assert s.snapshot()['connection']=='CONNECTED'
        assert s.snapshot()['candidates'][0]['reason']!='DATA_ERROR'
        scans=[e for e in p.snapshot(now=NOW)['events'] if e['kind']=='SCAN']
        assert len(scans)==2  # reason change within a minute must remain in audit.
    finally:
        p.db.close()


def test_incomplete_daily_response_is_not_cached_for_entire_day(tmp_path):
    class IncompleteMarket(Market):
        daily_calls = 0
        def bars(self,symbol,interval,now):
            bars = super().bars(symbol,interval,now)
            if interval == '1d':
                self.daily_calls += 1
                if self.daily_calls == 1:
                    return bars.iloc[:-1]
            return bars
    p = BollPortfolio(tmp_path/'s.db')
    try:
        m = IncompleteMarket()
        s = BollMarketService(p,market=m,clock=lambda:NOW)
        s.scan_once(now=NOW)
        assert s.snapshot()['connection'] == 'DEGRADED'
        s.scan_once(now=NOW)
        assert m.daily_calls == 2
        assert s.snapshot()['connection'] == 'CONNECTED'
        assert s.snapshot()['candidates'][0]['reason'] == 'daily_wr_not_overbought'
    finally:
        p.db.close()


def test_cached_daily_gate_tracks_closed_minute_price_in_both_directions(tmp_path):
    class MovingMarket(Market):
        price = 100.8
        def bars(self, symbol, interval, now):
            frame = super().bars(symbol, interval, now)
            if interval == '1m':
                frame.loc[frame.index[-2], 'close'] = self.price
                # The forming candle must never decide the gate.
                frame.loc[frame.index[-1], 'close'] = 100.99
            return frame
    p = BollPortfolio(tmp_path/'moving.db')
    try:
        m = MovingMarket()
        s = BollMarketService(p, market=m, clock=lambda:NOW)
        candidate = s._candidate(dict(symbol='AAAUSDT', rank=1), NOW)
        assert candidate['daily_wr'] == pytest.approx(-10.)
        assert candidate['stage'] != 'INELIGIBLE'
        m.price = 99.2
        candidate = s._candidate(dict(symbol='AAAUSDT', rank=1), NOW+60)
        assert candidate['daily_wr'] == pytest.approx(-90.)
        assert candidate['reason'] == 'daily_wr_not_overbought'
        assert s.daily['AAAUSDT'][1].iloc[-1]['close'] == 100.
    finally:
        p.db.close()


def test_stale_minutes_cannot_be_used_as_current_daily_price(tmp_path):
    class StaleMarket(Market):
        def bars(self, symbol, interval, now):
            return super().bars(symbol, interval, now-120 if interval == '1m' else now)
    p = BollPortfolio(tmp_path/'stale.db')
    try:
        s = BollMarketService(p, market=StaleMarket(), clock=lambda:NOW)
        result = s._candidate(dict(symbol='AAAUSDT', rank=1), NOW)
        assert result['reason'] == 'stale_minute_data'
        assert result['signal'] is None
    finally:
        p.db.close()
