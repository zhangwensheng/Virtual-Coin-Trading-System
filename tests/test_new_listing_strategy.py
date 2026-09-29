import json
import sqlite3
import threading
import time

import pandas as pd
import pytest

from futures_strategy.new_listing_signals import evaluate_breakdown, eligible_listings
from futures_strategy.new_listing_portfolio import NewListingPortfolio
from futures_strategy.new_listing_service import NewListingService
from futures_strategy.pair_portfolio import PairPortfolio
from futures_strategy.boll_short_portfolio import BollPortfolio
from futures_strategy.pair_web import create_server
from test_pair_workstation_http import request


def bars():
    day = pd.Timestamp('2026-09-17', tz='UTC')
    daily = pd.DataFrame([
        dict(open=100., high=112., low=95., close=105., volume=1000.),
        dict(open=106., high=110., low=90., close=94., volume=1000.),
    ], index=[day-pd.Timedelta(days=1), day])
    index = pd.date_range(day, periods=17, freq='min')
    minutes = pd.DataFrame([dict(open=106., high=110., low=100., close=104., volume=10.) for _ in index], index=index)
    minutes.loc[index[-1], ['open','high','low','close']] = [95.2, 95.4, 94.7, 94.9]
    return daily, minutes


def test_first_close_break_signal_and_filters():
    daily, minutes = bars()
    now = pd.Timestamp('2026-09-17T00:17:05Z').timestamp()
    result = evaluate_breakdown('NEWUSDT', pd.Timestamp('2026-09-14T08:00Z').timestamp(), daily, minutes, now)
    assert result['reason'] == 'SIGNAL_READY'
    assert result['signal']['signal_id'] == 'NEWUSDT:2026-09-17'
    assert result['signal']['break_depth_pct'] == pytest.approx(95/94.9-1)
    assert result['signal']['signal_time'] == pd.Timestamp('2026-09-17T00:17Z').timestamp()
    minutes.loc[minutes.index[-1], 'close'] = 94.5
    assert evaluate_breakdown('NEWUSDT', pd.Timestamp('2026-09-14T08:00Z').timestamp(), daily, minutes, now)['reason'] == 'BREAK_TOO_DEEP'
    minutes.loc[minutes.index[-1], 'close'] = 94.9
    minutes.loc[minutes.index[15], 'close'] = 94.8
    assert evaluate_breakdown('NEWUSDT', pd.Timestamp('2026-09-14T08:00Z').timestamp(), daily, minutes, now)['reason'] == 'FIRST_BREAK_ALREADY_PASSED'


def test_watch_candidate_exposes_previous_day_low_before_break():
    daily, minutes = bars()
    minutes.loc[minutes.index[-1], 'close'] = 96.
    now = pd.Timestamp('2026-09-17T00:17:05Z').timestamp()
    result = evaluate_breakdown('NEWUSDT',pd.Timestamp('2026-09-14T08:00Z').timestamp(),daily,minutes,now)
    assert result['reason'] == 'WAIT_BREAK'
    assert result['prev_low'] == 95.


def test_listing_universe_uses_official_onboard_date_and_contract_type():
    now = pd.Timestamp('2026-09-17T12:00Z').timestamp()
    def row(symbol, age, **extra):
        return dict(symbol=symbol, status='TRADING', contractType='PERPETUAL', quoteAsset='USDT',
                    onboardDate=int((now-age*86400)*1000), **extra)
    result = eligible_listings({'symbols':[row('NEWUSDT', 3), row('OLDUSDT', 31),
        dict(row('DELUSDT',2),status='PENDING_TRADING'),
        dict(row('QUARTERUSDT',2),contractType='CURRENT_QUARTER')]},now)
    assert [r['symbol'] for r in result] == ['NEWUSDT']


def quote(now, price=100.):
    return dict(bid=price-.01,ask=price+.01,mark=price,timestamp=now)


def test_independent_1000u_10x_account_entry_exit_and_persistence(tmp_path):
    path = tmp_path/'new.db'
    p = NewListingPortfolio(path)
    now = pd.Timestamp('2026-09-17T12:00Z').timestamp()
    p.update_quotes({'NEWUSDT':quote(now)},now=now)
    p.set_running(True,now=now)
    signal = dict(signal_id='NEWUSDT:2026-09-17',symbol='NEWUSDT',signal_time=now,
                  prev_low=100.,break_depth_pct=.001,listing_age_days=3)
    rules = dict(tick_size=.01,step_size=.001,min_qty=.001,max_qty=10000.,min_notional=5.)
    position = p.open_signal(signal,rules,now=now)
    assert position['leverage'] == 10
    assert position['initial_margin'] <= 100
    assert position['gross_notional'] <= 1000
    assert position['stop_price'] == pytest.approx(position['entry_price']*1.05)
    assert position['target_price'] == pytest.approx(position['entry_price']*.90)
    with pytest.raises(ValueError,match='DAILY_LIMIT|SIGNAL_USED'):
        p.open_signal(signal,rules,now=now)
    p.update_quotes({'NEWUSDT':quote(now+2,89.9)},now=now+2)
    p.monitor(now=now+2)
    assert p.snapshot(now=now+2)['account']['equity'] > 1000
    assert p.snapshot(now=now+2)['positions'] == []
    p.db.close()
    reopened = NewListingPortfolio(path)
    assert reopened.snapshot(now=now+3)['running'] is False
    assert len(reopened.snapshot(now=now+3)['closed_positions']) == 1
    reopened.db.close()


def test_listing_margin_per_trade_can_be_changed_and_persists(tmp_path):
    path = tmp_path/'configurable.db'
    p = NewListingPortfolio(path)
    now = pd.Timestamp('2026-09-17T12:00Z').timestamp()
    assert p.snapshot(now=now)['settings']['margin_per_trade'] == 100
    p.set_margin_per_trade(55,now=now)
    assert p.snapshot(now=now)['settings']['margin_per_trade'] == 55
    p.db.close()
    reopened = NewListingPortfolio(path)
    assert reopened.snapshot(now=now)['settings']['margin_per_trade'] == 55
    reopened.db.close()


def test_legacy_listing_margin_migrates_to_100u_default(tmp_path):
    path = tmp_path/'legacy.db'
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE account (id INTEGER PRIMARY KEY, body TEXT NOT NULL)')
    old = dict(cash=1000.,positions={},closed={},consumed=[],counts={},settled=[],
        running=False,risk_paused=False,day='',day_equity=1000.,
        settings=dict(initial_capital=1000.,leverage=10,margin_per_trade=10.,
            max_positions=5,daily_loss_limit=25.,stop_pct=.05,target_pct=.10,
            max_hold_seconds=86400,fee_rate=.0005,slippage_bps=2.,
            quote_max_age=5,max_spread_pct=.001,max_break_depth=.00262871,
            last_signal_minute_utc=1022))
    db.execute('INSERT INTO account VALUES(1,?)',(json.dumps(old),))
    db.commit();db.close()
    p = NewListingPortfolio(path)
    assert p.snapshot()['settings']['margin_per_trade'] == 100
    p.db.close()


def test_third_account_routes_are_independent(tmp_path):
    pair = PairPortfolio(tmp_path/'pair.db')
    boll = BollPortfolio(tmp_path/'boll.db')
    listing = NewListingPortfolio(tmp_path/'listing.db')
    server = create_server(pair,port=0,boll=boll,listing=listing)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        overview = json.load(request(base,'/api/overview'))
        assert overview['paper_initial_capital'] == 3000
        assert overview['equity'] == 3000
        assert overview['strategy_settings']['new_listing_breakdown']['margin_per_trade'] == 100
        request(base,'/api/listing/running',{'running':True},server.token,base).close()
        assert listing.snapshot()['running'] is True
        assert pair.snapshot()['running'] is False
        assert boll.snapshot()['running'] is False
        assert json.load(request(base,'/api/listing/state'))['strategy_id'] == 'new_listing_breakdown'
        assert request(base,'/listing.js').status == 200
        now = time.time()
        signal_id = f"NEWUSDT:{pd.Timestamp(now,unit='s',tz='UTC').date()}"
        listing.update_quotes({'NEWUSDT':quote(now)},now=now)
        listing.open_signal(dict(signal_id=signal_id,symbol='NEWUSDT',signal_time=now,
            prev_low=100.,break_depth_pct=.001,listing_age_days=3),
            dict(tick_size=.01,step_size=.001,min_qty=.001,max_qty=10000.,min_notional=5.),now=now)
        manual = json.load(request(base,'/api/listing/close-position',{'id':signal_id},server.token,base))['result']
        assert manual['exit_reason'] == 'MANUAL_CLOSE'
        assert listing.snapshot()['positions'] == []
        result = json.load(request(base,'/api/workstation/close-all',{},server.token,base))['result']
        assert set(result) == {'pair','boll_short','new_listing_breakdown'}
        assert listing.snapshot()['running'] is False
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)
        pair.db.close();boll.db.close();listing.db.close()


def test_listing_margin_setting_route_updates_future_entries(tmp_path):
    pair = PairPortfolio(tmp_path/'pair.db')
    boll = BollPortfolio(tmp_path/'boll.db')
    listing = NewListingPortfolio(tmp_path/'listing.db')
    server = create_server(pair,port=0,boll=boll,listing=listing)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        result = json.load(request(base,'/api/listing/settings',{'margin_per_trade':75},server.token,base))['result']
        assert result['settings']['margin_per_trade'] == 75
        now = time.time()
        signal_id = f"NEWUSDT:{pd.Timestamp(now,unit='s',tz='UTC').date()}"
        listing.update_quotes({'NEWUSDT':quote(now)},now=now)
        listing.set_running(True,now=now)
        opened = listing.open_signal(dict(signal_id=signal_id,symbol='NEWUSDT',signal_time=now,
            prev_low=100.,break_depth_pct=.001,listing_age_days=3),
            dict(tick_size=.01,step_size=.001,min_qty=.001,max_qty=10000.,min_notional=5.),now=now)
        assert opened['initial_margin'] <= 75
        assert opened['gross_notional'] <= 750
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)
        pair.db.close();boll.db.close();listing.db.close()


def test_live_scan_uses_closed_minutes_and_does_not_replay_old_signal(tmp_path):
    daily, minutes = bars()
    now = pd.Timestamp('2026-09-17T00:17:05Z').timestamp()
    info = {'symbols':[dict(symbol='NEWUSDT',status='TRADING',contractType='PERPETUAL',
        quoteAsset='USDT',onboardDate=int(pd.Timestamp('2026-09-14T08:00Z').timestamp()*1000),
        filters=[dict(filterType='LOT_SIZE',stepSize='.001',minQty='.001',maxQty='10000'),
            dict(filterType='PRICE_FILTER',tickSize='.01'),dict(filterType='MIN_NOTIONAL',notional='5')])]}
    class Market:
        def clock(self):return now
        def metadata(self):return info
        def candles(self,symbol,interval,start,end):
            if interval=='1d':return daily
            partial=pd.DataFrame([dict(open=94.9,high=100.,low=80.,close=80.,volume=10.)],
                index=[pd.Timestamp('2026-09-17T00:17Z')])
            return pd.concat([minutes,partial])
        def quotes(self,symbols):return {'NEWUSDT':quote(now,94.9)}
    p = NewListingPortfolio(tmp_path/'scan.db')
    p.set_running(True,now=now)
    service = NewListingService(p,market=Market(),clock=lambda:now)
    service.started_at=now-10
    service.scan_once(now=now)
    assert len(p.snapshot(now=now)['positions']) == 1
    assert service.snapshot()['candidates'][0]['reason'] == 'OPENED'
    assert service.minutes[('NEWUSDT',pd.Timestamp('2026-09-17T00:00Z'))].index[-1] == minutes.index[-1]
    p.db.close()

    old = NewListingPortfolio(tmp_path/'old.db')
    old.set_running(True,now=now)
    late = NewListingService(old,market=Market(),clock=lambda:now)
    late.scan_once(now=now)
    assert old.snapshot(now=now)['positions'] == []
    assert late.snapshot()['candidates'][0]['reason'] == 'WARMUP_NO_REPLAY'
    old.db.close()


def test_funding_is_logged_once_and_included_in_closed_pnl(tmp_path):
    p = NewListingPortfolio(tmp_path/'funding.db')
    now = pd.Timestamp('2026-09-17T12:00Z').timestamp()
    p.update_quotes({'NEWUSDT':quote(now)},now=now)
    p.set_running(True,now=now)
    signal = dict(signal_id='NEWUSDT:2026-09-17',symbol='NEWUSDT',signal_time=now,
                  prev_low=100.,break_depth_pct=.001,listing_age_days=3)
    rules = dict(tick_size=.01,step_size=.001,min_qty=.001,max_qty=10000.,min_notional=5.)
    opened = p.open_signal(signal,rules,now=now)
    p.update_quotes({'NEWUSDT':quote(now+60,99.)},now=now+60)
    closed = p.close_position(opened['id'],now=now+60)
    before = closed['net_pnl']
    p.settle_funding('NEWUSDT',now+30,.001,100.,now=now+60)
    after = p.snapshot(now=now+60)['closed_positions'][0]
    assert after['net_pnl'] == pytest.approx(before+opened['qty']*.1)
    p.settle_funding('NEWUSDT',now+30,.001,100.,now=now+60)
    assert p.snapshot(now=now+60)['closed_positions'][0]['net_pnl'] == pytest.approx(after['net_pnl'])
    assert any(e['kind']=='FUNDING' for e in p.snapshot(now=now+60)['events'])
    p.db.close()


def test_equity_snapshots_do_not_hide_signal_events(tmp_path):
    p = NewListingPortfolio(tmp_path/'events.db')
    now = pd.Timestamp('2026-09-17T12:00Z').timestamp()
    p.log('SCAN',{'candidates':[]},now=now)
    for offset in range(0,60,2):
        p.monitor(now=now+offset)
    events = p.snapshot(now=now+60)['events']
    assert sum(event['kind']=='EQUITY' for event in events) <= 1
    assert any(event['kind']=='SCAN' for event in events)
    p.db.close()


def test_history_panel_keeps_signal_events_visible_during_old_equity_backlog(tmp_path):
    p = NewListingPortfolio(tmp_path/'backlog.db')
    now = pd.Timestamp('2026-09-17T12:00Z').timestamp()
    p.log('SCAN',{'candidates':[]},now=now)
    for offset in range(150):
        p.log('EQUITY',{'equity':1000.,'position_count':0},now=now+offset)
    assert any(e['kind']=='SCAN' for e in p.snapshot(now=now+151)['events'])
    p.db.close()
