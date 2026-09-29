"""Public-market scan and independent exit supervision for the WR paper strategy."""
from __future__ import annotations

import copy
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from futures_strategy.binance_funding_arbitrage_data import BinanceFundingArbitrageDataClient, PERP_BASE_URL
from futures_strategy.pair_market_service import PublicPairMarket
from futures_strategy.boll_short_signals import evaluate_signal, rank_universe, _valid_closed_minutes

DAILY_DATA_ERRORS = {'invalid_daily_data','missing_current_daily_open',
                     'insufficient_daily_history','daily_data_not_continuous'}


def symbol_rules(row):
    filters = {f['filterType']:f for f in row['filters']}
    lot = filters['LOT_SIZE']
    market = filters.get('MARKET_LOT_SIZE',lot)
    # Binance occasionally provides zero market step; LOT_SIZE remains binding.
    step = float(market.get('stepSize',0)) or float(lot['stepSize'])
    step = max(step,float(lot['stepSize']))
    result = dict(tick_size=float(filters['PRICE_FILTER']['tickSize']),step_size=step,
        min_qty=max(float(lot['minQty']),float(market.get('minQty',0))),
        max_qty=min(float(lot['maxQty']),float(market.get('maxQty',0)) or float(lot['maxQty'])),
        min_notional=float(filters.get('MIN_NOTIONAL',filters.get('NOTIONAL',{})).get('notional',
            filters.get('MIN_NOTIONAL',filters.get('NOTIONAL',{})).get('minNotional',0))))
    if not all(math.isfinite(v) and v > 0 for v in result.values()):
        raise ValueError('INVALID_SYMBOL_RULES')
    return result


class PublicBollMarket:
    def _get(self,path):
        client = BinanceFundingArbitrageDataClient(timeout=8)
        try:
            return client._get(PERP_BASE_URL,path,{})
        finally:
            client.session.close()

    def metadata(self):
        return self._get('/fapi/v1/exchangeInfo')

    def tickers(self):
        return self._get('/fapi/v1/ticker/24hr')

    def clock(self):
        return int(self._get('/fapi/v1/time')['serverTime'])/1000

    def bars(self,symbol,interval,now):
        client = BinanceFundingArbitrageDataClient(timeout=8)
        try:
            start = now-(32*86400 if interval=='1d' else 3*3600)
            return client.fetch_perp_klines(symbol,interval,int(start*1000),int(now*1000))
        finally:
            client.session.close()

    def quotes(self,symbols):
        market = PublicPairMarket()
        try:
            return market.quotes(symbols)
        finally:
            market.client.session.close()


class BollMarketService:
    def __init__(self,portfolio,*,market=None,clock=time.time):
        self.portfolio = portfolio
        self.market = market or PublicBollMarket()
        self.clock = clock
        self.clock_offset = 0.
        self.portfolio.clock = self.market_time
        self.started_at = clock()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.threads = []
        self.status = dict(connection='WAITING',message='等待WR市场扫描',last_scan=None,last_quote=None,
                           candidates=[],universe=[],errors=[],rank_time=None)
        self.rank_time = 0
        self.ranking = []
        self.rank_since = {}
        self.info = {}
        self.daily = {}
        self.last_minute = None
        self.last_log_minute = None
        self.last_observation_key = None

    def market_time(self):
        return self.clock()+self.clock_offset

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.status)

    def _error(self,operation,exc):
        event = dict(timestamp=self.market_time(),message=operation,type=type(exc).__name__)
        if isinstance(exc,ValueError):
            event['reason'] = str(exc).split(':')[0][:80]
        with self.lock:
            self.status['connection'] = 'DEGRADED'
            self.status['message'] = operation+'；暂停使用过期信号'
            self.status['errors'] = (self.status['errors']+[event])[-20:]
        self.portfolio.log('MARKET_ERROR',event)

    def _candidate(self,row,now):
        symbol = row['symbol']
        day = int(now//86400)
        cached = self.daily.get(symbol)
        if not cached or cached[0] != day:
            self.daily[symbol] = (day,self.market.bars(symbol,'1d',now))
        daily = self.daily[symbol][1].copy()
        minutes = self.market.bars(symbol,'1m',now)
        timestamp = pd.Timestamp(now, unit='s', tz='UTC')
        closed, error = _valid_closed_minutes(minutes, timestamp)
        if error is not None:
            return dict(row,stage='INELIGIBLE',reason=error,signal=None)
        # Cache the completed daily range, never the intraday reference price.
        # Only a fresh, completed minute is available to both replay and scanning.
        day_start = timestamp.normalize()
        if day_start in daily.index:
            daily.loc[day_start, 'close'] = float(closed.iloc[-1]['close'])
        result = evaluate_signal(symbol,daily,minutes,now,rank=row['rank'])
        result.update(strategy_version='wr28-v4-top7-wick35',
                      daily_reference_price=float(closed.iloc[-1]['close']),
                      daily_reference_time=int((closed.index[-1]+pd.Timedelta(minutes=1)).timestamp()))
        if result.get('signal'):
            result['signal'].update(strategy_version=result['strategy_version'],
                entry_features={key: result.get(key) for key in
                    ('daily_wr','minute_wr','wr_drop','volume_ratio','daily_reference_price','daily_reference_time')},
                rank=row['rank'])
            # Preserve entry-only factors in SCAN and OPEN_POSITION for forward validation.
            last = closed.iloc[-1]
            width = float(last['high']-last['low'])
            vol_mean = float(closed['volume'].iloc[-21:-1].mean())
            factors = result['signal']['entry_features']
            factors.update(
                breakdown_lower_wick_fraction=float((min(last['open'],last['close'])-last['low'])/width) if width>0 else 0.,
                breakdown_volume_ratio=float(last['volume']/vol_mean) if vol_mean>0 else None,
                breakdown_body_pct=float((last['open']-last['close'])/last['open']*100))
            for period in (15,60):
                factors[f'return_{period}m_pct'] = (
                    float((last['close']/closed['close'].iloc[-1-period]-1)*100)
                    if len(closed)>period else None)
        if result.get('reason') in DAILY_DATA_ERRORS:
            self.daily.pop(symbol,None)
        return dict(row,**result)

    def scan_once(self,*,now=None):
        local_now = self.clock() if now is None else now
        now = local_now+self.clock_offset
        try:
            if now-self.rank_time >= 60 or not self.rank_time:
                before = self.clock()
                server_time = self.market.clock()
                after = self.clock()
                offset = server_time-(before+after)/2
                if not math.isfinite(server_time) or abs(offset) > 5 or after-before > 5:
                    raise ValueError('CLOCK_UNTRUSTED')
                self.clock_offset = offset
                now = local_now+offset
                info = self.market.metadata()
                ranked = rank_universe(info,self.market.tickers(),now)
                previous_symbols = {r['symbol'] for r in self.ranking}
                self.rank_since = {r['symbol']: self.rank_since.get(r['symbol'],0) if r['symbol'] in previous_symbols
                    else (now if self.rank_time else 0) for r in ranked}
                self.info = {r['symbol']:r for r in info['symbols']}
                self.ranking = ranked
                self.rank_time = now
            if self.market_time()-self.rank_time > 90:
                raise ValueError('STALE_RANKING')
            # Separate symbol I/O bounds latency without sharing a requests.Session.
            candidates = []
            with ThreadPoolExecutor(max_workers=5) as pool:
                jobs = [(row,pool.submit(self._candidate,row,now)) for row in self.ranking]
                for row,future in jobs:
                    try:
                        candidates.append(future.result())
                    except Exception as exc:
                        candidates.append(dict(row,stage='INELIGIBLE',reason='DATA_ERROR',signal=None,
                                               error_type=type(exc).__name__))
            finished = self.market_time()
            if finished-self.rank_time > 90:
                raise ValueError('STALE_RANKING')
            for candidate in candidates:
                sig = candidate.get('signal')
                if not sig:
                    continue
                if sig['pump_time'] < self.rank_since.get(sig['symbol'],now):
                    candidate['reason'] = 'RANKING_REENTRY_REQUIRES_NEW_PUMP'
                    continue
                if sig['signal_time'] <= self.started_at+self.clock_offset:
                    candidate['reason'] = 'WARMUP_NO_REPLAY'
                    continue
                if not 0 <= finished-sig['signal_time'] <= 15:
                    candidate['reason'] = 'SIGNAL_EXPIRED'
                    continue
                if not self.portfolio.snapshot()['running']:
                    candidate['reason'] = 'PAUSED'
                    continue
                try:
                    quote = self.market.quotes([sig['symbol']])
                    execution_time = self.market_time()
                    if execution_time-self.rank_time > 90:
                        raise ValueError('STALE_RANKING')
                    self.portfolio.update_quotes(quote,now=execution_time)
                    self.portfolio.open_signal(sig,symbol_rules(self.info[sig['symbol']]),now=execution_time)
                    candidate['reason'] = 'OPENED'
                except ValueError as exc:
                    candidate['reason'] = str(exc).split(':')[0]
                    self.portfolio.log('ENTRY_BLOCKED',dict(symbol=sig['symbol'],signal_id=sig['signal_id'],
                                                           reason=candidate['reason']))
            data_errors = {'DATA_ERROR','stale_minute_data','minute_data_not_continuous',
                           'invalid_minute_data','insufficient_minute_history'} | DAILY_DATA_ERRORS
            bad = any(c.get('reason') in data_errors for c in candidates)
            with self.lock:
                self.status.update(connection='DEGRADED' if bad else 'CONNECTED',last_scan=finished,
                    rank_time=self.rank_time,clock_offset=self.clock_offset,candidates=candidates,universe=[r['symbol'] for r in self.ranking],
                    message=f'涨幅榜 {len(candidates)} 币 · '+('部分数据不可用' if bad else '日线WR与分钟转弱监控'))
            observation_key = tuple((c['symbol'],c.get('stage'),c.get('reason')) for c in candidates)
            if self.last_log_minute != int(now//60) or observation_key != self.last_observation_key:
                self.portfolio.log('SCAN',dict(candidates=candidates,rank_time=self.rank_time),now=finished)
                self.last_log_minute = int(now//60)
                self.last_observation_key = observation_key
        except Exception as exc:
            self._error('WR扫描失败',exc)

    def _scan_loop(self):
        while not self.stop_event.is_set():
            now = self.clock()
            minute = int(self.market_time()//60)
            if minute != self.last_minute:
                self.scan_once(now=now)
                # Failed scans are retried; successful scans wait for another closed bar.
                status = self.snapshot()
                if status['connection']=='CONNECTED' and status['last_scan'] and status['last_scan'] >= now+self.clock_offset:
                    self.last_minute = minute
            self.stop_event.wait(2)

    def refresh_quotes(self,ids=None):
        positions = self.portfolio.snapshot()['positions']
        symbols = sorted({p['symbol'] for p in positions if ids is None or p['id'] in ids})
        if symbols:
            quotes = self.market.quotes(symbols)
            self.portfolio.update_quotes(quotes)
            with self.lock:
                self.status['last_quote'] = self.market_time()

    def _watch_loop(self):
        while not self.stop_event.is_set():
            try:
                self.refresh_quotes()
                self.portfolio.monitor()
            except Exception as exc:
                self._error('WR持仓行情更新失败',exc)
            self.stop_event.wait(2)

    def _funding_loop(self):
        client = BinanceFundingArbitrageDataClient(timeout=8)
        try:
            while not self.stop_event.is_set():
                for symbol,start in self.portfolio.funding_windows().items():
                    if self.stop_event.is_set():
                        return
                    try:
                        now = self.market_time()
                        rows = client.fetch_funding_history(symbol,int(start*1000),int(now*1000))
                        for ts,r in rows.iterrows():
                            self.portfolio.settle_funding(symbol,ts.timestamp(),float(r.funding_rate),float(r.mark_price))
                        self.portfolio.mark_funding_checked(symbol,now=now)
                    except Exception as exc:
                            self._error('WR资金费待补记',exc)
                self.stop_event.wait(60)
        finally:
            client.session.close()

    def close_position(self,pid):
        try:
            self.refresh_quotes([pid])
        except Exception as exc:
            self._error('平仓前刷新行情失败',exc)
        return self.portfolio.close_position(pid)

    def close_all(self):
        self.portfolio.set_running(False)
        try:
            self.refresh_quotes()
        except Exception as exc:
            self._error('批量平仓前刷新行情失败',exc)
        return self.portfolio.close_all()

    def start(self):
        for target in [self._scan_loop,self._watch_loop,self._funding_loop]:
            thread = threading.Thread(target=target,daemon=True)
            thread.start()
            self.threads.append(thread)

    def stop(self):
        self.stop_event.set()
