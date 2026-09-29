"""Public futures market scanner and position supervisor for new listings."""
from __future__ import annotations

import copy
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from futures_strategy.binance_funding_arbitrage_data import BinanceFundingArbitrageDataClient
from futures_strategy.boll_short_service import PublicBollMarket, symbol_rules
from futures_strategy.new_listing_signals import eligible_listings, evaluate_breakdown


class PublicListingMarket(PublicBollMarket):
    def candles(self,symbol,interval,start,now):
        client=BinanceFundingArbitrageDataClient(timeout=8)
        try:return client.fetch_perp_klines(symbol,interval,int(start*1000),int(now*1000))
        finally:client.session.close()


class NewListingService:
    def __init__(self,portfolio,*,market=None,clock=time.time):
        self.portfolio=portfolio
        self.market=market or PublicListingMarket()
        self.clock=clock
        self.offset=0.
        self.portfolio.clock=self.market_time
        self.started_at=clock()
        self.lock=threading.RLock()
        self.stop_event=threading.Event()
        self.threads=[]
        self.status=dict(connection='WAITING',message='等待新币市场扫描',last_scan=None,
            last_quote=None,candidates=[],universe=[],errors=[])
        self.info={}
        self.ranking=[]
        self.metadata_time=0.
        self.daily={}
        self.minutes={}
        self.last_minute=None
        self.last_log_key=None

    def market_time(self):return self.clock()+self.offset

    def snapshot(self):
        with self.lock:return copy.deepcopy(self.status)

    def _error(self,message,exc):
        event=dict(timestamp=self.market_time(),message=message,type=type(exc).__name__)
        if isinstance(exc,ValueError):event['reason']=str(exc).split(':')[0][:80]
        with self.lock:
            self.status['connection']='DEGRADED'
            self.status['message']=message+'；等待数据恢复'
            self.status['errors']=(self.status['errors']+[event])[-20:]
        self.portfolio.log('MARKET_ERROR',event)

    def _candidate(self,row,now):
        symbol=row['symbol']
        day=pd.Timestamp(now,unit='s',tz='UTC').normalize()
        daily_key=(symbol,day)
        if daily_key not in self.daily:
            self.daily[daily_key]=self.market.candles(symbol,'1d',now-3*86400,now)
        daily=self.daily[daily_key]
        previous=day-pd.Timedelta(days=1)
        if previous not in daily.index:
            return dict(row,stage='INELIGIBLE',reason='MISSING_PREVIOUS_DAY',signal=None)
        if float(daily.loc[previous,'close'])<=float(daily.loc[previous,'open']):
            return dict(row,stage='INELIGIBLE',reason='PREVIOUS_DAY_NOT_GREEN',signal=None)
        cache=self.minutes.get(daily_key)
        start=day.timestamp() if cache is None or cache.empty else (cache.index[-1]+pd.Timedelta(minutes=1)).timestamp()
        if start < now:
            update=self.market.candles(symbol,'1m',start,now)
            update=update[update.index+pd.Timedelta(minutes=1)<=pd.Timestamp(now,unit='s',tz='UTC')]
            if cache is None:cache=update
            elif not update.empty:cache=pd.concat([cache,update]).sort_index()
            if not cache.empty:cache=cache[~cache.index.duplicated(keep='last')]
            self.minutes[daily_key]=cache
        if cache is None or cache.empty:
            return dict(row,stage='INELIGIBLE',reason='INCOMPLETE_MINUTES',signal=None)
        result=evaluate_breakdown(symbol,row['onboard_date'],daily,cache,now)
        if result.get('signal') and now-result['signal']['signal_time']>15:
            result['reason']='SIGNAL_EXPIRED'
        return dict(row,**result)

    def scan_once(self,*,now=None):
        local_now=self.clock() if now is None else now
        try:
            if not self.metadata_time or local_now-self.metadata_time>=300:
                before=self.clock();server_time=self.market.clock();after=self.clock()
                offset=server_time-(before+after)/2
                if not math.isfinite(offset) or abs(offset)>5 or after-before>5:
                    raise ValueError('CLOCK_UNTRUSTED')
                self.offset=offset
                info=self.market.metadata()
                self.info={r['symbol']:r for r in info['symbols']}
                self.ranking=eligible_listings(info,local_now+offset)
                self.metadata_time=local_now
                active={r['symbol'] for r in self.ranking}
                day=pd.Timestamp(local_now+offset,unit='s',tz='UTC').normalize()
                self.daily={k:v for k,v in self.daily.items() if k[0] in active and k[1]==day}
                self.minutes={k:v for k,v in self.minutes.items() if k[0] in active and k[1]==day}
            now=local_now+self.offset
            if local_now-self.metadata_time>330:raise ValueError('STALE_METADATA')
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs=[(row,pool.submit(self._candidate,row,now)) for row in self.ranking]
                candidates=[]
                for row,future in jobs:
                    try:candidates.append(future.result())
                    except Exception as exc:
                        candidates.append(dict(row,stage='INELIGIBLE',reason='DATA_ERROR',
                            error_type=type(exc).__name__,signal=None))
            finished=self.market_time()
            for c in candidates:
                sig=c.get('signal')
                if not sig:continue
                if sig['signal_time']<=self.started_at+self.offset:
                    c['reason']='WARMUP_NO_REPLAY';continue
                if not 0<=finished-sig['signal_time']<=15:
                    c['reason']='SIGNAL_EXPIRED';continue
                if not self.portfolio.snapshot()['running']:
                    c['reason']='PAUSED';continue
                execution_time=finished
                try:
                    quote=self.market.quotes([sig['symbol']])
                    execution_time=self.market_time()
                    self.portfolio.update_quotes(quote,now=execution_time)
                    self.portfolio.open_signal(sig,symbol_rules(self.info[sig['symbol']]),now=execution_time)
                    c['reason']='OPENED'
                except ValueError as exc:
                    c['reason']=str(exc).split(':')[0]
                    self.portfolio.log('ENTRY_BLOCKED',dict(symbol=sig['symbol'],
                        signal_id=sig['signal_id'],reason=c['reason']),now=execution_time)
            bad=any(c.get('reason') in {'DATA_ERROR','INCOMPLETE_MINUTES','INVALID_MINUTES'} for c in candidates)
            with self.lock:
                self.status.update(connection='DEGRADED' if bad else 'CONNECTED',
                    last_scan=finished,candidates=candidates,
                    universe=[r['symbol'] for r in self.ranking],
                    message=f'上市 1–30 天永续合约 {len(candidates)} 币 · '+('部分行情不可用' if bad else '监控前日低点首次收盘跌破'))
            key=(int(now//60),tuple((c['symbol'],c.get('reason')) for c in candidates))
            if key!=self.last_log_key:
                self.portfolio.log('SCAN',dict(candidates=candidates),now=finished)
                self.last_log_key=key
        except Exception as exc:self._error('新币破位扫描失败',exc)

    def _scan_loop(self):
        while not self.stop_event.is_set():
            minute=int(self.market_time()//60)
            if minute!=self.last_minute:
                self.scan_once()
                if self.snapshot()['last_scan']:
                    self.last_minute=minute
            self.stop_event.wait(2)

    def refresh_quotes(self,ids=None):
        positions=self.portfolio.snapshot()['positions']
        symbols=sorted({p['symbol'] for p in positions if ids is None or p['id'] in ids})
        if symbols:
            quotes=self.market.quotes(symbols)
            self.portfolio.update_quotes(quotes)
            with self.lock:self.status['last_quote']=self.market_time()

    def _watch_loop(self):
        while not self.stop_event.is_set():
            try:
                self.refresh_quotes()
                self.portfolio.monitor()
            except Exception as exc:self._error('新币持仓行情更新失败',exc)
            self.stop_event.wait(2)

    def _funding_loop(self):
        client=BinanceFundingArbitrageDataClient(timeout=8)
        try:
            while not self.stop_event.is_set():
                for symbol,start in self.portfolio.funding_windows().items():
                    if self.stop_event.is_set():return
                    try:
                        now=self.market_time()
                        rows=client.fetch_funding_history(symbol,int(start*1000),int(now*1000))
                        for ts,row in rows.iterrows():
                            self.portfolio.settle_funding(symbol,ts.timestamp(),float(row.funding_rate),float(row.mark_price),now=now)
                        self.portfolio.mark_funding_checked(symbol,now=now)
                    except Exception as exc:self._error('新币资金费待补记',exc)
                self.stop_event.wait(60)
        finally:client.session.close()

    def close_position(self,pid):
        try:self.refresh_quotes([pid])
        except Exception as exc:self._error('平仓前刷新行情失败',exc)
        return self.portfolio.close_position(pid)

    def close_all(self):
        self.portfolio.set_running(False)
        try:self.refresh_quotes()
        except Exception as exc:self._error('批量平仓前刷新行情失败',exc)
        return self.portfolio.close_all()

    def start(self):
        for target in (self._scan_loop,self._watch_loop,self._funding_loop):
            thread=threading.Thread(target=target,daemon=True)
            thread.start();self.threads.append(thread)

    def stop(self):self.stop_event.set()
