"""Persistent 1000U paper account for configurable-margin / 10x new-listing shorts."""
from __future__ import annotations

import copy
import json
import math
import sqlite3
import threading
import time
from pathlib import Path

from futures_strategy.boll_short_portfolio import quantize
from futures_strategy.new_listing_signals import MAX_BREAK_DEPTH, LAST_SIGNAL_MINUTE_UTC

SETTINGS = dict(settings_version=2, initial_capital=1000., leverage=10, margin_per_trade=100.,
    max_positions=5, daily_loss_limit=25., stop_pct=.05, target_pct=.10,
    max_hold_seconds=86400, fee_rate=.0005, slippage_bps=2.,
    quote_max_age=5, max_spread_pct=.001, max_break_depth=MAX_BREAK_DEPTH,
    last_signal_minute_utc=LAST_SIGNAL_MINUTE_UTC)


def normalized_settings(stored=None):
    settings = dict(SETTINGS)
    if isinstance(stored, dict) and stored.get('settings_version') == SETTINGS['settings_version']:
        for key in SETTINGS:
            if key in stored:
                settings[key] = stored[key]
    return settings


class NewListingPortfolio:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.lock = threading.RLock()
        self.clock = time.time
        self.quotes = {}
        self.last_equity_event = 0.
        self.db = sqlite3.connect(self.path,check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS account (id INTEGER PRIMARY KEY, body TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, kind TEXT, body TEXT)')
        row = self.db.execute('SELECT body FROM account WHERE id=1').fetchone()
        self.state = json.loads(row[0]) if row else dict(cash=SETTINGS['initial_capital'],positions={},closed={},
            consumed=[],counts={},settled=[],running=False,risk_paused=False,day='',day_equity=1000.,
            settings=dict(SETTINGS))
        self.state['settings'] = normalized_settings(self.state.get('settings'))
        self.state.setdefault('settled',[])
        self.state['running'] = False
        self._save('STARTUP',{'message':'恢复新币破位模拟账户；新开仓暂停'})

    def _save(self,kind,payload,now=None):
        try:
            with self.db:
                self.db.execute('INSERT OR REPLACE INTO account VALUES(1,?)',(json.dumps(self.state,allow_nan=False),))
                self.db.execute('INSERT INTO events(timestamp,kind,body) VALUES(?,?,?)',
                    (self.clock() if now is None else now,kind,json.dumps(payload,allow_nan=False)))
        except Exception:
            row = self.db.execute('SELECT body FROM account WHERE id=1').fetchone()
            if row:self.state=json.loads(row[0])
            raise

    def log(self,kind,payload,*,now=None):
        with self.lock:self._save(kind,payload,now)

    def set_margin_per_trade(self,margin,*,now=None):
        now = self.clock() if now is None else now
        try:
            margin = float(margin)
        except (TypeError, ValueError):
            raise ValueError('INVALID_MARGIN_PER_TRADE') from None
        if not math.isfinite(margin) or margin < 1 or margin > self.state['settings']['initial_capital']:
            raise ValueError('INVALID_MARGIN_PER_TRADE')
        with self.lock:
            self.state['settings']['margin_per_trade'] = round(margin, 4)
            self._save('UPDATE_SETTINGS', {'margin_per_trade': self.state['settings']['margin_per_trade']}, now)
            return dict(self.state['settings'])

    def update_quotes(self,quotes,*,now=None):
        now = self.clock() if now is None else now
        with self.lock:
            for symbol,q in quotes.items():
                if (not all(math.isfinite(float(q[k])) and float(q[k])>0 for k in ('bid','ask','mark','timestamp'))
                        or q['bid']>q['ask'] or q['timestamp']>now+1):
                    raise ValueError('INVALID_QUOTE')
                if q['timestamp']>=self.quotes.get(symbol,{}).get('timestamp',0):
                    self.quotes[symbol]=dict(q)

    def _fresh(self,symbol,now):
        settings = self.state['settings']
        q=self.quotes.get(symbol)
        if not q or not -1<=now-q['timestamp']<=settings['quote_max_age']:
            raise ValueError('STALE_OR_MISSING_QUOTE')
        return q

    def _view(self,g,now):
        settings = self.state['settings']
        v=copy.deepcopy(g)
        q=self.quotes.get(g['symbol'])
        stale=not q or not -1<=now-q['timestamp']<=settings['quote_max_age']
        mark=q['mark'] if q else g.get('last_mark',g['entry_price'])
        pnl=(g['entry_price']-mark)*g['qty']
        fill=None if stale else q['ask']*(1+settings['slippage_bps']/10000)
        estimate=None if stale else (g['entry_price']-fill)*g['qty']-g['entry_fee']-fill*g['qty']*settings['fee_rate']+g['funding_pnl']
        v.update(mark_price=mark,price_pnl=pnl,net_pnl=pnl-g['entry_fee']+g['funding_pnl'],
            estimated_close_net=estimate,stale=bool(stale),holding_seconds=max(0,now-g['opened_at']),
            liquidation_price=None,liquidation_note='未模拟交易所强平与维持保证金')
        return v

    def _account(self,now):
        positions=[self._view(g,now) for g in self.state['positions'].values()]
        gross=sum(g['gross_notional'] for g in positions)
        margin=sum(g['initial_margin'] for g in positions)
        unrealized=sum(g['price_pnl'] for g in positions)
        equity=self.state['cash']+unrealized
        peak=max(self.state.get('peak',1000.),equity)
        return dict(cash=self.state['cash'],equity=equity,
            available=min(self.state['cash'],equity)-margin,gross_notional=gross,
            initial_margin=margin,unrealized=unrealized,
            realized=sum(g['net_pnl'] for g in self.state['closed'].values()),
            daily_pnl=equity-self.state['day_equity'],drawdown=max(0.,1-equity/peak),
            stale=any(g['stale'] for g in positions),position_count=len(positions))

    def _risk(self,now):
        settings = self.state['settings']
        account=self._account(now)
        if account['stale']:return
        day=time.strftime('%Y-%m-%d',time.gmtime(now))
        if self.state['day']!=day:
            self.state['day']=day
            self.state['day_equity']=self.state.get('last_equity',account['equity'])
            self.state['risk_paused']=False
        self.state['peak']=max(self.state.get('peak',1000.),account['equity'])
        self.state['last_equity']=account['equity']
        if account['equity']-self.state['day_equity']<=-settings['daily_loss_limit']:
            self.state['risk_paused']=True
            self.state['running']=False

    def set_running(self,running,*,now=None):
        now=self.clock() if now is None else now
        with self.lock:
            self._risk(now)
            if running and self.state['risk_paused']:raise ValueError('DAILY_LOSS')
            self.state['running']=bool(running)
            self._save('RESUME' if running else 'PAUSE',{'running':bool(running)},now)

    def open_signal(self,signal,rules,*,now=None):
        now=self.clock() if now is None else now
        with self.lock:
            settings = self.state['settings']
            self._risk(now)
            if self.state['risk_paused']:raise ValueError('DAILY_LOSS')
            if not self.state['running']:raise ValueError('PAUSED')
            if not isinstance(signal.get('signal_id'),str) or not isinstance(signal.get('symbol'),str):
                raise ValueError('INVALID_SIGNAL')
            if signal['signal_id'] in self.state['consumed']:raise ValueError('SIGNAL_USED')
            if not 0<=now-float(signal['signal_time'])<=15:raise ValueError('SIGNAL_EXPIRED')
            symbol=signal['symbol']
            if signal['signal_id']!=f"{symbol}:{time.strftime('%Y-%m-%d',time.gmtime(signal['signal_time']))}":
                raise ValueError('INVALID_SIGNAL')
            if not 1<=int(signal['listing_age_days'])<=30 or not 0<=float(signal['break_depth_pct'])<=settings['max_break_depth']:
                raise ValueError('INVALID_SIGNAL')
            if symbol in (g['symbol'] for g in self.state['positions'].values()):raise ValueError('SYMBOL_BUSY')
            if len(self.state['positions'])>=settings['max_positions']:raise ValueError('POSITION_LIMIT')
            if self.state['counts'].get(signal['signal_id'],0):raise ValueError('DAILY_LIMIT')
            account=self._account(now)
            if account['stale']:raise ValueError('STALE_PORTFOLIO')
            q=self._fresh(symbol,now)
            if (q['ask']-q['bid'])/((q['ask']+q['bid'])/2)>settings['max_spread_pct']:
                raise ValueError('SPREAD_TOO_WIDE')
            if not all(math.isfinite(float(rules[k])) and float(rules[k])>0 for k in ('tick_size','step_size','min_qty','max_qty','min_notional')):
                raise ValueError('INVALID_RULES')
            entry=q['bid']*(1-settings['slippage_bps']/10000)
            cap=settings['margin_per_trade']*settings['leverage']
            if account['available']<settings['margin_per_trade']:
                raise ValueError('CAPITAL_OR_RISK_BUDGET')
            qty=quantize(min(cap/entry,float(rules['max_qty'])),rules['step_size'])
            if qty<float(rules['min_qty']) or qty*entry<float(rules['min_notional']):
                raise ValueError('CAPITAL_OR_RISK_BUDGET')
            gross=qty*entry
            g=dict(id=signal['signal_id'],symbol=symbol,side='SHORT',status='OPEN',qty=qty,
                entry_price=entry,opened_at=now,gross_notional=gross,
                initial_margin=gross/settings['leverage'],leverage=settings['leverage'],
                entry_fee=gross*settings['fee_rate'],funding_pnl=0.,
                stop_price=entry*(1+settings['stop_pct']),target_price=entry*(1-settings['target_pct']),
                last_mark=q['mark'],signal=copy.deepcopy(signal),close_error=None)
            self.state['cash']-=g['entry_fee']
            self.state['positions'][g['id']]=g
            self.state['consumed'].append(g['id'])
            self.state['counts'][g['id']]=1
            self._save('OPEN_POSITION',g,now)
            return copy.deepcopy(g)

    def close_position(self,pid,*,now=None,reason='MANUAL_CLOSE'):
        now=self.clock() if now is None else now
        with self.lock:
            settings = self.state['settings']
            if pid in self.state['closed']:return copy.deepcopy(self.state['closed'][pid])
            if pid not in self.state['positions']:raise ValueError('POSITION_NOT_FOUND')
            g=self.state['positions'][pid]
            try:q=self._fresh(g['symbol'],now)
            except ValueError as exc:
                g['close_error']=str(exc)
                self._save('CLOSE_FAILED',{'id':pid,'reason':str(exc)},now)
                raise
            fill=q['ask']*(1+settings['slippage_bps']/10000)
            exit_fee=fill*g['qty']*settings['fee_rate']
            pnl=(g['entry_price']-fill)*g['qty']
            g=copy.deepcopy(g)
            g.update(status='CLOSED',closed_at=now,exit_price=fill,exit_fee=exit_fee,
                fees=g['entry_fee']+exit_fee,price_pnl=pnl,
                net_pnl=pnl-g['entry_fee']-exit_fee+g['funding_pnl'],exit_reason=reason,close_error=None)
            self.state['cash']+=pnl-exit_fee
            self.state['closed'][pid]=g
            del self.state['positions'][pid]
            self._risk(now)
            self._save('CLOSE_POSITION',g,now)
            return copy.deepcopy(g)

    def close_all(self,*,now=None):
        now=self.clock() if now is None else now
        with self.lock:
            self.state['running']=False
            self._save('CLOSE_ALL_REQUEST',{},now)
            result=dict(closed=[],failed=[])
            for pid in list(self.state['positions']):
                try:
                    self.close_position(pid,now=now,reason='MANUAL_CLOSE_ALL')
                    result['closed'].append(pid)
                except ValueError as exc:result['failed'].append(dict(id=pid,reason=str(exc)))
            return result

    def monitor(self,*,now=None):
        now=self.clock() if now is None else now
        with self.lock:
            settings = self.state['settings']
            self._risk(now)
            closed=[]
            for g in list(self.state['positions'].values()):
                try:q=self._fresh(g['symbol'],now)
                except ValueError:continue
                g['last_mark']=q['mark']
                buy=q['ask']
                reason=('STOP' if buy>=g['stop_price'] else 'TARGET' if buy<=g['target_price']
                    else 'TIME_EXIT' if now-g['opened_at']>=settings['max_hold_seconds'] else None)
                if reason:closed.append(self.close_position(g['id'],now=now,reason=reason))
            if now-self.last_equity_event>=60:
                self._save('EQUITY',self._account(now),now)
                self.last_equity_event=now
            return closed

    def settle_funding(self,symbol,timestamp,rate,mark_price,*,now=None):
        now=self.clock() if now is None else now
        if not all(math.isfinite(v) for v in (timestamp,rate,mark_price)) or mark_price<=0 or timestamp>now:
            raise ValueError('INVALID_FUNDING')
        with self.lock:
            key=f'{symbol}:{timestamp:.3f}'
            if key in self.state['settled']:return
            payments=[]
            for g in [*self.state['positions'].values(),*self.state['closed'].values()]:
                if g['symbol']==symbol and g['opened_at']<timestamp<=g.get('closed_at',now):
                    amount=g['qty']*mark_price*rate
                    g['funding_pnl']+=amount
                    self.state['cash']+=amount
                    if g['status']=='CLOSED':g['net_pnl']+=amount
                    payments.append(dict(id=g['id'],amount=amount))
            self.state['settled'].append(key)
            self._save('FUNDING',dict(symbol=symbol,timestamp=timestamp,payments=payments),now)

    def funding_windows(self):
        with self.lock:
            result={}
            for g in [*self.state['positions'].values(),*self.state['closed'].values()]:
                if not g.get('funding_finalized'):
                    result[g['symbol']]=min(result.get(g['symbol'],g['opened_at']),g['opened_at'])
            return result

    def mark_funding_checked(self,symbol,*,now):
        with self.lock:
            changed=False
            for g in self.state['closed'].values():
                if g['symbol']==symbol and not g.get('funding_finalized') and now>=g['closed_at']+72*3600:
                    g['funding_finalized']=True;changed=True
            if changed:self._save('FUNDING_RECONCILED',{'symbol':symbol},now)

    def snapshot(self,*,now=None):
        now=self.clock() if now is None else now
        with self.lock:
            important=list(self.db.execute("SELECT seq,timestamp,kind,body FROM events WHERE kind!='EQUITY' ORDER BY seq DESC LIMIT 80"))
            equity=list(self.db.execute("SELECT seq,timestamp,kind,body FROM events WHERE kind='EQUITY' ORDER BY seq DESC LIMIT 20"))
            event_rows=sorted(important+equity,key=lambda r:r[0],reverse=True)[:100]
            return dict(strategy_id='new_listing_breakdown',mode='PAPER',live_available=False,
                live_orders_sent=0,running=self.state['running'] and not self.state['risk_paused'],
                requested_running=self.state['running'],risk_paused=self.state['risk_paused'],
                settings=dict(self.state['settings']),timestamp=now,account=self._account(now),
                positions=[self._view(g,now) for g in self.state['positions'].values()],
                closed_positions=list(reversed(list(self.state['closed'].values())))[:100],
                events=[dict(seq=r[0],timestamp=r[1],kind=r[2],payload=json.loads(r[3])) for r in event_rows])

    def export_events(self):
        with self.lock:return list(self.db.execute('SELECT seq,timestamp,kind,body FROM events ORDER BY seq'))
