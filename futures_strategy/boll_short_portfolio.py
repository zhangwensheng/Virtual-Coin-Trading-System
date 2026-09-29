"""Independent, transactional single-short PAPER account. No exchange orders."""
from __future__ import annotations

import copy
import json
import math
import sqlite3
import threading
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from pathlib import Path

from futures_strategy.pair_portfolio import _day


SETTINGS = dict(initial_capital=1000., reserve=200., group_gross=320., max_gross=640.,
                max_positions=2, risk_per_trade=3., risk_fraction=.003, daily_loss_limit=10.,
                daily_coin_limit=3, cooldown_seconds=1800, max_hold_seconds=1800,
                fee_rate=.0005, slippage_bps=2., quote_max_age=5, leverage=1)


def quantize(value, step, *, up=False):
    a, b = Decimal(str(value)), Decimal(str(step))
    if not a.is_finite() or not b.is_finite() or b <= 0:
        raise ValueError('INVALID_PRECISION')
    return float((a/b).to_integral_value(rounding=ROUND_UP if up else ROUND_DOWN)*b)


class BollPortfolio:
    def __init__(self, path, *, leverage=1):
        if leverage != 1:
            raise ValueError('LEVERAGE_UNAVAILABLE: 维持保证金档位未接入，当前仅支持1倍模拟')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.clock = time.time
        self.quotes = {}
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS account (id INTEGER PRIMARY KEY, body TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, kind TEXT, body TEXT)')
        row = self.db.execute('SELECT body FROM account WHERE id=1').fetchone()
        self.state = json.loads(row[0]) if row else dict(cash=1000., positions={}, closed={},
            consumed=[], counts={}, cooldowns={}, settled=[], running=False, risk_paused=False,
            day='', day_equity=1000., peak=1000., settings=SETTINGS)
        if self.state['settings'] != SETTINGS:
            self.db.close()
            raise ValueError('SETTINGS_MISMATCH')
        self.state['running'] = False
        self._save('STARTUP', {'message': '恢复BOLL模拟账户；新开仓暂停'})

    def _save(self, kind, payload, now=None):
        try:
            with self.db:
                self.db.execute('INSERT OR REPLACE INTO account VALUES(1,?)', (json.dumps(self.state, allow_nan=False),))
                self.db.execute('INSERT INTO events(timestamp,kind,body) VALUES(?,?,?)',
                    (self.clock() if now is None else now, kind, json.dumps(payload, allow_nan=False)))
        except Exception:
            row = self.db.execute('SELECT body FROM account WHERE id=1').fetchone()
            if row:
                self.state = json.loads(row[0])
            raise

    def log(self, kind, payload, *, now=None):
        with self.lock:
            self._save(kind, payload, now)

    def update_quotes(self, quotes, *, now=None):
        now = self.clock() if now is None else now
        with self.lock:
            for symbol, q in quotes.items():
                if (not all(math.isfinite(float(q[k])) and float(q[k]) > 0 for k in ['bid','ask','mark','timestamp'])
                    or q['bid'] > q['ask'] or q['timestamp'] > now+1):
                    raise ValueError('INVALID_QUOTE')
            for symbol, q in quotes.items():
                if q['timestamp'] >= self.quotes.get(symbol, {}).get('timestamp', 0):
                    self.quotes[symbol] = dict(q)

    def _fresh(self, symbol, now):
        q = self.quotes.get(symbol)
        if not q or not -1 <= now-q['timestamp'] <= SETTINGS['quote_max_age']:
            raise ValueError('STALE_OR_MISSING_QUOTE')
        return q

    def _view(self, g, now):
        v = copy.deepcopy(g)
        q = self.quotes.get(g['symbol'])
        stale = not q or not -1 <= now-q['timestamp'] <= SETTINGS['quote_max_age']
        mark = q['mark'] if q else g.get('last_mark', g['entry_price'])
        pnl = (g['entry_price']-mark)*g['qty']
        fill = None if stale else q['ask']*(1+SETTINGS['slippage_bps']/10000)
        estimate = None if stale else (g['entry_price']-fill)*g['qty']-g['entry_fee']-fill*g['qty']*SETTINGS['fee_rate']+g['funding_pnl']
        v.update(mark_price=mark, price_pnl=pnl, net_pnl=pnl-g['entry_fee']+g['funding_pnl'],
                 estimated_close_net=estimate, stale=bool(stale), holding_seconds=max(0,now-g['opened_at']),
                 liquidation_price=None, liquidation_note='未接入交易所维持保证金档位，强平价不可用')
        return v

    def _account(self, now):
        positions = [self._view(g,now) for g in self.state['positions'].values()]
        gross = sum(g['gross_notional'] for g in positions)
        margin = sum(g['initial_margin'] for g in positions)
        unrealized = sum(g['price_pnl'] for g in positions)
        equity = self.state['cash']+unrealized
        peak = max(self.state['peak'], equity)
        return dict(cash=self.state['cash'], equity=equity, available=min(self.state['cash'],equity)-margin,
            gross_notional=gross, initial_margin=margin, unrealized=unrealized,
            realized=sum(g['net_pnl'] for g in self.state['closed'].values()),
            daily_pnl=equity-self.state['day_equity'], drawdown=max(0.,1-equity/peak),
            stale=any(g['stale'] for g in positions), position_count=len(positions))

    def _risk(self, now):
        a = self._account(now)
        if a['stale']:
            return
        if self.state['day'] != _day(now):
            # When holding over midnight, retain the last observed equity as baseline.
            self.state['day_equity'] = self.state.get('last_equity', a['equity'])
            self.state['day'] = _day(now)
            self.state['risk_paused'] = False
        self.state['peak'] = max(self.state['peak'], a['equity'])
        self.state['last_equity'] = a['equity']
        if a['equity']-self.state['day_equity'] <= -SETTINGS['daily_loss_limit']:
            self.state['risk_paused'] = True

    def set_running(self, running, *, now=None):
        now = self.clock() if now is None else now
        with self.lock:
            self._risk(now)
            if running and self.state['risk_paused']:
                self._save('ENTRY_BLOCKED', {'reason':'DAILY_LOSS'}, now)
                raise ValueError('DAILY_LOSS')
            self.state['running'] = bool(running)
            self._save('RESUME' if running else 'PAUSE', {'running':bool(running)}, now)

    def open_signal(self, signal, rules, *, now=None):
        now = self.clock() if now is None else now
        with self.lock:
            self._risk(now)
            if self.state['risk_paused']:
                self._save('ENTRY_BLOCKED', {'reason':'DAILY_LOSS'}, now)
                raise ValueError('DAILY_LOSS')
            if not self.state['running']:
                raise ValueError('PAUSED')
            if not isinstance(signal.get('signal_id'), str) or not signal['signal_id'] or not isinstance(signal.get('symbol'), str):
                raise ValueError('INVALID_SIGNAL')
            if signal['signal_id'] in self.state['consumed']:
                raise ValueError('SIGNAL_USED')
            if not all(math.isfinite(float(signal[k])) and float(signal[k]) > 0 for k in ['signal_time','peak','atr']):
                raise ValueError('INVALID_SIGNAL')
            if not 0 <= now-signal['signal_time'] <= 15:
                raise ValueError('SIGNAL_EXPIRED')
            symbol = signal['symbol']
            if any(g['symbol']==symbol for g in self.state['positions'].values()):
                raise ValueError('SYMBOL_BUSY')
            if len(self.state['positions']) >= 2:
                raise ValueError('POSITION_LIMIT')
            key = f'{_day(now)}:{symbol}'
            if self.state['counts'].get(key,0) >= 3:
                raise ValueError('DAILY_LIMIT')
            if now < self.state['cooldowns'].get(symbol,0):
                raise ValueError('COOLDOWN')
            account = self._account(now)
            if account['stale']:
                raise ValueError('STALE_PORTFOLIO')
            q = self._fresh(symbol, now)
            if (q['ask']-q['bid'])/((q['ask']+q['bid'])/2) > .001:
                raise ValueError('SPREAD_TOO_WIDE')
            if not all(math.isfinite(float(rules[k])) and float(rules[k]) > 0 for k in ['tick_size','step_size','min_qty','min_notional','max_qty']):
                raise ValueError('INVALID_RULES')
            fee, slip = SETTINGS['fee_rate'], SETTINGS['slippage_bps']/10000
            entry = q['bid']*(1-slip)
            stop = quantize(signal['peak']+max(.2*signal['atr'],2*rules['tick_size']),rules['tick_size'],up=True)
            r = stop-entry
            target = quantize(entry-2*r,rules['tick_size'],up=True)
            if r <= 0 or target <= 0 or target*(1+slip)*(1+fee) >= entry*(1-fee):
                raise ValueError('INVALID_STOP_OR_EDGE')
            unit_risk = stop*(1+slip)-entry+fee*(entry+stop*(1+slip))
            risk = min(3., account['equity']*.003)
            notional_cap = min(320.,640-account['gross_notional'],(account['available']-200)/(1+fee))
            qty = quantize(min(risk/unit_risk,notional_cap/entry,rules['max_qty']),rules['step_size'])
            if qty < rules['min_qty'] or qty*entry < rules['min_notional']:
                raise ValueError('CAPITAL_OR_RISK_BUDGET')
            gid = signal['signal_id']
            g = dict(id=gid,symbol=symbol,side='SHORT',status='OPEN',qty=qty,entry_price=entry,
                opened_at=now,gross_notional=qty*entry,initial_margin=qty*entry,leverage=1,
                entry_fee=qty*entry*fee,funding_pnl=0.,stop_price=stop,initial_stop=stop,
                target_price=target,risk_distance=r,planned_risk=qty*unit_risk,trailing=False,
                lowest_buy=None,last_mark=q['mark'],signal=copy.deepcopy(signal),close_error=None)
            self.state['cash'] -= g['entry_fee']
            self.state['positions'][gid] = g
            self.state['consumed'].append(gid)
            self.state['counts'][key] = self.state['counts'].get(key,0)+1
            self._save('OPEN_POSITION',g,now)
            return copy.deepcopy(g)

    def close_position(self, pid, *, now=None, reason='MANUAL_CLOSE'):
        now = self.clock() if now is None else now
        with self.lock:
            if pid in self.state['closed']:
                return copy.deepcopy(self.state['closed'][pid])
            if pid not in self.state['positions']:
                raise ValueError('POSITION_NOT_FOUND')
            g = self.state['positions'][pid]
            try:
                q = self._fresh(g['symbol'],now)
            except ValueError as exc:
                g['close_error'] = str(exc)
                self._save('CLOSE_FAILED',{'id':pid,'reason':str(exc)},now)
                raise
            fill = q['ask']*(1+SETTINGS['slippage_bps']/10000)
            exit_fee = fill*g['qty']*SETTINGS['fee_rate']
            pnl = (g['entry_price']-fill)*g['qty']
            g = copy.deepcopy(g)
            g.update(status='CLOSED',closed_at=now,exit_price=fill,exit_fee=exit_fee,
                fees=g['entry_fee']+exit_fee,price_pnl=pnl,net_pnl=pnl-g['entry_fee']-exit_fee+g['funding_pnl'],
                exit_reason=reason,close_error=None)
            self.state['cash'] += pnl-exit_fee
            self.state['closed'][pid] = g
            del self.state['positions'][pid]
            self.state['cooldowns'][g['symbol']] = now+1800
            self._risk(now)
            self._save('CLOSE_POSITION',g,now)
            return copy.deepcopy(g)

    def close_all(self, *, now=None):
        now = self.clock() if now is None else now
        with self.lock:
            self.state['running'] = False
            self._save('CLOSE_ALL_REQUEST',{},now)
            result = dict(closed=[],failed=[])
            for pid in list(self.state['positions']):
                try:
                    self.close_position(pid,now=now,reason='MANUAL_CLOSE_ALL')
                    result['closed'].append(pid)
                except ValueError as exc:
                    result['failed'].append(dict(id=pid,reason=str(exc)))
            return result

    def monitor(self, *, now=None):
        now = self.clock() if now is None else now
        with self.lock:
            self._risk(now)
            closed = []
            for g in list(self.state['positions'].values()):
                try:
                    q = self._fresh(g['symbol'],now)
                except ValueError:
                    continue
                g['last_mark'] = q['mark']
                buy = q['ask']
                reason = ('STOP' if buy >= g['stop_price'] else 'TARGET' if buy <= g['target_price']
                          else 'TIME_EXIT' if now-g['opened_at'] >= 1800 else None)
                if reason:
                    closed.append(self.close_position(g['id'],now=now,reason=reason))
                    continue
                if buy <= g['entry_price']-g['risk_distance']:
                    g['trailing'] = True
                if g['trailing']:
                    g['lowest_buy'] = min(g['lowest_buy'] or buy,buy)
                    g['stop_price'] = min(g['stop_price'],g['lowest_buy']+g['risk_distance'])
            self._save('EQUITY',self._account(now),now)
            return closed

    def settle_funding(self, symbol, timestamp, rate, mark_price, *, now=None):
        now = self.clock() if now is None else now
        if not all(math.isfinite(v) for v in [timestamp,rate,mark_price]) or mark_price <= 0 or timestamp > now:
            raise ValueError('INVALID_FUNDING')
        with self.lock:
            key = f'{symbol}:{timestamp:.3f}'
            if key in self.state['settled']:
                return
            payments = []
            for g in [*self.state['positions'].values(),*self.state['closed'].values()]:
                if g['symbol']==symbol and g['opened_at'] < timestamp <= g.get('closed_at',now):
                    amount = g['qty']*mark_price*rate
                    g['funding_pnl'] += amount
                    self.state['cash'] += amount
                    if g['status']=='CLOSED':
                        g['net_pnl'] += amount
                    payments.append(dict(id=g['id'],amount=amount))
            if payments:
                self.state['settled'].append(key)
                self._save('FUNDING',dict(symbol=symbol,timestamp=timestamp,payments=payments),now)

    def funding_windows(self):
        with self.lock:
            result = {}
            for g in [*self.state['positions'].values(),*self.state['closed'].values()]:
                if not g.get('funding_finalized'):
                    result[g['symbol']] = min(result.get(g['symbol'],g['opened_at']),g['opened_at'])
            return result

    def mark_funding_checked(self, symbol, *, now):
        with self.lock:
            changed = False
            for g in self.state['closed'].values():
                if g['symbol']==symbol and not g.get('funding_finalized') and now >= g['closed_at']+72*3600:
                    g['funding_finalized'] = True
                    changed = True
            if changed:
                self._save('FUNDING_RECONCILED',{'symbol':symbol},now)

    def snapshot(self, *, now=None):
        now = self.clock() if now is None else now
        with self.lock:
            return dict(strategy_id='boll_short',mode='PAPER',live_available=False,live_orders_sent=0,
                running=self.state['running'] and not self.state['risk_paused'],requested_running=self.state['running'],
                risk_paused=self.state['risk_paused'],settings=dict(SETTINGS),timestamp=now,
                account=self._account(now),positions=[self._view(g,now) for g in self.state['positions'].values()],
                closed_positions=list(reversed(list(self.state['closed'].values())))[:100],
                events=[dict(seq=r[0],timestamp=r[1],kind=r[2],payload=json.loads(r[3])) for r in
                        self.db.execute('SELECT seq,timestamp,kind,body FROM events ORDER BY seq DESC LIMIT 100')])

    def export_events(self):
        with self.lock:
            return list(self.db.execute('SELECT seq,timestamp,kind,body FROM events ORDER BY seq'))
