"""Transactional two-perpetual paper portfolio. No exchange order capability."""
from __future__ import annotations

import copy
import json
import math
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

from futures_strategy.pair_signals import entry_direction, spread_z


@dataclass(frozen=True)
class PairSettings:
    initial_capital: float = 1000.0
    reserve: float = 200.0
    max_gross: float = 800.0
    group_gross: float = 320.0
    min_group_gross: float = 50.0
    max_groups: int = 5
    daily_coin_limit: int = 3
    cooldown_seconds: int = 14400
    fee_rate: float = .0005
    slippage_bps: float = 2.0
    quote_max_age: int = 30
    entry_z: float = 2.2
    exit_z: float = .4
    stop_z: float = 4.0
    max_hold_hours: float = 48.0
    group_loss_limit: float = 3.0
    daily_loss_limit: float = 10.0
    max_drawdown: float = .05


def _day(now: float) -> str:
    return datetime.fromtimestamp(now, timezone(timedelta(hours=8))).date().isoformat()


class PairPortfolio:
    def __init__(self, path: str | Path, settings: PairSettings | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.settings = settings or PairSettings()
        self.lock = threading.RLock()
        self.quotes: dict = {}
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS portfolio (id INTEGER PRIMARY KEY, body TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, kind TEXT, body TEXT)')
        row = self.db.execute('SELECT body FROM portfolio WHERE id=1').fetchone()
        self.state = json.loads(row[0]) if row else dict(cash=self.settings.initial_capital,
            peak=self.settings.initial_capital, groups={}, closed={}, counts={}, cooldowns={},
            settled=[], running=False, circuit=False, day='', day_equity=self.settings.initial_capital,
            settings=asdict(self.settings))
        if row and self.state.get('settings') != asdict(self.settings):
            raise ValueError('SETTINGS_MISMATCH: use the original settings or a new output directory')
        self.state['running'] = False
        self._save('STARTUP', {'mode': 'PAPER', 'message': '恢复模拟账户；新开仓已暂停'})

    def _save(self, kind: str, payload: dict, now: float | None = None):
        """Commit account and audit together; roll memory back if disk write fails."""
        try:
            with self.db:
                self.db.execute('INSERT OR REPLACE INTO portfolio VALUES (1, ?)',
                                (json.dumps(self.state, allow_nan=False),))
                self.db.execute('INSERT INTO events(timestamp,kind,body) VALUES (?,?,?)',
                                (time.time() if now is None else now, kind, json.dumps(payload, allow_nan=False)))
        except Exception:
            row = self.db.execute('SELECT body FROM portfolio WHERE id=1').fetchone()
            if row:
                self.state = json.loads(row[0])
            raise

    def log(self, kind: str, payload: dict):
        with self.lock:
            self._save(kind, payload)

    def set_running(self, running: bool):
        with self.lock:
            if running and self.state['circuit']:
                raise ValueError('DRAWDOWN_CIRCUIT')
            self.state['running'] = bool(running)
            self._save('RESUME' if running else 'PAUSE', {'message': '开始扫描开仓' if running else '暂停新开仓，持仓退出监控继续'})

    def update_quotes(self, quotes: dict, *, now: float | None = None):
        now = time.time() if now is None else now
        with self.lock:
            for symbol, q in quotes.items():
                if (not all(math.isfinite(float(q[k])) and float(q[k]) > 0 for k in ['bid', 'ask', 'mark', 'timestamp'])
                    or q['bid'] > q['ask'] or q['timestamp'] > now+5):
                    raise ValueError(f'INVALID_QUOTE: {symbol}')
            for symbol, q in quotes.items():
                if q['timestamp'] >= self.quotes.get(symbol, {}).get('timestamp', 0):
                    self.quotes[symbol] = dict(q)

    def _fresh(self, symbol: str, now: float) -> dict:
        q = self.quotes.get(symbol)
        if not q or not -5 <= now-q['timestamp'] <= self.settings.quote_max_age:
            raise ValueError(f'STALE_OR_MISSING_QUOTE: {symbol}')
        return q

    def _fill(self, symbol: str, side: str, opening: bool, now: float) -> float:
        q = self._fresh(symbol, now)
        buy = (side == 'LONG') == opening
        return q['ask']*(1+self.settings.slippage_bps/10000) if buy else q['bid']*(1-self.settings.slippage_bps/10000)

    def _group_view(self, g: dict, now: float) -> dict:
        result = copy.deepcopy(g)
        price_pnl = exit_fees = exit_pnl = 0.0
        stale = False
        marks = []
        for leg in result['legs']:
            q = self.quotes.get(leg['symbol'])
            valid = q is not None and -5 <= now-q['timestamp'] <= self.settings.quote_max_age
            stale |= not valid
            mark = q['mark'] if q else leg.get('last_mark', leg['entry_price'])
            leg['mark_price'] = mark
            leg['stale'] = not valid
            leg['quote_time'] = q['timestamp'] if q else None
            sign = 1 if leg['side'] == 'LONG' else -1
            leg['pnl'] = sign*(mark-leg['entry_price'])*leg['qty']
            price_pnl += leg['pnl']
            marks.append(mark)
            if valid:
                fill = self._fill(leg['symbol'], leg['side'], False, now)
                exit_pnl += sign*(fill-leg['entry_price'])*leg['qty']
                exit_fees += fill*leg['qty']*self.settings.fee_rate
        result.update(price_pnl=price_pnl, net_pnl=price_pnl+g['funding_pnl']-g['entry_fee'],
                      estimated_close_net=None if stale else exit_pnl+g['funding_pnl']-g['entry_fee']-exit_fees,
                      estimated_exit_fee=None if stale else exit_fees,
                      stale=stale, z=None if stale else spread_z(g['model'], *marks),
                      holding_hours=max(0, now-g['opened_at'])/3600)
        return result

    def _account(self, now: float) -> dict:
        groups = [self._group_view(g, now) for g in self.state['groups'].values()]
        gross = sum(g['gross_notional'] for g in groups)
        equity = self.state['cash'] + sum(g['price_pnl'] for g in groups)
        peak = max(self.state['peak'], equity)
        return dict(equity=equity, cash=self.state['cash'], available=self.state['cash']-gross,
                    gross_notional=gross, unrealized=sum(g['price_pnl'] for g in groups),
                    realized=sum(g['net_pnl'] for g in self.state['closed'].values()),
                    drawdown=max(0, 1-equity/peak), daily_pnl=equity-self.state['day_equity'],
                    stale=any(g['stale'] for g in groups), group_count=len(groups))

    def _risk(self, now: float):
        account = self._account(now)
        if account['stale']:
            return
        self.state['peak'] = max(self.state['peak'], account['equity'])
        if self.state['day'] != _day(now):
            self.state['day'] = _day(now)
            self.state['day_equity'] = account['equity']
        if account['drawdown'] >= self.settings.max_drawdown:
            self.state['circuit'] = True
            self.state['running'] = False
        if account['equity']-self.state['day_equity'] <= -self.settings.daily_loss_limit:
            self.state['running'] = False

    def open_group(self, model: dict, *, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        with self.lock:
            self._risk(now)
            if not self.state['running']:
                self._save('ENTRY_BLOCKED', {'reason': 'PAUSED_OR_RISK'}, now)
                raise ValueError('PAUSED_OR_RISK')
            if self._account(now)['stale']:
                raise ValueError('STALE_PORTFOLIO_VALUATION')
            s = self.settings
            symbols = [model['symbol_a'], model['symbol_b']]
            if len(set(symbols)) != 2:
                raise ValueError('SAME_SYMBOL')
            occupied = {l['symbol'] for g in self.state['groups'].values() for l in g['legs']}
            if occupied.intersection(symbols):
                raise ValueError('SYMBOL_BUSY')
            if len(self.state['groups']) >= s.max_groups:
                raise ValueError('GROUP_LIMIT')
            keys = [f'{_day(now)}:{symbol}' for symbol in symbols]
            if any(self.state['counts'].get(k, 0) >= s.daily_coin_limit for k in keys):
                raise ValueError('DAILY_LIMIT')
            if any(now < self.state['cooldowns'].get(symbol, 0) for symbol in symbols):
                raise ValueError('COOLDOWN')
            beta = model['beta']
            if not all(math.isfinite(float(model[k])) for k in ['beta','alpha','mean','std','z','previous_z']):
                raise ValueError('INVALID_MODEL')
            if not .2 <= beta <= 5 or model['std'] <= 0:
                raise ValueError('INVALID_MODEL')
            direction = entry_direction(model['z'], model['previous_z'], s.entry_z, s.stop_z)
            if not direction:
                raise ValueError('NO_CONVERGENCE_SIGNAL')
            edge = (abs(model['z'])-s.exit_z)*model['std']/(1+beta)
            costs = 2*s.fee_rate+2*s.slippage_bps/10000+.0005
            if edge <= costs:
                raise ValueError('EDGE_BELOW_COST')
            gross_used = sum(g['gross_notional'] for g in self.state['groups'].values())
            gross = min(s.group_gross, 1000., s.max_gross-gross_used,
                        (self.state['cash']-gross_used-s.reserve)/(1+s.fee_rate))
            # Size such that expansion from entry Z to stop Z is inside group risk budget.
            stop_fraction = (s.stop_z-abs(model['z']))*model['std']/(1+beta)+costs
            gross = min(gross, s.group_loss_limit/stop_fraction)
            if gross < s.min_group_gross:
                raise ValueError('CAPITAL_OR_RISK_BUDGET')
            side_a = 'SHORT' if direction == 'SHORT_A_LONG_B' else 'LONG'
            sides = [side_a, 'LONG' if side_a == 'SHORT' else 'SHORT']
            fills = [self._fill(symbol, side, True, now) for symbol, side in zip(symbols, sides)]
            legs = [dict(symbol=symbol, side=side, qty=notional/fill, entry_price=fill, last_mark=self.quotes[symbol]['mark'])
                    for symbol, side, fill, notional in zip(symbols, sides, fills, [gross/(1+beta), gross*beta/(1+beta)])]
            group = dict(id=uuid.uuid4().hex[:16], opened_at=now, status='OPEN', model=dict(model), legs=legs,
                         entry_z=model['z'], gross_notional=gross, entry_fee=gross*s.fee_rate, funding_pnl=0., close_error=None)
            self.state['cash'] -= group['entry_fee']
            self.state['groups'][group['id']] = group
            for key in keys:
                self.state['counts'][key] = self.state['counts'].get(key, 0)+1
            self._save('OPEN_GROUP', group, now)
            return copy.deepcopy(group)

    def close_group(self, group_id: str, *, reason: str = 'MANUAL_CLOSE', now: float | None = None) -> dict:
        now = time.time() if now is None else now
        with self.lock:
            if group_id in self.state['closed']:
                return copy.deepcopy(self.state['closed'][group_id])
            if group_id not in self.state['groups']:
                raise ValueError('GROUP_NOT_FOUND')
            original = self.state['groups'][group_id]
            try:
                fills = [self._fill(l['symbol'], l['side'], False, now) for l in original['legs']]
            except ValueError as exc:
                original['close_error'] = str(exc)
                self._save('CLOSE_FAILED', {'id': group_id, 'reason': str(exc)}, now)
                raise
            g = copy.deepcopy(original)
            pnl = fees = 0.
            for leg, fill in zip(g['legs'], fills):
                leg['exit_price'] = fill
                leg['pnl'] = (1 if leg['side'] == 'LONG' else -1)*(fill-leg['entry_price'])*leg['qty']
                pnl += leg['pnl']
                fees += fill*leg['qty']*self.settings.fee_rate
                self.state['cooldowns'][leg['symbol']] = now+self.settings.cooldown_seconds
            g.update(status='CLOSED', closed_at=now, exit_reason=reason, price_pnl=pnl,
                     exit_fee=fees, fees=g['entry_fee']+fees, net_pnl=pnl+g['funding_pnl']-g['entry_fee']-fees, close_error=None)
            self.state['cash'] += pnl-fees
            del self.state['groups'][group_id]
            self.state['closed'][group_id] = g
            self._risk(now)
            self._save('CLOSE_GROUP', g, now)
            return copy.deepcopy(g)

    def close_all(self, *, now: float | None = None, reason: str = 'MANUAL_CLOSE_ALL') -> dict:
        now = time.time() if now is None else now
        with self.lock:
            self.state['running'] = False
            self._save('CLOSE_ALL_REQUEST', {'groups': list(self.state['groups'])}, now)
            result = {'closed': [], 'failed': []}
            for gid in list(self.state['groups']):
                try:
                    self.close_group(gid, reason=reason, now=now)
                    result['closed'].append(gid)
                except ValueError as exc:
                    result['failed'].append({'id': gid, 'reason': str(exc)})
            return result

    def settle_funding(self, symbol: str, timestamp: float, rate: float, mark_price: float, *, now: float | None = None):
        now = time.time() if now is None else now
        with self.lock:
            if not all(math.isfinite(v) for v in [timestamp, rate, mark_price]) or mark_price <= 0:
                raise ValueError('INVALID_FUNDING')
            if timestamp > now:
                raise ValueError('FUTURE_FUNDING')
            key = f'{symbol}:{timestamp:.3f}'
            if key in self.state['settled']:
                return
            payments = []
            for g in [*self.state['groups'].values(), *self.state['closed'].values()]:
                if g['opened_at'] >= timestamp or timestamp > g.get('closed_at', now):
                    continue
                for leg in g['legs']:
                    if leg['symbol'] == symbol:
                        amount = leg['qty']*mark_price*rate*(1 if leg['side'] == 'SHORT' else -1)
                        self.state['cash'] += amount
                        g['funding_pnl'] += amount
                        if g['status'] == 'CLOSED':
                            g['net_pnl'] += amount
                        payments.append({'group_id': g['id'], 'amount': amount})
            if payments:
                self.state['settled'].append(key)
                self._save('FUNDING', {'symbol': symbol, 'settlement_time': timestamp, 'rate': rate, 'payments': payments}, now)

    def funding_windows(self) -> dict[str, float]:
        """Complete durable backlog, independent of the UI's last-100 display limit."""
        with self.lock:
            since = {}
            for g in [*self.state['groups'].values(), *self.state['closed'].values()]:
                if g.get('funding_finalized'):
                    continue
                for leg in g['legs']:
                    since[leg['symbol']] = min(since.get(leg['symbol'], g['opened_at']), g['opened_at'])
            return since

    def mark_funding_checked(self, symbol: str, *, now: float):
        """Only finalize after both legs have a successful post-publication-window check."""
        with self.lock:
            changed = False
            for g in self.state['closed'].values():
                if g.get('funding_finalized') or now < g['closed_at']+72*3600:
                    continue
                for leg in g['legs']:
                    if leg['symbol'] == symbol:
                        leg['funding_checked_at'] = now
                        changed = True
                if all(l.get('funding_checked_at', 0) >= g['closed_at']+72*3600 for l in g['legs']):
                    g['funding_finalized'] = True
            if changed:
                self._save('FUNDING_RECONCILED', {'symbol': symbol, 'checked_until': now}, now)

    def monitor(self, *, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        with self.lock:
            self._risk(now)
            result = []
            for g in list(self.state['groups'].values()):
                view = self._group_view(g, now)
                if view['stale']:
                    continue
                for leg in g['legs']:
                    leg['last_mark'] = self.quotes[leg['symbol']]['mark']
                reason = None
                if self.state['circuit']:
                    reason = 'ACCOUNT_DRAWDOWN'
                elif view['estimated_close_net'] <= -self.settings.group_loss_limit:
                    reason = 'GROUP_LOSS_LIMIT'
                elif abs(view['z']) >= self.settings.stop_z:
                    reason = 'SPREAD_STOP'
                elif view['z']*(1 if g['entry_z'] > 0 else -1) <= self.settings.exit_z:
                    reason = 'MEAN_REVERSION'
                elif view['holding_hours'] >= self.settings.max_hold_hours:
                    reason = 'TIME_EXIT'
                if reason:
                    result.append(self.close_group(g['id'], reason=reason, now=now))
            self._save('EQUITY', self._account(now), now)
            return result

    def snapshot(self, *, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        with self.lock:
            events = [dict(seq=r[0], timestamp=r[1], kind=r[2], payload=json.loads(r[3]))
                      for r in self.db.execute('SELECT seq,timestamp,kind,body FROM events ORDER BY seq DESC LIMIT 100')]
            history = [dict(timestamp=r[0], equity=body['equity']) for r in
                       self.db.execute("SELECT timestamp,body FROM events WHERE kind='EQUITY' ORDER BY seq DESC LIMIT 180")
                       if not (body := json.loads(r[1])).get('stale')]
            return dict(mode='PAPER', live_orders_sent=0, running=self.state['running'], circuit=self.state['circuit'],
                        settings=asdict(self.settings), account=self._account(now), timestamp=now,
                        groups=[self._group_view(g, now) for g in self.state['groups'].values()],
                        closed_groups=list(reversed(list(self.state['closed'].values())))[:100],
                        events=events, equity_history=list(reversed(history)))

    def export_events(self) -> list:
        with self.lock:
            return list(self.db.execute('SELECT seq,timestamp,kind,body FROM events ORDER BY seq'))
