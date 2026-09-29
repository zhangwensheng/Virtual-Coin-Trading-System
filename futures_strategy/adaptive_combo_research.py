"""Isolated, bar-based research engine; never imported by the trading workstation.

All signals use completed candles. Fills occur at a subsequent minute open.
Pair regressions use the preceding seven days only and freeze on entry.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from futures_strategy.indicators import adx, atr, ema


def closed_bars(frame, rule):
    return frame.resample(rule, closed='left', label='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna()


def fill_price(price, direction, slippage):
    return price * (1 + direction * slippage)


def funding_cashflow(qty, mark, rates):
    return float(-np.sum(qty * mark * rates))


def bar_exit(side, open_, high, low, stop, target):
    # Adverse-first convention when both levels occur in an OHLC candle.
    if side == 1:
        if low <= stop:
            return min(open_, stop), 'stop'
        if high >= target:
            return target, 'target'
    else:
        if high >= stop:
            return max(open_, stop), 'stop'
        if low <= target:
            return target, 'target'
    return None


def can_enter(symbols, counts, occupied, daily_entries):
    return daily_entries < 20 and not any(s in occupied or counts.get(s, 0) >= 3 for s in symbols)


def prepare_features(frame):
    f = frame.copy()
    h = closed_bars(f, '1h')
    h['fast'] = ema(h.close, 20)
    h['slow'] = ema(h.close, 60)
    h['slope'] = h.fast.diff(3)
    h['ready'] = np.arange(len(h)) >= 60
    q = closed_bars(f, '15min')
    q['adx'] = adx(q, 14)
    q['trend'] = (q.adx > 25) & (q.adx.shift(1) > 25)
    q['range'] = (q.adx < 18) & (q.adx.shift(1) < 18)
    q['ceiling'] = q.high.rolling(32).max()
    q['floor'] = q.low.rolling(32).min()
    b = closed_bars(f, '5min')
    b['ema'] = ema(b.close, 20)
    b['atr'] = atr(b, 14)
    b['volume_ratio'] = b.volume / b.volume.rolling(20).median().shift(1)
    b['pull_long'] = ((b.low <= b.ema + .2 * b.atr) & (b.close > b.ema) & (b.volume_ratio < 1.2)).rolling(3).max()
    b['pull_short'] = ((b.high >= b.ema - .2 * b.atr) & (b.close < b.ema) & (b.volume_ratio < 1.2)).rolling(3).max()
    b['swing_low'] = b.low.rolling(3).min()
    b['swing_high'] = b.high.rolling(3).max()
    for prefix, source, columns in (
        ('h_', h, ['fast', 'slow', 'slope', 'ready']),
        ('q_', q, ['adx', 'trend', 'range', 'ceiling', 'floor']),
        ('b_', b, ['ema', 'atr', 'pull_long', 'pull_short', 'swing_low', 'swing_high']),
    ):
        aligned = source[columns].reindex(f.index, method='ffill')
        for col in columns:
            f[prefix + col] = aligned[col]
    f['atr'] = atr(f, 14)
    hi, lo = f.high.rolling(14).max(), f.low.rolling(14).min()
    f['wr'] = -100 * (hi - f.close) / (hi - lo).replace(0, np.nan)
    f['volume_ratio'] = f.volume / f.volume.rolling(20).median().shift(1)
    f['minute_high'] = f.high.rolling(3).max().shift(1)
    f['minute_low'] = f.low.rolling(3).min().shift(1)
    f['ready'] = f.h_ready.eq(True) & f.b_atr.notna()
    return f


def single_signals(symbol, f):
    """Return raw signals with their state eligibility for fixed/adaptive A/B."""
    signals = []
    for side in (1, -1):
        if side == 1:
            mask = (f.h_fast > f.h_slow) & (f.h_slope > 0) & f.b_pull_long.eq(1)
            mask &= (f.close > f.minute_high) & (f.close > f.b_ema)
            stops = f.b_swing_low - .2 * f.atr
        else:
            mask = (f.h_fast < f.h_slow) & (f.h_slope < 0) & f.b_pull_short.eq(1)
            mask &= (f.close < f.minute_low) & (f.close < f.b_ema)
            stops = f.b_swing_high + .2 * f.atr
        mask &= f.ready & ((f.close - f.b_ema).abs() < 1.5 * f.b_atr)
        for i in np.flatnonzero(mask.to_numpy()):
            risk = side * (f.close.iloc[i] - stops.iloc[i])
            if .0015 <= risk / f.close.iloc[i] <= .02:
                signals.append(dict(i=int(i), strategy='trend', symbols=[symbol], side=side,
                                    stop=float(stops.iloc[i]), target=float(f.close.iloc[i] + side * 3 * risk),
                                    eligible=bool(f.q_trend.iloc[i]), max_hold=240))
    # Sweep, reclaim, then a failed retest and break of the prior minute extreme.
    for side in (1, -1):
        if side == -1:
            sweep = (f.high > f.q_ceiling + .1 * f.atr) & (f.close < f.q_ceiling)
        else:
            sweep = (f.low < f.q_floor - .1 * f.atr) & (f.close > f.q_floor)
        sweep &= f.ready & (f.volume_ratio >= 1.5)
        used_until = -1
        for i in np.flatnonzero(sweep.to_numpy()):
            if i <= used_until:
                continue
            level = f.q_ceiling.iloc[i] if side == -1 else f.q_floor.iloc[i]
            target = float((f.q_ceiling.iloc[i] + f.q_floor.iloc[i]) / 2)
            for j in range(i + 1, min(i + 4, len(f))):
                if side == -1:
                    confirm = (f.high.iloc[j] >= level - .3 * f.atr.iloc[j]
                               and f.close.iloc[j] < f.low.iloc[j - 1] and f.wr.iloc[j] < -20)
                    stop = f.high.iloc[i:j + 1].max() + .2 * f.atr.iloc[j]
                else:
                    confirm = (f.low.iloc[j] <= level + .3 * f.atr.iloc[j]
                               and f.close.iloc[j] > f.high.iloc[j - 1] and f.wr.iloc[j] > -80)
                    stop = f.low.iloc[i:j + 1].min() - .2 * f.atr.iloc[j]
                risk = side * (f.close.iloc[j] - stop)
                reward = side * (target - f.close.iloc[j])
                if confirm and .0015 <= risk / f.close.iloc[j] <= .02 and reward >= risk:
                    signals.append(dict(i=int(j), strategy='reversal', symbols=[symbol], side=side,
                                        stop=float(stop), target=target, eligible=bool(f.q_range.iloc[j]), max_hold=90))
                    used_until = j
                    break
    return signals


def pair_signals(frames, pairs):
    """Daily trailing-window Engle-Granger checks; no full-sample pair selection."""
    signals, diagnostics, validity = [], [], {}
    index = next(iter(frames.values())).index
    for a, b in pairs:
        minute = pd.DataFrame({'a': frames[a].close, 'b': frames[b].close})
        quarter = minute.resample('15min', closed='left', label='right').last().dropna()
        logs = np.log(quarter)
        snapshots = []
        for day in pd.date_range(index[0].ceil('D'), index[-1].floor('D'), freq='D'):
            window = logs.loc[(logs.index > day - pd.Timedelta(days=7)) & (logs.index <= day)]
            if len(window) < 7 * 96:
                continue
            x, y = window.b.to_numpy(), window.a.to_numpy()
            beta, alpha = np.polyfit(x, y, 1)
            residual = y - alpha - beta * x
            sd = np.std(residual, ddof=1)
            corr = window.diff().a.corr(window.diff().b)
            phi = np.polyfit(residual[:-1], np.diff(residual), 1)[0]
            half_life = -np.log(2) / np.log(1 + phi) if -1 < phi < 0 else np.inf
            pvalue = coint(y, x, maxlag=1, autolag=None)[1]
            valid = bool(.25 < beta < 4 and corr > .65 and pvalue < .05 and 2 <= half_life <= 96 and sd > 0)
            row = dict(timestamp=day, a=a, b=b, alpha=alpha, beta=beta, sd=sd,
                       corr=corr, pvalue=pvalue, half_life=half_life, valid=valid)
            diagnostics.append(row)
            snapshots.append(row)
        if not snapshots:
            continue
        snap = pd.DataFrame(snapshots).set_index('timestamp').reindex(index, method='ffill')
        valid = snap.valid.eq(True).to_numpy()
        validity[(a, b)] = valid
        za = (np.log(minute.a) - snap.alpha - snap.beta * np.log(minute.b)) / snap.sd
        # A new regression must not manufacture a threshold crossing at midnight.
        stable = snap.alpha.eq(snap.alpha.shift(1))
        for side in (1, -1):
            cross = ((za.shift(1) * -side > 2.2) & (za * -side <= 2.2)
                     & (za * -side > .3) & stable & valid)
            for i in np.flatnonzero(cross.to_numpy()):
                row = snap.iloc[i]
                signals.append(dict(i=int(i), strategy='pair', symbols=[a, b], side=side,
                                    eligible=True, alpha=float(row.alpha), beta=float(row.beta), sd=float(row.sd),
                                    entry_z=float(za.iloc[i]), max_hold=int(np.clip(2 * row.half_life * 15, 60, 1440))))
    return signals, pd.DataFrame(diagnostics), validity


@dataclass
class Settings:
    capital: float = 1000.
    fee: float = .0005
    slip: float = .0002
    risk: float = .002
    delay: int = 1
    mode: str = 'adaptive'


def run_portfolio(frames, signals, funding, pair_validity, start, end, settings):
    symbols = list(frames)
    column = {s: i for i, s in enumerate(symbols)}
    index = frames[symbols[0]].index
    prices = {name: np.column_stack([frames[s][name].to_numpy() for s in symbols])
              for name in ('open', 'high', 'low', 'close', 'b_atr')}
    rates = np.column_stack([funding[s].reindex(index, fill_value=0).to_numpy() for s in symbols])
    lo = int(index.searchsorted(start))
    hi = int(index.searchsorted(end))
    events = {}
    rejection = Counter()
    for signal in signals:
        i = signal['i'] + settings.delay
        if lo <= i < hi:
            events.setdefault(i, []).append(signal)
    weights = {'trend': .45, 'reversal': .35, 'pair': .20}
    cash, peak, day_start = settings.capital, settings.capital, settings.capital
    positions, trades, curve, rejected = [], [], [], []
    counts, daily_entries, current_day = Counter(), 0, None
    daily_halt = hard_halt = False
    flatten_pending = None
    last_exit = {}
    worst_envelope_dd = 0.

    def equity(mark):
        return cash + sum(float(np.sum(p['qty'] * (mark[p['cols']] - p['entry']))) for p in positions)

    def gross(mark):
        return sum(float(np.sum(np.abs(p['qty']) * mark[p['cols']])) for p in positions)

    def close_position(p, raw, timestamp, reason, fraction=1.):
        nonlocal cash
        quantities = p['qty'] * fraction
        fill = fill_price(np.asarray(raw), -np.sign(quantities), settings.slip)
        pnl = float(np.sum(quantities * (fill - p['entry'])))
        fee = float(np.sum(np.abs(quantities) * fill) * settings.fee)
        cash += pnl - fee
        p['net'] += pnl - fee
        p['fees'] += fee
        p['gross_pnl'] += pnl
        p['qty'] -= quantities
        if fraction == 1.:
            trades.append(dict(entry_time=p['entry_time'], exit_time=timestamp, strategy=p['strategy'],
                               symbols='__'.join(p['symbols']), side=p['side'], net_pnl=p['net'],
                               price_pnl=p['gross_pnl'], fees=p['fees'], funding=p['funding'],
                               initial_risk=p['risk'], initial_notional=p['notional'],
                               holding_minutes=(timestamp - p['entry_time']).total_seconds() / 60,
                               exit_reason=reason, entry_state=p['eligible'],
                               max_adverse_pnl=p['mae'], max_favorable_pnl=p['mfe'],
                               signal_i=p['i'], candidate=p.get('candidate', 'original'),
                               entry_features=p.get('entry_features', {})))
            positions.remove(p)
            for s in p['symbols']:
                last_exit[s] = timestamp

    for i in range(lo, hi):
        t = index[i]
        op, high, low, close = (prices[name][i] for name in ('open', 'high', 'low', 'close'))
        day = (t + pd.Timedelta(hours=8)).date()
        if day != current_day:
            current_day = day
            day_start = equity(prices['close'][i - 1]) if i > lo else settings.capital
            counts, daily_entries, daily_halt = Counter(), 0, False
        # Funding is posted for positions already open at the settlement minute.
        if np.any(rates[i]):
            for p in positions:
                payment = funding_cashflow(p['qty'], op[p['cols']], rates[i, p['cols']])
                cash += payment
                p['net'] += payment
                p['funding'] += payment
        open_equity = equity(op)
        open_dd = 1 - open_equity / peak
        if open_dd >= .12:
            hard_halt = True
        if open_dd >= .15:
            flatten_pending = 'account_drawdown_exit'
        if open_equity <= day_start * .98:
            flatten_pending, daily_halt = 'daily_loss_exit', True
        if flatten_pending:
            for p in positions[:]:
                close_position(p, op[p['cols']], t, flatten_pending)
            flatten_pending = None
        # Prior-close exits execute before today's fresh entries.
        for p in positions[:]:
            reason = None
            if i - p['entry_i'] >= p['max_hold']:
                reason = 'time_exit'
            if p['strategy'] == 'pair':
                a, b = p['cols']
                z = (np.log(prices['close'][i - 1, a]) - p['alpha']
                     - p['beta'] * np.log(prices['close'][i - 1, b])) / p['sd']
                if z * -p['side'] <= .3:
                    reason = 'spread_reversion'
                elif abs(z) >= 3.5:
                    reason = 'spread_stop'
                elif not pair_validity[tuple(p['symbols'])][i - 1]:
                    reason = 'relationship_invalid'
            if reason:
                close_position(p, op[p['cols']], t, reason)
        for signal in sorted(events.get(i, []), key=lambda x: (x['strategy'], x['symbols'])):
            strategy = signal['strategy']
            if settings.mode in weights and strategy != settings.mode:
                continue
            reason = None
            if settings.mode != 'fixed' and not signal['eligible']:
                reason = 'regime_blocked'
            elif daily_halt or hard_halt:
                reason = 'account_paused'
            elif not can_enter(signal['symbols'], counts, {s for p in positions for s in p['symbols']}, daily_entries):
                reason = 'position_or_daily_limit'
            elif any(s in last_exit and (t - last_exit[s]).total_seconds() < 900 for s in signal['symbols']):
                reason = 'cooldown_15min'
            if reason:
                rejection[(strategy, reason)] += 1
                rejected.append((t, strategy, '__'.join(signal['symbols']), reason))
                continue
            eq = equity(op)
            dd = 1 - eq / peak
            scale = .5 if dd >= .08 else 1.
            budget = eq * settings.risk * scale * (.75 if strategy == 'pair' else 1)
            same_risk = sum(p['risk'] for p in positions if p['strategy'] == strategy)
            cap = eq * .01 * (weights[strategy] if settings.mode in ('adaptive', 'fixed') else 1)
            if same_risk + budget > cap or sum(p['risk'] for p in positions) + budget > eq * .01:
                rejection[(strategy, 'risk_budget')] += 1
                continue
            # Treat all single-coin directional exposures as one correlated cluster.
            if strategy != 'pair' and sum(p['risk'] for p in positions if p['strategy'] != 'pair' and p['side'] == signal['side']) + budget > eq * .005:
                rejection[(strategy, 'directional_risk')] += 1
                continue
            cols = np.array([column[s] for s in signal['symbols']])
            side = signal['side']
            directions = np.array([side, -side]) if strategy == 'pair' else np.array([side])
            entry = fill_price(op[cols], directions, settings.slip)
            remaining_gross = max(0., 2 * eq - gross(op))
            if strategy == 'pair':
                a, b = cols
                z = (np.log(op[a]) - signal['alpha'] - signal['beta'] * np.log(op[b])) / signal['sd']
                distance = (3.5 - abs(z)) * signal['sd']
                reward = (abs(z) - .3) * signal['sd'] / (1 + signal['beta'])
                if distance <= 0 or z * -side <= .3:
                    rejection[(strategy, 'entry_gap_invalid')] += 1
                    continue
                unit_a = budget / (distance + 2 * (settings.fee + settings.slip) * (1 + signal['beta']))
                unit_a = min(unit_a, eq / (1 + signal['beta']), remaining_gross / (1 + signal['beta']))
                notionals = np.array([unit_a, unit_a * signal['beta']])
                actual_risk = unit_a * (distance + 2 * (settings.fee + settings.slip) * (1 + signal['beta']))
            else:
                distance = side * (entry[0] - signal['stop'])
                reward = side * (signal['target'] - entry[0]) / entry[0]
                if distance <= 0 or reward <= 0:
                    rejection[(strategy, 'entry_gap_invalid')] += 1
                    continue
                loss_fraction = distance / entry[0] + 2 * settings.fee + settings.slip
                notionals = np.array([min(budget / loss_fraction, .5 * eq, remaining_gross)])
                actual_risk = notionals[0] * loss_fraction
            # Keep the entry filter fixed across cost stress runs to isolate cost effects.
            if reward < 3 * .0014 or notionals.sum() < 2:
                rejection[(strategy, 'insufficient_edge_over_cost')] += 1
                continue
            qty = directions * notionals / entry
            fee = float(notionals.sum() * settings.fee)
            cash -= fee
            p = dict(signal, cols=cols, entry=entry, qty=qty, entry_time=t, entry_i=i, risk=actual_risk,
                     notional=float(notionals.sum()), net=-fee, fees=fee, funding=0., gross_pnl=0.,
                     partial=False, mae=0., mfe=0.)
            if strategy == 'trend':
                p['partial_target'] = entry[0] + side * distance * p.get('partial_rr', 1.)
            positions.append(p)
            counts.update(signal['symbols'])
            daily_entries += 1
        # Envelope is a conservative OHLC bound, not a simultaneously observed mark.
        adverse_equity = cash
        for p in positions:
            adverse = np.where(p['qty'] > 0, low[p['cols']], high[p['cols']])
            adverse_equity += float(np.sum(p['qty'] * (adverse - p['entry'])))
        worst_envelope_dd = max(worst_envelope_dd, 1 - adverse_equity / peak)
        for p in positions[:]:
            cols = p['cols']
            floating = float(np.sum(p['qty'] * (close[cols] - p['entry'])))
            p['mae'], p['mfe'] = min(p['mae'], floating), max(p['mfe'], floating)
            if p['strategy'] == 'pair':
                adverse = np.where(p['qty'] > 0, low[cols], high[cols])
                # Conservative: if the joint adverse envelope breaches the cash stop,
                # liquidate at that envelope, explicitly penalizing unknown leg ordering.
                if p['net'] + float(np.sum(p['qty'] * (adverse - p['entry']))) <= -p['risk']:
                    close_position(p, adverse, t + pd.Timedelta(minutes=1), 'pair_cash_stop_envelope')
            else:
                k = cols[0]
                decision = bar_exit(p['side'], op[k], high[k], low[k], p['stop'], p['target'])
                if decision and decision[1] == 'stop':
                    close_position(p, [decision[0]], t + pd.Timedelta(minutes=1), 'stop')
                    continue
                if p['strategy'] == 'trend' and not p['partial']:
                    hit = high[k] >= p['partial_target'] if p['side'] == 1 else low[k] <= p['partial_target']
                    if hit:
                        ratio = p.get('partial_ratio', .5)
                        if ratio > 0:
                            close_position(p, [p['partial_target']], t + pd.Timedelta(minutes=1), 'partial', ratio)
                        p['partial'] = True
                if decision:
                    close_position(p, [decision[0]], t + pd.Timedelta(minutes=1), decision[1])
                elif p['strategy'] == 'trend' and p['partial']:
                    trailing = close[k] - p['side'] * p.get('trail_mult', 1.8) * prices['b_atr'][i, k]
                    p['stop'] = max(p['stop'], trailing) if p['side'] == 1 else min(p['stop'], trailing)
        eq = equity(close)
        peak = max(peak, eq)
        dd = 1 - eq / peak
        if dd >= .12:
            hard_halt = True
        if dd >= .15:
            flatten_pending = 'account_drawdown_exit'
        if eq <= day_start * .98:
            daily_halt, flatten_pending = True, 'daily_loss_exit'
        curve.append((t + pd.Timedelta(minutes=1), eq, dd * 100, len(positions), gross(close), hard_halt))
    for p in positions[:]:
        close_position(p, prices['close'][hi - 1, p['cols']], index[hi - 1] + pd.Timedelta(minutes=1), 'end_of_test')
    curve[-1] = (curve[-1][0], cash, (1 - cash / peak) * 100, 0, 0., hard_halt)
    equity_frame = pd.DataFrame(curve, columns=['timestamp', 'equity', 'drawdown_pct', 'positions', 'gross_notional', 'hard_halt']).set_index('timestamp')
    ledger = pd.DataFrame(trades)
    net = ledger.net_pnl if len(ledger) else pd.Series(dtype=float)
    days = (end - start).total_seconds() / 86400
    summary = dict(mode=settings.mode, initial_equity=settings.capital, final_equity=cash,
                   return_pct=(cash / settings.capital - 1) * 100,
                   max_drawdown_pct=max(0., float(equity_frame.drawdown_pct.max())),
                   conservative_intrabar_envelope_dd_pct=max(0., worst_envelope_dd * 100),
                   trades=len(ledger), trades_per_day=len(ledger) / days,
                   win_rate_pct=float((net > 0).mean() * 100) if len(net) else 0.,
                   profit_factor=float(net[net > 0].sum() / -net[net < 0].sum()) if (net < 0).any() else None,
                   fees=float(ledger.fees.sum()) if len(ledger) else 0.,
                   funding=float(ledger.funding.sum()) if len(ledger) else 0.,
                   avg_trade=float(net.mean()) if len(net) else 0.,
                   hard_halt=hard_halt, capital_reconciliation_error=float(cash - settings.capital - net.sum()),
                   fee_rate=settings.fee, slippage_bps=settings.slip * 10000, delay_minutes=settings.delay)
    rejects = pd.DataFrame([dict(strategy=s, reason=r, count=n) for (s, r), n in rejection.items()])
    return summary, ledger, equity_frame, rejects
