"""Closed-candle, first-break signal for newly listed USDT perpetuals."""
from __future__ import annotations

import math
import pandas as pd

MAX_AGE_DAYS = 30
MAX_BREAK_DEPTH = 0.00262871
LAST_SIGNAL_MINUTE_UTC = 1022
MIN_SESSION_DRAWDOWN = 0.03


def eligible_listings(info: dict, now: float) -> list[dict]:
    today = pd.Timestamp(now, unit='s', tz='UTC').normalize()
    result = []
    for row in info.get('symbols', []):
        try:
            if (row['status'] != 'TRADING' or row['contractType'] != 'PERPETUAL'
                    or row['quoteAsset'] != 'USDT' or not row['symbol'].endswith('USDT')):
                continue
            onboard = pd.Timestamp(int(row['onboardDate']), unit='ms', tz='UTC').normalize()
            age = int((today-onboard).days)
            if 1 <= age <= MAX_AGE_DAYS:
                result.append(dict(symbol=row['symbol'], onboard_date=onboard.timestamp(), age_days=age))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return sorted(result, key=lambda r:(r['age_days'],r['symbol']))


def evaluate_breakdown(symbol: str, onboard_time: float, daily: pd.DataFrame,
                       minutes: pd.DataFrame, now: float) -> dict:
    """Evaluate only the first eligible close break of the UTC day, without future bars."""
    day = pd.Timestamp(now, unit='s', tz='UTC').normalize()
    age = int((day-pd.Timestamp(onboard_time, unit='s', tz='UTC').normalize()).days)
    result = dict(symbol=symbol, listing_age_days=age, stage='WAIT', reason='WAIT_BREAK', signal=None)
    if not 1 <= age <= MAX_AGE_DAYS:
        result.update(stage='INELIGIBLE',reason='LISTING_AGE')
        return result
    prev = day-pd.Timedelta(days=1)
    if prev not in daily.index or not all(c in daily.columns for c in ('open','close','low')):
        result.update(stage='INELIGIBLE',reason='MISSING_PREVIOUS_DAY')
        return result
    p = daily.loc[prev]
    try:
        prev_open,prev_close,prev_low = (float(p[k]) for k in ('open','close','low'))
    except (TypeError, ValueError):
        result.update(stage='INELIGIBLE',reason='INVALID_DAILY')
        return result
    if not all(math.isfinite(v) and v>0 for v in (prev_open,prev_close,prev_low)):
        result.update(stage='INELIGIBLE',reason='INVALID_DAILY')
        return result
    if prev_close <= prev_open:
        result.update(stage='INELIGIBLE',reason='PREVIOUS_DAY_NOT_GREEN')
        return result
    result['prev_low'] = prev_low
    closed = minutes[(minutes.index >= day) & (minutes.index + pd.Timedelta(minutes=1) <= pd.Timestamp(now,unit='s',tz='UTC'))]
    if len(closed)<16 or closed.index[0] != day or not closed.index.is_unique or not closed.index.is_monotonic_increasing:
        result.update(stage='INELIGIBLE',reason='INCOMPLETE_MINUTES')
        return result
    if not (closed.index.to_series().diff().iloc[1:] == pd.Timedelta(minutes=1)).all():
        result.update(stage='INELIGIBLE',reason='INCOMPLETE_MINUTES')
        return result
    if not all(k in closed.columns for k in ('open','high','low','close')):
        result.update(stage='INELIGIBLE',reason='INVALID_MINUTES')
        return result
    if not (closed[['open','high','low','close']].apply(pd.to_numeric,errors='coerce')>0).all().all():
        result.update(stage='INELIGIBLE',reason='INVALID_MINUTES')
        return result
    first = closed.iloc[15:][closed['close'].iloc[15:] < prev_low]
    if first.empty:
        return result
    ts = first.index[0]
    bar = first.iloc[0]
    signal_time = (ts+pd.Timedelta(minutes=1)).timestamp()
    minute = int((ts-day).total_seconds()/60)
    session_high = float(closed.loc[:ts,'high'].max())
    close = float(bar['close'])
    depth = prev_low/close-1
    drawdown = 1-close/session_high
    result.update(stage='SIGNAL',prev_low=prev_low,signal_close=close,
                  break_depth_pct=depth,session_drawdown_pct=drawdown,
                  signal_minute_of_day=minute,signal_time=signal_time)
    if ts != closed.index[-1]:
        result['reason'] = 'FIRST_BREAK_ALREADY_PASSED'
    elif minute > LAST_SIGNAL_MINUTE_UTC:
        result['reason'] = 'LATE_UTC_SIGNAL'
    elif drawdown < MIN_SESSION_DRAWDOWN:
        result['reason'] = 'INSUFFICIENT_DRAWDOWN'
    elif depth > MAX_BREAK_DEPTH:
        result['reason'] = 'BREAK_TOO_DEEP'
    else:
        result['reason'] = 'SIGNAL_READY'
        result['signal'] = dict(signal_id=f'{symbol}:{day.date()}',symbol=symbol,
            signal_time=signal_time,listing_age_days=age,prev_low=prev_low,
            break_depth_pct=depth,session_drawdown_pct=drawdown,
            signal_minute_of_day=minute,signal_close=close,
            strategy_version='new-listing-break-v1')
    return result
