"""Closed-bar, cointegration-screened pair signals. Fits are frozen per position."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def entry_direction(z: float, previous_z: float, entry: float = 2.2, stop: float = 4.0) -> str | None:
    if not np.isfinite([z, previous_z]).all() or not entry <= abs(z) < stop:
        return None
    if z > 0 and previous_z > z:
        return 'SHORT_A_LONG_B'
    if z < 0 and previous_z < z:
        return 'LONG_A_SHORT_B'
    return None


def fit_pair(symbol_a: str, symbol_b: str, a: pd.DataFrame, b: pd.DataFrame, now: float) -> dict | None:
    cutoff = pd.Timestamp(now, unit='s', tz='UTC').floor('h')
    joined = pd.concat([a['close'].rename('a'), b['close'].rename('b')], axis=1)
    joined = joined.loc[(joined.index < cutoff) & (joined.index >= cutoff-pd.Timedelta(days=30))].sort_index()
    if len(joined) < 600 or joined.index.has_duplicates or joined.isna().any().any():
        return None
    if joined.index[-1] != cutoff-pd.Timedelta(hours=1):
        return None
    if (joined.index.to_series().diff().dropna() != pd.Timedelta(hours=1)).any():
        return None
    if not np.isfinite(joined.to_numpy()).all() or (joined <= 0).any().any():
        return None
    four = joined.resample('4h').last()
    counts = joined.resample('4h').count().min(axis=1)
    four = four.loc[(counts == 4) & (four.index + pd.Timedelta(hours=4) <= cutoff)]
    if len(four) < 120:
        return None
    logs = np.log(joined)
    corr = float(logs.diff().dropna().corr().iloc[0, 1])
    corr4 = float(np.log(four).diff().dropna().corr().iloc[0, 1])
    if not np.isfinite([corr, corr4]).all() or min(corr, corr4) < .7:
        return None
    beta, alpha = np.polyfit(logs['b'], logs['a'], 1)
    if not .2 <= beta <= 5:
        return None
    residual = logs['a'] - alpha - beta*logs['b']
    std = float(residual.std(ddof=1))
    if not np.isfinite(std) or std < 1e-6:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p1 = float(coint(logs['a'], logs['b'], maxlag=4, autolag='aic')[1])
        p4 = float(coint(np.log(four['a']), np.log(four['b']), maxlag=4, autolag='aic')[1])
    if not np.isfinite([p1, p4]).all() or max(p1, p4) > .05:
        return None
    return dict(symbol_a=symbol_a, symbol_b=symbol_b, beta=float(beta), alpha=float(alpha),
                mean=float(residual.mean()), std=std, correlation=corr, correlation_4h=corr4,
                pvalue_1h=p1, pvalue_4h=p4, fitted_at=cutoff.timestamp())


def spread_z(model: dict, price_a: float, price_b: float) -> float:
    return float((np.log(price_a)-model['alpha']-model['beta']*np.log(price_b)-model['mean'])/model['std'])


def signal_snapshot(model: dict, a: pd.DataFrame, b: pd.DataFrame, now: float) -> dict | None:
    cutoff = pd.Timestamp(now, unit='s', tz='UTC').floor('5min')
    data = pd.concat([a['close'].rename('a'), b['close'].rename('b')], axis=1).dropna()
    data = data.loc[data.index < cutoff].tail(2)
    if len(data) != 2 or data.index[-1] != cutoff-pd.Timedelta(minutes=5):
        return None
    if data.index[1]-data.index[0] != pd.Timedelta(minutes=5) or (data <= 0).any().any():
        return None
    zs = [spread_z(model, float(r.a), float(r.b)) for r in data.itertuples()]
    if not np.isfinite(zs).all():
        return None
    return dict(model, z=zs[-1], previous_z=zs[-2], signal_time=cutoff.timestamp(),
                direction=entry_direction(zs[-1], zs[-2]))
