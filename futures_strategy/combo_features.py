"""Causal entry features and conservative research admission rules."""
import numpy as np
import pandas as pd

from futures_strategy.adaptive_combo_research import closed_bars, prepare_features
from futures_strategy.indicators import atr, ema


def cost_risk_ratio(price, stop, round_trip_cost=.0014):
    return round_trip_cost / (abs(price - stop) / price)


def select_sleeves(records):
    return [r['name'] for r in records if r['train_net'] > 0 and r['valid_net'] > 0
            and r['train_n'] >= 30 and r['valid_n'] >= 20]


def enrich_features(raw):
    f = prepare_features(raw)
    for n in (15, 60, 240, 1440):
        f[f'ret_{n}'] = f.close.pct_change(n)
    f['efficiency_60'] = f.close.diff(60).abs() / f.close.diff().abs().rolling(60).sum()
    f['atr_pct'] = f.b_atr / f.close
    f['extension_atr'] = (f.close - f.b_ema) / f.b_atr
    f['trend_gap_atr'] = (f.h_fast - f.h_slow) / f.b_atr
    f['close_location'] = (f.close - f.low) / (f.high - f.low).replace(0, np.nan)
    f['flow_5'] = (2 * f.taker_buy_base.rolling(5).sum() / f.volume.rolling(5).sum() - 1) if 'taker_buy_base' in f else np.nan
    typical = (f.high + f.low + f.close) / 3
    f['vwap_60'] = (typical * f.volume).rolling(60).sum() / f.volume.rolling(60).sum()
    f['vwap_distance_atr'] = (f.close - f.vwap_60) / f.b_atr
    q = closed_bars(raw, '15min')
    q['atr15'] = atr(q, 14)
    q['donchian_high'] = q.high.rolling(16).max()
    q['donchian_low'] = q.low.rolling(16).min()
    q['ema15'] = ema(q.close, 20)
    q['previous_close'] = q.close
    for col in ('atr15', 'donchian_high', 'donchian_low', 'ema15', 'previous_close'):
        f[col] = q[col].reindex(f.index, method='ffill')
    f['market_vol_ratio'] = f.atr_pct / f.atr_pct.rolling(1440).median().shift(1)
    return f
