"""Small mechanism-driven candidate set; no coin/hour blacklist or broad grid."""
import numpy as np

from futures_strategy.adaptive_combo_research import closed_bars, single_signals
from futures_strategy.indicators import atr


def feature_log(f, i, side):
    keys = ['q_adx','volume_ratio','atr_pct','efficiency_60','flow_5','ret_60',
            'market_vol_ratio','extension_atr','wr']
    return {key: float(f[key].iloc[i]) for key in keys}


def generate_candidates(frames, paired):
    candidates = {name: [] for name in (
        'trend_entry_filter', 'trend_wider_exit', 'trend_filter_and_exit',
        'trend_15m_breakout', 'reversal_wider_stop', 'reversal_vwap_reclaim',
        'pair_original', 'pair_cost_filtered', 'trend_hourly_channel', 'trend_relative_strength')}
    btc = frames['BTCUSDT']
    for symbol,f in frames.items():
        original = single_signals(symbol,f)
        for raw in original:
            i, side = raw['i'],raw['side']
            price = float(f.close.iloc[i])
            distance = side * (price - raw['stop'])
            if raw['strategy'] == 'trend':
                # June evidence: avoid tiny movement budgets and terminal volume spikes.
                filtered = (f.volume_ratio.iloc[i] <= 2.5 and side*f.ret_60.iloc[i] >= .003
                            and f.efficiency_60.iloc[i] >= .1 and f.market_vol_ratio.iloc[i] <= 2.5)
                if filtered:
                    candidates['trend_entry_filter'].append(dict(raw))
                wider = max(distance, float(f.atr15.iloc[i])*1.5, price*.005)
                if wider/price <= .025:
                    signal = dict(raw, stop=price-side*wider, target=price+side*2.5*wider,
                                  partial_ratio=0., partial_rr=1.5, trail_mult=3., max_hold=720)
                    candidates['trend_wider_exit'].append(signal)
                    if filtered:
                        candidates['trend_filter_and_exit'].append(dict(signal))
            else:
                # WR logs do not support a universal volume cutoff for reversals.
                wider = max(distance, float(f.b_atr.iloc[i])*1.5,price*.005)
                reward = side*(raw['target']-price)
                if reward >= 1.2*wider and wider/price <= .025:
                    candidates['reversal_wider_stop'].append(dict(raw,stop=price-side*wider,max_hold=180))
        for side in (1,-1):
            boundary = f.donchian_high if side == 1 else f.donchian_low
            crossed = (side*(f.close-boundary)>0)&(side*(f.close.shift(1)-boundary)<=0)
            mask = crossed & f.ready & f.q_trend.eq(True)
            mask &= side*(f.h_fast-f.h_slow)>0
            mask &= side*btc.ret_60 >= 0
            mask &= f.volume_ratio.between(1.,2.5) & f.market_vol_ratio.between(.7,2.5)
            for i in np.flatnonzero(mask.to_numpy()):
                price = float(f.close.iloc[i])
                risk = max(float(f.atr15.iloc[i])*1.5,price*.005)
                if risk/price <= .025:
                    candidates['trend_15m_breakout'].append(dict(i=int(i),strategy='trend',symbols=[symbol],side=side,
                        stop=price-side*risk,target=price+side*2.5*risk,eligible=True,max_hold=720,
                        partial_ratio=0.,partial_rr=1.5,trail_mult=3.))
            # Range extension then reclaim: causal band and historical extreme only.
            distance = (f.close-f.vwap_60)/f.b_atr
            old_distance = distance.shift(1)
            mask = f.ready & f.q_range.eq(True)
            mask &= (-side*old_distance > 2) & (-side*distance <= 2)
            mask &= side*f.close.diff() > 0
            mask &= f.market_vol_ratio.between(.7,2.)
            mask &= f.ret_60.abs() < .01
            for i in np.flatnonzero(mask.to_numpy()):
                price = float(f.close.iloc[i])
                extreme = float(f.low.iloc[max(0,i-10):i+1].min() if side==1 else f.high.iloc[max(0,i-10):i+1].max())
                risk = max(side*(price-extreme)+.2*float(f.atr.iloc[i]),price*.005)
                reward = side*(float(f.vwap_60.iloc[i])-price)
                if reward >= risk and risk/price <= .025:
                    candidates['reversal_vwap_reclaim'].append(dict(i=int(i),strategy='reversal',symbols=[symbol],side=side,
                        stop=price-side*risk,target=float(f.vwap_60.iloc[i]),eligible=True,max_hold=180))
        # Slower signal horizon motivated by the daily-momentum literature and cost/R diagnostics.
        # One-minute execution is retained, but exits have enough room for larger moves.
        h = closed_bars(f, '1h')
        h['hour_atr'] = atr(h,14)
        h['prior_high'] = h.high.rolling(20).max().shift(1)
        h['prior_low'] = h.low.rolling(20).min().shift(1)
        for side in (1,-1):
            boundary = h.prior_high if side==1 else h.prior_low
            mask = (side*(h.close-boundary)>0)&(side*(h.close.shift(1)-boundary)<=0)
            for timestamp in h.index[mask]:
                # The hourly candle closes here; place its signal on the final minute
                # so the existing next-open engine executes at this completed boundary.
                i = int(f.index.searchsorted(timestamp))-1
                if i<0 or i>=len(f) or not f.ready.iloc[i]:
                    continue
                price=float(f.close.iloc[i]); risk=max(float(h.loc[timestamp,'hour_atr'])*2,price*.008)
                if risk/price <= .04:
                    candidates['trend_hourly_channel'].append(dict(i=i,strategy='trend',symbols=[symbol],side=side,
                        stop=price-side*risk,target=price+side*3*risk,eligible=True,max_hold=1440,
                        partial_ratio=0.,partial_rr=2.,trail_mult=8.))
    # Cross-sectional ranking is calculated among the same fixed five coins at each
    # completed four-hour boundary; this is directional momentum, not guaranteed hedging.
    import pandas as pd
    returns=pd.DataFrame({s:f.ret_1440 for s,f in frames.items()})
    for i in np.flatnonzero((returns.index.minute==59)&(returns.index.hour%4==3)):
        row=returns.iloc[i]
        if row.isna().any():
            continue
        for symbol,side in ((row.idxmax(),1),(row.idxmin(),-1)):
            f=frames[symbol]; price=float(f.close.iloc[i])
            if side*row[symbol] < .01 or not f.ready.iloc[i]:
                continue
            risk=max(float(f.atr15.iloc[i])*3,price*.008)
            if risk/price <= .04:
                candidates['trend_relative_strength'].append(dict(i=int(i),strategy='trend',symbols=[symbol],side=side,
                    stop=price-side*risk,target=price+side*2.5*risk,eligible=True,max_hold=480,
                    partial_ratio=0.,partial_rr=2.,trail_mult=6.))
    candidates['pair_original'] = [dict(s) for s in paired]
    for s in paired:
        edge = (abs(s['entry_z'])-.3)*s['sd']/(1+s['beta'])
        if edge >= .0014*5:
            candidates['pair_cost_filtered'].append(dict(s))
    for name,signals in candidates.items():
        for s in signals:
            s['candidate'] = name
            s['entry_features'] = feature_log(frames[s['symbols'][0]],s['i'],s['side'])
    return candidates
