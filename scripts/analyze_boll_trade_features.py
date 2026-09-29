"""Entry-only diagnostics for the saved PRB backtest; no trading side effects."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from futures_strategy.boll_short_signals import evaluate_signal

SOURCE = ROOT / 'outputs/boll_short_prb_backtest_2026-01_2026-03_current_rules'
DATA = ROOT / 'data/top_gainers_boll_short_1m/realtime_enriched'
OUT = ROOT / 'outputs/boll_short_feature_diagnostics_20260911'


def stats(g):
    p = g.net_pnl
    return dict(n=len(g), wins=int((p > 0).sum()), win_pct=float((p > 0).mean()*100),
                net=float(p.sum()), avg=float(p.mean()),
                pf=float(p[p > 0].sum() / -p[p < 0].sum()) if (p < 0).any() else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(SOURCE / 'trades.csv')
    for c in ['entry_time', 'exit_time', 'breakdown_time']:
        t[c] = pd.to_datetime(t[c], utc=True)
    assert not t.duplicated(['symbol', 'entry_time']).any()
    assert np.allclose(t.price_pnl-t.entry_fee-t.exit_fee, t.net_pnl)
    t['month'] = t.entry_time.dt.strftime('%Y-%m')
    t['day'] = t.entry_time.dt.strftime('%Y-%m-%d')
    t['hour_utc'] = t.entry_time.dt.hour
    t['stop_pct'] = t.risk_distance / t.entry_price * 100
    t['atr_pct'] = t.atr / t.entry_price * 100
    t['stop_atr'] = t.risk_distance / t.atr
    t['daily_gap_pct'] = (t.daily_open/t.daily_upper-1)*100
    t['day_gain_pct'] = (t.entry_price/t.daily_open-1)*100
    t['fees_risk_pct'] = (t.entry_fee+t.exit_fee)/t.planned_risk*100
    t['holding_minutes'] = (t.exit_time-t.entry_time).dt.total_seconds()/60
    errors = []
    for k, (symbol, group) in enumerate(t.groupby('symbol')):
        files = list(DATA.glob(symbol.lower() + '_1m_*.csv'))
        assert files, symbol
        d = pd.concat([pd.read_csv(file) for file in sorted(files)], ignore_index=True)
        d.index = pd.to_datetime(d.pop('timestamp'), utc=True)
        d = d.sort_index()
        assert d.index.is_unique and d.index.is_monotonic_increasing
        c, v = d.close, d.volume
        mid = c.rolling(20).mean()
        upper = mid+2*c.rolling(20).std(ddof=0)
        lower = mid-2*c.rolling(20).std(ddof=0)
        vr = v/v.shift(1).rolling(20).mean()
        for i, trade in group.iterrows():
            b = d.index.get_loc(trade.breakdown_time)
            window = d.iloc[max(0, b-180):b+1]
            if len(window) < 121 or not (window.index.to_series().diff().dropna() == pd.Timedelta(minutes=1)).all():
                errors.append([int(i), 'insufficient_or_discontinuous_context'])
                continue
            # No candle at/after entry is used for any predictor.
            row = d.iloc[b]
            assert window.index[-1]+pd.Timedelta(minutes=1) == trade.entry_time
            assert np.isclose(row.close*(1-.0002), trade.entry_price)
            for n in [5, 15, 60, 120]:
                t.loc[i, f'return_{n}m_pct'] = (c.iloc[b]/c.iloc[b-n]-1)*100
            t.loc[i, 'bandwidth_pct'] = (upper.iloc[b]-lower.iloc[b])/mid.iloc[b]*100
            t.loc[i, 'upper_slope_5m_pct'] = (upper.iloc[b]/upper.iloc[b-5]-1)*100
            t.loc[i, 'mid_slope_5m_pct'] = (mid.iloc[b]/mid.iloc[b-5]-1)*100
            t.loc[i, 'above_upper_count_20'] = int((c.iloc[b-19:b+1]>upper.iloc[b-19:b+1]).sum())
            t.loc[i, 'entry_band_position'] = (c.iloc[b]-lower.iloc[b])/(upper.iloc[b]-lower.iloc[b])
            t.loc[i, 'breakdown_volume_ratio'] = vr.iloc[b]
            t.loc[i, 'breakdown_body_pct'] = (row.open-row.close)/row.open*100
            t.loc[i, 'breakdown_lower_wick_fraction'] = (min(row.open,row.close)-row.low)/(row.high-row.low) if row.high>row.low else 0
            t.loc[i, 'taker_buy_share_5m'] = d.taker_buy_quote.iloc[b-4:b+1].sum()/d.quote_volume.iloc[b-4:b+1].sum()
            t.loc[i, 'quote_volume_60m'] = d.quote_volume.iloc[b-59:b+1].sum()
            # Recreate frozen daily gate solely to recover exact P/R timestamps.
            daily_idx = pd.date_range(trade.entry_time.normalize()-pd.Timedelta(days=20), periods=21, freq='D')
            daily = pd.DataFrame({name:trade.daily_upper for name in ['open','high','low','close','volume']}, index=daily_idx)
            daily.loc[daily_idx[-1], 'open'] = trade.daily_open
            result = evaluate_signal(symbol, daily, window, trade.entry_time, int(trade.daily_rank))
            s = result.get('signal')
            if not s or not np.isclose(s['peak'], trade.peak) or not np.isclose(s['atr'],trade.atr):
                errors.append([int(i), 'signal_reconstruction_mismatch'])
                continue
            p = d.index.get_loc(pd.Timestamp(s['pump_time'], unit='s', tz='UTC'))
            r = d.index.get_loc(pd.Timestamp(s['reversal_time'], unit='s', tz='UTC'))
            pr, rr = d.iloc[p], d.iloc[r]
            t.loc[i,'pump_to_entry_minutes'] = b-p+1
            t.loc[i,'reversal_to_breakdown_minutes'] = b-r
            t.loc[i,'pump_body_pct'] = (pr.close/pr.open-1)*100
            t.loc[i,'pump_upper_wick_fraction'] = (pr.high-max(pr.open,pr.close))/(pr.high-pr.low)
            t.loc[i,'reversal_upper_wick_fraction'] = (rr.high-max(rr.open,rr.close))/(rr.high-rr.low)
            t.loc[i,'pump_over_upper_pct'] = (pr.close/upper.iloc[p]-1)*100
            t.loc[i,'breakdown_to_pump_volume'] = row.volume/pr.volume
            t.loc[i,'reversal_to_pump_volume'] = rr.volume/pr.volume
            t.loc[i,'rise_before_pump_15m_pct'] = (pr.open/c.iloc[p-15]-1)*100
        if k % 15 == 0:
            print(f'FEATURES {k+1} symbols processed', flush=True)
    t.to_csv(OUT/'trade_features.csv',index=False)
    # Use January only to choose quartile edges; February/March are temporal checks.
    features = ['volume_ratio','daily_rank','hour_utc','stop_pct','atr_pct','stop_atr','daily_gap_pct','day_gain_pct',
                'return_5m_pct','return_15m_pct','return_60m_pct','return_120m_pct','bandwidth_pct',
                'upper_slope_5m_pct','mid_slope_5m_pct','above_upper_count_20','entry_band_position',
                'breakdown_volume_ratio','breakdown_body_pct','breakdown_lower_wick_fraction',
                'taker_buy_share_5m','quote_volume_60m','pump_to_entry_minutes','reversal_to_breakdown_minutes',
                'pump_body_pct','pump_upper_wick_fraction','reversal_upper_wick_fraction','pump_over_upper_pct',
                'breakdown_to_pump_volume','reversal_to_pump_volume','rise_before_pump_15m_pct']
    cuts, comparisons, definitions = [], [], {}
    train = t[t.month == '2026-01']
    for f in features:
        edges = np.unique(train[f].dropna().quantile([.25,.5,.75]).values)
        edges = [-np.inf]+list(edges)+[np.inf]
        definitions[f] = edges
        bins = pd.cut(t[f], edges, duplicates='drop')
        for period, mask in [('ALL',t.index==t.index)]+[(m,t.month==m) for m in sorted(t.month.unique())]:
            for bucket, g in t[mask].groupby(bins[mask], observed=True):
                cuts.append(dict(feature=f, bucket=str(bucket), period=period, **stats(g)))
        comparisons.append(dict(feature=f, losing_median=float(t.loc[t.net_pnl<0,f].median()), winning_median=float(t.loc[t.net_pnl>0,f].median())))
    pd.DataFrame(cuts).to_csv(OUT/'feature_buckets.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'winner_loser_medians.csv',index=False)
    # Outcome-only diagnostics never used as entry filters.
    exits = {reason:stats(g) for reason,g in t.groupby('exit_reason')}
    audit = dict(baseline=stats(t), monthly={m:stats(g) for m,g in t.groupby('month')},
                 exits=exits, symbols=int(t.symbol.nunique()), days=int(t.day.nunique()), reconstruction_errors=errors,
                 features=features, cut_edges={f:[None if not np.isfinite(x) else float(x) for x in e] for f,e in definitions.items()},
                 notes=['Entry features end at breakdown candle close; outcome columns excluded from predictors.',
                        'Quartile thresholds computed on January only. All months were already inspected in the preceding backtest; checks are exploratory, not untouched OOS.',
                        'Subset PnL is attribution for original accepted trades, not a new portfolio backtest.',
                        'Historical UTC-day ranking differs from rolling 24h live rank; original universe coverage and listing/liquidity eligibility require separate validation.'])
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
