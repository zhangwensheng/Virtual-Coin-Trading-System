"""Reproducible, offline WR signal-subset research. Never writes account state.

The saved signal pool cannot validate newly admitted pumps/daily gates or a full
market universe. All Jan-Mar months have been inspected before: NOT untouched OOS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.optimize_wr_short_strategy import simulate_portfolio

SOURCE = ROOT/'outputs/wr_short_backtest_2026-01_2026-03_current_rules/signals.csv'
DATA = ROOT/'data/top_gainers_boll_short_1m/realtime_enriched'
FEE = .0005


def strategy_mask(pool: pd.DataFrame, *, max_rank: int) -> pd.Series:
    """Return the rows admitted by the proposed WR strategy."""
    if max_rank < 1:
        raise ValueError('max_rank_must_be_positive')
    entry_hours = pool.entry_time.dt.hour
    pump_hours = pool.pump_time.dt.hour
    reversal_hours = pool.reversal_time.dt.hour
    return (
        pool.daily_rank.between(1, max_rank)
        & pool.wr_drop.between(30, 50)
        & (pool.breakdown_lower_wick_fraction <= .35)
        & ~entry_hours.isin([6, 7])
        & ~pump_hours.isin([6, 7])
        & ~reversal_hours.isin([6, 7])
        & (entry_hours < 14)
    )


def metrics(trades):
    if trades.empty:
        return dict(trades=0,net_pnl=0.,win_rate_pct=0.,profit_factor=None,
                    closed_trade_drawdown_pct=0.,avg_trade=0.)
    ordered = trades.sort_values('exit_time')
    pnl = ordered.net_pnl
    equity = np.r_[1000., 1000.+pnl.cumsum().to_numpy()]
    dd = 1-equity/np.maximum.accumulate(equity)
    loss = -pnl[pnl<0].sum()
    return dict(trades=len(trades),net_pnl=float(pnl.sum()),
                win_rate_pct=float((pnl>0).mean()*100),
                profit_factor=float(pnl[pnl>0].sum()/loss) if loss>0 else None,
                closed_trade_drawdown_pct=float(dd.max()*100),avg_trade=float(pnl.mean()))


def entry_features(frame, signal):
    history = frame.loc[:signal['breakdown_time']].tail(121)
    if len(history)<121 or not (history.index.to_series().diff().dropna()==pd.Timedelta(minutes=1)).all():
        raise ValueError('insufficient_or_discontinuous_feature_history')
    row = history.iloc[-1]
    previous_vol = history.volume.iloc[-21:-1].mean()
    if previous_vol<=0:
        raise ValueError('zero_previous_volume')
    pump = history.loc[signal['pump_time']]
    result = dict(stop_pct=100*signal['risk_distance']/signal['entry_price'],
                  atr_pct=100*signal['atr']/signal['entry_price'],
                  pump_body_pct=100*(pump.close/pump.open-1),
                  breakdown_volume_ratio=float(row.volume/previous_vol),
                  breakdown_lower_wick_fraction=float((min(row.open,row.close)-row.low)/(row.high-row.low)) if row.high>row.low else 0.,
                  pump_to_entry_minutes=float((signal['breakdown_time']-signal['pump_time']).total_seconds()/60+1))
    for n in (5,15,60,120):
        result[f'return_{n}m_pct'] = float((row.close/history.close.iloc[-1-n]-1)*100)
    return result


def replay_exit(frame, signal, *, breakeven=False, slip=.0002):
    """Next-bar market entry; gap-aware stop-first OHLC; new stops next bar only."""
    window = frame.loc[signal['entry_time']:].head(30)
    expected = pd.date_range(signal['entry_time'],periods=30,freq='min')
    if not window.index.equals(expected):
        raise ValueError('missing_exit_bars')
    entry = float(window.iloc[0].open)*(1-slip)
    stop = float(signal['stop_price'])
    risk = stop-entry
    target = entry-2*risk
    if risk<=0 or target<=0:
        raise ValueError('invalid_entry_gap')
    live_stop, lowest, trailing = stop, entry, False
    exit_price = float(window.iloc[-1].close)*(1+slip)
    exit_time = window.index[-1]+pd.Timedelta(minutes=1)
    reason = 'TIME_EXIT'
    for t, bar in window.iterrows():
        if bar.high>=live_stop:
            exit_price = max(float(bar.open),live_stop)*(1+slip)
            reason = 'TRAIL_STOP' if trailing else 'STOP'
        elif bar.low<=target:
            exit_price = target*(1+slip)
            reason = 'TARGET'
        else:
            if bar.low<=entry-risk:
                trailing = True
            if trailing:
                lowest = min(lowest,float(bar.low))
                live_stop = min(live_stop,lowest+risk)
                if breakeven:
                    net_zero_ask = entry*(1-FEE)/((1+slip)*(1+FEE))
                    live_stop = min(live_stop,net_zero_ask)
            continue
        exit_time = t+pd.Timedelta(minutes=1)
        break
    return dict(entry_price=entry,stop_price=stop,target_price=target,risk_distance=risk,
                exit_price=exit_price,exit_time=exit_time,exit_reason=reason,
                unit_risk=stop*(1+slip)-entry+FEE*(entry+stop*(1+slip)),
                unit_price_pnl=entry-exit_price)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/wr_factor_research_20260912')
    parser.add_argument('--max-rank',type=int,default=20)
    args = parser.parse_args()
    if args.max_rank < 1:
        parser.error('--max-rank must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    signals = pd.read_csv(SOURCE)
    for col in ('entry_time','exit_time','breakdown_time','pump_time','reversal_time'):
        signals[col] = pd.to_datetime(signals[col],utc=True)
    signals = signals[signals.daily_rank<=args.max_rank].copy()
    signals['entry_hour_utc'] = signals.entry_time.dt.hour
    signals['month'] = signals.entry_time.dt.strftime('%Y-%m')
    enriched, errors = [], []
    variants = {'causal':[], 'net_breakeven':[], 'slip_5bps':[]}
    for n,(symbol,group) in enumerate(signals.groupby('symbol')):
        files = sorted(DATA.glob(f'{symbol.lower()}_1m_*.csv'))
        if not files:
            errors.append(dict(symbol=symbol,error='missing_symbol'))
            continue
        frame = pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
        frame.index = pd.to_datetime(frame.pop('timestamp'),utc=True)
        frame = frame[~frame.index.duplicated()].sort_index()
        for _,row in group.iterrows():
            record = row.to_dict()
            try:
                record.update(entry_features(frame,record))
                priced = {name:dict(record,**replay_exit(frame,record,breakeven=name=='net_breakeven',
                           slip=.0005 if name=='slip_5bps' else .0002)) for name in variants}
            except ValueError as exc:
                errors.append(dict(symbol=symbol,entry_time=str(row.entry_time),error=str(exc)))
                continue
            enriched.append(record)
            for name in variants:
                variants[name].append(priced[name])
        if n%10==0:
            print(f'features: {n+1} symbols',flush=True)
    features = pd.DataFrame(enriched)
    features.to_csv(args.output/'signal_features.csv',index=False)
    frames = dict(saved_fills=features,**{name:pd.DataFrame(rows) for name,rows in variants.items()})
    comparisons = []
    for execution, pool in frames.items():
        for max_drop in (35,40,45,50,60,100):
            for blocked in (True,False):
                mask = pool.wr_drop.between(30,max_drop)
                if blocked:
                    mask &= ~pool.entry_hour_utc.isin([6,7])
                    # Observations cannot span a blocked pump/reversal candle.
                    mask &= ~pool.pump_time.dt.hour.isin([6,7])
                    mask &= ~pool.reversal_time.dt.hour.isin([6,7])
                mask &= pool.entry_hour_utc<14
                trades = simulate_portfolio(pool[mask])
                result = dict(execution=execution,max_wr_drop=max_drop,block_6_7=blocked,
                              raw_signals=int(mask.sum()),**metrics(trades))
                for month in ('2026-01','2026-02','2026-03'):
                    selected = trades[trades.entry_time.dt.strftime('%Y-%m')==month] if not trades.empty else trades
                    result[month+'_net'] = metrics(selected)['net_pnl']
                    result[month+'_trades'] = len(selected)
                if not trades.empty:
                    gaps = trades.sort_values('entry_time').entry_time.diff().dt.total_seconds()/3600
                    result['max_observed_gap_hours'] = float(gaps.max())
                comparisons.append(result)
                if execution=='causal' and blocked and max_drop in (35,50):
                    trades.to_csv(args.output/f'trades_causal_wr30_{max_drop}.csv',index=False)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(args.output/'variant_comparison.csv',index=False)

    strategy_comparisons = []
    for execution, pool in frames.items():
        for max_rank in sorted({5, 7, args.max_rank}):
            selection = strategy_mask(pool, max_rank=max_rank)
            trades = simulate_portfolio(pool[selection])
            record = dict(
                execution=execution,
                max_rank=max_rank,
                raw_signals=int(selection.sum()),
                **metrics(trades),
            )
            for month in ('2026-01','2026-02','2026-03'):
                part = trades[trades.entry_time.dt.strftime('%Y-%m')==month] if not trades.empty else trades
                record[month+'_net'] = metrics(part)['net_pnl']
                record[month+'_trades'] = len(part)
            if not trades.empty:
                gaps = trades.sort_values('entry_time').entry_time.diff().dt.total_seconds()/3600
                record['max_observed_gap_hours'] = float(gaps.max())
            strategy_comparisons.append(record)
            if execution == 'causal' and max_rank in {5, 7, args.max_rank}:
                trades.to_csv(args.output/f'trades_proposed_top{max_rank}.csv',index=False)
    pd.DataFrame(strategy_comparisons).to_csv(args.output/'strategy_comparison.csv',index=False)

    rank_sweep = []
    for execution in ('causal', 'slip_5bps'):
        pool = frames[execution]
        for max_rank in range(1, args.max_rank + 1):
            selection = strategy_mask(pool, max_rank=max_rank)
            trades = simulate_portfolio(pool[selection])
            record = dict(
                execution=execution,
                max_rank=max_rank,
                raw_signals=int(selection.sum()),
                **metrics(trades),
            )
            for month in ('2026-01','2026-02','2026-03'):
                part = trades[trades.entry_time.dt.strftime('%Y-%m')==month] if not trades.empty else trades
                record[month+'_net'] = metrics(part)['net_pnl']
                record[month+'_trades'] = len(part)
            rank_sweep.append(record)
    pd.DataFrame(rank_sweep).to_csv(args.output/'rank_cap_sweep.csv',index=False)

    # Rounded, interpretable follow-up hypotheses; exploratory and not promoted.
    factor_comparisons = []
    for execution,pool in frames.items():
        base = (pool.wr_drop.between(30,50)&~pool.entry_hour_utc.isin([6,7])
                &~pool.pump_time.dt.hour.isin([6,7])&~pool.reversal_time.dt.hour.isin([6,7])
                &(pool.entry_hour_utc<14))
        filters = dict(wide_base=base,wick_le_35pct=base&(pool.breakdown_lower_wick_fraction<=.35),
                       stop_le_1_25pct=base&(pool.stop_pct<=1.25),daily_wr_le_minus12=base&(pool.daily_wr<=-12))
        for name,selection in filters.items():
            trades = simulate_portfolio(pool[selection])
            record = dict(execution=execution,factor=name,**metrics(trades))
            for month in ('2026-01','2026-02','2026-03'):
                part = trades[trades.month==month] if not trades.empty else trades
                record[month+'_net'] = metrics(part)['net_pnl']
                record[month+'_trades'] = len(part)
            factor_comparisons.append(record)
            if execution=='causal':
                trades.to_csv(args.output/f'trades_factor_{name}.csv',index=False)
    pd.DataFrame(factor_comparisons).to_csv(args.output/'factor_candidate_comparison.csv',index=False)

    # Descriptive attribution on the wider 30..50 cohort, not a new optimized portfolio.
    causal = frames['causal']
    mask = causal.wr_drop.between(30,50)&~causal.entry_hour_utc.isin([6,7])&(causal.entry_hour_utc<14)
    accepted = simulate_portfolio(causal[mask])
    factor_names = ['wr_drop','daily_wr','volume_ratio','stop_pct','atr_pct','pump_body_pct',
                    'breakdown_body_pct','breakdown_volume_ratio','breakdown_lower_wick_fraction',
                    'return_5m_pct','return_15m_pct','return_60m_pct','return_120m_pct','pump_to_entry_minutes']
    buckets, contrasts, thresholds = [], [], {}
    jan = accepted[accepted.entry_time.dt.month==1]
    for factor in factor_names:
        cuts = sorted(set(float(v) for v in jan[factor].quantile([.25,.5,.75]).dropna()))
        thresholds[factor] = cuts
        categories = pd.cut(accepted[factor],[-np.inf,*cuts,np.inf],duplicates='drop')
        for period in ('ALL','2026-01','2026-02','2026-03'):
            selection = accepted if period=='ALL' else accepted[accepted.month==period]
            for bucket,group in selection.groupby(categories,observed=True):
                buckets.append(dict(factor=factor,period=period,bucket=str(bucket),**metrics(group)))
        contrasts.append(dict(factor=factor,
                              winner_median=float(accepted.loc[accepted.net_pnl>0,factor].median()),
                              loser_median=float(accepted.loc[accepted.net_pnl<=0,factor].median())))
    pd.DataFrame(buckets).to_csv(args.output/'factor_buckets.csv',index=False)
    pd.DataFrame(contrasts).to_csv(args.output/'winner_loser_features.csv',index=False)
    audit = dict(source=str(SOURCE),max_rank=args.max_rank,source_signals=len(signals),usable_signals=len(features),errors=errors,
                 january_quantile_edges=thresholds,notes=[
        'Signal-subset replay only: no new candidate search. Cannot assess WR<30, daily-gate, volume-min or observation-window relaxation.',
        'Historical universe preselected by old BOLL caches; UTC-day rank approximates rolling 24h live ranking. Not full market.',
        'All Jan-Mar months previously inspected; February/March are exploratory temporal checks, NOT untouched out-of-sample.',
        'Causal fills: next-minute open, stop-first, adverse stop gaps, new trailing stop active next bar; intrabar order is unknown.',
        '0.05% fees per side; 2bps slippage (stress 5bps). Funding, spread, exchange quantity rounding not modeled.',
        '1000U, 3U planned risk, 320U notional/position, 2 positions. Uses prior approximate cash-based portfolio replay.',
        'Daily loss and counts in this research replay use UTC, whereas paper uses Beijing day; floating-loss risk not modeled.',
        'Drawdown is closed-trade equity including initial capital; not intratrade mark-to-market drawdown.',
        'Bucket PnL is descriptive attribution, not portfolio rerun; many comparisons and small samples can overfit.',
        'No parameters are automatically promoted to the running strategy.'
    ])
    (args.output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(comparison[(comparison.block_6_7)&comparison.max_wr_drop.isin([35,50])].to_string(index=False))
    print('errors',errors,flush=True)


if __name__=='__main__':
    main()
