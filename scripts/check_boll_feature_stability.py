"""Exploratory sensitivity checks on entry-time features, not portfolio results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyze_boll_trade_features import stats, OUT

t = pd.read_csv(OUT/'trade_features.csv')
rules = {
    'late_utc_14_24': t.hour_utc >= 14,
    'weak_pump_body_lt_0_65pct': t.pump_body_pct < .65,
    'large_breakdown_body_gt_0_7pct': t.breakdown_body_pct > .7,
    'buy_share_gt_55pct': t.taker_buy_share_5m > .55,
    'pump_volume_le_5x': t.volume_ratio <= 5,
    'no_reversal_upper_wick_lt_3pct': t.reversal_upper_wick_fraction < .03,
    'fast_rise_15m_gt_2_5pct': t.return_15m_pct > 2.5,
    'bandwidth_gt_5_6pct': t.bandwidth_pct > 5.6,
    'pump_to_entry_ge_5m': t.pump_to_entry_minutes >= 5,
}
rng = np.random.default_rng(20260911)
rows, summary = [], {}
for name,mask in rules.items():
    for period, pm in [('ALL', np.ones(len(t),dtype=bool))]+[(m,t.month==m) for m in sorted(t.month.unique())]:
        for label, flag in [('feature',mask),('complement',~mask)]:
            g=t[pm & flag]
            rows.append(dict(rule=name, period=period, group=label, days=g.day.nunique(),symbols=g.symbol.nunique(),**stats(g)))
    groups=t.assign(flag=mask.astype(int)).groupby(['day','flag']).net_pnl.agg(['sum','count']).unstack(fill_value=0)
    diffs=[]
    for _ in range(3000):
        sample=groups.iloc[rng.integers(0,len(groups),size=len(groups))].sum()
        if sample['count',1] and sample['count',0]:
            diffs.append(sample['sum',1]/sample['count',1]-sample['sum',0]/sample['count',0])
    both=[]
    for symbol,g in t.assign(flag=mask).groupby('symbol'):
        if g.flag.nunique()==2:
            both.append(g.loc[g.flag,'net_pnl'].mean()-g.loc[~g.flag,'net_pnl'].mean())
    worst=t.groupby('symbol').net_pnl.sum().sort_values().head(5).index
    clean=t[~t.symbol.isin(worst)]
    clean_mask=mask.loc[clean.index]
    summary[name] = dict(day_cluster_bootstrap_difference_ci95=np.quantile(diffs,[.025,.975]).tolist(),
                         difference_avg=stats(t[mask])['avg']-stats(t[~mask])['avg'],
                         shared_symbols=len(both),same_symbol_negative_fraction=float(np.mean(np.array(both)<0)),
                         same_symbol_equal_weight_mean_difference=float(np.mean(both)),
                         excluded_worst_five=list(worst),excluding_worst_five_feature=stats(clean[clean_mask]),
                         excluding_worst_five_complement=stats(clean[~clean_mask]))
pd.DataFrame(rows).to_csv(OUT/'rule_stability.csv',index=False)
(OUT/'stability_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).query("period == 'ALL'").to_string(index=False))
print(json.dumps(summary,indent=2))
print('TIME BY HOUR',t.groupby('hour_utc').net_pnl.agg(['count','sum','mean']).to_string())
print('DURATION',t.groupby('pump_to_entry_minutes').net_pnl.agg(['count','sum','mean']).to_string())
print('OUTCOME MEDIANS',t.assign(outcome=np.where(t.net_pnl>0,'WIN','LOSS')).groupby('outcome')[['holding_minutes','volume_ratio','pump_body_pct','breakdown_body_pct','taker_buy_share_5m','fees_risk_pct']].median().to_string())
