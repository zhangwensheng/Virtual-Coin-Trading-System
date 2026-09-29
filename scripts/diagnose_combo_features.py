"""Rebuild entry-time features for non-overlapping trade ledgers."""
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from futures_strategy.data import load_csv
from futures_strategy.combo_features import enrich_features


def main():
    out = Path('outputs/combo_optimization_20260915')
    out.mkdir(exist_ok=True)
    frames = {}
    symbols = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT']
    for s in symbols:
        raw = load_csv(Path('data/adaptive_combo_research') / f'{s.lower()}_1m_2026-05_2026-08.csv')
        frames[s] = enrich_features(raw)
        # Derived local cache only, not a source of future information.
        frames[s].to_pickle(out / f'features_{s}.pkl')
        print('features', s, flush=True)
    paths = ['outputs/adaptive_combo_research_20260915/adaptive_combo/trades.csv',
             'outputs/adaptive_combo_research_20260915/july_fresh_account/adaptive_combo/trades.csv',
             'outputs/adaptive_combo_research_20260915/august_fresh_account/adaptive_combo/trades.csv']
    records = []
    columns = ['q_adx','volume_ratio','atr_pct','extension_atr','trend_gap_atr','flow_5',
               'efficiency_60','vwap_distance_atr','ret_15','ret_60','ret_240','ret_1440','wr','market_vol_ratio']
    for path in paths:
        trades = pd.read_csv(path)
        for t in trades.to_dict('records'):
            if '__' in t['symbols']:
                continue
            ts = pd.Timestamp(t['entry_time']) - pd.Timedelta(minutes=1)
            f = frames[t['symbols']].loc[ts]
            btc = frames['BTCUSDT'].loc[ts]
            row = dict(t, source=path, month=ts.strftime('%Y-%m'), win=t['net_pnl'] > 0)
            row.update({col:float(f[col]) for col in columns})
            row['cost_risk'] = .0014 / (t['initial_risk']/t['initial_notional'])
            row['net_r'] = t['net_pnl']/t['initial_risk']
            row['btc_aligned'] = bool(t['side'] * (btc.h_fast-btc.h_slow) > 0)
            row['flow_aligned'] = float(t['side'] * f.flow_5)
            row['momentum_aligned'] = float(t['side'] * f.ret_60)
            row['giveback_close_r'] = t['max_favorable_pnl']/t['initial_risk']
            records.append(row)
    data = pd.DataFrame(records)
    data.to_csv(out/'baseline_trade_features.csv', index=False)
    # Outcome grouping describes differences; only pre-entry features can be used as filters.
    stats = data.groupby(['month','strategy','win'])[columns+['cost_risk','flow_aligned','momentum_aligned','net_r','giveback_close_r']].mean()
    stats.to_csv(out/'win_loss_feature_means.csv')
    cuts = {'cost_risk': [0,.2,.35,.5,1,100], 'q_adx':[0,18,25,35,50,100],
            'atr_pct':[0,.001,.002,.004,.008,1], 'flow_aligned':[-1,-.1,0,.1,.3,1],
            'momentum_aligned':[-1,-.003,0,.003,.01,1], 'efficiency_60':[0,.1,.2,.35,.5,1],
            'volume_ratio':[0,.7,1,1.5,2.5,100], 'market_vol_ratio':[0,.7,1,1.5,2.5,100]}
    rows = []
    for feature,bins in cuts.items():
        buckets = pd.cut(data[feature],bins,include_lowest=True)
        for (month,strategy,bucket), group in data.groupby([data.month,data.strategy,buckets],observed=True):
            rows.append(dict(feature=feature,bucket=str(bucket),month=month,strategy=strategy,n=len(group),
                             net_pnl=group.net_pnl.sum(),mean_r=group.net_r.mean(),win_rate=group.win.mean(),
                             fees=group.fees.sum()))
    buckets = pd.DataFrame(rows)
    buckets.to_csv(out/'factor_buckets.csv',index=False)
    print('JUNE TREND FACTORS', buckets[(buckets.month=='2026-06')&(buckets.strategy=='trend')].to_string(index=False),flush=True)
    print('JUNE SIDES', data[data.month=='2026-06'].groupby(['strategy','side']).net_pnl.agg(['count','sum']).to_string(),flush=True)
    wr = pd.read_csv('outputs/wr_short_backtest_2026-01_2026-03_current_rules/trades.csv')
    wr['month'] = wr.entry_time.str[:7]
    wr['win'] = wr.net_pnl > 0
    wr['stop_pct'] = abs(wr.entry_price-wr.stop_price)/wr.entry_price
    wr['fee_risk'] = (wr.entry_fee+wr.exit_fee)/wr.planned_risk
    wr.groupby(['month','win'])[['volume_ratio','daily_wr','pump_wr','breakdown_wr','wr_drop','breakdown_body_pct','stop_pct','fee_risk','net_pnl']].agg(['count','mean']).to_csv(out/'historical_wr_features.csv')
    # One canonical WR ledger; do not pool overlapping optimization variants.
    paper = []
    for name,table in [('boll_short_workstation','account'),('pair_workstation','portfolio')]:
        path = Path('outputs')/name/'paper.sqlite3'
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as conn:
            body = json.loads(conn.execute('SELECT body FROM '+table+' WHERE id=1').fetchone()[0])
            closed = body['closed']
            values = list(closed.values()) if isinstance(closed,dict) else closed
            # Closed position data only, never account configuration/credentials.
            for p in values:
                paper.append(dict(source=name,position=p))
    (out/'paper_closed_positions.json').write_text(json.dumps(paper,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PAPER_ROWS',len(paper),'KEYS',[(p['source'],list(p['position'])) for p in paper[:6]],flush=True)


if __name__ == '__main__':
    main()
