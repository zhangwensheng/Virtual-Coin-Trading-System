"""Diagnose -> finite candidates -> chronological selection -> locked evaluation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from functools import lru_cache

import pandas as pd

from futures_strategy.adaptive_combo_research import Settings, pair_signals, run_portfolio, single_signals
from futures_strategy.combo_candidates import generate_candidates
from futures_strategy.combo_features import select_sleeves


OUT = Path('outputs/combo_optimization_20260915')
SYMBOLS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT']


@lru_cache(maxsize=1)
def input_signature():
    digest=hashlib.sha256()
    paths=[OUT/f'features_{s}.pkl' for s in SYMBOLS]
    paths += [Path(f'data/binance_funding_arbitrage/{s}_funding.csv') for s in SYMBOLS]
    paths += list(Path('data/adaptive_combo_research').glob('*_funding_2026-08.csv'))
    for path in sorted(paths):
        with path.open('rb') as file:
            digest.update(hashlib.file_digest(file,'sha256').digest())
    return digest.hexdigest()


def load_inputs(extend_funding=False):
    frames = {s:pd.read_pickle(OUT/f'features_{s}.pkl') for s in SYMBOLS}
    rates = {}
    for s in SYMBOLS:
        f = pd.read_csv(f'data/binance_funding_arbitrage/{s}_funding.csv')
        ts = pd.to_datetime(f.funding_time,utc=True,format='mixed').dt.floor('min')
        rates[s] = pd.Series(f.funding_rate.to_numpy(),index=ts)
        if extend_funding:
            extra=pd.read_csv(f'data/adaptive_combo_research/{s}_funding_2026-08.csv')
            et=pd.to_datetime(extra.funding_time,utc=True,format='mixed').dt.floor('min')
            series=pd.Series(extra.funding_rate.to_numpy(),index=et)
            common=rates[s].index.intersection(series.index)
            if not (abs(rates[s].loc[common]-series.loc[common])<1e-12).all():
                raise ValueError('Funding sources disagree')
            merged=pd.concat([rates[s],series])
            rates[s]=merged[~merged.index.duplicated(keep='last')].sort_index()
            expected=pd.date_range('2026-08-27','2026-09-01',inclusive='left',freq='8h',tz='UTC')
            if len(expected.difference(rates[s].index)):
                raise ValueError('Incomplete holdout funding')
    return frames,rates


def run_one(name,signals,frames,rates,validity,start,end,config=None):
    config = config or Settings(mode='fixed')
    start,end = pd.Timestamp(start,tz='UTC'),pd.Timestamp(end,tz='UTC')
    signature=hashlib.sha256(json.dumps(dict(signals=signals,settings=asdict(config),start=str(start),end=str(end),
        inputs=input_signature(),engine=hashlib.sha256(Path('futures_strategy/adaptive_combo_research.py').read_bytes()).hexdigest()),
        sort_keys=True,default=str).encode()).hexdigest()
    dest=OUT/'runs'/name
    previous=dest/'summary.json'
    if previous.exists():
        cached=json.loads(previous.read_text())
        if cached.get('run_signature')==signature:
            print('verified cache',name,flush=True)
            return cached
    summary,trades,curve,rejects = run_portfolio(frames,signals,rates,validity,start,end,config)
    if abs(summary['capital_reconciliation_error'])>1e-7:
        raise AssertionError('account reconciliation')
    dest.mkdir(parents=True,exist_ok=True)
    trades.to_csv(dest/'trades.csv',index=False)
    curve.to_csv(dest/'equity.csv')
    rejects.to_csv(dest/'rejections.csv',index=False)
    summary.update(run=name,start=str(start),end_exclusive=str(end),run_signature=signature)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(name,round(summary['return_pct'],4),summary['trades'],round(summary['max_drawdown_pct'],3),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['search','evaluate','holdout'],default='search')
    args=parser.parse_args()
    frames,rates=load_inputs(extend_funding=args.phase=='holdout')
    pfile=OUT/'pair_signals.pkl'
    pair_meta=OUT/'pair_cache_metadata.json'
    pair_signature=hashlib.sha256((input_signature()+hashlib.sha256(Path('futures_strategy/adaptive_combo_research.py').read_bytes()).hexdigest()).encode()).hexdigest()
    if pfile.exists() and pair_meta.exists() and json.loads(pair_meta.read_text()).get('signature')==pair_signature:
        paired,diag,validity=pd.read_pickle(pfile)
    else:
        paired,diag,validity=pair_signals(frames,[('BTCUSDT','ETHUSDT'),('SOLUSDT','ETHUSDT'),('BNBUSDT','ETHUSDT')])
        pd.to_pickle((paired,diag,validity),pfile)
        pair_meta.write_text(json.dumps({'signature':pair_signature}),encoding='utf-8')
        diag.to_csv(OUT/'pair_daily_features.csv',index=False)
    candidates=generate_candidates(frames,paired)
    (OUT/'candidate_counts.json').write_text(json.dumps({k:len(v) for k,v in candidates.items()},indent=2),encoding='utf-8')
    if args.phase=='search':
        rows,checks=[],[]
        for name,signals in candidates.items():
            # Preserve state gates already present on original-style signals.
            signals=[s for s in signals if s['eligible']]
            a=run_one(name+'_train',signals,frames,rates,validity,'2026-06-01','2026-07-01')
            b=run_one(name+'_validation',signals,frames,rates,validity,'2026-07-01','2026-08-01')
            rows += [a,b]
            checks.append(dict(name=name,train_net=a['final_equity']-1000,valid_net=b['final_equity']-1000,
                               train_n=a['trades'],valid_n=b['trades'],train_pf=a['profit_factor'],valid_pf=b['profit_factor']))
            pd.DataFrame(rows).to_csv(OUT/'candidate_results.csv',index=False)
        passed=select_sleeves(checks)
        # At most one rule per sleeve; choose on train+validation only.
        selected=[]
        for family in ('trend','reversal','pair'):
            choices=[r for r in checks if r['name'] in passed and r['name'].startswith(family)]
            if choices:
                selected.append(max(choices,key=lambda r:min(r['train_net'],r['valid_net']))['name'])
        shadow=max(checks,key=lambda r:min(r['train_net'],r['valid_net']))['name']
        decision=dict(selected=selected,shadow_best_worst_month=shadow,checks=checks,
                      admission='Positive net PnL June AND July; >=30 June trades and >=20 July trades. Then pick best minimum monthly net within each sleeve.',
                      train='2026-06',validation='2026-07',evaluation='2026-08-01 through 2026-08-26',
                      caveat='August was viewed in previous baseline research. Not a truly untouched holdout.',
                      development_note='Eight initial mechanisms were examined on June/July, then two slower momentum mechanisms were added. July therefore participated in development, not strict independent validation. Ten candidates in total.',
                      candidate_source_sha256=hashlib.sha256(Path('futures_strategy/combo_candidates.py').read_bytes()).hexdigest())
        (OUT/'locked_selection.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
        print('LOCKED',json.dumps(decision,ensure_ascii=False),flush=True)
    else:
        decision=json.loads((OUT/'locked_selection.json').read_text(encoding='utf-8'))
        digest=hashlib.sha256(Path('futures_strategy/combo_candidates.py').read_bytes()).hexdigest()
        if digest!=decision['candidate_source_sha256']:
            raise AssertionError('Candidate changed after selection lock')
        selected=[s for name in decision['selected'] for s in candidates[name] if s['eligible']]
        shadow=[s for s in candidates[decision['shadow_best_worst_month']] if s['eligible']]
        original=[s for symbol,f in frames.items() for s in single_signals(symbol,f)]+paired
        rows=[]
        period_start='2026-08-27' if args.phase=='holdout' else '2026-08-01'
        period_end='2026-09-01' if args.phase=='holdout' else '2026-08-27'
        suffix='fresh_aug27_31' if args.phase=='holdout' else 'august'
        for name,sigs,cfg in [('selected_combo',selected,Settings()),('selection_reference',shadow,Settings(mode='fixed')),
                               ('old_combo',original,Settings()),('selected_double_cost',selected,Settings(fee=.001,slip=.0004)),
                               ('selected_delay',selected,Settings(delay=2))]:
            rows.append(run_one(name+'_'+suffix,sigs,frames,rates,validity,period_start,period_end,cfg))
        pd.DataFrame(rows).to_csv(OUT/('fresh_holdout_results.csv' if args.phase=='holdout' else 'evaluation_results.csv'),index=False)


if __name__=='__main__':
    main()
