"""Reproducible fixed-parameter experiment, separate from paper/live services."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from futures_strategy.adaptive_combo_research import (
    Settings, pair_signals, prepare_features, run_portfolio, single_signals,
)
from futures_strategy.data import load_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2026-06-01')
    parser.add_argument('--end', default='2026-08-27', help='Exclusive UTC end')
    parser.add_argument('--output', default='outputs/adaptive_combo_research_20260915')
    parser.add_argument('--experiments', default=None, help='Optional comma-separated experiment names')
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start, tz='UTC'), pd.Timestamp(args.end, tz='UTC')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
    frames, funding, manifest, signals = {}, {}, [], []
    for s in symbols:
        path = Path('data/adaptive_combo_research') / f'{s.lower()}_1m_2026-05_2026-08.csv'
        raw = load_csv(path)
        raw = raw.loc[(raw.index >= start - pd.Timedelta(days=10)) & (raw.index < end)]
        expected = pd.date_range(start - pd.Timedelta(days=10), end, inclusive='left', freq='min')
        missing = expected.difference(raw.index)
        if len(missing) or raw.index.has_duplicates or not raw.index.equals(expected):
            raise ValueError(f'{s}: non-contiguous minute data: {len(missing)} missing')
        invalid = ((raw[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
                   | (raw.high < raw[['open', 'close', 'low']].max(axis=1))
                   | (raw.low > raw[['open', 'close', 'high']].min(axis=1)))
        if invalid.any():
            raise ValueError(f'{s}: invalid OHLC candles')
        fpath = Path('data/binance_funding_arbitrage') / f'{s}_funding.csv'
        ff = pd.read_csv(fpath)
        ff['timestamp'] = pd.to_datetime(ff.funding_time, utc=True, format='mixed').dt.floor('min')
        if ff.timestamp.duplicated().any():
            raise ValueError(f'{s}: duplicate funding settlement')
        coverage = ff.set_index('timestamp').sort_index().funding_rate
        if coverage.index.min() > start or coverage.index.max() < end:
            raise ValueError(f'{s}: incomplete funding coverage')
        sub = coverage.loc[(coverage.index >= start) & (coverage.index < end)]
        gaps = sub.index.to_series().diff().dt.total_seconds().dropna()
        if len(gaps) and gaps.max() > 8 * 3600 + 60:
            raise ValueError(f'{s}: funding settlement gap')
        funding[s] = coverage
        frames[s] = prepare_features(raw)
        current = single_signals(s, frames[s])
        signals.extend(current)
        manifest.append(dict(symbol=s, path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             rows=len(raw), start=str(raw.index[0]), end=str(raw.index[-1]), missing_minutes=len(missing),
                             funding_path=str(fpath.resolve()), funding_sha256=hashlib.sha256(fpath.read_bytes()).hexdigest(),
                             funding_events=len(sub), raw_single_signals=len(current)))
        print(f'prepared {s}: {len(raw)} minutes, {len(current)} raw signals', flush=True)
    paired, diagnostics, validity = pair_signals(frames, [('BTCUSDT', 'ETHUSDT'), ('SOLUSDT', 'ETHUSDT'), ('BNBUSDT', 'ETHUSDT')])
    signals.extend(paired)
    print(f'pair signals: {len(paired)}; daily relationships checked: {len(diagnostics)}', flush=True)
    diagnostics.to_csv(out / 'pair_diagnostics.csv', index=False)
    # Signal indices refer to candle OPEN times. A +1 minute delay is the first
    # executable open following that candle's close, not a same-close fill.
    signal_rows = [dict(s, signal_candle_open=str(next(iter(frames.values())).index[s['i']])) for s in signals]
    pd.DataFrame(signal_rows).to_csv(out / 'raw_signals.csv', index=False)
    metadata = dict(start=str(start), end_exclusive=str(end), manifest=manifest,
                    engine_sha256=hashlib.sha256(Path('futures_strategy/adaptive_combo_research.py').read_bytes()).hexdigest(),
                    limitations=[
                        'Fixed five major coins, not point-in-time exchange-wide top-five or top-twenty rankings.',
                        'No parameter search; the final month is a chronological holdout, not untouched by all prior project research.',
                        'Funding timestamp rounded down to minute; notional uses settlement-minute trade open, not exact exchange mark.',
                        'Five/15/60-minute candles become usable only after closing; completed daily pair calibration uses preceding seven days.',
                        'Pair cash-stop execution uses the joint adverse OHLC envelope; pessimistic but cannot model actual leg fill sequencing.',
                        'No order book, queue position, exchange minimum lot rounding, outage or liquidation reconstruction.',
                        'Intrabar equity envelope sums leg extremes that need not occur simultaneously and is not observed mark-price drawdown.',
                        'Borrowing/leverage limited by gross notional <= 2x equity at entry; no exchange margin ledger.',
                        'At 12% drawdown new entries are latched off for the rest of the run; at 15% flatten next minute.',
                        'BTC market-state overlay and live volatility/spread anomaly filter are not implemented in this first prototype.',
                    ],
                    settings=asdict(Settings()),
                    risk_budget={'trend': .45, 'reversal': .35, 'pair': .20},
                    execution='adverse-first OHLC stop; taker both sides; partial trend 50% at 1R then 3R/trailing; 15-minute post-exit cooldown')
    (out / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    experiments = [
        ('trend_only', Settings(mode='trend')),
        ('reversal_only', Settings(mode='reversal')),
        ('pair_only', Settings(mode='pair')),
        ('fixed_combo', Settings(mode='fixed')),
        ('adaptive_combo', Settings()),
        ('adaptive_double_cost', Settings(fee=.001, slip=.0004)),
        ('adaptive_delay_1_extra_minute', Settings(delay=2)),
    ]
    if args.experiments:
        selected = set(args.experiments.split(','))
        known = {name for name, _ in experiments}
        if selected - known:
            raise ValueError(f'Unknown experiments: {selected - known}')
        experiments = [(name, config) for name, config in experiments if name in selected]
    summaries, monthly, sleeves = [], [], []
    for name, config in experiments:
        print(f'running {name}', flush=True)
        summary, ledger, curve, rejects = run_portfolio(frames, signals, funding, validity, start, end, config)
        summary['experiment'] = name
        if abs(summary['capital_reconciliation_error']) > 1e-7:
            raise AssertionError(f'Unreconciled account: {summary}')
        dest = out / name
        dest.mkdir(exist_ok=True)
        ledger.to_csv(dest / 'trades.csv', index=False)
        curve.to_csv(dest / 'equity_curve.csv')
        rejects.to_csv(dest / 'rejection_counts.csv', index=False)
        (dest / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        previous = config.capital
        # Curve records end-of-minute; assign midnight closes to preceding minute's month.
        month_keys = (curve.index - pd.Timedelta(nanoseconds=1)).strftime('%Y-%m')
        for month, block in curve.groupby(month_keys):
            final = float(block.equity.iloc[-1])
            monthly.append(dict(experiment=name, month=month, return_pct=(final / previous - 1) * 100,
                                net_pnl=final - previous, end_equity=final))
            previous = final
        if len(ledger):
            # Check caps from actual accepted entries, including both pair legs.
            expanded = []
            for row in ledger.itertuples():
                for symbol in row.symbols.split('__'):
                    expanded.append(((row.entry_time + pd.Timedelta(hours=8)).date(), symbol))
            if pd.Series(expanded).value_counts().max() > 3:
                raise AssertionError('Cross-strategy per-symbol daily limit exceeded')
            for strategy, block in ledger.groupby('strategy'):
                sleeves.append(dict(experiment=name, strategy=strategy, trades=len(block), net_pnl=block.net_pnl.sum(),
                                    price_pnl=block.price_pnl.sum(), fees=block.fees.sum(), funding=block.funding.sum(),
                                    avg_holding_minutes=block.holding_minutes.mean()))
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        pd.DataFrame(summaries).to_csv(out / 'comparison.csv', index=False)
        pd.DataFrame(monthly).to_csv(out / 'monthly.csv', index=False)
        pd.DataFrame(sleeves).to_csv(out / 'strategy_contributions.csv', index=False)
    table = pd.DataFrame(summaries)
    labels = {'trend_only': '趋势单策略', 'reversal_only': '反转单策略', 'pair_only': '配对单策略',
              'fixed_combo': '固定组合', 'adaptive_combo': '行情切换组合', 'adaptive_double_cost': '切换组合·双倍成本',
              'adaptive_delay_1_extra_minute': '切换组合·额外延迟一分钟'}
    lines = ['# 三策略组合：首轮固定参数回测', '',
             f'区间：{start.date()} 至 {(end - pd.Timedelta(days=1)).date()}，初始权益 1000 USDT。',
             '固定币池：BTC、ETH、SOL、BNB、XRP；真实一分钟 K 线，包含手续费、滑点及历史资金费。', '',
             '| 实验 | 收益 | 分钟收盘最大回撤 | 保守盘中回撤上界 | 交易组数 | 日均交易 | 盈利因子 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in table.itertuples():
        pf = f'{row.profit_factor:.3f}' if pd.notna(row.profit_factor) else '无亏损样本/无交易'
        lines.append(f'| {labels[row.experiment]} | {row.return_pct:.2f}% | {row.max_drawdown_pct:.2f}% | {row.conservative_intrabar_envelope_dd_pct:.2f}% | {row.trades} | {row.trades_per_day:.2f} | {pf} |')
    lines += ['', '## 验证与限制', '',
              '- 这是设计方案的首轮研究原型，未接入模拟盘或实盘。BTC 大盘叠加过滤、实时点差/异常波动过滤尚未实现。',
              '- 固定参数，不在测试结果出来后选择最赚钱参数；逐月结果见 monthly.csv。不能把三个多月视为长期稳定证明。',
              '- 固定五币池，并非全市场历史成交额排名；缺乏下架币和更广币池的覆盖。',
              '- 资金费事件取历史数据，但以结算分钟开盘价近似标记价格，结算时间向下取整到分钟。',
              '- 止盈止损同分钟发生时先计止损；配对金额止损按两腿共同不利极值成交，偏保守，无法还原真实成交先后。',
              '- 分钟收盘回撤包含浮盈浮亏。盘中上界是各腿最不利极值相加，不表示这些极值同时发生，也不是交易所标记价格回撤。',
              '- 每币每日最多三单（北京时间，跨策略及配对腿共享）；全账户每日最多二十组；名义敞口入场时上限权益二倍。',
              '- 账户回撤 8% 降低新风险，12% 停止新仓，15% 触发下一分钟平仓。停机可能降低交易频率；不能据此声称稳定盈利。',
              '- 不模拟盘口、最小交易量取整、断网、强平和精确保证金；所有结果仍需更长样本与纸盘复核。',
              '- 所有原始信号、日配对检验、拒绝统计、成交日志、分钟净值以及数据哈希保存在本目录。', '',
              '## 复现', '',
              '`C:/ProgramData/anaconda3/python.exe run_adaptive_combo_research.py`', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'DONE: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
