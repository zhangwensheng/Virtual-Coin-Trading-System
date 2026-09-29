from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests
import yaml

from download_binance_archive import download_month, month_range
from futures_strategy.backtest import run_backtest
from futures_strategy.config import AppConfig, load_config
from futures_strategy.data import load_csv
from futures_strategy.factor_short_optimization import (
    binance_native_to_unified_symbol,
    month_slices,
    normalize_profit_factor,
    recommended_warmup_bars,
    to_builtin,
)
from futures_strategy.strategy import prepare_market_data


DEFAULT_UNIVERSAL_SHORT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "BNBUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "FILUSDT",
    "BCHUSDT",
    "LTCUSDT",
    "TRXUSDT",
    "UNIUSDT",
    "AAVEUSDT",
    "NEARUSDT",
    "FETUSDT",
    "APTUSDT",
    "SUIUSDT",
    "WLDUSDT",
    "1000PEPEUSDT",
]

DEFAULT_STRATEGY_CONFIGS = [
    "configs/binance_universal_short_trend_defensive_1h.yaml",
    "configs/binance_universal_short_trend_aggressive_1h.yaml",
    "configs/binance_universal_short_breakout_1h.yaml",
    "configs/binance_universal_short_bear_rally_1h.yaml",
]


@dataclass
class StrategyCandidate:
    name: str
    config_path: Path
    config: AppConfig


def archive_output_path(
    data_dir: str | Path,
    symbol: str,
    timeframe: str,
    start_month: str,
    end_month: str,
) -> Path:
    return Path(data_dir) / f"{symbol.lower()}_{timeframe}_{start_month}_{end_month}.csv"


def ensure_archive_csv(
    symbol: str,
    timeframe: str,
    start_month: str,
    end_month: str,
    data_dir: str | Path,
    *,
    market: str = "um",
    session: requests.Session | None = None,
    force: bool = False,
    logger: Callable[[str], None] | None = None,
) -> Path:
    output_path = archive_output_path(data_dir, symbol, timeframe, start_month, end_month)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path

    http = session or requests.Session()
    frames: list[pd.DataFrame] = []
    for month in month_range(start_month, end_month):
        frame = download_month(symbol, timeframe, month, market, http)
        frames.append(frame)
        if logger is not None:
            logger(f"downloaded {symbol} {month} rows={len(frame)}")

    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    merged.to_csv(output_path, index=False)
    return output_path


def load_strategy_candidates(config_paths: list[str | Path]) -> list[StrategyCandidate]:
    candidates: list[StrategyCandidate] = []
    for raw_path in config_paths:
        path = Path(raw_path)
        config = load_config(path)
        candidates.append(
            StrategyCandidate(
                name=path.stem,
                config_path=path,
                config=config,
            )
        )
    return candidates


def run_price_strategy_backtest(
    raw_frame: pd.DataFrame,
    base_config: AppConfig,
    symbol: str,
    *,
    eval_start: pd.Timestamp | None = None,
    eval_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config.exchange.symbol = binance_native_to_unified_symbol(symbol)

    model_frame = raw_frame
    if eval_start is not None and eval_end is not None:
        eval_start = pd.Timestamp(eval_start)
        eval_end = pd.Timestamp(eval_end)
        warmup_bars = recommended_warmup_bars(config)
        eval_start_loc = raw_frame.index.searchsorted(eval_start)
        warmup_loc = max(0, eval_start_loc - warmup_bars)
        warmup_start = raw_frame.index[warmup_loc]
        model_frame = raw_frame[(raw_frame.index >= warmup_start) & (raw_frame.index < eval_end)].copy()
    else:
        model_frame = raw_frame.copy()

    try:
        prepared = prepare_market_data(model_frame, config.strategy)
    except ValueError:
        prepared = pd.DataFrame()

    if prepared.empty:
        return {
            "initial_capital": round(config.risk.initial_capital, 2),
            "final_equity": round(config.risk.initial_capital, 2),
            "net_profit": 0.0,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
            "prepared_rows": 0,
        }

    if eval_start is not None:
        signal_columns = [
            column
            for column in ["long_signal", "short_signal", "exit_long_signal", "exit_short_signal"]
            if column in prepared.columns
        ]
        if signal_columns:
            prepared.loc[prepared.index < eval_start, signal_columns] = False

    result = run_backtest(prepared, config)
    summary = dict(result.summary)
    summary["prepared_rows"] = len(prepared)
    return summary


def compute_survivor_score(
    *,
    full_return_pct: float,
    profitable_month_ratio: float,
    active_month_ratio: float,
    mean_month_return_pct: float,
    avg_month_drawdown_pct: float,
    avg_month_profit_factor: float,
    worst_month_return_pct: float,
    full_profit_factor: float,
) -> float:
    return (
        (full_return_pct * 0.45)
        + (mean_month_return_pct * 0.95)
        + (profitable_month_ratio * 4.0)
        + (active_month_ratio * 1.1)
        + ((avg_month_profit_factor - 1.0) * 1.6)
        + ((full_profit_factor - 1.0) * 0.9)
        - (avg_month_drawdown_pct * 0.35)
        - (abs(min(worst_month_return_pct, 0.0)) * 0.12)
    )


def is_survivor_candidate(row: pd.Series) -> bool:
    return bool(
        (float(row["full_return_pct"]) > 0.0)
        and (float(row["profitable_month_ratio"]) >= 0.5)
        and (float(row["active_month_ratio"]) >= 0.35)
        and (float(row["mean_month_return_pct"]) >= 0.1)
        and (float(row["avg_month_profit_factor"]) >= 1.05)
        and (float(row["avg_month_drawdown_pct"]) <= 12.0)
        and (float(row["worst_month_return_pct"]) >= -12.0)
        and (float(row["full_profit_factor"]) >= 1.05)
        and (int(row["full_total_trades"]) >= 8)
        and (int(row["months_tested"]) >= 18)
    )


def dump_survivor_configs(
    summary: pd.DataFrame,
    candidates: dict[str, StrategyCandidate],
    dataset_paths: dict[str, Path],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for _, row in summary.iterrows():
        if not bool(row["recommended"]):
            continue
        strategy_name = str(row["strategy_name"])
        symbol = str(row["symbol"])
        candidate = candidates[strategy_name]
        config = deepcopy(candidate.config)
        config.exchange.symbol = binance_native_to_unified_symbol(symbol)
        config.exchange.csv_path = str(dataset_paths[symbol])
        slug = f"{symbol.lower()}_{strategy_name}"
        config.output.output_dir = f"outputs/{slug}"
        payload = to_builtin(
            {
                "exchange": asdict(config.exchange),
                "strategy": asdict(config.strategy),
                "risk": asdict(config.risk),
                "output": asdict(config.output),
            }
        )
        output_path = output_dir / f"{slug}.yaml"
        output_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        created.append(output_path)
    return created


def render_universal_short_report(results: dict[str, Any]) -> str:
    lines = [
        "# 通用空头策略两年滚动筛选报告",
        "",
        f"- 数据窗口: `{results['start_month']} -> {results['end_month']}`",
        f"- 币种数量: `{len(results['symbols'])}`",
        f"- 候选策略: `{len(results['strategy_candidates'])}`",
        "",
        "## 策略总览",
        "",
    ]
    strategy_summary = results["strategy_summary"]
    for _, row in strategy_summary.iterrows():
        lines.append(
            f"- `{row['strategy_name']}`: survivors=`{int(row['survivor_count'])}` "
            f"avg_full_return=`{row['avg_full_return_pct']}%` avg_month_return=`{row['avg_mean_month_return_pct']}%` "
            f"profitable_ratio=`{row['avg_profitable_month_ratio']}`"
        )

    lines.extend(["", "## 幸存组合", ""])
    survivor_summary = results["survivor_summary"]
    if survivor_summary.empty:
        lines.append("- 当前没有达到跨月份生存阈值的策略-币种组合。")
    else:
        for _, row in survivor_summary.iterrows():
            lines.append(
                f"- `{row['symbol']}` + `{row['strategy_name']}`: full=`{row['full_return_pct']}%` "
                f"month_mean=`{row['mean_month_return_pct']}%` month_win_ratio=`{row['profitable_month_ratio']}` "
                f"full_pf=`{row['full_profit_factor']}` score=`{row['survivor_score']}`"
            )

    lines.extend(["", "## 每币最优组合", ""])
    best_per_symbol = results["best_per_symbol"]
    for _, row in best_per_symbol.iterrows():
        lines.append(
            f"- `{row['symbol']}` best=`{row['strategy_name']}` recommended=`{bool(row['recommended'])}` "
            f"full=`{row['full_return_pct']}%` month_mean=`{row['mean_month_return_pct']}%` "
            f"month_win_ratio=`{row['profitable_month_ratio']}`"
        )

    return "\n".join(lines) + "\n"


def screen_universal_short_strategies(
    *,
    symbols: list[str],
    strategy_config_paths: list[str | Path],
    data_dir: str | Path,
    start_month: str,
    end_month: str,
    timeframe: str = "1h",
    market: str = "um",
    force_download: bool = False,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    session = requests.Session()
    dataset_paths: dict[str, Path] = {}
    datasets: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        csv_path = ensure_archive_csv(
            symbol,
            timeframe,
            start_month,
            end_month,
            data_dir,
            market=market,
            session=session,
            force=force_download,
            logger=logger,
        )
        dataset_paths[symbol] = csv_path
        datasets[symbol] = load_csv(csv_path)
        if logger is not None:
            logger(f"loaded archive {symbol} rows={len(datasets[symbol])}")

    candidates = load_strategy_candidates(strategy_config_paths)
    candidate_map = {candidate.name: candidate for candidate in candidates}

    summary_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if logger is not None:
            logger(f"screening strategy {candidate.name}")
        for symbol, raw_frame in datasets.items():
            full_summary = run_price_strategy_backtest(raw_frame, candidate.config, symbol)

            symbol_monthly_rows: list[dict[str, Any]] = []
            for month_window in month_slices(raw_frame):
                monthly_summary = run_price_strategy_backtest(
                    raw_frame,
                    candidate.config,
                    symbol,
                    eval_start=month_window["start_time"],
                    eval_end=month_window["end_time"],
                )
                payload = {
                    "strategy_name": candidate.name,
                    "strategy_config_path": str(candidate.config_path),
                    "symbol": symbol,
                    "month": month_window["month"],
                    "csv_path": str(dataset_paths[symbol]),
                    **monthly_summary,
                }
                monthly_rows.append(payload)
                symbol_monthly_rows.append(payload)

            monthly_frame = pd.DataFrame(symbol_monthly_rows)
            profitable_month_ratio = float((monthly_frame["return_pct"] > 0).mean())
            active_month_ratio = float((monthly_frame["total_trades"] > 0).mean())
            avg_month_pf = float(monthly_frame["profit_factor"].map(normalize_profit_factor).mean())
            avg_month_dd = float(monthly_frame["max_drawdown_pct"].abs().mean())
            mean_month_return = float(monthly_frame["return_pct"].mean())
            median_month_return = float(monthly_frame["return_pct"].median())
            worst_month_return = float(monthly_frame["return_pct"].min())
            best_month_return = float(monthly_frame["return_pct"].max())
            full_pf = normalize_profit_factor(full_summary["profit_factor"])
            survivor_score = compute_survivor_score(
                full_return_pct=float(full_summary["return_pct"]),
                profitable_month_ratio=profitable_month_ratio,
                active_month_ratio=active_month_ratio,
                mean_month_return_pct=mean_month_return,
                avg_month_drawdown_pct=avg_month_dd,
                avg_month_profit_factor=avg_month_pf,
                worst_month_return_pct=worst_month_return,
                full_profit_factor=full_pf,
            )
            row = {
                "strategy_name": candidate.name,
                "strategy_config_path": str(candidate.config_path),
                "symbol": symbol,
                "csv_path": str(dataset_paths[symbol]),
                "months_tested": int(len(monthly_frame)),
                "profitable_month_ratio": round(profitable_month_ratio, 4),
                "active_month_ratio": round(active_month_ratio, 4),
                "mean_month_return_pct": round(mean_month_return, 4),
                "median_month_return_pct": round(median_month_return, 4),
                "avg_month_drawdown_pct": round(avg_month_dd, 4),
                "avg_month_profit_factor": round(avg_month_pf, 4),
                "worst_month_return_pct": round(worst_month_return, 4),
                "best_month_return_pct": round(best_month_return, 4),
                "full_return_pct": round(float(full_summary["return_pct"]), 4),
                "full_max_drawdown_pct": round(float(full_summary["max_drawdown_pct"]), 4),
                "full_profit_factor": round(float(full_pf), 4),
                "full_total_trades": int(full_summary["total_trades"]),
                "full_win_rate_pct": round(float(full_summary["win_rate_pct"]), 4),
                "survivor_score": round(survivor_score, 4),
            }
            row["recommended"] = is_survivor_candidate(pd.Series(row))
            summary_rows.append(row)

    monthly_results = pd.DataFrame(monthly_rows).sort_values(
        ["strategy_name", "symbol", "month"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["recommended", "survivor_score", "full_return_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    survivor_summary = summary[summary["recommended"]].reset_index(drop=True)
    best_per_symbol = (
        summary.sort_values(["symbol", "recommended", "survivor_score"], ascending=[True, False, False])
        .groupby("symbol", sort=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    strategy_summary_rows: list[dict[str, Any]] = []
    for strategy_name, group in summary.groupby("strategy_name", sort=False):
        strategy_summary_rows.append(
            {
                "strategy_name": strategy_name,
                "survivor_count": int(group["recommended"].sum()),
                "avg_full_return_pct": round(float(group["full_return_pct"].mean()), 4),
                "avg_mean_month_return_pct": round(float(group["mean_month_return_pct"].mean()), 4),
                "avg_profitable_month_ratio": round(float(group["profitable_month_ratio"].mean()), 4),
                "avg_survivor_score": round(float(group["survivor_score"].mean()), 4),
            }
        )
    strategy_summary = pd.DataFrame(strategy_summary_rows).sort_values(
        ["survivor_count", "avg_survivor_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    best_survivor_per_symbol = (
        survivor_summary.sort_values(["symbol", "survivor_score"], ascending=[True, False])
        .groupby("symbol", sort=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
        if not survivor_summary.empty
        else survivor_summary.copy()
    )

    return {
        "start_month": start_month,
        "end_month": end_month,
        "symbols": symbols,
        "strategy_candidates": [candidate.name for candidate in candidates],
        "dataset_paths": dataset_paths,
        "datasets": datasets,
        "summary": summary,
        "monthly_results": monthly_results,
        "survivor_summary": survivor_summary,
        "best_per_symbol": best_per_symbol,
        "best_survivor_per_symbol": best_survivor_per_symbol,
        "strategy_summary": strategy_summary,
        "candidate_map": candidate_map,
    }
