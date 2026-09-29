from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

from download_binance_archive import download_month, month_range
from futures_strategy.backtest import run_backtest
from futures_strategy.config import AppConfig
from futures_strategy.data import load_csv
from futures_strategy.strategy import prepare_market_data

DEFAULT_SMALLCAP_SYMBOLS = [
    "1000PEPEUSDT",
    "1000BONKUSDT",
    "1000FLOKIUSDT",
    "1000SHIBUSDT",
    "WIFUSDT",
    "ENAUSDT",
]

DEFAULT_PARAM_GRID = {
    "intraday_day_return_threshold": [0.03, 0.05],
    "intraday_volume_spike_threshold": [1.6, 2.0],
    "intraday_sell_volume_ratio": [1.5],
    "intraday_upper_wick_ratio": [0.2, 0.3],
    "intraday_drop_from_peak_threshold": [0.01],
}


def expand_months(spec: str | None) -> list[str]:
    if spec is None:
        return []

    months: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            start_month, end_month = [part.strip() for part in token.split(":", 1)]
            months.extend(month_range(start_month, end_month))
        else:
            months.append(token)

    unique_months: list[str] = []
    seen: set[str] = set()
    for month in months:
        if month not in seen:
            unique_months.append(month)
            seen.add(month)
    return unique_months


def parse_csv_list(raw: str | None, default: list[str] | None = None) -> list[str]:
    if raw is None:
        return list(default or [])
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or list(default or [])


def parse_float_list(raw: str | None, default: list[float]) -> list[float]:
    if raw is None:
        return list(default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [float(value) for value in values] if values else list(default)


def build_param_grid(grid_values: dict[str, list[float]]) -> list[dict[str, float]]:
    ordered_keys = list(grid_values.keys())
    combos: list[dict[str, float]] = []
    for values in product(*(grid_values[key] for key in ordered_keys)):
        combos.append({key: value for key, value in zip(ordered_keys, values)})
    return combos


def binance_native_to_unified_symbol(native_symbol: str) -> str:
    for quote in ("USDT", "USDC", "BUSD"):
        if native_symbol.endswith(quote):
            base = native_symbol[: -len(quote)]
            return f"{base}/{quote}:{quote}"
    return native_symbol


def monthly_cache_path(cache_dir: Path, symbol: str, timeframe: str, month: str) -> Path:
    return cache_dir / f"{symbol.lower()}_{timeframe}_{month}.csv"


def ensure_binance_month_csv(
    symbol: str,
    timeframe: str,
    month: str,
    cache_dir: Path,
    market: str,
    session: requests.Session,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = monthly_cache_path(cache_dir, symbol, timeframe, month)
    if output_path.exists():
        return output_path

    try:
        frame = download_month(symbol, timeframe, month, market, session)
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code == 404:
            raise FileNotFoundError(f"Archive not found for {symbol} {timeframe} {month}") from exc
        raise

    frame.to_csv(output_path, index=False)
    return output_path


def load_symbol_months(
    symbol: str,
    timeframe: str,
    months: list[str],
    cache_dir: Path,
    market: str,
    session: requests.Session,
) -> pd.DataFrame:
    if not months:
        raise ValueError("months cannot be empty")

    frames: list[pd.DataFrame] = []
    for month in months:
        csv_path = ensure_binance_month_csv(symbol, timeframe, month, cache_dir, market, session)
        frames.append(load_csv(csv_path))

    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="first")]
    return merged


def empty_summary(initial_capital: float) -> dict[str, Any]:
    return {
        "initial_capital": round(initial_capital, 2),
        "final_equity": round(initial_capital, 2),
        "net_profit": 0.0,
        "return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "total_trades": 0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "avg_trade_pnl": 0.0,
    }


def run_symbol_backtest(
    raw_frame: pd.DataFrame,
    base_config: AppConfig,
    symbol: str,
    params: dict[str, float],
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config.exchange.name = "binance"
    config.exchange.symbol = binance_native_to_unified_symbol(symbol)
    config.exchange.timeframe = "1m"
    config.exchange.csv_path = None

    for key, value in params.items():
        setattr(config.strategy, key, value)

    try:
        prepared = prepare_market_data(raw_frame, config.strategy)
    except ValueError:
        prepared = pd.DataFrame()

    if prepared.empty:
        summary = empty_summary(config.risk.initial_capital)
        summary["prepared_rows"] = 0
        return summary

    result = run_backtest(prepared, config)
    summary = dict(result.summary)
    summary["prepared_rows"] = len(prepared)
    return summary


def normalize_profit_factor(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    if value == "inf":
        return 5.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not pd.notna(parsed):
        return 0.0
    if parsed == float("inf"):
        return 5.0
    return max(0.0, min(parsed, 5.0))


def compute_robustness_score(
    mean_return_pct: float,
    median_return_pct: float,
    min_return_pct: float,
    profitable_ratio: float,
    activity_ratio: float,
    avg_profit_factor: float,
    avg_max_drawdown_pct: float,
    worst_max_drawdown_pct: float,
) -> float:
    pf_edge = max(-1.0, min(avg_profit_factor - 1.0, 2.0))
    downside_penalty = max(0.0, -min_return_pct) * 0.35
    return (
        (mean_return_pct * 0.8)
        + (median_return_pct * 0.6)
        + (profitable_ratio * 2.5)
        + (activity_ratio * 1.5)
        + (pf_edge * 1.2)
        - (avg_max_drawdown_pct * 0.7)
        - (worst_max_drawdown_pct * 0.3)
        - downside_penalty
    )


def aggregate_combo_metrics(symbol_results: pd.DataFrame) -> pd.DataFrame:
    if symbol_results.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    param_columns = [column for column in symbol_results.columns if column.startswith("param_")]

    for (combo_id, split_name), group in symbol_results.groupby(["combo_id", "split"], sort=False):
        profit_factors = group["profit_factor"].map(normalize_profit_factor)
        max_drawdown_abs = group["max_drawdown_pct"].abs()
        profitable_symbols = int((group["net_profit"] > 0).sum())
        zero_trade_symbols = int((group["total_trades"] == 0).sum())
        symbols_tested = int(len(group))
        profitable_ratio = profitable_symbols / symbols_tested if symbols_tested else 0.0
        activity_ratio = 1 - (zero_trade_symbols / symbols_tested) if symbols_tested else 0.0

        avg_max_drawdown_pct = float(max_drawdown_abs.mean()) if symbols_tested else 0.0
        worst_max_drawdown_pct = float(max_drawdown_abs.max()) if symbols_tested else 0.0
        mean_return_pct = float(group["return_pct"].mean()) if symbols_tested else 0.0
        median_return_pct = float(group["return_pct"].median()) if symbols_tested else 0.0
        min_return_pct = float(group["return_pct"].min()) if symbols_tested else 0.0
        score = compute_robustness_score(
            mean_return_pct=mean_return_pct,
            median_return_pct=median_return_pct,
            min_return_pct=min_return_pct,
            profitable_ratio=profitable_ratio,
            activity_ratio=activity_ratio,
            avg_profit_factor=float(profit_factors.mean()) if symbols_tested else 0.0,
            avg_max_drawdown_pct=avg_max_drawdown_pct,
            worst_max_drawdown_pct=worst_max_drawdown_pct,
        )

        row = {
            "combo_id": int(combo_id),
            "split": split_name,
            "symbols_tested": symbols_tested,
            "profitable_symbols": profitable_symbols,
            "zero_trade_symbols": zero_trade_symbols,
            "profitable_ratio": round(profitable_ratio, 4),
            "activity_ratio": round(activity_ratio, 4),
            "mean_return_pct": round(mean_return_pct, 4),
            "median_return_pct": round(median_return_pct, 4),
            "min_return_pct": round(min_return_pct, 4),
            "max_return_pct": round(float(group["return_pct"].max()) if symbols_tested else 0.0, 4),
            "avg_max_drawdown_pct": round(avg_max_drawdown_pct, 4),
            "worst_max_drawdown_pct": round(worst_max_drawdown_pct, 4),
            "avg_profit_factor": round(float(profit_factors.mean()) if symbols_tested else 0.0, 4),
            "median_profit_factor": round(float(profit_factors.median()) if symbols_tested else 0.0, 4),
            "mean_win_rate_pct": round(float(group["win_rate_pct"].mean()) if symbols_tested else 0.0, 4),
            "mean_total_trades": round(float(group["total_trades"].mean()) if symbols_tested else 0.0, 4),
            "robustness_score": round(score, 4),
        }
        for column in param_columns:
            row[column] = group.iloc[0][column]
        rows.append(row)

    aggregated = pd.DataFrame(rows)
    if aggregated.empty:
        return aggregated
    return aggregated.sort_values(["split", "robustness_score"], ascending=[True, False]).reset_index(drop=True)


def build_selection_table(aggregate_results: pd.DataFrame) -> pd.DataFrame:
    if aggregate_results.empty:
        return pd.DataFrame()

    param_columns = [column for column in aggregate_results.columns if column.startswith("param_")]
    selection = aggregate_results[["combo_id", *param_columns]].drop_duplicates("combo_id").set_index("combo_id")
    metrics = [
        "symbols_tested",
        "profitable_ratio",
        "mean_return_pct",
        "median_return_pct",
        "min_return_pct",
        "avg_max_drawdown_pct",
        "avg_profit_factor",
        "mean_total_trades",
        "robustness_score",
    ]
    for split_name in aggregate_results["split"].drop_duplicates():
        split_rows = aggregate_results[aggregate_results["split"] == split_name].set_index("combo_id")
        for metric in metrics:
            selection[f"{split_name}_{metric}"] = split_rows[metric]

    if "valid_robustness_score" in selection.columns:
        selection["selection_score"] = (
            selection["valid_robustness_score"].fillna(-999.0) * 0.65
            + selection.get("train_robustness_score", 0.0).fillna(0.0) * 0.35
        )
    else:
        selection["selection_score"] = selection.get("train_robustness_score", 0.0).fillna(0.0)

    return selection.reset_index().sort_values("selection_score", ascending=False).reset_index(drop=True)


def optimize_across_symbols(
    base_config: AppConfig,
    symbols: list[str],
    split_months: dict[str, list[str]],
    param_grid: list[dict[str, float]],
    cache_dir: Path,
    market: str = "um",
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    required_splits = [split_name for split_name, months in split_months.items() if months]
    if not required_splits:
        raise ValueError("At least one split with months is required.")
    if not param_grid:
        raise ValueError("Parameter grid is empty.")

    session = requests.Session()
    symbol_frames: dict[str, dict[str, pd.DataFrame]] = {}
    skipped_symbols: list[dict[str, str]] = []

    for symbol in symbols:
        split_frames: dict[str, pd.DataFrame] = {}
        failed = False
        for split_name in required_splits:
            months = split_months[split_name]
            try:
                frame = load_symbol_months(
                    symbol=symbol,
                    timeframe=base_config.exchange.timeframe,
                    months=months,
                    cache_dir=cache_dir,
                    market=market,
                    session=session,
                )
            except Exception as exc:
                skipped_symbols.append({"symbol": symbol, "split": split_name, "reason": str(exc)})
                failed = True
                break
            if frame.empty:
                skipped_symbols.append({"symbol": symbol, "split": split_name, "reason": "empty frame"})
                failed = True
                break
            split_frames[split_name] = frame

        if not failed:
            symbol_frames[symbol] = split_frames
            if logger is not None:
                logger(f"已加载 {symbol}，覆盖分组: {', '.join(split_frames.keys())}")

    if not symbol_frames:
        raise RuntimeError("No symbols available for optimization after loading data.")

    symbol_rows: list[dict[str, Any]] = []
    total_steps = len(param_grid) * len(symbol_frames) * len(required_splits)
    current_step = 0

    for combo_id, params in enumerate(param_grid, start=1):
        if logger is not None:
            logger(f"开始参数组 {combo_id}/{len(param_grid)}: {params}")
        for symbol, frames_by_split in symbol_frames.items():
            for split_name, raw_frame in frames_by_split.items():
                summary = run_symbol_backtest(raw_frame, base_config, symbol, params)
                row = {
                    "combo_id": combo_id,
                    "split": split_name,
                    "symbol": symbol,
                    **summary,
                }
                for key, value in params.items():
                    row[f"param_{key}"] = value
                symbol_rows.append(row)
                current_step += 1
        if logger is not None:
            logger(f"已完成参数组 {combo_id}/{len(param_grid)}，累计 {current_step}/{total_steps} 次回测")

    symbol_results = pd.DataFrame(symbol_rows)
    aggregate_results = aggregate_combo_metrics(symbol_results)
    selection_table = build_selection_table(aggregate_results)

    return {
        "symbol_results": symbol_results,
        "aggregate_results": aggregate_results,
        "selection_table": selection_table,
        "usable_symbols": list(symbol_frames.keys()),
        "skipped_symbols": skipped_symbols,
        "required_splits": required_splits,
    }


def best_params_payload(
    selection_table: pd.DataFrame,
    aggregate_results: pd.DataFrame,
    symbol_results: pd.DataFrame,
    split_months: dict[str, list[str]],
    symbols: list[str],
) -> dict[str, Any]:
    if selection_table.empty:
        return {
            "symbols": symbols,
            "split_months": split_months,
            "best_combo": None,
        }

    best_row = selection_table.iloc[0].to_dict()
    combo_id = int(best_row["combo_id"])
    aggregate_subset = aggregate_results[aggregate_results["combo_id"] == combo_id]
    symbol_subset = symbol_results[symbol_results["combo_id"] == combo_id]
    params = {
        column.replace("param_", ""): best_row[column]
        for column in selection_table.columns
        if column.startswith("param_")
    }

    split_metrics = {
        split_name: aggregate_subset[aggregate_subset["split"] == split_name].iloc[0].to_dict()
        for split_name in aggregate_subset["split"].drop_duplicates()
    }
    per_symbol = {
        split_name: (
            symbol_subset[symbol_subset["split"] == split_name]
            .sort_values("return_pct", ascending=False)[
                ["symbol", "return_pct", "max_drawdown_pct", "total_trades", "profit_factor", "win_rate_pct"]
            ]
            .to_dict(orient="records")
        )
        for split_name in symbol_subset["split"].drop_duplicates()
    }

    return {
        "symbols": symbols,
        "split_months": split_months,
        "best_combo": {
            "combo_id": combo_id,
            "selection_score": round(float(best_row["selection_score"]), 4),
            "params": params,
            "split_metrics": split_metrics,
            "per_symbol": per_symbol,
        },
    }


def render_markdown_report(payload: dict[str, Any], selection_table: pd.DataFrame) -> str:
    best_combo = payload.get("best_combo")
    lines = [
        "# 小币种空头跨币种优化报告",
        "",
        f"- 覆盖币种: {', '.join(payload.get('symbols', []))}",
        f"- 训练月份: {', '.join(payload.get('split_months', {}).get('train', [])) or '无'}",
        f"- 验证月份: {', '.join(payload.get('split_months', {}).get('valid', [])) or '无'}",
        "",
    ]

    if not best_combo:
        lines.append("没有可用的最优参数结果。")
        return "\n".join(lines)

    lines.extend(["## 最优参数", ""])
    for key, value in best_combo["params"].items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## 分组表现", ""])
    for split_name, metrics in best_combo["split_metrics"].items():
        lines.append(
            f"- `{split_name}`: score={metrics['robustness_score']}, "
            f"mean_return={metrics['mean_return_pct']}%, profitable_ratio={metrics['profitable_ratio']}, "
            f"avg_dd={metrics['avg_max_drawdown_pct']}%, avg_pf={metrics['avg_profit_factor']}, "
            f"mean_trades={metrics['mean_total_trades']}"
        )

    lines.extend(["", "## 验证集币种明细", ""])
    valid_rows = best_combo["per_symbol"].get("valid", best_combo["per_symbol"].get("train", []))
    for row in valid_rows:
        lines.append(
            f"- `{row['symbol']}`: return={row['return_pct']}%, max_dd={row['max_drawdown_pct']}%, "
            f"trades={row['total_trades']}, pf={row['profit_factor']}, win_rate={row['win_rate_pct']}%"
        )

    lines.extend(["", "## 前五参数组", ""])
    for _, row in selection_table.head(5).iterrows():
        params = ", ".join(
            f"{column.replace('param_', '')}={row[column]}"
            for column in selection_table.columns
            if column.startswith("param_")
        )
        train_score = row.get("train_robustness_score", "n/a")
        valid_score = row.get("valid_robustness_score", "n/a")
        lines.append(
            f"- combo {int(row['combo_id'])}: selection={round(float(row['selection_score']), 4)}, "
            f"train={train_score}, valid={valid_score}; {params}"
        )

    return "\n".join(lines)
