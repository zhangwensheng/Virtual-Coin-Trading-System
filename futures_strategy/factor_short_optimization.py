from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests
import yaml

from futures_strategy.backtest import run_backtest
from futures_strategy.binance_factors import build_factor_dataset, factor_window, save_factor_dataset
from futures_strategy.config import AppConfig
from futures_strategy.data import load_csv
from futures_strategy.strategy import prepare_market_data

DEFAULT_FACTOR_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

DEFAULT_FACTOR_PARAM_GRID = {
    "enhanced_min_funding_rate": [-0.00005, -0.00001],
    "enhanced_min_funding_change": [0.0, 0.00001],
    "enhanced_min_oi_change_pct": [0.005, 0.01],
    "enhanced_min_price_bounce_pct": [0.003, 0.006],
}

DEFAULT_AGGRESSIVE_PARAM_GRID = {
    "enhanced_entry_min_score": [1, 2, 3],
    "enhanced_risk_step": [0.3, 0.4, 0.5],
    "enhanced_max_risk_multiplier": [2.0, 2.5],
    "risk_per_trade": [0.02, 0.025, 0.03],
}


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


def parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if raw is None:
        return list(default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [int(value) for value in values] if values else list(default)


def build_param_grid(grid_values: dict[str, list[Any]]) -> list[dict[str, Any]]:
    ordered_keys = list(grid_values.keys())
    combos: list[dict[str, Any]] = []
    for values in product(*(grid_values[key] for key in ordered_keys)):
        combos.append({key: value for key, value in zip(ordered_keys, values)})
    return combos


def binance_native_to_unified_symbol(native_symbol: str) -> str:
    for quote in ("USDT", "USDC", "BUSD"):
        if native_symbol.endswith(quote):
            base = native_symbol[: -len(quote)]
            return f"{base}/{quote}:{quote}"
    return native_symbol


def factor_cache_path(cache_dir: Path, symbol: str, interval: str, end_label: str) -> Path:
    return cache_dir / f"{symbol.lower()}_{interval}_{end_label}_factors.csv"


def ensure_factor_dataset(
    symbol: str,
    interval: str,
    days_back: int,
    end_time: str | None,
    cache_dir: Path,
    session: requests.Session,
) -> Path:
    window = factor_window(days_back=days_back, end_time=end_time)
    end_label = window.end_time.strftime("%Y-%m-%d")
    cache_path = factor_cache_path(cache_dir, symbol, interval, end_label)
    if cache_path.exists():
        return cache_path

    dataset = build_factor_dataset(
        symbol=symbol,
        interval=interval,
        days_back=days_back,
        end_time=end_time,
        session=session,
    )
    save_factor_dataset(dataset, cache_path)
    return cache_path


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


def build_rolling_windows(
    datasets: dict[str, pd.DataFrame],
    window_days: int,
    step_days: int,
) -> list[dict[str, Any]]:
    if not datasets:
        return []
    common_start = max(frame.index.min() for frame in datasets.values())
    common_end = min(frame.index.max() for frame in datasets.values())
    window_delta = pd.Timedelta(days=max(1, window_days))
    step_delta = pd.Timedelta(days=max(1, step_days))

    windows: list[dict[str, Any]] = []
    cursor = common_start.floor("h")
    window_id = 1
    while cursor + window_delta <= common_end:
        end_cursor = cursor + window_delta
        windows.append(
            {
                "window_id": window_id,
                "start_time": cursor,
                "end_time": end_cursor,
                "label": f"{cursor.strftime('%Y-%m-%d')}__{(end_cursor - pd.Timedelta(hours=1)).strftime('%Y-%m-%d')}",
                "month": cursor.strftime("%Y-%m"),
            }
        )
        cursor += step_delta
        window_id += 1
    return windows


def timeframe_to_minutes(timeframe: str) -> int:
    suffix = timeframe[-1].lower()
    value = int(timeframe[:-1])
    multiplier = {
        "m": 1,
        "h": 60,
        "d": 60 * 24,
        "w": 60 * 24 * 7,
    }.get(suffix)
    if multiplier is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return value * multiplier


def recommended_warmup_bars(config: AppConfig) -> int:
    base_minutes = timeframe_to_minutes(config.exchange.timeframe)
    htf_minutes = timeframe_to_minutes(config.strategy.higher_timeframe)
    htf_ratio = max(1, htf_minutes // base_minutes)
    return max(
        config.strategy.volume_sma,
        config.strategy.swing_lookback,
        config.strategy.rsi_period,
        config.strategy.atr_period,
        config.strategy.ltf_ema,
        config.strategy.enhanced_funding_lookback,
        config.strategy.enhanced_oi_lookback,
        config.strategy.enhanced_price_lookback,
        config.strategy.htf_slow_ema * htf_ratio,
    )


def month_slices(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    month_index = frame.index.tz_localize(None).to_period("M")
    for month, group in frame.groupby(month_index):
        month_frame = group.copy()
        rows.append(
            {
                "month": str(month),
                "start_time": month_frame.index.min(),
                "end_time": month_frame.index.max() + pd.Timedelta(hours=1),
            }
        )
    return rows


def slice_frame(frame: pd.DataFrame, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
    return frame[(frame.index >= start_time) & (frame.index < end_time)].copy()


def normalize_profit_factor(value: Any) -> float:
    if value in (None, "", "inf"):
        return 5.0 if value == "inf" else 0.0
    try:
        return max(0.0, min(float(value), 5.0))
    except (TypeError, ValueError):
        return 0.0


def compute_selection_score(mean_return_pct: float, profitable_ratio: float, avg_drawdown_pct: float, avg_pf: float) -> float:
    return (
        (mean_return_pct * 0.8)
        + (profitable_ratio * 2.5)
        + ((avg_pf - 1.0) * 1.2)
        - (avg_drawdown_pct * 0.45)
    )


def compute_profit_focus_score(
    mean_return_pct: float,
    profitable_ratio: float,
    avg_drawdown_pct: float,
    avg_pf: float,
    avg_trades: float,
) -> float:
    return (
        (mean_return_pct * 1.25)
        + (profitable_ratio * 1.2)
        + ((avg_pf - 1.0) * 0.85)
        + (min(avg_trades, 8.0) * 0.08)
        - (avg_drawdown_pct * 0.35)
    )


def apply_tunable_params(config: AppConfig, params: dict[str, Any]) -> None:
    for key, value in params.items():
        if hasattr(config.strategy, key):
            setattr(config.strategy, key, value)
            continue
        if hasattr(config.risk, key):
            setattr(config.risk, key, value)
            continue
        raise AttributeError(f"Unknown tunable parameter: {key}")


def run_factor_backtest(
    raw_frame: pd.DataFrame,
    base_config: AppConfig,
    symbol: str,
    params: dict[str, Any],
    eval_start: pd.Timestamp | None = None,
    eval_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config.exchange.symbol = binance_native_to_unified_symbol(symbol)
    apply_tunable_params(config, params)

    model_frame = raw_frame
    if eval_start is not None and eval_end is not None:
        eval_end = pd.Timestamp(eval_end)
        eval_start = pd.Timestamp(eval_start)
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
        summary = empty_summary(config.risk.initial_capital)
        summary["prepared_rows"] = 0
        return summary

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


def optimize_factor_strategy(
    base_config: AppConfig,
    symbols: list[str],
    interval: str,
    days_back: int,
    end_time: str | None,
    cache_dir: Path,
    param_grid: list[dict[str, Any]],
    window_days: int,
    step_days: int,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not param_grid:
        raise ValueError("Parameter grid is empty.")

    session = requests.Session()
    datasets: dict[str, pd.DataFrame] = {}
    dataset_paths: dict[str, Path] = {}
    for symbol in symbols:
        csv_path = ensure_factor_dataset(symbol, interval, days_back, end_time, cache_dir, session)
        dataset_paths[symbol] = csv_path
        datasets[symbol] = load_csv(csv_path)
        if logger is not None:
            logger(f"已准备增强数据集 {symbol}: rows={len(datasets[symbol])}")

    windows = build_rolling_windows(datasets, window_days=window_days, step_days=step_days)
    if not windows:
        raise RuntimeError("No rolling windows available for factor optimization.")

    result_rows: list[dict[str, Any]] = []
    for combo_id, params in enumerate(param_grid, start=1):
        if logger is not None:
            logger(f"开始增强策略参数组 {combo_id}/{len(param_grid)}: {params}")
        for symbol, frame in datasets.items():
            for window in windows:
                summary = run_factor_backtest(
                    raw_frame=frame,
                    base_config=base_config,
                    symbol=symbol,
                    params=params,
                    eval_start=window["start_time"],
                    eval_end=window["end_time"],
                )
                row = {
                    "combo_id": combo_id,
                    "symbol": symbol,
                    "window_id": window["window_id"],
                    "window_label": window["label"],
                    "window_month": window["month"],
                    "start_time": window["start_time"].isoformat(),
                    "end_time": window["end_time"].isoformat(),
                    **summary,
                }
                for key, value in params.items():
                    row[f"param_{key}"] = value
                result_rows.append(row)

    window_results = pd.DataFrame(result_rows)
    param_columns = [column for column in window_results.columns if column.startswith("param_")]

    aggregate_rows: list[dict[str, Any]] = []
    for combo_id, group in window_results.groupby("combo_id", sort=False):
        avg_pf = group["profit_factor"].map(normalize_profit_factor).mean()
        avg_dd = group["max_drawdown_pct"].abs().mean()
        profitable_ratio = float((group["return_pct"] > 0).mean())
        mean_return = float(group["return_pct"].mean())
        selection_score = compute_selection_score(
            mean_return_pct=mean_return,
            profitable_ratio=profitable_ratio,
            avg_drawdown_pct=float(avg_dd),
            avg_pf=float(avg_pf),
        )
        row = {
            "combo_id": int(combo_id),
            "mean_return_pct": round(mean_return, 4),
            "profitable_ratio": round(profitable_ratio, 4),
            "avg_max_drawdown_pct": round(float(avg_dd), 4),
            "avg_profit_factor": round(float(avg_pf), 4),
            "avg_total_trades": round(float(group["total_trades"].mean()), 4),
            "selection_score": round(selection_score, 4),
        }
        for column in param_columns:
            row[column] = group.iloc[0][column]
        aggregate_rows.append(row)

    rolling_summary = pd.DataFrame(aggregate_rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    if rolling_summary.empty:
        raise RuntimeError("No rolling summary generated for factor optimization.")

    best_combo_id = int(rolling_summary.iloc[0]["combo_id"])
    best_params = {
        column.removeprefix("param_"): rolling_summary.iloc[0][column]
        for column in param_columns
    }

    symbol_rows: list[dict[str, Any]] = []
    for (symbol, combo_id), group in window_results.groupby(["symbol", "combo_id"], sort=False):
        avg_pf = group["profit_factor"].map(normalize_profit_factor).mean()
        avg_dd = group["max_drawdown_pct"].abs().mean()
        profitable_ratio = float((group["return_pct"] > 0).mean())
        mean_return = float(group["return_pct"].mean())
        avg_trades = float(group["total_trades"].mean())
        selection_score = compute_selection_score(
            mean_return_pct=mean_return,
            profitable_ratio=profitable_ratio,
            avg_drawdown_pct=float(avg_dd),
            avg_pf=float(avg_pf),
        )
        profit_focus_score = compute_profit_focus_score(
            mean_return_pct=mean_return,
            profitable_ratio=profitable_ratio,
            avg_drawdown_pct=float(avg_dd),
            avg_pf=float(avg_pf),
            avg_trades=avg_trades,
        )
        row = {
            "symbol": symbol,
            "combo_id": int(combo_id),
            "mean_return_pct": round(mean_return, 4),
            "profitable_ratio": round(profitable_ratio, 4),
            "avg_max_drawdown_pct": round(float(avg_dd), 4),
            "avg_profit_factor": round(float(avg_pf), 4),
            "avg_total_trades": round(avg_trades, 4),
            "selection_score": round(selection_score, 4),
            "profit_focus_score": round(profit_focus_score, 4),
            "csv_path": str(dataset_paths[symbol]),
        }
        for column in param_columns:
            row[column] = group.iloc[0][column]
        symbol_rows.append(row)

    symbol_rolling_summary = pd.DataFrame(symbol_rows).sort_values(
        ["symbol", "profit_focus_score"],
        ascending=[True, False],
    ).reset_index(drop=True)
    symbol_best_summary = (
        symbol_rolling_summary.groupby("symbol", sort=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    symbol_backtest_rows: list[dict[str, Any]] = []
    for _, row in symbol_best_summary.iterrows():
        symbol = str(row["symbol"])
        symbol_params = {
            column.removeprefix("param_"): row[column]
            for column in param_columns
        }
        summary = run_factor_backtest(
            raw_frame=datasets[symbol],
            base_config=base_config,
            symbol=symbol,
            params=symbol_params,
        )
        normalized_pf = normalize_profit_factor(summary["profit_factor"])
        full_profit_focus_score = compute_profit_focus_score(
            mean_return_pct=float(summary["return_pct"]),
            profitable_ratio=1.0 if float(summary["return_pct"]) > 0 else 0.0,
            avg_drawdown_pct=abs(float(summary["max_drawdown_pct"])),
            avg_pf=normalized_pf,
            avg_trades=float(summary["total_trades"]),
        )
        recommended = (
            float(summary["return_pct"]) > 0
            and normalized_pf >= 1.15
            and abs(float(summary["max_drawdown_pct"])) <= 6.0
            and int(summary["total_trades"]) >= 2
        )
        payload = {
            "symbol": symbol,
            "combo_id": int(row["combo_id"]),
            "csv_path": str(dataset_paths[symbol]),
            "full_profit_focus_score": round(full_profit_focus_score, 4),
            "recommended": recommended,
            **summary,
        }
        for column in param_columns:
            payload[column] = row[column]
        symbol_backtest_rows.append(payload)

    symbol_best_backtests = pd.DataFrame(symbol_backtest_rows).sort_values(
        ["recommended", "full_profit_focus_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    shortlist = symbol_best_backtests[symbol_best_backtests["recommended"]].reset_index(drop=True)

    month_rows: list[dict[str, Any]] = []
    for symbol, frame in datasets.items():
        for month_window in month_slices(frame):
            summary = run_factor_backtest(
                raw_frame=frame,
                base_config=base_config,
                symbol=symbol,
                params=best_params,
                eval_start=month_window["start_time"],
                eval_end=month_window["end_time"],
            )
            suitability_score = compute_selection_score(
                mean_return_pct=float(summary["return_pct"]),
                profitable_ratio=1.0 if summary["return_pct"] > 0 else 0.0,
                avg_drawdown_pct=abs(float(summary["max_drawdown_pct"])),
                avg_pf=normalize_profit_factor(summary["profit_factor"]),
            )
            month_rows.append(
                {
                    "month": month_window["month"],
                    "symbol": symbol,
                    "suitability_score": round(suitability_score, 4),
                    **summary,
                }
            )

    month_symbol_scores = pd.DataFrame(month_rows).sort_values(
        ["month", "suitability_score"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return {
        "datasets": datasets,
        "windows": windows,
        "window_results": window_results,
        "rolling_summary": rolling_summary,
        "symbol_rolling_summary": symbol_rolling_summary,
        "symbol_best_summary": symbol_best_summary,
        "symbol_best_backtests": symbol_best_backtests,
        "shortlist": shortlist,
        "month_symbol_scores": month_symbol_scores,
        "best_combo_id": best_combo_id,
        "best_params": best_params,
    }


def render_factor_report(results: dict[str, Any]) -> str:
    best_row = results["rolling_summary"].iloc[0]
    lines = [
        "# 增强版空头策略滚动回测报告",
        "",
        "## 最优参数",
        "",
    ]
    for key, value in results["best_params"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## 滚动窗口结果",
            "",
            f"- `selection_score`: `{best_row['selection_score']}`",
            f"- `mean_return_pct`: `{best_row['mean_return_pct']}%`",
            f"- `profitable_ratio`: `{best_row['profitable_ratio']}`",
            f"- `avg_max_drawdown_pct`: `{best_row['avg_max_drawdown_pct']}%`",
            f"- `avg_profit_factor`: `{best_row['avg_profit_factor']}`",
            "",
            "## 月份/币种筛选",
            "",
        ]
    )
    for _, row in results["month_symbol_scores"].iterrows():
        lines.append(
            f"- `{row['month']}` `{row['symbol']}`: score=`{row['suitability_score']}` "
            f"return=`{row['return_pct']}%` dd=`{row['max_drawdown_pct']}%` trades=`{row['total_trades']}`"
        )
    if not results["symbol_best_backtests"].empty:
        lines.extend(
            [
                "",
                "## 激进利润候选",
                "",
            ]
        )
        for _, row in results["symbol_best_backtests"].iterrows():
            lines.append(
                f"- `{row['symbol']}`: recommended=`{bool(row['recommended'])}` "
                f"return=`{row['return_pct']}%` dd=`{row['max_drawdown_pct']}%` "
                f"trades=`{row['total_trades']}` score=`{row['full_profit_focus_score']}`"
            )
    return "\n".join(lines) + "\n"


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def dump_best_config(base_config: AppConfig, best_params: dict[str, float], output_path: Path) -> Path:
    config = deepcopy(base_config)
    apply_tunable_params(config, best_params)
    payload = to_builtin(
        {
        "exchange": asdict(config.exchange),
        "strategy": asdict(config.strategy),
        "risk": asdict(config.risk),
        "output": asdict(config.output),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return output_path


def dump_symbol_configs(base_config: AppConfig, symbol_best_backtests: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    param_columns = [column for column in symbol_best_backtests.columns if column.startswith("param_")]
    for _, row in symbol_best_backtests.iterrows():
        config = deepcopy(base_config)
        config.exchange.symbol = binance_native_to_unified_symbol(str(row["symbol"]))
        csv_path = row.get("csv_path")
        if csv_path:
            config.exchange.csv_path = str(csv_path)
        params = {
            column.removeprefix("param_"): row[column]
            for column in param_columns
        }
        apply_tunable_params(config, params)
        symbol_slug = str(row["symbol"]).lower()
        config.output.output_dir = f"outputs/{symbol_slug}_funding_oi_short_profit"
        payload = to_builtin(
            {
                "exchange": asdict(config.exchange),
                "strategy": asdict(config.strategy),
                "risk": asdict(config.risk),
                "output": asdict(config.output),
            }
        )
        output_path = output_dir / f"{symbol_slug}_profit.yaml"
        output_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        created.append(output_path)
    return created
