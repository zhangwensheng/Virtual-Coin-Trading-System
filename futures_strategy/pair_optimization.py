from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

from futures_strategy.optimization import (
    compute_robustness_score,
    load_symbol_months,
    normalize_profit_factor,
)
from futures_strategy.pairs import PairConfig, prepare_pair_data, run_pair_backtest

DEFAULT_PAIR_UNIVERSE = [
    ("BTCUSDT", "ETHUSDT"),
    ("SOLUSDT", "ETHUSDT"),
    ("BNBUSDT", "ETHUSDT"),
]


def parse_pair_list(raw: str | None, default: list[tuple[str, str]] | None = None) -> list[tuple[str, str]]:
    if raw is None:
        return list(default or [])

    pairs: list[tuple[str, str]] = []
    for token in raw.split(","):
        token = token.strip().upper()
        if not token:
            continue
        if ":" in token:
            leg_a, leg_b = token.split(":", 1)
        elif "/" in token:
            leg_a, leg_b = token.split("/", 1)
        else:
            raise ValueError(f"Invalid pair token: {token}. Use BTCUSDT:ETHUSDT style.")
        pairs.append((leg_a.strip(), leg_b.strip()))
    return pairs or list(default or [])


def pair_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]}__{pair[1]}"


def month_slice_bounds(months: list[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Period(months[0], freq="M").to_timestamp(how="start").tz_localize("UTC")
    end = (pd.Period(months[-1], freq="M") + 1).to_timestamp(how="start").tz_localize("UTC")
    return start, end


def slice_frame_by_months(frame: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    if not months:
        return frame.iloc[0:0].copy()
    start, end = month_slice_bounds(months)
    return frame[(frame.index >= start) & (frame.index < end)].copy()


def build_rolling_windows(months: list[str], train_size: int, valid_size: int) -> list[dict[str, Any]]:
    if train_size <= 0 or valid_size <= 0:
        raise ValueError("train_size and valid_size must be positive integers.")

    unique_months: list[str] = []
    seen: set[str] = set()
    for month in months:
        if month not in seen:
            unique_months.append(month)
            seen.add(month)

    if len(unique_months) < train_size + valid_size:
        raise ValueError("Not enough months to build rolling windows.")

    windows: list[dict[str, Any]] = []
    max_start = len(unique_months) - train_size - valid_size + 1
    for index in range(max_start):
        train_months = unique_months[index : index + train_size]
        valid_months = unique_months[index + train_size : index + train_size + valid_size]
        windows.append(
            {
                "window_id": index + 1,
                "train_months": train_months,
                "valid_months": valid_months,
                "train_label": f"{train_months[0]}__{train_months[-1]}",
                "valid_label": f"{valid_months[0]}__{valid_months[-1]}",
            }
        )
    return windows


def apply_pair_params(config: PairConfig, params: dict[str, Any]) -> PairConfig:
    updated = deepcopy(config)
    for key, value in params.items():
        setattr(updated.strategy, key, value)
    return updated


def empty_pair_summary(initial_capital: float) -> dict[str, Any]:
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
        "avg_bars_held": 0.0,
        "prepared_rows": 0,
    }


def run_pair_split_backtest(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    base_config: PairConfig,
    params: dict[str, Any],
    split_months: list[str],
) -> dict[str, Any]:
    config = apply_pair_params(base_config, params)
    try:
        prepared = prepare_pair_data(frame_a, frame_b, config.strategy)
    except ValueError:
        return empty_pair_summary(config.risk.initial_capital)

    split_prepared = slice_frame_by_months(prepared, split_months)
    if split_prepared.empty:
        return empty_pair_summary(config.risk.initial_capital)

    result = run_pair_backtest(split_prepared, config)
    summary = dict(result.summary)
    summary["prepared_rows"] = len(split_prepared)
    return summary


def aggregate_pair_combo_metrics(cell_results: pd.DataFrame) -> pd.DataFrame:
    if cell_results.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    param_columns = [column for column in cell_results.columns if column.startswith("param_")]

    for (combo_id, split_name), group in cell_results.groupby(["combo_id", "split"], sort=False):
        profit_factors = group["profit_factor"].map(normalize_profit_factor)
        max_drawdown_abs = group["max_drawdown_pct"].abs()
        profitable_cells = int((group["net_profit"] > 0).sum())
        zero_trade_cells = int((group["total_trades"] == 0).sum())
        cells_tested = int(len(group))

        by_pair = (
            group.groupby("pair")
            .agg(
                mean_return_pct=("return_pct", "mean"),
                mean_total_trades=("total_trades", "mean"),
            )
            .reset_index()
        )
        profitable_pair_ratio = float((by_pair["mean_return_pct"] > 0).mean()) if not by_pair.empty else 0.0
        active_pair_ratio = float((by_pair["mean_total_trades"] > 0).mean()) if not by_pair.empty else 0.0
        profitable_cell_ratio = profitable_cells / cells_tested if cells_tested else 0.0
        activity_cell_ratio = 1 - (zero_trade_cells / cells_tested) if cells_tested else 0.0

        avg_max_drawdown_pct = float(max_drawdown_abs.mean()) if cells_tested else 0.0
        worst_max_drawdown_pct = float(max_drawdown_abs.max()) if cells_tested else 0.0
        mean_return_pct = float(group["return_pct"].mean()) if cells_tested else 0.0
        median_return_pct = float(group["return_pct"].median()) if cells_tested else 0.0
        min_return_pct = float(group["return_pct"].min()) if cells_tested else 0.0
        return_std_pct = float(group["return_pct"].std(ddof=0)) if cells_tested else 0.0
        score = compute_robustness_score(
            mean_return_pct=mean_return_pct,
            median_return_pct=median_return_pct,
            min_return_pct=min_return_pct,
            profitable_ratio=(profitable_cell_ratio + profitable_pair_ratio) / 2,
            activity_ratio=(activity_cell_ratio + active_pair_ratio) / 2,
            avg_profit_factor=float(profit_factors.mean()) if cells_tested else 0.0,
            avg_max_drawdown_pct=avg_max_drawdown_pct,
            worst_max_drawdown_pct=worst_max_drawdown_pct,
        ) - (return_std_pct * 0.2)

        row = {
            "combo_id": int(combo_id),
            "split": split_name,
            "cells_tested": cells_tested,
            "pairs_tested": int(by_pair["pair"].nunique()) if not by_pair.empty else 0,
            "profitable_cells": profitable_cells,
            "zero_trade_cells": zero_trade_cells,
            "profitable_cell_ratio": round(profitable_cell_ratio, 4),
            "activity_cell_ratio": round(activity_cell_ratio, 4),
            "profitable_pair_ratio": round(profitable_pair_ratio, 4),
            "activity_pair_ratio": round(active_pair_ratio, 4),
            "mean_return_pct": round(mean_return_pct, 4),
            "median_return_pct": round(median_return_pct, 4),
            "min_return_pct": round(min_return_pct, 4),
            "max_return_pct": round(float(group["return_pct"].max()) if cells_tested else 0.0, 4),
            "return_std_pct": round(return_std_pct, 4),
            "avg_max_drawdown_pct": round(avg_max_drawdown_pct, 4),
            "worst_max_drawdown_pct": round(worst_max_drawdown_pct, 4),
            "avg_profit_factor": round(float(profit_factors.mean()) if cells_tested else 0.0, 4),
            "median_profit_factor": round(float(profit_factors.median()) if cells_tested else 0.0, 4),
            "mean_win_rate_pct": round(float(group["win_rate_pct"].mean()) if cells_tested else 0.0, 4),
            "mean_total_trades": round(float(group["total_trades"].mean()) if cells_tested else 0.0, 4),
            "robustness_score": round(score, 4),
        }
        for column in param_columns:
            row[column] = group.iloc[0][column]
        rows.append(row)

    aggregated = pd.DataFrame(rows)
    if aggregated.empty:
        return aggregated
    return aggregated.sort_values(["split", "robustness_score"], ascending=[True, False]).reset_index(drop=True)


def build_pair_selection_table(aggregate_results: pd.DataFrame) -> pd.DataFrame:
    if aggregate_results.empty:
        return pd.DataFrame()

    param_columns = [column for column in aggregate_results.columns if column.startswith("param_")]
    selection = aggregate_results[["combo_id", *param_columns]].drop_duplicates("combo_id").set_index("combo_id")
    metrics = [
        "cells_tested",
        "pairs_tested",
        "profitable_cell_ratio",
        "profitable_pair_ratio",
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
            selection["valid_robustness_score"].fillna(-999.0) * 0.7
            + selection.get("train_robustness_score", 0.0).fillna(0.0) * 0.3
        )
    else:
        selection["selection_score"] = selection.get("train_robustness_score", 0.0).fillna(0.0)

    return selection.reset_index().sort_values("selection_score", ascending=False).reset_index(drop=True)


def optimize_pair_universe(
    base_config: PairConfig,
    pairs: list[tuple[str, str]],
    rolling_windows: list[dict[str, Any]],
    param_grid: list[dict[str, Any]],
    cache_dir: Path,
    market: str = "um",
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("pairs cannot be empty.")
    if not rolling_windows:
        raise ValueError("rolling_windows cannot be empty.")
    if not param_grid:
        raise ValueError("param_grid cannot be empty.")

    all_months: list[str] = []
    for window in rolling_windows:
        all_months.extend(window["train_months"])
        all_months.extend(window["valid_months"])
    unique_months = list(dict.fromkeys(all_months))

    session = requests.Session()
    unique_symbols = sorted({symbol for pair in pairs for symbol in pair})
    symbol_frames: dict[str, pd.DataFrame] = {}
    skipped_symbols: list[dict[str, str]] = []
    for symbol in unique_symbols:
        try:
            frame = load_symbol_months(
                symbol=symbol,
                timeframe=base_config.pair.leg_a.timeframe,
                months=unique_months,
                cache_dir=cache_dir,
                market=market,
                session=session,
            )
        except Exception as exc:
            skipped_symbols.append({"symbol": symbol, "reason": str(exc)})
            continue
        if frame.empty:
            skipped_symbols.append({"symbol": symbol, "reason": "empty frame"})
            continue
        symbol_frames[symbol] = frame
        if logger is not None:
            logger(f"已加载 {symbol}，月份覆盖: {unique_months[0]} -> {unique_months[-1]}")

    usable_pairs: list[tuple[str, str]] = []
    pair_window_frames: dict[tuple[str, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
    skipped_pairs: list[dict[str, Any]] = []
    for pair in pairs:
        if pair[0] not in symbol_frames or pair[1] not in symbol_frames:
            skipped_pairs.append({"pair": pair_label(pair), "reason": "missing symbol history"})
            continue
        pair_ok = True
        for window in rolling_windows:
            combined_months = [*window["train_months"], *window["valid_months"]]
            frame_a = slice_frame_by_months(symbol_frames[pair[0]], combined_months)
            frame_b = slice_frame_by_months(symbol_frames[pair[1]], combined_months)
            if frame_a.empty or frame_b.empty:
                skipped_pairs.append(
                    {
                        "pair": pair_label(pair),
                        "window_id": window["window_id"],
                        "reason": "empty combined frame",
                    }
                )
                pair_ok = False
                break
            pair_window_frames[(pair_label(pair), window["window_id"])] = (frame_a, frame_b)
        if pair_ok:
            usable_pairs.append(pair)
            if logger is not None:
                logger(f"已准备配对 {pair_label(pair)}，滚动窗口数: {len(rolling_windows)}")

    if not usable_pairs:
        raise RuntimeError("No usable pairs available for optimization.")

    cell_rows: list[dict[str, Any]] = []
    total_steps = len(param_grid) * len(usable_pairs) * len(rolling_windows) * 2
    current_step = 0

    for combo_id, params in enumerate(param_grid, start=1):
        if logger is not None:
            logger(f"开始参数组 {combo_id}/{len(param_grid)}: {params}")
        for pair in usable_pairs:
            label = pair_label(pair)
            for window in rolling_windows:
                frame_a, frame_b = pair_window_frames[(label, window["window_id"])]
                for split_name in ("train", "valid"):
                    months = window[f"{split_name}_months"]
                    summary = run_pair_split_backtest(
                        frame_a=frame_a,
                        frame_b=frame_b,
                        base_config=base_config,
                        params=params,
                        split_months=months,
                    )
                    row = {
                        "combo_id": combo_id,
                        "split": split_name,
                        "pair": label,
                        "window_id": window["window_id"],
                        "window_train_label": window["train_label"],
                        "window_valid_label": window["valid_label"],
                        **summary,
                    }
                    for key, value in params.items():
                        row[f"param_{key}"] = value
                    cell_rows.append(row)
                    current_step += 1
        if logger is not None:
            logger(f"已完成参数组 {combo_id}/{len(param_grid)}，累计 {current_step}/{total_steps} 次回测")

    cell_results = pd.DataFrame(cell_rows)
    aggregate_results = aggregate_pair_combo_metrics(cell_results)
    selection_table = build_pair_selection_table(aggregate_results)

    return {
        "cell_results": cell_results,
        "aggregate_results": aggregate_results,
        "selection_table": selection_table,
        "usable_pairs": [pair_label(pair) for pair in usable_pairs],
        "skipped_symbols": skipped_symbols,
        "skipped_pairs": skipped_pairs,
        "rolling_windows": rolling_windows,
    }


def best_pair_payload(
    selection_table: pd.DataFrame,
    aggregate_results: pd.DataFrame,
    cell_results: pd.DataFrame,
    rolling_windows: list[dict[str, Any]],
    pairs: list[str],
) -> dict[str, Any]:
    if selection_table.empty:
        return {
            "pairs": pairs,
            "rolling_windows": rolling_windows,
            "best_combo": None,
        }

    best_row = selection_table.iloc[0].to_dict()
    combo_id = int(best_row["combo_id"])
    aggregate_subset = aggregate_results[aggregate_results["combo_id"] == combo_id]
    cell_subset = cell_results[cell_results["combo_id"] == combo_id]
    params = {
        column.replace("param_", ""): best_row[column]
        for column in selection_table.columns
        if column.startswith("param_")
    }

    split_metrics = {
        split_name: aggregate_subset[aggregate_subset["split"] == split_name].iloc[0].to_dict()
        for split_name in aggregate_subset["split"].drop_duplicates()
    }

    pair_tables: dict[str, list[dict[str, Any]]] = {}
    for split_name in cell_subset["split"].drop_duplicates():
        split_rows = cell_subset[cell_subset["split"] == split_name]
        grouped = (
            split_rows.groupby("pair")
            .agg(
                mean_return_pct=("return_pct", "mean"),
                min_return_pct=("return_pct", "min"),
                avg_max_drawdown_pct=("max_drawdown_pct", lambda values: values.abs().mean()),
                total_trades=("total_trades", "sum"),
                profitable_windows=("net_profit", lambda values: int((values > 0).sum())),
                windows_tested=("window_id", "nunique"),
            )
            .reset_index()
            .sort_values("mean_return_pct", ascending=False)
        )
        pair_tables[split_name] = grouped.to_dict(orient="records")

    return {
        "pairs": pairs,
        "rolling_windows": rolling_windows,
        "best_combo": {
            "combo_id": combo_id,
            "selection_score": round(float(best_row["selection_score"]), 4),
            "params": params,
            "split_metrics": split_metrics,
            "per_pair": pair_tables,
        },
    }


def render_pair_markdown_report(payload: dict[str, Any], selection_table: pd.DataFrame) -> str:
    best_combo = payload.get("best_combo")
    lines = [
        "# 多配对滚动回测报告",
        "",
        f"- 覆盖配对: {', '.join(payload.get('pairs', []))}",
        f"- 滚动窗口数: {len(payload.get('rolling_windows', []))}",
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
            f"mean_return={metrics['mean_return_pct']}%, profitable_pair_ratio={metrics['profitable_pair_ratio']}, "
            f"avg_dd={metrics['avg_max_drawdown_pct']}%, avg_pf={metrics['avg_profit_factor']}, "
            f"mean_trades={metrics['mean_total_trades']}"
        )

    lines.extend(["", "## 验证集配对明细", ""])
    valid_rows = best_combo["per_pair"].get("valid", best_combo["per_pair"].get("train", []))
    for row in valid_rows:
        lines.append(
            f"- `{row['pair']}`: mean_return={round(float(row['mean_return_pct']), 4)}%, "
            f"min_return={round(float(row['min_return_pct']), 4)}%, "
            f"avg_dd={round(float(row['avg_max_drawdown_pct']), 4)}%, "
            f"trades={int(row['total_trades'])}, profitable_windows={int(row['profitable_windows'])}/{int(row['windows_tested'])}"
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
