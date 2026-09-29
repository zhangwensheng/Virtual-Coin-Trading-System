from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from futures_strategy.liquidity_retest import LiquidityRetestSettings, normalize_timestamp
from run_liquidity_retest_backtest import (
    load_archive_frames,
    load_settings_config,
    parse_symbols,
    run_backtest_from_frames,
)


SIGNAL_VARIATIONS: tuple[tuple[str, dict[str, float | int]], ...] = (
    ("breakout_loose", {"breakout_buffer_atr": 0.10, "breakout_body_atr": 0.30, "breakout_min_quote_volume_ratio": 1.0}),
    ("breakout_base", {"breakout_buffer_atr": 0.15, "breakout_body_atr": 0.40, "breakout_min_quote_volume_ratio": 1.2}),
    ("breakout_strict", {"breakout_buffer_atr": 0.20, "breakout_body_atr": 0.50, "breakout_min_quote_volume_ratio": 1.4}),
    ("retest_fast", {"retest_tolerance_atr": 0.15, "retest_max_5m_bars": 12, "confirm_min_quote_volume_ratio": 1.1}),
    ("retest_base", {"retest_tolerance_atr": 0.20, "retest_max_5m_bars": 24, "confirm_min_quote_volume_ratio": 1.0}),
    ("retest_wide", {"retest_tolerance_atr": 0.25, "retest_max_5m_bars": 36, "confirm_min_quote_volume_ratio": 1.2}),
)

EXIT_VARIATIONS: tuple[tuple[str, dict[str, float | int]], ...] = (
    ("exit_fast", {"partial_close_ratio": 0.50, "trailing_atr": 1.5, "max_hold_5m_bars": 48}),
    ("exit_base", {"partial_close_ratio": 0.50, "trailing_atr": 2.0, "max_hold_5m_bars": 72}),
    ("exit_scale70", {"partial_close_ratio": 0.70, "trailing_atr": 2.0, "max_hold_5m_bars": 72}),
    ("exit_slow", {"partial_close_ratio": 0.70, "trailing_atr": 2.5, "max_hold_5m_bars": 96}),
)


def _normalized_frames(frames_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    normalized: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames_by_symbol.items():
        if frame.empty:
            continue
        data = frame.copy().sort_index()
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC")
        else:
            data.index = data.index.tz_convert("UTC")
        normalized[symbol.upper()] = data
    if not normalized:
        raise ValueError("No usable frames for short optimization")
    return normalized


def split_train_oos(
    frames_by_symbol: dict[str, pd.DataFrame],
    *,
    train_days: int = 20,
    oos_days: int = 10,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    frames = _normalized_frames(frames_by_symbol)
    common_end = min(frame.index.max() for frame in frames.values()) + pd.Timedelta(minutes=5)
    oos_end = normalize_timestamp(common_end)
    train_start = oos_end - pd.Timedelta(days=train_days + oos_days)
    train_end = oos_end - pd.Timedelta(days=oos_days)
    train = {symbol: frame[(frame.index >= train_start) & (frame.index < train_end)].copy() for symbol, frame in frames.items()}
    oos = {symbol: frame[(frame.index >= train_end) & (frame.index < oos_end)].copy() for symbol, frame in frames.items()}
    train = {symbol: frame for symbol, frame in train.items() if not frame.empty}
    oos = {symbol: frame for symbol, frame in oos.items() if not frame.empty}
    if not train or not oos:
        raise ValueError("Frames do not cover both train and out-of-sample windows")
    meta = {
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "oos_start": train_end.isoformat(),
        "oos_end": oos_end.isoformat(),
        "train_days": int(train_days),
        "oos_days": int(oos_days),
    }
    return train, oos, meta


def compute_performance_metrics(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_equity: float,
) -> dict[str, float | int]:
    pnl = pd.to_numeric(trades.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl < 0].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    net_profit = float(pnl.sum())
    equities = pd.to_numeric(equity_curve.get("equity", pd.Series(dtype=float)), errors="coerce").dropna()
    if equities.empty:
        equities = pd.Series([float(initial_equity)])
    peaks = equities.cummax()
    drawdowns = ((peaks - equities) / peaks.replace(0.0, pd.NA)).fillna(0.0)
    weekly_profitable = 0
    if not trades.empty and "entry_time" in trades.columns:
        dated = trades.copy()
        dated["entry_time"] = pd.to_datetime(dated["entry_time"], utc=True, errors="coerce")
        dated["pnl"] = pd.to_numeric(dated.get("pnl", 0.0), errors="coerce").fillna(0.0)
        weekly = dated.dropna(subset=["entry_time"]).set_index("entry_time")["pnl"].resample("7D").sum()
        weekly_profitable = int((weekly > 0).sum())
    return {
        "net_profit": net_profit,
        "return_pct": (net_profit / float(initial_equity)) * 100.0 if initial_equity else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown_pct": float(drawdowns.max()) * 100.0,
        "trade_count": int(len(pnl)),
        "win_rate_pct": float((pnl > 0).mean() * 100.0) if len(pnl) else 0.0,
        "profitable_weeks": weekly_profitable,
    }


def candidate_parameter_sets(baseline: LiquidityRetestSettings) -> list[tuple[str, LiquidityRetestSettings]]:
    forced = replace(
        baseline,
        trade_transform_mode="mirror_original_longs_short_only",
        max_drawdown_pct=0.20,
    )
    candidates: list[tuple[str, LiquidityRetestSettings]] = [("baseline", forced)]
    seen = {json.dumps(asdict(forced), sort_keys=True, default=str)}
    for parameter_id, values in (*SIGNAL_VARIATIONS, *EXIT_VARIATIONS):
        candidate = replace(forced, **values)
        fingerprint = json.dumps(asdict(candidate), sort_keys=True, default=str)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        candidates.append((parameter_id, candidate))
    return candidates


def _frames_with_warmup(
    frames_by_symbol: dict[str, pd.DataFrame],
    trade_start: pd.Timestamp,
    trade_end: pd.Timestamp,
    *,
    warmup_days: int = 21,
) -> dict[str, pd.DataFrame]:
    frames = _normalized_frames(frames_by_symbol)
    warmup_start = trade_start - pd.Timedelta(days=warmup_days)
    return {
        symbol: frame[(frame.index >= warmup_start) & (frame.index < trade_end)].copy()
        for symbol, frame in frames.items()
        if not frame[(frame.index >= warmup_start) & (frame.index < trade_end)].empty
    }


def _settings_payload(settings: LiquidityRetestSettings) -> dict[str, Any]:
    payload = asdict(settings)
    payload["stablecoin_bases"] = list(payload["stablecoin_bases"])
    return payload


def _run_window(
    frames_by_symbol: dict[str, pd.DataFrame],
    *,
    settings: LiquidityRetestSettings,
    output_dir: Path,
    initial_equity: float,
    run_id: str,
    trade_start: pd.Timestamp,
    trade_end: pd.Timestamp,
) -> tuple[Any, dict[str, float | int]]:
    frames = _frames_with_warmup(frames_by_symbol, trade_start, trade_end)
    result = run_backtest_from_frames(
        frames,
        settings=settings,
        output_dir=output_dir,
        initial_equity=initial_equity,
        run_id=run_id,
        trade_start=trade_start,
        trade_end=trade_end,
    )
    return result, compute_performance_metrics(result.trades, result.equity_curve, initial_equity)


def run_short_optimization(
    frames_by_symbol: dict[str, pd.DataFrame],
    baseline: LiquidityRetestSettings,
    output_dir: str | Path,
    initial_equity: float = 10_000.0,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(f"Optimization output directory must be empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    frames = _normalized_frames(frames_by_symbol)
    _, _, split_meta = split_train_oos(frames, train_days=20, oos_days=10)
    train_start = normalize_timestamp(split_meta["train_start"])
    train_end = normalize_timestamp(split_meta["train_end"])
    oos_end = normalize_timestamp(split_meta["oos_end"])

    rows: list[dict[str, Any]] = []
    candidates = candidate_parameter_sets(baseline)
    for parameter_id, settings in candidates:
        _, metrics = _run_window(
            frames,
            settings=settings,
            output_dir=output_path / "train_runs" / parameter_id,
            initial_equity=initial_equity,
            run_id=f"short-opt-train-{parameter_id}",
            trade_start=train_start,
            trade_end=train_end,
        )
        rows.append({"parameter_id": parameter_id, **metrics, "settings": _settings_payload(settings)})

    results = pd.DataFrame([{key: value for key, value in row.items() if key != "settings"} for row in rows])
    results.to_csv(output_path / "optimization_results.csv", index=False)
    eligible = [row for row in rows if int(row["trade_count"]) >= 20 and float(row["max_drawdown_pct"]) < 20.0]
    selection_pool = eligible or rows
    selected = max(
        selection_pool,
        key=lambda row: (float(row["profit_factor"]), float(row["return_pct"]), int(row["profitable_weeks"])),
    )
    selected_id = str(selected["parameter_id"])
    settings_by_id = dict(candidates)
    selected_settings = settings_by_id[selected_id]

    _, oos_metrics = _run_window(
        frames,
        settings=selected_settings,
        output_dir=output_path / "out_of_sample",
        initial_equity=initial_equity,
        run_id=f"short-opt-oos-{selected_id}",
        trade_start=train_end,
        trade_end=oos_end,
    )
    oos_payload = {"parameter_id": selected_id, **oos_metrics, "settings": _settings_payload(selected_settings)}

    rolling_specs = (
        ("rolling_1", train_start + pd.Timedelta(days=14), train_start + pd.Timedelta(days=21)),
        ("rolling_2", train_start + pd.Timedelta(days=21), train_start + pd.Timedelta(days=28)),
    )
    rolling_rows: list[dict[str, Any]] = []
    for window_id, validation_start, validation_end in rolling_specs:
        _, metrics = _run_window(
            frames,
            settings=selected_settings,
            output_dir=output_path / "rolling_runs" / window_id,
            initial_equity=initial_equity,
            run_id=f"short-opt-{window_id}-{selected_id}",
            trade_start=validation_start,
            trade_end=validation_end,
        )
        rolling_rows.append({"window_id": window_id, "trade_start": validation_start.isoformat(), "trade_end": validation_end.isoformat(), **metrics})
    pd.DataFrame(rolling_rows).to_csv(output_path / "rolling_validation.csv", index=False)
    profitable_rolling = sum(float(row["net_profit"]) > 0 for row in rolling_rows)
    base_signal_qualified = bool(
        float(oos_metrics["return_pct"]) > 0
        and float(oos_metrics["profit_factor"]) >= 1.30
        and float(oos_metrics["max_drawdown_pct"]) < 20.0
        and profitable_rolling >= 2
    )

    selected_payload = {"parameter_id": selected_id, "settings": _settings_payload(selected_settings)}
    (output_path / "selected_settings.json").write_text(json.dumps(selected_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_path / "out_of_sample_summary.json").write_text(json.dumps(oos_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "selection_source": "train",
        "selected_parameter_id": selected_id,
        "selected_settings": _settings_payload(selected_settings),
        "train": {key: value for key, value in selected.items() if key != "settings"},
        "out_of_sample": oos_payload,
        "rolling_validation": rolling_rows,
        "profitable_rolling_windows": profitable_rolling,
        "total_rolling_windows": len(rolling_rows),
        "base_signal_qualified": base_signal_qualified,
        "split": split_meta,
        "candidate_pool_bias": "current_top30_proxy",
    }
    report_lines = [
        "# Short-Only Liquidity Retest Optimization",
        "",
        f"- Selected parameter: `{selected_id}`",
        f"- Train return: {float(selected['return_pct']):.4f}%",
        f"- OOS return: {float(oos_metrics['return_pct']):.4f}%",
        f"- OOS profit factor: {float(oos_metrics['profit_factor']):.4f}",
        f"- OOS max drawdown: {float(oos_metrics['max_drawdown_pct']):.4f}%",
        f"- Profitable rolling windows: {profitable_rolling}/{len(rolling_rows)}",
        f"- Base signal qualified: {base_signal_qualified}",
        "- Candidate-pool caveat: historical universe is approximated by the supplied current candidate pool.",
    ]
    (output_path / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the short-only mirrored liquidity-retest strategy.")
    parser.add_argument("--start-month", required=True)
    parser.add_argument("--end-month", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--cache-dir", default="data/liquidity_retest_5m")
    parser.add_argument("--output-dir", default="outputs/liquidity_retest_short_optimization")
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--config", default="configs/binance_liquidity_retest_short_only.yaml")
    args = parser.parse_args()

    settings = load_settings_config(args.config)
    frames = load_archive_frames(parse_symbols(args.symbols), args.start_month, args.end_month, Path(args.cache_dir))
    report = run_short_optimization(frames, settings, args.output_dir, initial_equity=args.initial_equity)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
