from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import optimize_liquidity_retest_short as optimizer_module
from futures_strategy.liquidity_retest import LiquidityRetestBacktestResult, LiquidityRetestSettings
from optimize_liquidity_retest_short import (
    compute_performance_metrics,
    run_short_optimization,
    split_train_oos,
)


def make_30_day_frame() -> pd.DataFrame:
    index = pd.date_range("2026-07-28T00:00:00Z", periods=30 * 288, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 10.0,
            "quote_volume": 1_000.0,
        },
        index=index,
    )


def test_split_train_oos_uses_strict_chronological_boundary() -> None:
    frames = {"AAAUSDT": make_30_day_frame()}

    train, oos, meta = split_train_oos(frames, train_days=20, oos_days=10)

    assert train["AAAUSDT"].index.max() < oos["AAAUSDT"].index.min()
    assert meta["train_end"] == oos["AAAUSDT"].index.min().isoformat()
    assert len(train["AAAUSDT"]) == 20 * 288
    assert len(oos["AAAUSDT"]) == 10 * 288


def test_compute_performance_metrics_reports_peak_drawdown_and_profit_factor() -> None:
    trades = pd.DataFrame({"pnl": [100.0, -50.0, 25.0]})
    equity = pd.DataFrame({"equity": [10_000.0, 10_100.0, 9_900.0, 10_075.0]})

    result = compute_performance_metrics(trades, equity, 10_000.0)

    assert result["net_profit"] == pytest.approx(75.0)
    assert result["profit_factor"] == pytest.approx(2.5)
    assert result["max_drawdown_pct"] == pytest.approx((10_100.0 - 9_900.0) / 10_100.0 * 100.0)


def fake_result_for(buffer: float, split: str) -> LiquidityRetestBacktestResult:
    if split == "train" and buffer == 0.10:
        pnl = ([10.0] * 20) + ([-5.0] * 5)
    elif split == "train":
        pnl = ([5.0] * 20) + ([-5.0] * 5)
    else:
        pnl = [4.0, -2.0, 3.0]
    trades = pd.DataFrame(
        {
            "pnl": pnl,
            "entry_time": pd.date_range("2026-07-28", periods=len(pnl), freq="12h", tz="UTC").astype(str),
        }
    )
    equity_values = [10_000.0]
    for value in pnl:
        equity_values.append(equity_values[-1] + value)
    equity = pd.DataFrame({"equity": equity_values})
    return LiquidityRetestBacktestResult({}, trades, equity, pd.DataFrame())


def test_optimizer_selects_on_train_then_evaluates_oos_once(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, float, pd.Timestamp, pd.Timestamp]] = []
    baseline = LiquidityRetestSettings(trade_transform_mode="mirror_original_longs_short_only")
    candidates = [
        ("buffer_010", replace(baseline, breakout_buffer_atr=0.10)),
        ("buffer_020", replace(baseline, breakout_buffer_atr=0.20)),
    ]

    def fake_run(
        frames,
        *,
        settings,
        output_dir,
        initial_equity,
        run_id,
        trade_start,
        trade_end,
    ):
        split = "oos" if "out_of_sample" in str(output_dir) else "train"
        calls.append((split, settings.breakout_buffer_atr, trade_start, trade_end))
        return fake_result_for(settings.breakout_buffer_atr, split)

    monkeypatch.setattr(optimizer_module, "candidate_parameter_sets", lambda _: candidates)
    monkeypatch.setattr(optimizer_module, "run_backtest_from_frames", fake_run)

    report = run_short_optimization({"AAAUSDT": make_30_day_frame()}, baseline, tmp_path)

    assert sum(split == "oos" for split, *_ in calls) == 1
    assert report["selection_source"] == "train"
    assert report["selected_parameter_id"] == "buffer_010"
    assert report["out_of_sample"]["parameter_id"] == "buffer_010"
    assert all(start < end for _, _, start, end in calls)
    assert (tmp_path / "optimization_results.csv").exists()
    assert (tmp_path / "out_of_sample_summary.json").exists()
