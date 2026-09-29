from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from futures_strategy.liquidity_retest import LiquidityRetestSettings, SignalCandidate, Zone
from run_liquidity_retest_paper_trader import PaperRuntime, PaperState, run_paper_cycle_from_frames


def fixture_module():
    spec = importlib.util.spec_from_file_location("liquidity_retest_fixtures", Path("tests/test_liquidity_retest.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_flat_frame(*, quote_volume: float):
    index = pd.date_range("2026-01-01", periods=12 * 12, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0] * len(index),
            "high": [100.2] * len(index),
            "low": [99.8] * len(index),
            "close": [100.0] * len(index),
            "volume": [quote_volume / 100.0] * len(index),
            "quote_volume": [quote_volume] * len(index),
        },
        index=index,
    )


def test_paper_cycle_is_idempotent_for_same_closed_bar(tmp_path: Path) -> None:
    fixtures = fixture_module()
    settings = LiquidityRetestSettings(
        pivot_left_bars=1,
        pivot_right_bars=1,
        zone_min_touches=2,
        breakout_volume_window=2,
        confirm_volume_window=2,
        breakout_min_quote_volume_ratio=1.05,
        confirm_min_quote_volume_ratio=1.0,
        min_margin_usdt=0.1,
    )
    runtime = PaperRuntime(output_dir=str(tmp_path), once=True, initial_equity=10_000.0)
    state = PaperState()
    frames = {"AAAUSDT": fixtures.make_long_breakout_retest_frame()}
    frames["AAAUSDT"]["quote_volume"] = 1000.0
    first = run_paper_cycle_from_frames(frames, settings=settings, runtime=runtime, state=state)
    second = run_paper_cycle_from_frames(frames, settings=settings, runtime=runtime, state=state)
    assert first["signals_processed"] >= 0
    assert second["signals_processed"] == 0
    assert (tmp_path / "latest_status.json").exists()
    assert (tmp_path / "state.json").exists()


def test_paper_cycle_applies_same_short_only_transform_before_risk_gate(tmp_path: Path, monkeypatch) -> None:
    settings = LiquidityRetestSettings(
        universe_size=1,
        trade_transform_mode="mirror_original_longs_short_only",
        pivot_left_bars=1,
        pivot_right_bars=1,
        zone_min_touches=2,
        breakout_volume_window=2,
        confirm_volume_window=2,
        breakout_min_quote_volume_ratio=1.05,
        confirm_min_quote_volume_ratio=1.0,
        min_margin_usdt=0.1,
    )
    frame = make_flat_frame(quote_volume=1_000.0)
    timestamp = frame.index[0]
    raw = SignalCandidate(
        signal_id="raw-long",
        symbol="AAAUSDT",
        side="long",
        signal_time=timestamp,
        zone=Zone("resistance", "1h", 98.0, 100.0, 2, 4.0, timestamp - pd.Timedelta(hours=2)),
        planned_entry=100.0,
        planned_stop=99.0,
        atr_5m=0.5,
        universe_rank=1,
        quote_volume_24h=1_000_000.0,
    )
    emitted = False

    def fake_on_bar(self, symbol, history_5m, top_symbols, as_of, universe_row=None):
        nonlocal emitted
        if emitted:
            return []
        emitted = True
        return [raw]

    monkeypatch.setattr("run_liquidity_retest_paper_trader.LiquidityRetestSignalEngine.on_bar", fake_on_bar)

    status = run_paper_cycle_from_frames(
        {"AAAUSDT": frame},
        settings=settings,
        runtime=PaperRuntime(output_dir=str(tmp_path), once=True),
        state=PaperState(),
    )

    signals = pd.read_csv(tmp_path / "signals.csv")
    accepted = signals.loc[signals["status"] == "accepted"]
    assert not accepted.empty
    assert set(accepted["side"]) == {"short"}
    assert set(accepted["original_side"]) == {"long"}
    assert (accepted["planned_stop"] > accepted["planned_entry"]).all()
    assert (accepted["planned_tp1"] < accepted["planned_entry"]).all()
    assert status["signals_processed"] == 1


def test_paper_state_persists_equity_and_transform_mode() -> None:
    state = PaperState(
        cash=9_500.0,
        peak_equity=12_000.0,
        last_marked_equity=9_800.0,
        transform_mode="mirror_original_longs_short_only",
    )

    restored = PaperState.from_jsonable(state.to_jsonable())

    assert restored.cash == 9_500.0
    assert restored.peak_equity == 12_000.0
    assert restored.last_marked_equity == 9_800.0
    assert restored.transform_mode == "mirror_original_longs_short_only"


def test_paper_cycle_keeps_open_position_for_next_cycle(tmp_path: Path, monkeypatch) -> None:
    settings = LiquidityRetestSettings(
        universe_size=1,
        trade_transform_mode="mirror_original_longs_short_only",
        fee_rate=0.0,
        slippage_bps=0.0,
        min_margin_usdt=0.1,
    )
    frame = make_flat_frame(quote_volume=1_000.0).iloc[:3].copy()
    timestamp = frame.index[0]
    raw = SignalCandidate(
        signal_id="persistent-long",
        symbol="AAAUSDT",
        side="long",
        signal_time=timestamp,
        zone=Zone("resistance", "1h", 98.0, 100.0, 2, 4.0, timestamp - pd.Timedelta(hours=2)),
        planned_entry=100.0,
        planned_stop=99.0,
        atr_5m=0.5,
        universe_rank=1,
        quote_volume_24h=1_000_000.0,
    )
    emitted = False

    def fake_on_bar(self, symbol, history_5m, top_symbols, as_of, universe_row=None):
        nonlocal emitted
        if emitted:
            return []
        emitted = True
        return [raw]

    monkeypatch.setattr("run_liquidity_retest_paper_trader.LiquidityRetestSignalEngine.on_bar", fake_on_bar)
    state = PaperState(transform_mode=settings.trade_transform_mode)
    runtime = PaperRuntime(output_dir=str(tmp_path), once=True)

    run_paper_cycle_from_frames({"AAAUSDT": frame}, settings=settings, runtime=runtime, state=state)
    assert "AAAUSDT" in state.positions

    next_time = frame.index[-1] + pd.Timedelta(minutes=5)
    next_bar = pd.DataFrame(
        {"open": [100.0], "high": [102.0], "low": [99.8], "close": [101.5], "volume": [10.0], "quote_volume": [1_000.0]},
        index=[next_time],
    )
    second_frame = pd.concat([frame, next_bar])
    run_paper_cycle_from_frames({"AAAUSDT": second_frame}, settings=settings, runtime=runtime, state=state)

    assert "AAAUSDT" not in state.positions
    trades = pd.read_csv(tmp_path / "trades.csv")
    assert trades.iloc[-1]["exit_reason"] == "stop_loss"
