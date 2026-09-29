from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from futures_strategy.liquidity_retest import LiquidityRetestSettings, SignalCandidate, Zone
from run_liquidity_retest_backtest import load_settings_config, run_backtest_from_frames


def fixture_module():
    spec = importlib.util.spec_from_file_location("liquidity_retest_fixtures", Path("tests/test_liquidity_retest.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_flat_frame(*, quote_volume: float) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=12 * 12, freq="5min", tz="UTC")
    close = [100.0] * len(index)
    return pd.DataFrame(
        {
            "open": close,
            "high": [100.2] * len(index),
            "low": [99.8] * len(index),
            "close": close,
            "volume": [quote_volume / 100.0] * len(index),
            "quote_volume": [quote_volume] * len(index),
        },
        index=index,
    )


def test_run_backtest_from_frames_generates_logs_and_respects_top5(tmp_path: Path) -> None:
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
    frames = {
        "AAAUSDT": fixtures.make_long_breakout_retest_frame(),
        "BBBUSDT": fixtures.make_short_breakout_retest_frame(),
        "CCCUSDT": make_flat_frame(quote_volume=400.0),
        "DDDUSDT": make_flat_frame(quote_volume=300.0),
        "EEEUSDT": make_flat_frame(quote_volume=200.0),
        "FFFUSDT": fixtures.make_long_breakout_retest_frame(),
    }
    frames["AAAUSDT"]["quote_volume"] = 600.0
    frames["BBBUSDT"]["quote_volume"] = 500.0
    frames["FFFUSDT"]["quote_volume"] = 10.0

    result = run_backtest_from_frames(frames, settings=settings, output_dir=tmp_path, initial_equity=10_000.0)

    assert "FFFUSDT" not in set(result.trades.get("symbol", []))
    assert (tmp_path / "events.jsonl").exists()
    assert (tmp_path / "signals.csv").exists()
    assert (tmp_path / "trades.csv").exists()
    assert (tmp_path / "daily_summary.csv").exists()
    assert (tmp_path / "equity_curve.csv").exists()
    assert (tmp_path / "summary.json").exists()


def test_load_settings_config_overrides_defaults(tmp_path: Path) -> None:
    config = tmp_path / "liquidity.yaml"
    config.write_text(
        "liquidity_retest:\n  min_margin_usdt: 2.5\n  max_symbol_entries_per_day: 4\n",
        encoding="utf-8",
    )
    settings = load_settings_config(config)
    assert settings.min_margin_usdt == 2.5
    assert settings.max_symbol_entries_per_day == 4


def test_backtest_applies_short_only_transform_before_risk_gate(tmp_path: Path, monkeypatch) -> None:
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

    monkeypatch.setattr("run_liquidity_retest_backtest.LiquidityRetestSignalEngine.on_bar", fake_on_bar)

    run_backtest_from_frames({"AAAUSDT": frame}, settings=settings, output_dir=tmp_path)

    signals = pd.read_csv(tmp_path / "signals.csv")
    accepted = signals.loc[signals["status"] == "accepted"]
    assert not accepted.empty
    assert set(accepted["side"]) == {"short"}
    assert set(accepted["original_side"]) == {"long"}
    assert (accepted["planned_stop"] > accepted["planned_entry"]).all()
    assert (accepted["planned_tp1"] < accepted["planned_entry"]).all()


def test_short_only_config_loads_transform_and_drawdown_limit() -> None:
    settings = load_settings_config("configs/binance_liquidity_retest_short_only.yaml")

    assert settings.trade_transform_mode == "mirror_original_longs_short_only"
    assert settings.max_drawdown_pct == 0.20
    assert settings.max_symbol_entries_per_day == 3
    assert settings.max_account_entries_per_day == 6
    assert settings.min_margin_usdt == 1.0
