from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pandas as pd
import pytest

from futures_strategy.liquidity_retest import (
    EntryCounters,
    EventLogger,
    LiquidityRetestPortfolioSimulator,
    LiquidityRetestSettings,
    RiskGate,
    SignalCandidate,
    Zone,
)


def make_candidate(
    symbol: str,
    side: str,
    timestamp: pd.Timestamp,
    *,
    entry: float = 100.0,
    stop: float | None = None,
) -> SignalCandidate:
    if stop is None:
        stop = 99.0 if side == "long" else 101.0
    return SignalCandidate(
        signal_id=f"{symbol}-{side}-{timestamp.isoformat()}",
        symbol=symbol,
        side=side,
        signal_time=timestamp,
        zone=Zone(
            side="resistance" if side == "long" else "support",
            timeframe="1h",
            low=98.0 if side == "long" else 100.0,
            high=100.0 if side == "long" else 102.0,
            touches=2,
            score=4.0,
            last_touch_time=timestamp - pd.Timedelta(hours=2),
        ),
        planned_entry=entry,
        planned_stop=stop,
        atr_5m=0.5,
        universe_rank=1,
        quote_volume_24h=1_000_000.0,
    )


def make_exit_frame_long() -> pd.DataFrame:
    index = pd.date_range("2026-01-01T00:10:00Z", periods=5, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 101.7, 101.6, 101.2],
            "high": [101.0, 101.7, 102.1, 101.8, 101.3],
            "low": [99.8, 100.9, 101.4, 101.2, 100.4],
            "close": [101.0, 101.6, 101.8, 101.25, 100.7],
            "atr": [0.5, 0.5, 0.5, 0.5, 0.5],
        },
        index=index,
    )


def test_simulator_uses_candidate_explicit_tp1(tmp_path: Path) -> None:
    settings = LiquidityRetestSettings(fee_rate=0.0, slippage_bps=0.0, min_margin_usdt=0.1)
    candidate = replace(
        make_candidate("AAAUSDT", "short", pd.Timestamp("2026-01-01T00:00:00Z"), entry=100.0, stop=103.0),
        planned_tp1=98.0,
    )
    simulator = LiquidityRetestPortfolioSimulator(settings, logger=EventLogger(tmp_path, run_id="explicit-tp1"))

    assert simulator.open_from_candidate(candidate, fill_price=100.0, fill_time=candidate.signal_time)
    assert simulator.positions["AAAUSDT"].tp1_price == pytest.approx(98.0)
    assert simulator.positions["AAAUSDT"].qty == pytest.approx((10_000.0 * 0.005) / 3.0)


def test_drawdown_limit_blocks_new_entries_but_keeps_existing_position_management(tmp_path: Path) -> None:
    settings = LiquidityRetestSettings(
        max_drawdown_pct=0.20,
        fee_rate=0.0,
        slippage_bps=0.0,
        min_margin_usdt=0.1,
    )
    simulator = LiquidityRetestPortfolioSimulator(
        settings,
        logger=EventLogger(tmp_path, run_id="drawdown"),
        initial_equity=10_000.0,
    )
    candidate = make_candidate(
        "AAAUSDT",
        "short",
        pd.Timestamp("2026-01-01T00:00:00Z"),
        entry=100.0,
        stop=101.0,
    )

    assert simulator.open_from_candidate(candidate, fill_price=100.0, fill_time=candidate.signal_time)
    simulator.cash = 7_999.0
    blocked = make_candidate(
        "BBBUSDT",
        "short",
        pd.Timestamp("2026-01-01T00:05:00Z"),
        entry=100.0,
        stop=101.0,
    )
    assert simulator.open_from_candidate(blocked, fill_price=100.0, fill_time=blocked.signal_time) is False
    signals = pd.read_csv(tmp_path / "signals.csv")
    assert signals.iloc[-1]["block_reason"] == "BLOCKED_DRAWDOWN_LIMIT"

    simulator.manage_positions(
        pd.Timestamp("2026-01-01T00:10:00Z"),
        {"AAAUSDT": pd.Series({"high": 102.0, "low": 99.0, "close": 101.0, "atr": 1.0})},
    )
    assert "AAAUSDT" not in simulator.positions


def test_risk_gate_blocks_fourth_symbol_trade_and_seventh_account_trade_on_beijing_day() -> None:
    settings = LiquidityRetestSettings(max_symbol_entries_per_day=3, max_account_entries_per_day=6)
    gate = RiskGate(settings)
    counters = EntryCounters()
    day = pd.Timestamp("2026-01-01T16:30:00Z")
    for index in range(3):
        counters.record_entry("AAAUSDT", day + pd.Timedelta(minutes=index))
    decision = gate.evaluate(make_candidate("AAAUSDT", "long", day), equity=10_000.0, open_positions={}, counters=counters, cooldowns={})
    assert decision.status == "blocked"
    assert decision.reason == "BLOCKED_SYMBOL_DAILY_LIMIT"
    counters = EntryCounters()
    for symbol in ["B1USDT", "B2USDT", "B3USDT", "B4USDT", "B5USDT", "B6USDT"]:
        counters.record_entry(symbol, day)
    decision = gate.evaluate(make_candidate("B7USDT", "short", day), equity=10_000.0, open_positions={}, counters=counters, cooldowns={})
    assert decision.status == "blocked"
    assert decision.reason == "BLOCKED_ACCOUNT_DAILY_LIMIT"


def test_entry_counters_reset_at_asia_shanghai_midnight() -> None:
    counters = EntryCounters(timezone="Asia/Shanghai")
    counters.record_entry("AAAUSDT", pd.Timestamp("2026-01-01T15:55:00Z"))
    counters.record_entry("AAAUSDT", pd.Timestamp("2026-01-01T16:05:00Z"))
    assert counters.symbol_count("AAAUSDT", pd.Timestamp("2026-01-01T15:59:00Z")) == 1
    assert counters.symbol_count("AAAUSDT", pd.Timestamp("2026-01-01T16:05:00Z")) == 1


def test_risk_gate_rejects_safe_position_below_one_usdt_margin_without_inflating_size() -> None:
    settings = LiquidityRetestSettings(risk_per_trade=0.005, leverage=3.0, min_margin_usdt=1.0, max_stop_pct=1.0)
    gate = RiskGate(settings)
    candidate = make_candidate("AAAUSDT", "long", pd.Timestamp("2026-01-01T00:00:00Z"), entry=100.0, stop=50.0)
    decision = gate.evaluate(candidate, equity=100.0, open_positions={}, counters=EntryCounters(), cooldowns={})
    assert decision.status == "blocked"
    assert decision.reason == "BLOCKED_MIN_MARGIN"
    assert decision.margin_usdt < 1.0


def test_risk_gate_rejects_invalid_or_too_wide_stop() -> None:
    gate = RiskGate(LiquidityRetestSettings(max_stop_pct=0.03))
    invalid = gate.evaluate(
        make_candidate("AAAUSDT", "long", pd.Timestamp("2026-01-01T00:00:00Z"), entry=100.0, stop=101.0),
        equity=10_000.0,
        open_positions={},
        counters=EntryCounters(),
        cooldowns={},
    )
    too_wide = gate.evaluate(
        make_candidate("AAAUSDT", "short", pd.Timestamp("2026-01-01T00:05:00Z"), entry=100.0, stop=104.0),
        equity=10_000.0,
        open_positions={},
        counters=EntryCounters(),
        cooldowns={},
    )
    assert invalid.reason == "BLOCKED_INVALID_STOP"
    assert too_wide.reason == "BLOCKED_INVALID_STOP"


def test_portfolio_simulator_opens_partial_exits_and_trails_remaining_position(tmp_path: Path) -> None:
    settings = LiquidityRetestSettings(fee_rate=0.0, slippage_bps=0.0, min_margin_usdt=0.1)
    logger = EventLogger(tmp_path, run_id="test-run")
    simulator = LiquidityRetestPortfolioSimulator(settings, logger=logger, initial_equity=10_000.0)
    candidate = make_candidate("AAAUSDT", "long", pd.Timestamp("2026-01-01T00:00:00Z"), entry=100.0, stop=99.0)
    assert simulator.open_from_candidate(candidate, fill_price=100.0, fill_time=pd.Timestamp("2026-01-01T00:05:00Z"))
    for timestamp, row in make_exit_frame_long().iterrows():
        simulator.manage_positions(timestamp, {"AAAUSDT": row})
    simulator.close_all(pd.Timestamp("2026-01-01T00:35:00Z"), {"AAAUSDT": 100.7}, reason="end_of_test")
    trades = pd.read_csv(tmp_path / "trades.csv")
    assert trades.iloc[0]["exit_reason"] in {"trail_stop", "time_exit", "end_of_test"}
    assert bool(trades.iloc[0]["scaled_out"]) is True
