from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from futures_strategy.funding_arbitrage import ArbitrageCandidate, FundingArbitrageSettings
from futures_strategy.funding_arbitrage_simulator import (
    FundingArbitragePortfolioSimulator,
    FundingSettlement,
)


def make_candidate(symbol: str = "AAAUSDT") -> ArbitrageCandidate:
    return ArbitrageCandidate(
        symbol=symbol,
        timestamp=pd.Timestamp("2026-01-01T00:05:00Z"),
        spot_price=100.0,
        perp_price=100.1,
        entry_basis_pct=0.001,
        liquidity_score=50_000_000.0,
        funding_lookback=(0.001, 0.001, 0.001),
        funding_interval_hours=8.0,
        predicted_funding_rate=0.021,
        round_trip_cost_rate=0.0038,
        expected_rebalance_cost_rate=0.0002,
        basis_risk_buffer_rate=0.001,
        predicted_net_rate=0.016,
        score=0.016,
    )


def make_open_simulator(tmp_path: Path, settings: FundingArbitrageSettings | None = None) -> FundingArbitragePortfolioSimulator:
    simulator = FundingArbitragePortfolioSimulator(settings or FundingArbitrageSettings(), output_dir=tmp_path)
    assert simulator.open_pair(make_candidate(), 200.0, pd.Timestamp("2026-01-01T00:05:00Z"))
    return simulator


def make_settlement(rate: float = 0.001) -> FundingSettlement:
    return FundingSettlement(
        symbol="AAAUSDT",
        funding_time=pd.Timestamp("2026-01-01T08:00:00Z"),
        funding_rate=rate,
        mark_price=100.1,
    )


def test_open_and_close_charge_each_leg_fees_and_slippage(tmp_path: Path) -> None:
    simulator = make_open_simulator(tmp_path)
    simulator.close_pair(
        "AAAUSDT",
        spot_price=100.0,
        perp_price=100.1,
        timestamp=pd.Timestamp("2026-01-01T08:05:00Z"),
        reason="TEST_CLOSE",
    )
    assert simulator.realized_fees_usdt > 0.0
    assert simulator.realized_slippage_usdt > 0.0
    assert "AAAUSDT" not in simulator.positions


def test_equal_spot_and_perp_move_has_near_zero_gross_price_pnl(tmp_path: Path) -> None:
    settings = FundingArbitrageSettings(spot_fee_rate=0.0, perp_fee_rate=0.0, slippage_bps=0.0)
    simulator = make_open_simulator(tmp_path, settings)
    simulator.mark_to_market(
        timestamp=pd.Timestamp("2026-01-01T04:05:00Z"),
        prices={"AAAUSDT": (110.0, 110.11)},
    )
    assert simulator.unrealized_price_pnl_usdt == pytest.approx(0.0, abs=0.03)


def test_positive_funding_credits_short_notional_once(tmp_path: Path) -> None:
    simulator = make_open_simulator(tmp_path)
    first = simulator.apply_funding(make_settlement())
    second = simulator.apply_funding(make_settlement())
    assert first == pytest.approx(0.2)
    assert second == pytest.approx(0.0)
    assert simulator.funding_pnl_usdt == pytest.approx(0.2)


def test_position_opened_at_settlement_time_cannot_receive_settlement(tmp_path: Path) -> None:
    simulator = FundingArbitragePortfolioSimulator(FundingArbitrageSettings(), output_dir=tmp_path)
    candidate = make_candidate()
    assert simulator.open_pair(candidate, 200.0, pd.Timestamp("2026-01-01T08:00:00Z"))
    assert simulator.apply_funding(make_settlement()) == pytest.approx(0.0)


def test_negative_funding_is_charged_then_position_is_closed(tmp_path: Path) -> None:
    simulator = make_open_simulator(tmp_path)
    simulator.apply_funding(make_settlement(rate=-0.001))
    assert simulator.funding_pnl_usdt == pytest.approx(-0.2)
    assert "AAAUSDT" not in simulator.positions
    assert simulator.closed_trades[-1]["exit_reason"] == "NEGATIVE_FUNDING"


def test_hedge_drift_over_limit_rebalances_perpetual_leg(tmp_path: Path) -> None:
    settings = FundingArbitrageSettings(
        spot_fee_rate=0.0,
        perp_fee_rate=0.0,
        slippage_bps=0.0,
        max_abs_entry_basis_pct=1.0,
        max_basis_widening_pct=1.0,
    )
    simulator = make_open_simulator(tmp_path, settings)
    simulator.mark_to_market(
        timestamp=pd.Timestamp("2026-01-01T04:05:00Z"),
        prices={"AAAUSDT": (110.0, 100.0)},
        rebalance=True,
    )
    position = simulator.positions["AAAUSDT"]
    spot_notional = position.spot_quantity * 110.0
    perp_notional = position.perp_quantity * 100.0
    assert abs(spot_notional - perp_notional) / max(spot_notional, perp_notional) <= 0.005
    assert simulator.rebalance_count == 1


def test_rebalance_reducing_short_realizes_perp_pnl_to_cash(tmp_path: Path) -> None:
    settings = FundingArbitrageSettings(
        spot_fee_rate=0.0,
        perp_fee_rate=0.0,
        slippage_bps=0.0,
        max_abs_entry_basis_pct=1.0,
        max_basis_widening_pct=1.0,
    )
    simulator = make_open_simulator(tmp_path, settings)
    before = simulator.positions["AAAUSDT"]
    cash_before = simulator.cash_usdt

    simulator.mark_to_market(
        timestamp=pd.Timestamp("2026-01-01T04:05:00Z"),
        prices={"AAAUSDT": (90.0, 99.0)},
        rebalance=True,
    )

    after = simulator.positions["AAAUSDT"]
    closed_quantity = before.perp_quantity - after.perp_quantity
    realized_pnl = closed_quantity * (before.perp_entry_price - 99.0)
    released_margin = before.perp_margin_usdt - after.perp_margin_usdt
    assert simulator.realized_price_pnl_usdt == pytest.approx(realized_pnl)
    assert simulator.cash_usdt == pytest.approx(cash_before + released_margin + realized_pnl)


def test_rebalance_adding_short_uses_slipped_fill_in_average_entry(tmp_path: Path) -> None:
    settings = FundingArbitrageSettings(
        spot_fee_rate=0.0,
        perp_fee_rate=0.0,
        slippage_bps=10.0,
        max_abs_entry_basis_pct=1.0,
        max_basis_widening_pct=1.0,
    )
    simulator = make_open_simulator(tmp_path, settings)
    before = simulator.positions["AAAUSDT"]

    simulator.mark_to_market(
        timestamp=pd.Timestamp("2026-01-01T04:05:00Z"),
        prices={"AAAUSDT": (110.0, 100.0)},
        rebalance=True,
    )

    after = simulator.positions["AAAUSDT"]
    added_quantity = after.perp_quantity - before.perp_quantity
    expected_entry = (before.perp_quantity * before.perp_entry_price + added_quantity * 99.9) / after.perp_quantity
    assert after.perp_entry_price == pytest.approx(expected_entry)


def test_closed_trade_net_pnl_deducts_open_and_close_fees(tmp_path: Path) -> None:
    settings = FundingArbitrageSettings(spot_fee_rate=0.001, perp_fee_rate=0.001, slippage_bps=0.0)
    simulator = make_open_simulator(tmp_path, settings)

    trade = simulator.close_pair(
        "AAAUSDT",
        spot_price=100.0,
        perp_price=100.1,
        timestamp=pd.Timestamp("2026-01-01T08:05:00Z"),
        reason="TEST_CLOSE",
    )

    assert trade is not None
    assert trade["gross_price_pnl_usdt"] == pytest.approx(0.0, abs=1e-9)
    assert trade["net_pnl_usdt"] == pytest.approx(-simulator.realized_fees_usdt)


def test_second_leg_failure_emergency_closes_first_leg(tmp_path: Path) -> None:
    simulator = FundingArbitragePortfolioSimulator(FundingArbitrageSettings(), output_dir=tmp_path)
    opened = simulator.open_pair(
        make_candidate(),
        200.0,
        pd.Timestamp("2026-01-01T00:05:00Z"),
        second_leg_succeeds=False,
    )
    assert opened is False
    assert simulator.positions == {}
    assert simulator.closed_trades[-1]["exit_reason"] == "ONE_LEG_FAILURE"
    assert simulator.realized_slippage_usdt > 0.0


def test_five_percent_drawdown_closes_positions_and_blocks_new_entries(tmp_path: Path) -> None:
    settings = FundingArbitrageSettings(
        spot_fee_rate=0.0,
        perp_fee_rate=0.0,
        slippage_bps=0.0,
        max_drawdown_pct=0.05,
    )
    simulator = make_open_simulator(tmp_path, settings)
    simulator.peak_equity_usdt = 1100.0
    simulator.mark_to_market(
        timestamp=pd.Timestamp("2026-01-01T04:05:00Z"),
        prices={"AAAUSDT": (100.0, 100.1)},
    )
    assert simulator.circuit_breaker is True
    assert simulator.positions == {}
    assert simulator.open_pair(make_candidate("BBBUSDT"), 100.0, pd.Timestamp("2026-01-01T05:05:00Z")) is False


def test_maximum_holding_period_closes_pair(tmp_path: Path) -> None:
    simulator = make_open_simulator(tmp_path)
    simulator.mark_to_market(
        timestamp=pd.Timestamp("2026-01-08T00:05:00Z"),
        prices={"AAAUSDT": (100.0, 100.1)},
    )
    assert "AAAUSDT" not in simulator.positions
    assert simulator.closed_trades[-1]["exit_reason"] == "MAX_HOLD_TIME"
