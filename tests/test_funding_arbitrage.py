from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from futures_strategy.funding_arbitrage import (
    ArbitrageCandidate,
    FundingArbitrageSettings,
    FundingObservation,
    MarketSnapshot,
    allocate_candidates,
    evaluate_candidate,
    forecast_seven_day_funding_rate,
    infer_funding_interval_hours,
    round_trip_cost_rate,
)


def make_observations(
    rates: list[float],
    *,
    symbol: str = "AAAUSDT",
    hours: int = 8,
    start: str = "2026-01-01T00:00:00Z",
) -> list[FundingObservation]:
    first = pd.Timestamp(start)
    return [
        FundingObservation(
            symbol=symbol,
            funding_time=first + timedelta(hours=index * hours),
            funding_rate=rate,
            mark_price=100.0,
        )
        for index, rate in enumerate(rates)
    ]


def make_snapshot(symbol: str = "AAAUSDT", timestamp: str = "2026-01-02T00:01:00Z") -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        timestamp=pd.Timestamp(timestamp),
        spot_price=100.0,
        perp_price=100.1,
        spot_quote_volume_24h=50_000_000.0,
        perp_quote_volume_24h=80_000_000.0,
        spot_tradable=True,
        perp_tradable=True,
    )


def test_round_trip_cost_includes_four_fills() -> None:
    settings = FundingArbitrageSettings()
    assert round_trip_cost_rate(settings) == pytest.approx(0.0038)


def test_forecast_uses_lowest_positive_rate_and_observed_interval() -> None:
    history = make_observations([0.0006, 0.0004, 0.0005], hours=4)
    assert infer_funding_interval_hours(history) == pytest.approx(4.0)
    assert forecast_seven_day_funding_rate(history) == pytest.approx(0.0004 * 42)


def test_candidate_rejects_non_positive_lookback_rate() -> None:
    result = evaluate_candidate(
        make_snapshot(),
        make_observations([0.0006, -0.0001, 0.0007]),
        FundingArbitrageSettings(),
    )
    assert result.accepted is False
    assert result.reason == "NON_POSITIVE_FUNDING_LOOKBACK"


def test_candidate_requires_twice_full_round_trip_cost() -> None:
    result = evaluate_candidate(
        make_snapshot(),
        make_observations([0.0001, 0.0001, 0.0001]),
        FundingArbitrageSettings(),
    )
    assert result.accepted is False
    assert result.reason == "INSUFFICIENT_COST_COVERAGE"


def test_candidate_rejects_future_funding_observation() -> None:
    future_snapshot = make_snapshot(timestamp="2026-01-01T12:00:00Z")
    history = make_observations([0.001, 0.001, 0.001], hours=8)
    with pytest.raises(ValueError, match="FUTURE_FUNDING_OBSERVATION"):
        evaluate_candidate(future_snapshot, history, FundingArbitrageSettings())


def test_candidate_net_score_deducts_risk_and_rebalance_buffers() -> None:
    settings = FundingArbitrageSettings()
    result = evaluate_candidate(
        make_snapshot(),
        make_observations([0.001, 0.001, 0.001]),
        settings,
    )
    assert result.accepted is True
    assert result.candidate is not None
    assert result.candidate.predicted_funding_rate == pytest.approx(0.021)
    assert result.candidate.predicted_net_rate == pytest.approx(0.016)


def test_allocator_respects_total_group_and_position_caps() -> None:
    settings = FundingArbitrageSettings()
    candidates: list[ArbitrageCandidate] = []
    for index in range(7):
        symbol = f"A{index}USDT"
        result = evaluate_candidate(
            make_snapshot(symbol),
            make_observations([0.001, 0.001, 0.001], symbol=symbol),
            settings,
        )
        assert result.candidate is not None
        candidates.append(result.candidate)

    allocations = allocate_candidates(candidates, 800.0, frozenset(), settings)

    assert len(allocations) == 5
    assert sum(item.notional_usdt for item in allocations) == pytest.approx(800.0)
    assert all(50.0 <= item.notional_usdt <= 320.0 for item in allocations)


def test_allocator_is_deterministic_and_excludes_open_symbol() -> None:
    settings = FundingArbitrageSettings(max_positions=2)
    candidates: list[ArbitrageCandidate] = []
    for symbol in ("BBBUSDT", "AAAUSDT", "CCCUSDT"):
        result = evaluate_candidate(
            make_snapshot(symbol),
            make_observations([0.001, 0.001, 0.001], symbol=symbol),
            settings,
        )
        assert result.candidate is not None
        candidates.append(result.candidate)

    allocations = allocate_candidates(candidates, 200.0, frozenset({"AAAUSDT"}), settings)

    assert [item.symbol for item in allocations] == ["BBBUSDT"]
    assert allocations[0].notional_usdt == pytest.approx(200.0)
