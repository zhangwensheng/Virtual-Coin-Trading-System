from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from futures_strategy.funding_arbitrage import (
    ArbitrageCandidate,
    CandidateEvaluation,
    FundingArbitrageSettings,
)
from futures_strategy.funding_arbitrage_logging import FundingArbitrageLogger
from futures_strategy.funding_arbitrage_simulator import (
    FundingArbitragePortfolioSimulator,
    FundingSettlement,
)


def make_candidate() -> ArbitrageCandidate:
    return ArbitrageCandidate(
        symbol="AAAUSDT",
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


def test_candidate_log_contains_inputs_costs_and_reason(tmp_path: Path) -> None:
    logger = FundingArbitrageLogger(tmp_path, run_id="audit")
    evaluation = CandidateEvaluation(
        symbol="AAAUSDT",
        timestamp=pd.Timestamp("2026-01-01T00:05:00Z"),
        accepted=False,
        reason="INSUFFICIENT_COST_COVERAGE",
        candidate=None,
        funding_lookback=(0.001, 0.001, 0.001),
        round_trip_cost_rate=0.0038,
        predicted_funding_rate=0.0063,
        predicted_net_rate=0.0013,
    )
    logger.log_candidate(evaluation)
    row = pd.read_csv(tmp_path / "candidates.csv").iloc[0]
    assert row["symbol"] == "AAAUSDT"
    assert row["funding_lookback"] == "0.00100000|0.00100000|0.00100000"
    assert row["round_trip_cost_rate"] == 0.0038
    assert row["accepted"] in (False, 0)
    assert row["reason"] == "INSUFFICIENT_COST_COVERAGE"


def test_jsonl_event_has_stable_audit_envelope(tmp_path: Path) -> None:
    logger = FundingArbitrageLogger(tmp_path, run_id="audit")
    logger.log_event(
        event_type="TEST_EVENT",
        timestamp=pd.Timestamp("2026-01-01T00:05:00Z"),
        symbol="AAAUSDT",
        payload={"amount": 1.25},
    )
    event = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert event["run_id"] == "audit"
    assert event["event_type"] == "TEST_EVENT"
    assert event["symbol"] == "AAAUSDT"
    assert event["payload"] == {"amount": 1.25}
    assert event["event_id"]


def test_state_round_trip_preserves_positions_and_processed_funding_keys(tmp_path: Path) -> None:
    settings = FundingArbitrageSettings()
    simulator = FundingArbitragePortfolioSimulator(settings, output_dir=tmp_path)
    assert simulator.open_pair(make_candidate(), 200.0, pd.Timestamp("2026-01-01T00:05:00Z"))
    settlement = FundingSettlement(
        symbol="AAAUSDT",
        funding_time=pd.Timestamp("2026-01-01T08:00:00Z"),
        funding_rate=0.001,
        mark_price=100.1,
    )
    simulator.apply_funding(settlement)
    simulator.save_state()

    restored = FundingArbitragePortfolioSimulator.load_state(settings, output_dir=tmp_path)

    assert restored.processed_funding_keys == simulator.processed_funding_keys
    assert restored.positions.keys() == simulator.positions.keys()
    assert restored.positions["AAAUSDT"].cumulative_funding_usdt == 0.2
    assert restored.cash_usdt == simulator.cash_usdt


def test_reloading_and_replaying_settlement_is_idempotent(tmp_path: Path) -> None:
    settings = FundingArbitrageSettings()
    simulator = FundingArbitragePortfolioSimulator(settings, output_dir=tmp_path)
    assert simulator.open_pair(make_candidate(), 200.0, pd.Timestamp("2026-01-01T00:05:00Z"))
    settlement = FundingSettlement(
        symbol="AAAUSDT",
        funding_time=pd.Timestamp("2026-01-01T08:00:00Z"),
        funding_rate=0.001,
        mark_price=100.1,
    )
    simulator.apply_funding(settlement)
    simulator.save_state()
    restored = FundingArbitragePortfolioSimulator.load_state(settings, output_dir=tmp_path)

    assert restored.apply_funding(settlement) == 0.0
    assert restored.funding_pnl_usdt == 0.2


def test_state_persistence_can_be_disabled_for_backtests(tmp_path: Path) -> None:
    simulator = FundingArbitragePortfolioSimulator(
        FundingArbitrageSettings(),
        output_dir=tmp_path,
        persist_state=False,
    )

    simulator.save_state()

    assert not (tmp_path / "state.json").exists()
