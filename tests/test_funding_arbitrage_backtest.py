from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from futures_strategy.funding_arbitrage import (
    FundingArbitrageSettings,
    FundingObservation,
    MarketSnapshot,
)
from futures_strategy.funding_arbitrage_simulator import (
    FundingArbitragePortfolioSimulator,
    FundingSettlement,
)
from run_funding_arbitrage_backtest import (
    _filter_symbols_for_window,
    _read_indexed_csv,
    build_period_summary,
    run_backtest_cycle,
    settings_from_config,
)


def positive_history(symbol: str = "AAAUSDT") -> list[FundingObservation]:
    return [
        FundingObservation(symbol, pd.Timestamp("2026-01-01T00:00:00Z"), 0.001, 100.0),
        FundingObservation(symbol, pd.Timestamp("2026-01-01T08:00:00Z"), 0.001, 100.0),
        FundingObservation(symbol, pd.Timestamp("2026-01-01T16:00:00Z"), 0.001, 100.0),
    ]


def market(timestamp: str = "2026-01-02T00:00:00Z") -> MarketSnapshot:
    return MarketSnapshot(
        symbol="AAAUSDT",
        timestamp=pd.Timestamp(timestamp),
        spot_price=100.0,
        perp_price=100.1,
        spot_quote_volume_24h=10_000_000.0,
        perp_quote_volume_24h=20_000_000.0,
        spot_tradable=True,
        perp_tradable=True,
    )


def test_cycle_excludes_funding_published_at_same_decision_time(tmp_path: Path) -> None:
    settings = FundingArbitrageSettings()
    simulator = FundingArbitragePortfolioSimulator(settings, output_dir=tmp_path)
    timestamp = pd.Timestamp("2026-01-02T00:00:00Z")
    history = positive_history() + [FundingObservation("AAAUSDT", timestamp, -0.01, 100.1)]

    opened = run_backtest_cycle(
        timestamp=timestamp,
        snapshots={"AAAUSDT": market()},
        funding_history={"AAAUSDT": history},
        settlements=[],
        simulator=simulator,
        settings=settings,
        allow_new_entries=True,
    )

    assert opened == ["AAAUSDT"]


def test_cycle_does_not_credit_new_position_for_same_time_settlement(tmp_path: Path) -> None:
    settings = FundingArbitrageSettings()
    simulator = FundingArbitragePortfolioSimulator(settings, output_dir=tmp_path)
    timestamp = pd.Timestamp("2026-01-02T00:00:00Z")
    settlement = FundingSettlement("AAAUSDT", timestamp, 0.001, 100.1)

    run_backtest_cycle(
        timestamp=timestamp,
        snapshots={"AAAUSDT": market()},
        funding_history={"AAAUSDT": positive_history()},
        settlements=[settlement],
        simulator=simulator,
        settings=settings,
        allow_new_entries=True,
    )

    assert "AAAUSDT" in simulator.positions
    assert simulator.funding_pnl_usdt == 0.0
    assert settlement.key in simulator.processed_funding_keys


def test_cycle_manages_existing_position_when_new_entries_are_disabled(tmp_path: Path) -> None:
    settings = FundingArbitrageSettings()
    simulator = FundingArbitragePortfolioSimulator(settings, output_dir=tmp_path)
    run_backtest_cycle(
        timestamp=pd.Timestamp("2026-01-02T00:00:00Z"),
        snapshots={"AAAUSDT": market()},
        funding_history={"AAAUSDT": positive_history()},
        settlements=[],
        simulator=simulator,
        settings=settings,
        allow_new_entries=True,
    )
    run_backtest_cycle(
        timestamp=pd.Timestamp("2026-01-02T08:00:00Z"),
        snapshots={"AAAUSDT": market("2026-01-02T08:00:00Z")},
        funding_history={"AAAUSDT": positive_history()},
        settlements=[FundingSettlement("AAAUSDT", pd.Timestamp("2026-01-02T08:00:00Z"), 0.001, 100.1)],
        simulator=simulator,
        settings=settings,
        allow_new_entries=False,
    )
    assert simulator.funding_pnl_usdt > 0.0
    assert len(simulator.positions) == 1


def test_period_summary_uses_actual_equity_at_period_boundary() -> None:
    index = pd.to_datetime(
        ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-03-01T00:00:00Z"]
    )
    equity = pd.DataFrame({"equity_usdt": [1000.0, 1100.0, 1155.0]}, index=index)
    summary = build_period_summary(
        equity,
        trades=pd.DataFrame(),
        settlements=pd.DataFrame(),
        start=pd.Timestamp("2026-02-01T00:00:00Z"),
        end=pd.Timestamp("2026-03-01T00:00:00Z"),
    )
    assert summary["starting_equity_usdt"] == 1100.0
    assert summary["ending_equity_usdt"] == 1155.0
    assert summary["net_return_pct"] == pytest.approx(0.05)
    assert summary["max_drawdown_pct"] == 0.0


def test_period_summary_calculates_drawdown_and_verified_settlements() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC")
    equity = pd.DataFrame({"equity_usdt": [1000.0, 1100.0, 990.0, 1050.0]}, index=index)
    settlements = pd.DataFrame(
        {
            "funding_time": ["2026-01-02T00:00:00.000000+00:00", "2026-01-03T00:00:00+00:00"],
            "status": ["APPLIED", "NOT_ELIGIBLE"],
            "cashflow_usdt": [1.0, 0.0],
        }
    )
    summary = build_period_summary(equity, pd.DataFrame(), settlements, start=index[0], end=index[-1])
    assert summary["max_drawdown_pct"] == pytest.approx(0.1)
    assert summary["verified_funding_settlements"] == 1


def test_config_rejects_any_attempt_to_enable_live_ordering(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "strategy:\n  initial_capital_usdt: 1000\npaper:\n  live_ordering_enabled: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="LIVE_ORDERING_NOT_SUPPORTED"):
        settings_from_config(config)


def test_indexed_csv_reader_accepts_mixed_timestamp_precision(tmp_path: Path) -> None:
    path = tmp_path / "funding.csv"
    path.write_text(
        "funding_time,funding_rate,mark_price\n"
        "2026-05-24 08:00:00.000000+00:00,0.001,100\n"
        "2026-05-24 16:00:00+00:00,0.001,101\n",
        encoding="utf-8",
    )

    frame = _read_indexed_csv(path)

    assert list(frame.index) == [
        pd.Timestamp("2026-05-24T08:00:00Z"),
        pd.Timestamp("2026-05-24T16:00:00Z"),
    ]


def test_window_filter_excludes_newly_listed_symbols_without_blocking_backtest() -> None:
    start = pd.Timestamp("2026-01-02T00:00:00Z")
    end = pd.Timestamp("2026-01-03T00:00:00Z")
    full_index = pd.date_range("2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z", freq="5min")
    short_index = pd.date_range("2026-01-02T12:00:00Z", "2026-01-03T00:00:00Z", freq="5min")
    full = pd.DataFrame({"close": 100.0, "quote_volume": 1_000.0}, index=full_index)
    short = pd.DataFrame({"close": 10.0, "quote_volume": 1_000.0}, index=short_index)

    spot, perp, funding, exclusions = _filter_symbols_for_window(
        {"BTCUSDT": full, "AEROUSDT": short},
        {"BTCUSDT": full, "AEROUSDT": short},
        {"BTCUSDT": positive_history("BTCUSDT"), "AEROUSDT": positive_history("AEROUSDT")},
        start=start,
        end=end,
    )

    assert set(spot) == {"BTCUSDT"}
    assert set(perp) == {"BTCUSDT"}
    assert set(funding) == {"BTCUSDT"}
    assert exclusions == [
        {
            "symbol": "AEROUSDT",
            "reason": "INSUFFICIENT_WINDOW_COVERAGE",
            "spot_start": short_index[0].isoformat(),
            "spot_end": short_index[-1].isoformat(),
            "perp_start": short_index[0].isoformat(),
            "perp_end": short_index[-1].isoformat(),
        }
    ]
