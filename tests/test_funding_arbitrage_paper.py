from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from futures_strategy.funding_arbitrage import (
    FundingArbitrageSettings,
    FundingObservation,
    MarketSnapshot,
)
from run_funding_arbitrage_paper_trader import PaperCycleData, run_paper_cycle


class FakePaperProvider:
    def __init__(self, symbols: list[str], *, stale_minutes: int = 0) -> None:
        self.symbols = symbols
        self.stale_minutes = stale_minutes

    def get_cycle_data(self, now: pd.Timestamp) -> PaperCycleData:
        snapshot_time = now - pd.Timedelta(minutes=self.stale_minutes)
        snapshots = {
            symbol: MarketSnapshot(
                symbol=symbol,
                timestamp=snapshot_time,
                spot_price=100.0,
                perp_price=100.1,
                spot_quote_volume_24h=20_000_000.0 - index,
                perp_quote_volume_24h=30_000_000.0 - index,
                spot_tradable=True,
                perp_tradable=True,
            )
            for index, symbol in enumerate(self.symbols)
        }
        history = {
            symbol: [
                FundingObservation(symbol, now - pd.Timedelta(hours=24), 0.001, 100.1),
                FundingObservation(symbol, now - pd.Timedelta(hours=16), 0.001, 100.1),
                FundingObservation(symbol, now - pd.Timedelta(hours=8), 0.001, 100.1),
            ]
            for symbol in self.symbols
        }
        return PaperCycleData(snapshots=snapshots, funding_history=history, settlements=[])


def test_first_paper_cycle_emits_two_leg_orders_and_never_live_orders(tmp_path: Path) -> None:
    now = pd.Timestamp("2026-01-02T00:00:00Z")
    summary = run_paper_cycle(
        provider=FakePaperProvider(["AAAUSDT"]),
        settings=FundingArbitrageSettings(),
        output_dir=tmp_path,
        now=now,
    )
    orders = pd.read_csv(tmp_path / "orders.csv")
    assert summary["mode"] == "paper"
    assert summary["live_orders_sent"] == 0
    assert summary["new_position_groups"] == 1
    assert set(zip(orders["market"], orders["side"])) == {("SPOT", "BUY"), ("PERP", "SELL")}
    assert orders["group_id"].nunique() == 1


def test_restarting_same_market_state_does_not_duplicate_position(tmp_path: Path) -> None:
    settings = FundingArbitrageSettings()
    provider = FakePaperProvider(["AAAUSDT"])
    now = pd.Timestamp("2026-01-02T00:00:00Z")
    first = run_paper_cycle(provider=provider, settings=settings, output_dir=tmp_path, now=now)
    second = run_paper_cycle(
        provider=provider,
        settings=settings,
        output_dir=tmp_path,
        now=now + pd.Timedelta(minutes=1),
    )
    orders = pd.read_csv(tmp_path / "orders.csv")
    assert first["open_position_groups"] == 1
    assert second["new_position_groups"] == 0
    assert second["open_position_groups"] == 1
    assert len(orders) == 2


def test_stale_market_data_blocks_new_entries_but_writes_summary(tmp_path: Path) -> None:
    summary = run_paper_cycle(
        provider=FakePaperProvider(["AAAUSDT"], stale_minutes=20),
        settings=FundingArbitrageSettings(),
        output_dir=tmp_path,
        now=pd.Timestamp("2026-01-02T00:00:00Z"),
        stale_after_minutes=15,
    )
    assert summary["new_position_groups"] == 0
    assert summary["cycle_status"] == "STALE_MARKET_DATA"
    assert (tmp_path / "summary.json").exists()


def test_paper_cycle_never_exceeds_five_groups(tmp_path: Path) -> None:
    settings = FundingArbitrageSettings(spot_fee_rate=0.0, perp_fee_rate=0.0, slippage_bps=0.0)
    symbols = [f"A{index}USDT" for index in range(6)]
    summary = run_paper_cycle(
        provider=FakePaperProvider(symbols),
        settings=settings,
        output_dir=tmp_path,
        now=pd.Timestamp("2026-01-02T00:00:00Z"),
    )
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert summary["open_position_groups"] == 5
    assert len(state["positions"]) == 5
    assert sum(item["initial_notional_usdt"] for item in state["positions"].values()) <= 800.0
