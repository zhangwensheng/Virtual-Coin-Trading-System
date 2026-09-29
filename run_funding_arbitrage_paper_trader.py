"""Signal-only paper trader for Binance funding arbitrage.

This module deliberately has no authenticated client and cannot place orders.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import pandas as pd

from futures_strategy.binance_funding_arbitrage_data import BinanceFundingArbitrageDataClient
from futures_strategy.funding_arbitrage import (
    FundingArbitrageSettings,
    FundingObservation,
    MarketSnapshot,
)
from futures_strategy.funding_arbitrage_simulator import (
    FundingArbitragePortfolioSimulator,
    FundingSettlement,
)
from run_funding_arbitrage_backtest import run_backtest_cycle, settings_from_config


@dataclass(frozen=True)
class PaperCycleData:
    snapshots: Mapping[str, MarketSnapshot]
    funding_history: Mapping[str, Sequence[FundingObservation]]
    settlements: Sequence[FundingSettlement]


class PaperMarketProvider(Protocol):
    def get_cycle_data(self, now: pd.Timestamp) -> PaperCycleData:
        raise NotImplementedError


class BinancePublicPaperProvider:
    def __init__(self, *, universe_size: int = 20, client: BinanceFundingArbitrageDataClient | None = None) -> None:
        self.universe_size = universe_size
        self.client = client or BinanceFundingArbitrageDataClient()

    def get_cycle_data(self, now: pd.Timestamp) -> PaperCycleData:
        common = {item.symbol for item in self.client.fetch_common_tradable_symbols()}
        spot_tickers = self.client.fetch_24h_tickers("spot")
        perp_tickers = self.client.fetch_24h_tickers("perp")
        ranked = []
        for symbol in common & spot_tickers.keys() & perp_tickers.keys():
            liquidity = min(
                spot_tickers[symbol]["quote_volume"],
                perp_tickers[symbol]["quote_volume"],
            )
            ranked.append((symbol, liquidity))
        top = [symbol for symbol, _ in sorted(ranked, key=lambda item: (-item[1], item[0]))[: self.universe_size]]
        snapshots: dict[str, MarketSnapshot] = {}
        history: dict[str, list[FundingObservation]] = {}
        settlements: list[FundingSettlement] = []
        start_ms = int((now - pd.Timedelta(days=4)).timestamp() * 1000)
        end_ms = int(now.timestamp() * 1000)
        for symbol in top:
            snapshots[symbol] = MarketSnapshot(
                symbol=symbol,
                timestamp=now,
                spot_price=spot_tickers[symbol]["last_price"],
                perp_price=perp_tickers[symbol]["last_price"],
                spot_quote_volume_24h=spot_tickers[symbol]["quote_volume"],
                perp_quote_volume_24h=perp_tickers[symbol]["quote_volume"],
                spot_tradable=True,
                perp_tradable=True,
            )
            frame = self.client.fetch_funding_history(symbol, start_ms, end_ms)
            items = [
                FundingObservation(
                    symbol=symbol,
                    funding_time=timestamp,
                    funding_rate=float(row["funding_rate"]),
                    mark_price=float(row["mark_price"]),
                )
                for timestamp, row in frame.iterrows()
            ]
            history[symbol] = items
            settlements.extend(
                FundingSettlement(symbol, item.funding_time, item.funding_rate, item.mark_price)
                for item in items
                if item.funding_time <= now
            )
        return PaperCycleData(snapshots=snapshots, funding_history=history, settlements=settlements)


def _atomic_summary(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def run_paper_cycle(
    *,
    provider: PaperMarketProvider,
    settings: FundingArbitrageSettings,
    output_dir: Path,
    now: pd.Timestamp | None = None,
    stale_after_minutes: int = 15,
) -> dict[str, object]:
    timestamp = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    if state_path.exists():
        simulator = FundingArbitragePortfolioSimulator.load_state(settings, output_dir=output_dir)
    else:
        simulator = FundingArbitragePortfolioSimulator(settings, output_dir=output_dir, run_id="paper")
    before = set(simulator.positions)
    data = provider.get_cycle_data(timestamp)
    if not data.snapshots:
        cycle_status = "DATA_UNAVAILABLE"
        allow_entries = False
    else:
        newest = max(snapshot.timestamp for snapshot in data.snapshots.values())
        stale = timestamp - newest > pd.Timedelta(minutes=stale_after_minutes)
        cycle_status = "STALE_MARKET_DATA" if stale else "OK"
        allow_entries = not stale
    opened = run_backtest_cycle(
        timestamp=timestamp,
        snapshots=data.snapshots,
        funding_history=data.funding_history,
        settlements=data.settlements,
        simulator=simulator,
        settings=settings,
        allow_new_entries=allow_entries,
    )
    simulator.save_state()
    summary: dict[str, object] = {
        "mode": "paper",
        "timestamp": timestamp.isoformat(),
        "cycle_status": cycle_status,
        "message": "模拟盘，无真实下单",
        "live_orders_sent": 0,
        "new_position_groups": len(opened),
        "new_symbols": opened,
        "open_position_groups": len(simulator.positions),
        "open_symbols": sorted(simulator.positions),
        "closed_this_cycle": len(before - set(simulator.positions)),
        "equity_usdt": simulator.equity_usdt,
        "cash_usdt": simulator.cash_usdt,
        "funding_pnl_usdt": simulator.funding_pnl_usdt,
        "fees_usdt": simulator.realized_fees_usdt,
        "slippage_usdt": simulator.realized_slippage_usdt,
        "circuit_breaker": simulator.circuit_breaker,
    }
    _atomic_summary(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binance 资金费率套利模拟盘信号")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings, config = settings_from_config(args.config)
    paper_config = config.get("paper", {})
    stale_after = int(paper_config.get("stale_after_minutes", 15))
    poll_seconds = int(args.poll_seconds or paper_config.get("poll_seconds", 60))
    provider = BinancePublicPaperProvider(universe_size=settings.universe_size)
    print("模拟盘模式：只生成信号，不发送真实订单。")
    while True:
        try:
            summary = run_paper_cycle(
                provider=provider,
                settings=settings,
                output_dir=args.output_dir,
                stale_after_minutes=stale_after,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            print(json.dumps({"mode": "paper", "cycle_status": "ERROR", "error": str(exc)}, ensure_ascii=False))
            if args.once:
                return 2
        if args.once:
            return 0
        try:
            time.sleep(max(1, poll_seconds))
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
