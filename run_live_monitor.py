from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from futures_strategy.config import load_config
from futures_strategy.data import load_market_data
from futures_strategy.realtime import append_event_jsonl, stream_klines, upsert_candle
from futures_strategy.strategy import latest_signal_snapshot, prepare_market_data


async def runner(args) -> None:
    config = load_config(args.config)
    if args.warmup_limit:
        config.exchange.limit = args.warmup_limit

    raw = load_market_data(config.exchange)
    prepared = prepare_market_data(raw, config.strategy)
    initial_signal = latest_signal_snapshot(prepared, config.strategy)
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_signal_path = output_dir / "latest_signal.json"
    live_signal_path = output_dir / "live_signals.jsonl"
    live_kline_path = output_dir / "live_klines.jsonl"

    latest_signal_path.write_text(json.dumps(initial_signal, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(initial_signal, ensure_ascii=False))

    async for event in stream_klines(
        config.exchange.name,
        config.exchange.symbol,
        config.exchange.timeframe,
        closed_only=True,
        reconnect=not args.no_reconnect,
    ):
        raw = upsert_candle(raw, event, max_rows=config.exchange.limit + 300)
        prepared = prepare_market_data(raw, config.strategy)
        signal = latest_signal_snapshot(prepared, config.strategy)
        latest_signal_path.write_text(json.dumps(signal, ensure_ascii=False, indent=2), encoding="utf-8")
        append_jsonl(live_signal_path, signal)
        append_event_jsonl(live_kline_path, event)
        print(json.dumps(signal, ensure_ascii=False))

def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the strategy on realtime closed candles")
    parser.add_argument("--config", default="configs/binance_btc.yaml", help="YAML config path")
    parser.add_argument("--warmup-limit", type=int, default=0, help="Override historical warmup candle count")
    parser.add_argument("--no-reconnect", action="store_true", help="Disable websocket reconnects")
    args = parser.parse_args()
    asyncio.run(runner(args))


if __name__ == "__main__":
    main()
