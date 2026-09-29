from __future__ import annotations

import argparse
import asyncio
import json

from futures_strategy.realtime import append_event_jsonl, stream_klines


async def runner(args) -> None:
    count = 0
    async for event in stream_klines(
        args.exchange,
        args.symbol,
        args.timeframe,
        closed_only=args.closed_only,
        reconnect=not args.no_reconnect,
    ):
        payload = event.as_dict()
        print(json.dumps(payload, ensure_ascii=False))
        if args.output:
            append_event_jsonl(args.output, event)
        count += 1
        if args.max_events and count >= args.max_events:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Subscribe to realtime futures klines from supported exchanges")
    parser.add_argument("--exchange", required=True, help="binance / okx / bybit / bitget / gate / deribit / hyperliquid")
    parser.add_argument("--symbol", required=True, help="Trading symbol")
    parser.add_argument("--timeframe", default="1h", help="Kline timeframe")
    parser.add_argument("--output", default=None, help="Optional JSONL output path")
    parser.add_argument("--max-events", type=int, default=0, help="Stop after N events, 0 means keep running")
    parser.add_argument("--closed-only", action="store_true", help="Only emit finalized candles")
    parser.add_argument("--no-reconnect", action="store_true", help="Disable reconnect after socket errors")
    args = parser.parse_args()
    asyncio.run(runner(args))


if __name__ == "__main__":
    main()
