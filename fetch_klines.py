from __future__ import annotations

import argparse

from futures_strategy.config import ExchangeSettings
from futures_strategy.data import load_market_data, save_ohlcv_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical futures klines from supported exchanges")
    parser.add_argument("--exchange", required=True, help="demo / binance / okx / bybit / bitget / gate / kucoinfutures / deribit / hyperliquid")
    parser.add_argument("--symbol", required=True, help="CCXT unified symbol, such as BTC/USDT:USDT")
    parser.add_argument("--timeframe", default="1h", help="1m / 5m / 15m / 1h / 4h / 1d ...")
    parser.add_argument("--limit", type=int, default=1500, help="Number of candles to fetch")
    parser.add_argument("--since", default=None, help="Optional start time, for example 2025-01-01 or 2025-01-01T00:00:00Z")
    parser.add_argument("--output", required=True, help="CSV output path")
    args = parser.parse_args()

    settings = ExchangeSettings(
        name=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
        since=args.since,
    )
    frame = load_market_data(settings)
    output_path = save_ohlcv_csv(frame, args.output)

    print(f"exchange: {args.exchange}")
    print(f"symbol: {args.symbol}")
    print(f"timeframe: {args.timeframe}")
    print(f"rows: {len(frame)}")
    print(f"start: {frame.index[0].isoformat()}")
    print(f"end: {frame.index[-1].isoformat()}")
    print(f"output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
