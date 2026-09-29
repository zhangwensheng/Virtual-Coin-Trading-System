from __future__ import annotations

import argparse
from pathlib import Path

import requests

from futures_strategy.binance_factors import build_factor_dataset, save_factor_dataset


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Binance futures factor datasets with klines, funding rate and open interest")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT", help="Comma separated Binance native symbols")
    parser.add_argument("--interval", default="1h", help="Kline/OI interval, such as 1h")
    parser.add_argument("--days-back", default=30, type=int, help="Lookback days, capped by Binance OI history limits")
    parser.add_argument("--end-time", default=None, help="Optional UTC end time, such as 2026-03-18T00:00:00Z")
    parser.add_argument("--output-dir", default="data/binance_factor_bundle", help="Folder for enriched CSV datasets")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    for symbol in parse_csv_list(args.symbols):
        dataset = build_factor_dataset(
            symbol=symbol,
            interval=args.interval,
            days_back=args.days_back,
            end_time=args.end_time,
            session=session,
        )
        end_label = dataset.index[-1].strftime("%Y-%m-%d")
        output_path = output_dir / f"{symbol.lower()}_{args.interval}_{end_label}_factors.csv"
        saved = save_factor_dataset(dataset, output_path)
        print(f"symbol: {symbol}")
        print(f"rows: {len(dataset)}")
        print(f"start: {dataset.index[0].isoformat()}")
        print(f"end: {dataset.index[-1].isoformat()}")
        print(f"output: {saved.resolve()}")
        print("")


if __name__ == "__main__":
    main()
