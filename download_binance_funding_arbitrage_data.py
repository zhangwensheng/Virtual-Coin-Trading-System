"""Download cached Binance public data for funding-arbitrage research."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from futures_strategy.binance_funding_arbitrage_data import (
    BinanceFundingArbitrageDataClient,
    build_historical_liquidity_universe,
    merge_cached_frame,
    validate_symbol_history,
)


def _milliseconds(timestamp: pd.Timestamp) -> int:
    return int(timestamp.timestamp() * 1000)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def download_dataset(
    *,
    days: int,
    interval: str,
    output_dir: Path,
    client: BinanceFundingArbitrageDataClient | None = None,
    end: pd.Timestamp | None = None,
) -> dict[str, object]:
    data_client = client or BinanceFundingArbitrageDataClient()
    end_time = pd.Timestamp.now(tz="UTC").floor("5min") if end is None else pd.Timestamp(end)
    end_time = end_time.tz_localize("UTC") if end_time.tzinfo is None else end_time.tz_convert("UTC")
    start_time = end_time - pd.Timedelta(days=days)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = data_client.fetch_common_tradable_symbols()
    metadata_by_symbol = {item.symbol: item for item in metadata_rows}
    spot_hourly: dict[str, pd.DataFrame] = {}
    perp_hourly: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}

    for symbol in sorted(metadata_by_symbol):
        try:
            spot = data_client.fetch_spot_klines(
                symbol, "1h", _milliseconds(start_time), _milliseconds(end_time)
            )
            perp = data_client.fetch_perp_klines(
                symbol, "1h", _milliseconds(start_time), _milliseconds(end_time)
            )
            spot_hourly[symbol] = merge_cached_frame(output_dir / f"{symbol}_spot_1h.csv", spot)
            perp_hourly[symbol] = merge_cached_frame(output_dir / f"{symbol}_perp_1h.csv", perp)
        except Exception as exc:
            failures[symbol] = f"HOURLY_DOWNLOAD_FAILED: {type(exc).__name__}: {exc}"

    hourly_universe = build_historical_liquidity_universe(
        spot_hourly,
        perp_hourly,
        universe_size=20,
    )
    selected_symbols = sorted({symbol for symbols in hourly_universe.values() for symbol in symbols})
    quality: dict[str, object] = {}
    completed_symbols: list[str] = []
    for symbol in selected_symbols:
        try:
            spot = data_client.fetch_spot_klines(
                symbol, interval, _milliseconds(start_time), _milliseconds(end_time)
            )
            perp = data_client.fetch_perp_klines(
                symbol, interval, _milliseconds(start_time), _milliseconds(end_time)
            )
            funding = data_client.fetch_funding_history(
                symbol, _milliseconds(start_time), _milliseconds(end_time)
            )
            spot = merge_cached_frame(output_dir / f"{symbol}_spot_{interval}.csv", spot)
            perp = merge_cached_frame(output_dir / f"{symbol}_perp_{interval}.csv", perp)
            merge_cached_frame(output_dir / f"{symbol}_funding.csv", funding)
            spot_quality = validate_symbol_history(spot, expected_interval=interval)
            perp_quality = validate_symbol_history(perp, expected_interval=interval)
            quality[symbol] = {
                "spot": asdict(spot_quality),
                "perp": asdict(perp_quality),
            }
            if spot_quality.valid and perp_quality.valid:
                completed_symbols.append(symbol)
        except Exception as exc:
            failures[symbol] = f"DETAILED_DOWNLOAD_FAILED: {type(exc).__name__}: {exc}"

    payload: dict[str, object] = {
        "downloaded_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "days": days,
        "interval": interval,
        "common_symbol_count": len(metadata_by_symbol),
        "hourly_complete_symbol_count": len(spot_hourly.keys() & perp_hourly.keys()),
        "historical_top20_union": selected_symbols,
        "detailed_complete_symbols": completed_symbols,
        "symbol_metadata": {symbol: asdict(item) for symbol, item in metadata_by_symbol.items()},
        "quality": quality,
        "failures": failures,
        "current_listing_universe_bias": True,
        "survivorship_bias": len(spot_hourly.keys() & perp_hourly.keys()) < len(metadata_by_symbol),
        "endpoints": {
            "spot": "https://api.binance.com/api/v3",
            "perpetual": "https://fapi.binance.com/fapi/v1",
        },
    }
    _atomic_json(output_dir / "metadata.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="下载 Binance 资金费率套利回测数据")
    parser.add_argument("--days", type=int, default=95)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--output-dir", type=Path, default=Path("data/binance_funding_arbitrage"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata = download_dataset(days=args.days, interval=args.interval, output_dir=args.output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0 if metadata["detailed_complete_symbols"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
