from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests


ARCHIVE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def month_range(start_month: str, end_month: str) -> list[str]:
    months = pd.period_range(start=start_month, end=end_month, freq="M")
    return [str(month) for month in months]


def build_url(symbol: str, timeframe: str, month: str, market: str) -> str:
    year, month_value = month.split("-")
    return (
        f"https://data.binance.vision/data/futures/{market}/monthly/klines/"
        f"{symbol}/{timeframe}/{symbol}-{timeframe}-{year}-{month_value}.zip"
    )


def download_month(symbol: str, timeframe: str, month: str, market: str, session: requests.Session) -> pd.DataFrame:
    url = build_url(symbol, timeframe, month, market)
    response = session.get(url, timeout=30)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    name = archive.namelist()[0]
    frame = pd.read_csv(archive.open(name), header=None)
    if len(frame.columns) >= len(ARCHIVE_COLUMNS):
        frame = frame.iloc[:, : len(ARCHIVE_COLUMNS)].copy()
        frame.columns = ARCHIVE_COLUMNS
    if not frame.empty and str(frame.iloc[0, 0]).lower() in {"open_time", "timestamp"}:
        frame = frame.iloc[1:].reset_index(drop=True)
    return normalize(frame)


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if "open_time" not in frame.columns and "timestamp" not in frame.columns:
        if len(frame.columns) >= len(ARCHIVE_COLUMNS):
            renamed_input = frame.iloc[:, : len(ARCHIVE_COLUMNS)].copy()
            renamed_input.columns = ARCHIVE_COLUMNS
            frame = renamed_input

    renamed = frame.rename(columns={"open_time": "timestamp"}).copy()
    renamed["timestamp"] = pd.to_datetime(
        pd.to_numeric(renamed["timestamp"], errors="coerce"),
        unit="ms",
        utc=True,
    )
    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    normalized = renamed[columns].copy()
    for column in columns[1:]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna().drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance futures archive klines")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance native symbol, such as BTCUSDT")
    parser.add_argument("--timeframe", default="1h", help="1m / 5m / 15m / 1h / 4h / 1d ...")
    parser.add_argument("--start-month", required=True, help="YYYY-MM")
    parser.add_argument("--end-month", required=True, help="YYYY-MM")
    parser.add_argument("--market", default="um", choices=["um", "cm"], help="um for USD-M, cm for coin-M")
    parser.add_argument("--output", required=True, help="CSV output path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    frames: list[pd.DataFrame] = []
    downloaded_months: list[str] = []

    for month in month_range(args.start_month, args.end_month):
        frame = download_month(args.symbol, args.timeframe, month, args.market, session)
        frames.append(frame)
        downloaded_months.append(month)
        print(f"downloaded: {month} rows={len(frame)}")

    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    merged.to_csv(output_path, index=False)

    print(f"symbol: {args.symbol}")
    print(f"timeframe: {args.timeframe}")
    print(f"months: {downloaded_months[0]} -> {downloaded_months[-1]}")
    print(f"rows: {len(merged)}")
    print(f"start: {merged['timestamp'].iloc[0]}")
    print(f"end: {merged['timestamp'].iloc[-1]}")
    print(f"output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
