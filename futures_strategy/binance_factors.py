from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BINANCE_FAPI_BASE_URL = "https://fapi.binance.com"
# Binance docs state "latest 1 month data", but the live endpoint rejects
# ranges slightly shorter than 30 calendar days on some dates, so keep a
# conservative cap for stable automation.
MAX_OI_LOOKBACK_DAYS = 27

KLINE_INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


@dataclass(frozen=True)
class FactorWindow:
    start_time: pd.Timestamp
    end_time: pd.Timestamp


def ensure_utc_timestamp(value: str | datetime | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC")
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def factor_window(days_back: int = MAX_OI_LOOKBACK_DAYS, end_time: str | datetime | pd.Timestamp | None = None) -> FactorWindow:
    end_ts = ensure_utc_timestamp(end_time).floor("h")
    capped_days = max(1, min(int(days_back), MAX_OI_LOOKBACK_DAYS))
    start_ts = (end_ts - pd.Timedelta(days=capped_days)).floor("h")
    return FactorWindow(start_time=start_ts, end_time=end_ts)


def request_json(
    path: str,
    params: dict[str, Any],
    session: requests.Session | None = None,
    timeout: int = 30,
) -> Any:
    http = session or requests.Session()
    response = http.get(f"{BINANCE_FAPI_BASE_URL}{path}", params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_klines(
    symbol: str,
    interval: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    if interval not in KLINE_INTERVAL_MS:
        supported = ", ".join(sorted(KLINE_INTERVAL_MS))
        raise ValueError(f"Unsupported interval: {interval}. Supported: {supported}")

    step_ms = KLINE_INTERVAL_MS[interval]
    cursor = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    rows: list[list[Any]] = []

    while cursor < end_ms:
        payload = request_json(
            "/fapi/v1/klines",
            {
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            },
            session=session,
        )
        if not payload:
            break
        if rows:
            last_open_time = rows[-1][0]
            payload = [row for row in payload if row[0] > last_open_time]
        if not payload:
            break
        rows.extend(payload)
        cursor = int(payload[-1][0]) + step_ms
        if len(payload) < 1500:
            break

    frame = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
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
        ],
    )
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.set_index("timestamp").sort_index()
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["open", "high", "low", "close", "volume"]].dropna()


def fetch_funding_rate_history(
    symbol: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    cursor = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    rows: list[dict[str, Any]] = []

    while cursor <= end_ms:
        payload = request_json(
            "/fapi/v1/fundingRate",
            {
                "symbol": symbol.upper(),
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            session=session,
        )
        if not payload:
            break
        if rows:
            last_funding_time = rows[-1]["fundingTime"]
            payload = [row for row in payload if int(row["fundingTime"]) > last_funding_time]
        if not payload:
            break
        rows.extend(payload)
        cursor = int(payload[-1]["fundingTime"]) + 1
        if len(payload) < 1000:
            break

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["funding_rate", "mark_price"])
    frame["timestamp"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True)
    frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
    frame["mark_price"] = pd.to_numeric(frame["markPrice"], errors="coerce")
    frame = frame.set_index("timestamp").sort_index()
    return frame[["funding_rate", "mark_price"]].dropna()


def fetch_open_interest_history(
    symbol: str,
    period: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    if start_time < (end_time - pd.Timedelta(days=MAX_OI_LOOKBACK_DAYS)):
        start_time = (end_time - pd.Timedelta(days=MAX_OI_LOOKBACK_DAYS)).floor("h")

    period_ms = KLINE_INTERVAL_MS.get(period)
    if period_ms is None:
        supported = ", ".join(sorted(KLINE_INTERVAL_MS))
        raise ValueError(f"Unsupported OI period: {period}. Supported: {supported}")

    cursor = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    rows: list[dict[str, Any]] = []

    while cursor <= end_ms:
        payload = request_json(
            "/futures/data/openInterestHist",
            {
                "symbol": symbol.upper(),
                "period": period,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 500,
            },
            session=session,
        )
        if not payload:
            break
        if rows:
            last_timestamp = rows[-1]["timestamp"]
            payload = [row for row in payload if int(row["timestamp"]) > last_timestamp]
        if not payload:
            break
        rows.extend(payload)
        cursor = int(payload[-1]["timestamp"]) + period_ms
        if len(payload) < 500:
            break

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["oi_contracts", "oi_value"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame["oi_contracts"] = pd.to_numeric(frame["sumOpenInterest"], errors="coerce")
    frame["oi_value"] = pd.to_numeric(frame["sumOpenInterestValue"], errors="coerce")
    frame = frame.set_index("timestamp").sort_index()
    return frame[["oi_contracts", "oi_value"]].dropna()


def merge_factor_dataset(
    klines: pd.DataFrame,
    funding_rates: pd.DataFrame,
    open_interest: pd.DataFrame,
) -> pd.DataFrame:
    if klines.empty:
        raise ValueError("No klines available for factor dataset")

    merged = klines.reset_index().sort_values("timestamp")

    if not funding_rates.empty:
        funding_frame = funding_rates.reset_index().sort_values("timestamp")
        merged = pd.merge_asof(
            merged,
            funding_frame,
            on="timestamp",
            direction="backward",
        )
    else:
        merged["funding_rate"] = pd.NA
        merged["mark_price"] = pd.NA

    if not open_interest.empty:
        oi_frame = open_interest.reset_index().sort_values("timestamp")
        merged = pd.merge_asof(
            merged,
            oi_frame,
            on="timestamp",
            direction="backward",
        )
    else:
        merged["oi_contracts"] = pd.NA
        merged["oi_value"] = pd.NA

    merged = merged.sort_values("timestamp").set_index("timestamp")
    merged["funding_rate"] = pd.to_numeric(merged["funding_rate"], errors="coerce").ffill()
    merged["mark_price"] = pd.to_numeric(merged["mark_price"], errors="coerce").ffill()
    merged["oi_contracts"] = pd.to_numeric(merged["oi_contracts"], errors="coerce").ffill()
    merged["oi_value"] = pd.to_numeric(merged["oi_value"], errors="coerce").ffill()
    return merged.dropna(subset=["open", "high", "low", "close", "volume", "funding_rate", "oi_value"])


def build_factor_dataset(
    symbol: str,
    interval: str = "1h",
    days_back: int = MAX_OI_LOOKBACK_DAYS,
    end_time: str | datetime | pd.Timestamp | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    window = factor_window(days_back=days_back, end_time=end_time)
    klines = fetch_klines(symbol, interval, window.start_time, window.end_time, session=session)
    funding_rates = fetch_funding_rate_history(symbol, window.start_time, window.end_time, session=session)
    open_interest = fetch_open_interest_history(symbol, interval, window.start_time, window.end_time, session=session)
    return merge_factor_dataset(klines, funding_rates, open_interest)


def save_factor_dataset(dataset: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.reset_index().to_csv(output_path, index=False)
    return output_path
