from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from futures_strategy.config import ExchangeSettings
from futures_strategy.exchanges import get_exchange_spec


def build_exchange(settings: ExchangeSettings):
    name = settings.name.lower()
    if name == "demo":
        raise ValueError("demo source does not use a live exchange client")
    spec = get_exchange_spec(name)
    return spec.build_ccxt_exchange()


def load_market_data(settings: ExchangeSettings) -> pd.DataFrame:
    if settings.name.lower() == "demo":
        return generate_demo_ohlcv(settings.limit)
    if settings.csv_path:
        return load_csv(Path(settings.csv_path))
    return fetch_ohlcv(settings)


def fetch_ohlcv(settings: ExchangeSettings) -> pd.DataFrame:
    spec = get_exchange_spec(settings.name)
    exchange = build_exchange(settings)
    timeframe_ms = exchange.parse_timeframe(settings.timeframe) * 1000
    if settings.since:
        since_input = settings.since
        if "T" not in since_input:
            since_input = f"{since_input}T00:00:00Z"
        since_ms = exchange.parse8601(since_input)
    else:
        since_ms = exchange.milliseconds() - (settings.limit * timeframe_ms)

    candles: list[list[float]] = []
    next_since = since_ms

    try:
        while len(candles) < settings.limit:
            batch_limit = min(spec.rest_limit_cap, settings.limit - len(candles))
            batch = exchange.fetch_ohlcv(
                settings.symbol,
                timeframe=settings.timeframe,
                since=next_since,
                limit=batch_limit,
            )
            if not batch:
                break
            if candles:
                last_open = candles[-1][0]
                batch = [row for row in batch if row[0] > last_open]
            if not batch:
                break
            candles.extend(batch)
            next_since = candles[-1][0] + timeframe_ms
            if len(batch) < batch_limit:
                break
    except Exception as exc:
        raise RuntimeError(
            "交易所历史数据拉取失败。请检查网络/API 可达性，或在配置里设置 csv_path 走本地 K 线回测。"
        ) from exc

    if not candles:
        raise RuntimeError("没有拿到任何 K 线数据，请检查 symbol / timeframe / 网络配置。")
    return ohlcv_to_frame(candles)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    frame = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV 缺少字段: {sorted(missing)}")

    timestamp = frame["timestamp"]
    if is_numeric_dtype(timestamp):
        unit = "ms" if timestamp.iloc[0] > 10_000_000_000 else "s"
        parsed = pd.to_datetime(timestamp, unit=unit, utc=True)
    else:
        parsed = pd.to_datetime(timestamp, utc=True)

    normalized = frame.copy()
    normalized["timestamp"] = parsed
    normalized = normalized.set_index("timestamp").sort_index()
    core_columns = ["open", "high", "low", "close", "volume"]
    for column in core_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    extra_columns = [column for column in normalized.columns if column not in core_columns]
    for column in extra_columns:
        try:
            normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        except (TypeError, ValueError):
            continue
    normalized = normalized.dropna(subset=core_columns)
    ordered_columns = [*core_columns, *extra_columns]
    return normalized[ordered_columns]


def ohlcv_to_frame(candles: list[list[float]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.set_index("timestamp").sort_index()
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    return frame[["open", "high", "low", "close", "volume"]]


def save_ohlcv_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.reset_index().rename(columns={"index": "timestamp"})
    payload.to_csv(output_path, index=False)
    return output_path


def generate_demo_ohlcv(limit: int) -> pd.DataFrame:
    periods = max(limit, 600)
    index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    pivots = np.array([0.0, 2_000.0, 500.0, 3_200.0, 1_400.0])
    points = np.linspace(0, periods - 1, len(pivots))
    trend = np.interp(np.arange(periods), points, pivots)
    waves = (
        (600 * np.sin(np.arange(periods) / 9))
        - (240 * np.sin(np.arange(periods) / 3))
        + (160 * np.sin(np.arange(periods) / 17))
    )
    close = 50_000 + trend + waves
    open_ = np.roll(close, 1)
    open_[0] = close[0] - 30
    high = np.maximum(open_, close) + 90
    low = np.minimum(open_, close) - 90
    volume = 1_200 + (100 * np.sin(np.arange(periods) / 7))

    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame.iloc[-limit:]
