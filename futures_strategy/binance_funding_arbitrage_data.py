"""Binance public-market data adapter for funding arbitrage."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Mapping

import pandas as pd
import requests


SPOT_BASE_URL = "https://api.binance.com"
PERP_BASE_URL = "https://fapi.binance.com"
STABLE_BASE_ASSETS = frozenset({"USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "PYUSD"})
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


@dataclass(frozen=True)
class CommonSymbolMetadata:
    symbol: str
    base_asset: str
    quote_asset: str
    spot_tick_size: float
    spot_step_size: float
    spot_min_quantity: float
    spot_min_notional: float
    perp_tick_size: float
    perp_step_size: float
    perp_min_quantity: float
    perp_min_notional: float


@dataclass(frozen=True)
class HistoryValidation:
    valid: bool
    reason: str | None
    rows: int
    duplicate_rows: int
    max_gap_minutes: float


def _filter_value(symbol: Mapping[str, object], filter_type: str, *names: str) -> float:
    for item in symbol.get("filters", []):
        if isinstance(item, Mapping) and item.get("filterType") == filter_type:
            for name in names:
                value = item.get(name)
                if value not in (None, ""):
                    return float(value)
    return 0.0


def _tradable_rows(payload: object, *, perpetual: bool) -> dict[str, Mapping[str, object]]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("symbols"), list):
        raise ValueError("INVALID_EXCHANGE_INFO_RESPONSE")
    rows: dict[str, Mapping[str, object]] = {}
    for raw in payload["symbols"]:
        if not isinstance(raw, Mapping):
            continue
        symbol = str(raw.get("symbol", "")).upper()
        base = str(raw.get("baseAsset", "")).upper()
        quote = str(raw.get("quoteAsset", "")).upper()
        if raw.get("status") != "TRADING" or quote != "USDT":
            continue
        if perpetual and raw.get("contractType") != "PERPETUAL":
            continue
        if base in STABLE_BASE_ASSETS or any(base.endswith(suffix) for suffix in LEVERAGED_SUFFIXES):
            continue
        rows[symbol] = raw
    return rows


class BinanceFundingArbitrageDataClient:
    def __init__(self, *, session: object | None = None, timeout: int = 30) -> None:
        self.session = session or requests.Session()
        self.timeout = int(timeout)

    def _get(self, base_url: str, path: str, params: dict[str, object]) -> object:
        response = self.session.get(f"{base_url}{path}", params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def fetch_common_tradable_symbols(self) -> list[CommonSymbolMetadata]:
        spot = _tradable_rows(self._get(SPOT_BASE_URL, "/api/v3/exchangeInfo", {}), perpetual=False)
        perp = _tradable_rows(self._get(PERP_BASE_URL, "/fapi/v1/exchangeInfo", {}), perpetual=True)
        common: list[CommonSymbolMetadata] = []
        for symbol in sorted(spot.keys() & perp.keys()):
            spot_row = spot[symbol]
            perp_row = perp[symbol]
            common.append(
                CommonSymbolMetadata(
                    symbol=symbol,
                    base_asset=str(spot_row["baseAsset"]),
                    quote_asset="USDT",
                    spot_tick_size=_filter_value(spot_row, "PRICE_FILTER", "tickSize"),
                    spot_step_size=_filter_value(spot_row, "LOT_SIZE", "stepSize"),
                    spot_min_quantity=_filter_value(spot_row, "LOT_SIZE", "minQty"),
                    spot_min_notional=_filter_value(spot_row, "MIN_NOTIONAL", "minNotional", "notional")
                    or _filter_value(spot_row, "NOTIONAL", "minNotional", "notional"),
                    perp_tick_size=_filter_value(perp_row, "PRICE_FILTER", "tickSize"),
                    perp_step_size=_filter_value(perp_row, "LOT_SIZE", "stepSize"),
                    perp_min_quantity=_filter_value(perp_row, "LOT_SIZE", "minQty"),
                    perp_min_notional=_filter_value(perp_row, "MIN_NOTIONAL", "notional", "minNotional")
                    or _filter_value(perp_row, "NOTIONAL", "notional", "minNotional"),
                )
            )
        return common

    def fetch_24h_tickers(self, market: str) -> dict[str, dict[str, float]]:
        if market == "spot":
            payload = self._get(SPOT_BASE_URL, "/api/v3/ticker/24hr", {})
        elif market == "perp":
            payload = self._get(PERP_BASE_URL, "/fapi/v1/ticker/24hr", {})
        else:
            raise ValueError(f"UNSUPPORTED_MARKET: {market}")
        if not isinstance(payload, list):
            raise ValueError(f"INVALID_24H_TICKER_RESPONSE: {market}")
        result: dict[str, dict[str, float]] = {}
        for row in payload:
            if not isinstance(row, Mapping):
                continue
            symbol = str(row.get("symbol", "")).upper()
            try:
                last_price = float(row["lastPrice"])
                quote_volume = float(row["quoteVolume"])
            except (KeyError, TypeError, ValueError):
                continue
            if symbol and isfinite(last_price) and isfinite(quote_volume):
                result[symbol] = {"last_price": last_price, "quote_volume": quote_volume}
        return result

    def _fetch_klines(
        self,
        *,
        base_url: str,
        path: str,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
    ) -> pd.DataFrame:
        rows: list[list[object]] = []
        cursor = int(start_time)
        while cursor < int(end_time):
            payload = self._get(
                base_url,
                path,
                {
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": int(end_time),
                    "limit": 1000,
                },
            )
            if not isinstance(payload, list):
                raise ValueError(f"INVALID_KLINE_RESPONSE: {symbol} {path}")
            page = [item for item in payload if isinstance(item, list) and len(item) >= 9]
            if not page:
                break
            rows.extend(page)
            next_cursor = int(page[-1][0]) + 1
            if next_cursor <= cursor:
                raise ValueError(f"NON_ADVANCING_KLINE_PAGE: {symbol} {path}")
            cursor = next_cursor
            if len(page) < 1000:
                break
        if not rows:
            empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "quote_volume", "trades"])
            empty.index = pd.DatetimeIndex([], tz="UTC", name="open_time")
            return empty
        frame = pd.DataFrame(
            {
                "open_time": [int(row[0]) for row in rows],
                "open": [float(row[1]) for row in rows],
                "high": [float(row[2]) for row in rows],
                "low": [float(row[3]) for row in rows],
                "close": [float(row[4]) for row in rows],
                "volume": [float(row[5]) for row in rows],
                "quote_volume": [float(row[7]) for row in rows],
                "trades": [int(row[8]) for row in rows],
            }
        )
        frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        return frame.drop_duplicates("open_time", keep="last").sort_values("open_time").set_index("open_time")

    def fetch_spot_klines(
        self, symbol: str, interval: str, start_time: int, end_time: int
    ) -> pd.DataFrame:
        return self._fetch_klines(
            base_url=SPOT_BASE_URL,
            path="/api/v3/klines",
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
        )

    def fetch_perp_klines(
        self, symbol: str, interval: str, start_time: int, end_time: int
    ) -> pd.DataFrame:
        return self._fetch_klines(
            base_url=PERP_BASE_URL,
            path="/fapi/v1/klines",
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
        )

    def fetch_funding_history(self, symbol: str, start_time: int, end_time: int) -> pd.DataFrame:
        rows: list[Mapping[str, object]] = []
        cursor = int(start_time)
        while cursor < int(end_time):
            payload = self._get(
                PERP_BASE_URL,
                "/fapi/v1/fundingRate",
                {
                    "symbol": symbol.upper(),
                    "startTime": cursor,
                    "endTime": int(end_time),
                    "limit": 1000,
                },
            )
            if not isinstance(payload, list):
                raise ValueError(f"INVALID_FUNDING_RESPONSE: {symbol}")
            page = [item for item in payload if isinstance(item, Mapping)]
            if not page:
                break
            rows.extend(page)
            next_cursor = int(page[-1]["fundingTime"]) + 1
            if next_cursor <= cursor:
                raise ValueError(f"NON_ADVANCING_FUNDING_PAGE: {symbol}")
            cursor = next_cursor
            if len(page) < 1000:
                break
        if not rows:
            empty = pd.DataFrame(columns=["funding_rate", "mark_price"])
            empty.index = pd.DatetimeIndex([], tz="UTC", name="funding_time")
            return empty
        frame = pd.DataFrame(
            {
                "funding_time": [int(row["fundingTime"]) for row in rows],
                "funding_rate": [float(row["fundingRate"]) for row in rows],
                "mark_price": [float(row.get("markPrice", "nan")) for row in rows],
            }
        )
        frame["funding_time"] = pd.to_datetime(frame["funding_time"], unit="ms", utc=True)
        return frame.drop_duplicates("funding_time", keep="last").sort_values("funding_time").set_index("funding_time")


def build_historical_liquidity_universe(
    spot_frames: Mapping[str, pd.DataFrame],
    perp_frames: Mapping[str, pd.DataFrame],
    *,
    universe_size: int = 20,
) -> dict[pd.Timestamp, list[str]]:
    scores: dict[str, pd.Series] = {}
    for symbol in sorted(spot_frames.keys() & perp_frames.keys()):
        spot_volume = spot_frames[symbol]["quote_volume"].astype(float).sort_index()
        perp_volume = perp_frames[symbol]["quote_volume"].astype(float).sort_index()
        common_index = spot_volume.index.intersection(perp_volume.index)
        spot_prior = spot_volume.reindex(common_index).rolling(24, min_periods=24).sum().shift(1)
        perp_prior = perp_volume.reindex(common_index).rolling(24, min_periods=24).sum().shift(1)
        scores[symbol] = pd.concat([spot_prior, perp_prior], axis=1).min(axis=1, skipna=False)
    if not scores:
        return {}
    score_frame = pd.DataFrame(scores).sort_index()
    result: dict[pd.Timestamp, list[str]] = {}
    for timestamp, row in score_frame.iterrows():
        visible = [(symbol, float(value)) for symbol, value in row.items() if pd.notna(value)]
        if not visible:
            continue
        ranked = sorted(visible, key=lambda item: (-item[1], item[0]))[:universe_size]
        result[pd.Timestamp(timestamp)] = [symbol for symbol, _ in ranked]
    return result


def validate_symbol_history(frame: pd.DataFrame, *, expected_interval: str = "5min") -> HistoryValidation:
    if frame.empty:
        return HistoryValidation(False, "EMPTY_HISTORY", 0, 0, 0.0)
    duplicate_rows = int(frame.index.duplicated().sum())
    if duplicate_rows:
        return HistoryValidation(False, "DUPLICATE_TIMESTAMPS", len(frame), duplicate_rows, 0.0)
    ordered = frame.sort_index()
    numeric_columns = [column for column in ("open", "high", "low", "close", "volume", "quote_volume") if column in ordered]
    values = ordered[numeric_columns].to_numpy(dtype=float)
    if not all(isfinite(float(value)) and float(value) >= 0.0 for value in values.ravel()):
        return HistoryValidation(False, "INVALID_NUMERIC_VALUE", len(frame), 0, 0.0)
    differences = ordered.index.to_series().diff().dropna().dt.total_seconds() / 60.0
    max_gap = float(differences.max()) if not differences.empty else 0.0
    expected_minutes = pd.Timedelta(expected_interval).total_seconds() / 60.0
    if max_gap > max(15.0, expected_minutes * 3.0):
        return HistoryValidation(False, "UNRECOVERABLE_KLINE_GAP", len(frame), 0, max_gap)
    return HistoryValidation(True, None, len(frame), 0, max_gap)


def merge_cached_frame(path: Path | str, incoming: pd.DataFrame) -> pd.DataFrame:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    if destination.exists() and destination.stat().st_size > 0:
        cached = pd.read_csv(destination, index_col=0, parse_dates=[0])
        cached.index = pd.to_datetime(cached.index, utc=True)
        cached.index.name = incoming.index.name or cached.index.name
        frames.append(cached)
    frames.append(incoming.copy())
    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    merged.to_csv(temporary)
    temporary.replace(destination)
    return merged


__all__ = [
    "BinanceFundingArbitrageDataClient",
    "CommonSymbolMetadata",
    "HistoryValidation",
    "build_historical_liquidity_universe",
    "merge_cached_frame",
    "validate_symbol_history",
]
