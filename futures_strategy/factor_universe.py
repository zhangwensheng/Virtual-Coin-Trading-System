from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from futures_strategy.binance_factors import build_factor_dataset, save_factor_dataset
from futures_strategy.config import AppConfig
from futures_strategy.factor_short_optimization import (
    apply_tunable_params,
    binance_native_to_unified_symbol,
    to_builtin,
)


BINANCE_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_24H_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"
DEFAULT_STRICT_UNIVERSE_SIZE = 30
DEFAULT_MIN_LISTING_DAYS = 45
DEFAULT_MIN_QUOTE_VOLUME = 25_000_000.0

PREFERRED_STRICT_SHORT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "BNBUSDT",
    "ADAUSDT",
    "SUIUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "ENAUSDT",
    "DOTUSDT",
    "FILUSDT",
    "BCHUSDT",
    "APTUSDT",
    "NEARUSDT",
    "LTCUSDT",
    "TRXUSDT",
    "FETUSDT",
    "AAVEUSDT",
    "TAOUSDT",
    "1000PEPEUSDT",
    "WLDUSDT",
    "UNIUSDT",
    "INJUSDT",
    "ATOMUSDT",
    "HBARUSDT",
    "ARBUSDT",
    "OPUSDT",
    "SEIUSDT",
    "JTOUSDT",
    "XLMUSDT",
    "ETCUSDT",
    "1000SHIBUSDT",
    "CRVUSDT",
    "SANDUSDT",
    "GALAUSDT",
    "WIFUSDT",
    "1000BONKUSDT",
    "ORDIUSDT",
]

EXCLUDED_SYMBOLS = {
    "PAXGUSDT",
    "USDCUSDT",
    "FDUSDUSDT",
    "TUSDUSDT",
    "BUSDUSDT",
    "USDPUSDT",
    "BTCDOMUSDT",
    "DEFIUSDT",
    "BLUEBIRDUSDT",
}

STRICT_SHARED_PARAMS = {
    "enhanced_min_funding_rate": -0.00005,
    "enhanced_min_funding_change": -0.00001,
    "enhanced_min_oi_change_pct": 0.0,
    "enhanced_min_price_bounce_pct": 0.0,
    "enhanced_entry_min_score": 2,
    "enhanced_risk_step": 0.5,
    "enhanced_max_risk_multiplier": 2.0,
}


def _fetch_json(url: str, session: requests.Session | None = None, timeout: int = 30) -> Any:
    http = session or requests.Session()
    response = http.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_liquid_usdt_perpetual_rows(
    *,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    exchange_info = _fetch_json(BINANCE_EXCHANGE_INFO_URL, session=session)
    tickers = _fetch_json(BINANCE_24H_TICKER_URL, session=session)

    symbol_meta = {
        item["symbol"]: item
        for item in exchange_info.get("symbols", [])
        if item.get("contractType") == "PERPETUAL"
        and item.get("quoteAsset") == "USDT"
        and item.get("status") == "TRADING"
        and item.get("symbol") not in EXCLUDED_SYMBOLS
    }

    now_utc = pd.Timestamp.now(tz="UTC")
    rows: list[dict[str, Any]] = []
    for item in tickers:
        symbol = str(item.get("symbol", "")).upper()
        meta = symbol_meta.get(symbol)
        if meta is None:
            continue
        onboard_ms = meta.get("onboardDate")
        onboard_time = pd.to_datetime(onboard_ms, unit="ms", utc=True) if onboard_ms else pd.NaT
        listing_days = None
        if pd.notna(onboard_time):
            listing_days = max(0, int((now_utc - onboard_time).total_seconds() // 86400))
        rows.append(
            {
                "symbol": symbol,
                "quote_volume": float(item.get("quoteVolume", 0.0) or 0.0),
                "base_volume": float(item.get("volume", 0.0) or 0.0),
                "last_price": float(item.get("lastPrice", 0.0) or 0.0),
                "price_change_pct_24h": float(item.get("priceChangePercent", 0.0) or 0.0),
                "listing_days": listing_days,
                "onboard_date": "" if pd.isna(onboard_time) else onboard_time.strftime("%Y-%m-%d"),
            }
        )

    rows.sort(key=lambda item: item["quote_volume"], reverse=True)
    return rows


def select_strict_short_universe(
    *,
    size: int = DEFAULT_STRICT_UNIVERSE_SIZE,
    min_listing_days: int = DEFAULT_MIN_LISTING_DAYS,
    min_quote_volume: float = DEFAULT_MIN_QUOTE_VOLUME,
    preferred_symbols: list[str] | None = None,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    preferred = [symbol.upper() for symbol in (preferred_symbols or PREFERRED_STRICT_SHORT_SYMBOLS)]
    preferred_rank = {symbol: index for index, symbol in enumerate(preferred)}
    available_rows = fetch_liquid_usdt_perpetual_rows(session=session)

    eligible = [
        row
        for row in available_rows
        if row["quote_volume"] >= float(min_quote_volume)
        and (row["listing_days"] is None or row["listing_days"] >= int(min_listing_days))
    ]

    preferred_rows = [row for row in eligible if row["symbol"] in preferred_rank]
    preferred_rows.sort(
        key=lambda row: (
            preferred_rank[row["symbol"]],
            -row["quote_volume"],
        )
    )
    selected_symbols = {row["symbol"] for row in preferred_rows[:size]}
    selected_rows = preferred_rows[:size]

    if len(selected_rows) < size:
        fallback_rows = [row for row in eligible if row["symbol"] not in selected_symbols]
        fallback_rows.sort(key=lambda row: row["quote_volume"], reverse=True)
        selected_rows.extend(fallback_rows[: max(0, size - len(selected_rows))])

    return selected_rows[:size]


def refresh_factor_datasets_for_symbols(
    symbols: list[str],
    *,
    interval: str,
    days_back: int,
    end_time: str | None,
    cache_dir: Path,
    session: requests.Session | None = None,
) -> dict[str, Path]:
    http = session or requests.Session()
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: dict[str, Path] = {}
    for symbol in symbols:
        dataset = build_factor_dataset(
            symbol=symbol,
            interval=interval,
            days_back=days_back,
            end_time=end_time,
            session=http,
        )
        end_label = dataset.index[-1].strftime("%Y-%m-%d")
        output_path = cache_dir / f"{symbol.lower()}_{interval}_{end_label}_factors.csv"
        csv_paths[symbol] = save_factor_dataset(dataset, output_path)
    return csv_paths


def build_shared_symbol_configs(
    base_config: AppConfig,
    *,
    symbols: list[str],
    output_dir: Path,
    shared_params: dict[str, Any] | None = None,
    csv_paths: dict[str, Path] | None = None,
    output_name_suffix: str = "funding_oi_strict",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    params = shared_params or {}

    for symbol in symbols:
        config = deepcopy(base_config)
        config.exchange.symbol = binance_native_to_unified_symbol(symbol)
        if csv_paths and symbol in csv_paths:
            config.exchange.csv_path = str(csv_paths[symbol])
        apply_tunable_params(config, params)
        symbol_slug = symbol.lower()
        config.output.output_dir = f"outputs/{symbol_slug}_{output_name_suffix}"
        payload = to_builtin(
            {
                "exchange": asdict(config.exchange),
                "strategy": asdict(config.strategy),
                "risk": asdict(config.risk),
                "output": asdict(config.output),
            }
        )
        config_path = output_dir / f"{symbol_slug}_strict.yaml"
        config_path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        created.append(config_path)

    return created
