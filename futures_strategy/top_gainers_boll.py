from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from download_binance_archive import download_month, month_range
from futures_strategy.config import AppConfig
from futures_strategy.factor_short_optimization import binance_native_to_unified_symbol
from futures_strategy.factor_universe import fetch_liquid_usdt_perpetual_rows
from futures_strategy.indicators import bollinger_bands


def month_start(month: str) -> pd.Timestamp:
    return pd.Timestamp(f"{month}-01", tz="UTC")


def next_month_start(month: str) -> pd.Timestamp:
    period = pd.Period(month, freq="M") + 1
    return pd.Timestamp(period.start_time).tz_localize("UTC")


def archive_cache_path(cache_dir: Path, symbol: str, timeframe: str, start_month: str, end_month: str) -> Path:
    suffix = f"{symbol.lower()}_{timeframe}_{start_month}_{end_month}.csv".replace(":", "_")
    return cache_dir / suffix


def download_archive_range(
    symbol: str,
    timeframe: str,
    start_month: str,
    end_month: str,
    *,
    cache_dir: Path,
    market: str = "um",
    session: requests.Session | None = None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = archive_cache_path(cache_dir, symbol, timeframe, start_month, end_month)
    if output_path.exists():
        return output_path

    http = session or requests.Session()
    frames: list[pd.DataFrame] = []
    for month in month_range(start_month, end_month):
        frames.append(download_month(symbol, timeframe, month, market, http))

    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    merged.to_csv(output_path, index=False)
    return output_path


def load_archive_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    for column in frame.columns:
        if column == "symbol":
            continue
        if column == "selected_day":
            continue
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError):
            continue
    return frame


def fetch_active_usdt_perpetual_symbols(session: requests.Session | None = None) -> list[str]:
    rows = fetch_liquid_usdt_perpetual_rows(session=session)
    return [row["symbol"] for row in rows]


def compute_daily_selection(
    daily_frames: dict[str, pd.DataFrame],
    *,
    start_month: str,
    end_month: str,
    daily_boll_window: int,
    daily_boll_std: float,
    top_n: int,
    min_daily_return_pct: float,
) -> pd.DataFrame:
    start_time = month_start(start_month)
    end_time = next_month_start(end_month)
    rows: list[pd.DataFrame] = []

    for symbol, frame in daily_frames.items():
        if frame.empty:
            continue
        daily = frame.copy().sort_index()
        middle, upper, lower = bollinger_bands(daily["close"], daily_boll_window, daily_boll_std)
        daily["daily_bb_mid"] = middle
        daily["daily_bb_upper"] = upper
        daily["daily_bb_lower"] = lower
        daily["daily_return_pct"] = (daily["close"] / daily["open"]) - 1
        daily["trade_date"] = daily.index.normalize()
        daily["symbol"] = symbol
        filtered = daily[(daily.index >= start_time) & (daily.index < end_time)].copy()
        filtered = filtered.dropna(subset=["daily_bb_upper"])
        if filtered.empty:
            continue
        rows.append(
            filtered[
                [
                    "symbol",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "daily_return_pct",
                    "daily_bb_mid",
                    "daily_bb_upper",
                    "daily_bb_lower",
                ]
            ].rename(
                columns={
                    "open": "daily_open",
                    "high": "daily_high",
                    "low": "daily_low",
                    "close": "daily_close",
                    "volume": "daily_volume",
                }
            )
        )

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    combined["daily_rank"] = combined.groupby("trade_date")["daily_return_pct"].rank(method="first", ascending=False)
    combined["selected_day"] = (
        (combined["daily_rank"] <= float(top_n))
        & (combined["daily_close"] > combined["daily_bb_upper"])
        & (combined["daily_return_pct"] >= float(min_daily_return_pct))
    )
    return combined.sort_values(["trade_date", "daily_rank", "symbol"]).reset_index(drop=True)


def selected_symbols(selection_frame: pd.DataFrame) -> list[str]:
    if selection_frame.empty:
        return []
    return sorted(selection_frame.loc[selection_frame["selected_day"], "symbol"].unique().tolist())


def _daily_boll_lookup(
    daily_frame: pd.DataFrame,
    *,
    daily_boll_window: int,
    daily_boll_std: float,
) -> pd.DataFrame:
    daily = daily_frame.copy().sort_index()
    middle, upper, lower = bollinger_bands(daily["close"], daily_boll_window, daily_boll_std)
    lookup = pd.DataFrame(
        {
            "trade_date": daily.index.normalize(),
            "daily_bb_mid": middle.shift(1),
            "daily_bb_upper": upper.shift(1),
            "daily_bb_lower": lower.shift(1),
            "prev_daily_close": daily["close"].shift(1),
        }
    )
    return lookup.dropna(subset=["daily_bb_upper"])


def compute_realtime_intraday_selection(
    intraday_frames: dict[str, pd.DataFrame],
    daily_frames: dict[str, pd.DataFrame],
    *,
    start_month: str,
    end_month: str,
    daily_boll_window: int,
    daily_boll_std: float,
    top_n: int,
    min_daily_return_pct: float,
) -> dict[str, pd.DataFrame]:
    start_time = month_start(start_month)
    end_time = next_month_start(end_month)
    enriched_by_symbol: dict[str, pd.DataFrame] = {}
    rank_rows: list[pd.DataFrame] = []

    for symbol, intraday_frame in intraday_frames.items():
        daily_frame = daily_frames.get(symbol)
        if daily_frame is None or daily_frame.empty or intraday_frame.empty:
            continue

        intraday = intraday_frame[(intraday_frame.index >= start_time) & (intraday_frame.index < end_time)].copy()
        if intraday.empty:
            continue

        intraday = intraday.reset_index().rename(columns={"index": "timestamp"})
        intraday["trade_date"] = intraday["timestamp"].dt.normalize()
        session = intraday["trade_date"]
        intraday["daily_open"] = intraday.groupby(session)["open"].transform("first")
        intraday["daily_high"] = intraday.groupby(session)["high"].cummax()
        intraday["daily_low"] = intraday.groupby(session)["low"].cummin()
        intraday["daily_close"] = intraday["close"]
        intraday["daily_volume"] = intraday.groupby(session)["volume"].cumsum()
        intraday["daily_return_pct"] = (intraday["close"] / intraday["daily_open"]) - 1

        lookup = _daily_boll_lookup(
            daily_frame,
            daily_boll_window=daily_boll_window,
            daily_boll_std=daily_boll_std,
        )
        intraday = intraday.merge(lookup, on="trade_date", how="left")
        intraday["daily_bb_mid"] = pd.to_numeric(intraday["daily_bb_mid"], errors="coerce")
        intraday["daily_bb_upper"] = pd.to_numeric(intraday["daily_bb_upper"], errors="coerce")
        intraday["daily_bb_lower"] = pd.to_numeric(intraday["daily_bb_lower"], errors="coerce")
        intraday["daily_upper_break"] = intraday["close"] > intraday["daily_bb_upper"]
        intraday["symbol"] = symbol

        rank_rows.append(
            intraday[
                [
                    "timestamp",
                    "symbol",
                    "daily_return_pct",
                    "daily_upper_break",
                ]
            ].copy()
        )
        enriched_by_symbol[symbol] = intraday

    if not rank_rows:
        return {}

    rank_frame = pd.concat(rank_rows, ignore_index=True)
    rank_frame["daily_rank"] = rank_frame.groupby("timestamp")["daily_return_pct"].rank(method="first", ascending=False)
    rank_frame["selected_day"] = (
        (rank_frame["daily_rank"] <= float(top_n))
        & (rank_frame["daily_return_pct"] >= float(min_daily_return_pct))
        & rank_frame["daily_upper_break"].fillna(False)
    )

    selected_by_symbol: dict[str, pd.DataFrame] = {
        symbol: group[["timestamp", "daily_rank", "selected_day"]].copy()
        for symbol, group in rank_frame.groupby("symbol", sort=False)
    }

    output: dict[str, pd.DataFrame] = {}
    for symbol, intraday in enriched_by_symbol.items():
        selection = selected_by_symbol.get(symbol)
        if selection is None:
            continue
        merged = intraday.merge(selection, on="timestamp", how="left")
        merged["daily_rank"] = merged["daily_rank"].fillna(999.0)
        merged["selected_day"] = merged["selected_day"].fillna(False).astype(bool)
        merged = merged.set_index("timestamp").sort_index()
        output[symbol] = merged.drop(columns=["trade_date", "daily_upper_break"], errors="ignore")
    return output


def enrich_intraday_with_daily_selection(
    intraday_frame: pd.DataFrame,
    *,
    symbol: str,
    selection_frame: pd.DataFrame,
    start_month: str,
    end_month: str,
) -> pd.DataFrame:
    start_time = month_start(start_month)
    end_time = next_month_start(end_month)
    intraday = intraday_frame[(intraday_frame.index >= start_time) & (intraday_frame.index < end_time)].copy()
    if intraday.empty:
        return intraday

    symbol_daily = selection_frame[selection_frame["symbol"] == symbol].copy()
    if symbol_daily.empty:
        intraday["selected_day"] = False
        intraday["daily_rank"] = 999.0
        intraday["daily_return_pct"] = 0.0
        intraday["daily_close"] = intraday["close"]
        intraday["daily_bb_upper"] = intraday["close"]
        intraday["daily_bb_mid"] = intraday["close"]
        intraday["daily_bb_lower"] = intraday["close"]
        intraday["daily_open"] = intraday["open"]
        intraday["daily_high"] = intraday["high"]
        intraday["daily_low"] = intraday["low"]
        intraday["daily_volume"] = intraday["volume"]
        intraday["symbol"] = symbol
        return intraday

    intraday = intraday.reset_index().rename(columns={"index": "timestamp"})
    intraday["trade_date"] = intraday["timestamp"].dt.normalize()
    merged = intraday.merge(
        symbol_daily,
        on="trade_date",
        how="left",
        suffixes=("", "_daily"),
    )
    merged["selected_day"] = merged["selected_day"].apply(lambda value: bool(value) if pd.notna(value) else False)
    merged["daily_rank"] = merged["daily_rank"].fillna(999.0)
    merged["daily_return_pct"] = merged["daily_return_pct"].fillna(0.0)
    for column in [
        "daily_close",
        "daily_bb_upper",
        "daily_bb_mid",
        "daily_bb_lower",
        "daily_open",
        "daily_high",
        "daily_low",
        "daily_volume",
    ]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["symbol"] = symbol
    merged = merged.set_index("timestamp").sort_index()
    return merged.drop(columns=["trade_date"])


def write_symbol_configs(
    base_config: AppConfig,
    csv_paths: dict[str, Path],
    *,
    output_dir: Path,
    output_name_suffix: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for symbol, csv_path in csv_paths.items():
        config = deepcopy(base_config)
        config.exchange.symbol = binance_native_to_unified_symbol(symbol)
        config.exchange.csv_path = str(csv_path)
        config.output.output_dir = f"outputs/{symbol.lower()}_{output_name_suffix}"
        payload = {
            "exchange": asdict(config.exchange),
            "strategy": asdict(config.strategy),
            "risk": asdict(config.risk),
            "output": asdict(config.output),
        }
        config_path = output_dir / f"{symbol.lower()}_{output_name_suffix}.yaml"
        config_path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        created.append(config_path)
    return created


def build_top_gainers_boll_datasets(
    *,
    base_config: AppConfig,
    start_month: str,
    end_month: str,
    symbols: list[str] | None = None,
    cache_dir: Path,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    http = session or requests.Session()
    universe = [symbol.upper() for symbol in (symbols or fetch_active_usdt_perpetual_symbols(session=http))]
    if not universe:
        raise RuntimeError("No Binance perpetual symbols available for top gainers scan.")

    daily_cache_dir = cache_dir / "daily"
    intraday_cache_dir = cache_dir / "intraday"
    enriched_cache_dir = cache_dir / "enriched"
    daily_cache_dir.mkdir(parents=True, exist_ok=True)
    intraday_cache_dir.mkdir(parents=True, exist_ok=True)
    enriched_cache_dir.mkdir(parents=True, exist_ok=True)

    warmup_start_month = str(pd.Period(start_month, freq="M") - 1)
    daily_frames: dict[str, pd.DataFrame] = {}
    skipped_daily: list[dict[str, str]] = []
    for symbol in universe:
        try:
            daily_path = download_archive_range(
                symbol,
                "1d",
                warmup_start_month,
                end_month,
                cache_dir=daily_cache_dir,
                session=http,
            )
            daily_frame = load_archive_csv(daily_path)
        except Exception as exc:
            skipped_daily.append({"symbol": symbol, "reason": str(exc)})
            continue
        if daily_frame.empty:
            skipped_daily.append({"symbol": symbol, "reason": "empty_daily"})
            continue
        daily_frames[symbol] = daily_frame

    selection_frame = compute_daily_selection(
        daily_frames,
        start_month=start_month,
        end_month=end_month,
        daily_boll_window=base_config.strategy.top_gainers_daily_boll_window,
        daily_boll_std=base_config.strategy.top_gainers_daily_boll_std,
        top_n=base_config.strategy.top_gainers_daily_top_n,
        min_daily_return_pct=base_config.strategy.top_gainers_daily_min_return_pct,
    )
    chosen_symbols = selected_symbols(selection_frame)

    enriched_paths: dict[str, Path] = {}
    skipped_intraday: list[dict[str, str]] = []
    expected_start = month_start(start_month)
    expected_end = next_month_start(end_month)
    for symbol in chosen_symbols:
        try:
            intraday_path = download_archive_range(
                symbol,
                base_config.exchange.timeframe,
                start_month,
                end_month,
                cache_dir=intraday_cache_dir,
                session=http,
            )
            intraday_frame = load_archive_csv(intraday_path)
        except Exception as exc:
            skipped_intraday.append({"symbol": symbol, "reason": str(exc)})
            continue
        if intraday_frame.empty:
            skipped_intraday.append({"symbol": symbol, "reason": "empty_intraday"})
            continue
        if intraday_frame.index.min() > expected_start or intraday_frame.index.max() < (expected_end - pd.Timedelta(minutes=1)):
            skipped_intraday.append({"symbol": symbol, "reason": "incomplete_intraday_range"})
            continue
        enriched = enrich_intraday_with_daily_selection(
            intraday_frame,
            symbol=symbol,
            selection_frame=selection_frame,
            start_month=start_month,
            end_month=end_month,
        )
        if enriched.empty:
            skipped_intraday.append({"symbol": symbol, "reason": "empty_enriched"})
            continue
        enriched_path = archive_cache_path(
            enriched_cache_dir,
            symbol,
            base_config.exchange.timeframe,
            start_month,
            end_month,
        )
        enriched.reset_index().to_csv(enriched_path, index=False)
        enriched_paths[symbol] = enriched_path

    return {
        "universe": universe,
        "selection_frame": selection_frame,
        "selected_symbols": chosen_symbols,
        "enriched_paths": enriched_paths,
        "skipped_daily": skipped_daily,
        "skipped_intraday": skipped_intraday,
        "cache_dir": cache_dir,
    }


def build_realtime_top_gainers_boll_datasets(
    *,
    base_config: AppConfig,
    start_month: str,
    end_month: str,
    symbols: list[str] | None = None,
    cache_dir: Path,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    http = session or requests.Session()
    universe = [symbol.upper() for symbol in (symbols or fetch_active_usdt_perpetual_symbols(session=http))]
    if not universe:
        raise RuntimeError("No Binance perpetual symbols available for realtime top gainers scan.")

    daily_cache_dir = cache_dir / "daily"
    intraday_cache_dir = cache_dir / "intraday"
    enriched_cache_dir = cache_dir / "realtime_enriched"
    daily_cache_dir.mkdir(parents=True, exist_ok=True)
    intraday_cache_dir.mkdir(parents=True, exist_ok=True)
    enriched_cache_dir.mkdir(parents=True, exist_ok=True)

    warmup_start_month = str(pd.Period(start_month, freq="M") - 1)
    expected_start = month_start(start_month)
    expected_end = next_month_start(end_month)

    daily_frames: dict[str, pd.DataFrame] = {}
    intraday_frames: dict[str, pd.DataFrame] = {}
    skipped_daily: list[dict[str, str]] = []
    skipped_intraday: list[dict[str, str]] = []

    for symbol in universe:
        try:
            daily_path = download_archive_range(
                symbol,
                "1d",
                warmup_start_month,
                end_month,
                cache_dir=daily_cache_dir,
                session=http,
            )
            daily_frame = load_archive_csv(daily_path)
        except Exception as exc:
            skipped_daily.append({"symbol": symbol, "reason": str(exc)})
            continue
        if daily_frame.empty:
            skipped_daily.append({"symbol": symbol, "reason": "empty_daily"})
            continue
        daily_frames[symbol] = daily_frame

        try:
            intraday_path = download_archive_range(
                symbol,
                base_config.exchange.timeframe,
                start_month,
                end_month,
                cache_dir=intraday_cache_dir,
                session=http,
            )
            intraday_frame = load_archive_csv(intraday_path)
        except Exception as exc:
            skipped_intraday.append({"symbol": symbol, "reason": str(exc)})
            continue
        if intraday_frame.empty:
            skipped_intraday.append({"symbol": symbol, "reason": "empty_intraday"})
            continue
        if intraday_frame.index.min() > expected_start or intraday_frame.index.max() < (expected_end - pd.Timedelta(minutes=1)):
            skipped_intraday.append({"symbol": symbol, "reason": "incomplete_intraday_range"})
            continue
        intraday_frames[symbol] = intraday_frame

    enriched_frames = compute_realtime_intraday_selection(
        intraday_frames,
        daily_frames,
        start_month=start_month,
        end_month=end_month,
        daily_boll_window=base_config.strategy.top_gainers_daily_boll_window,
        daily_boll_std=base_config.strategy.top_gainers_daily_boll_std,
        top_n=base_config.strategy.top_gainers_daily_top_n,
        min_daily_return_pct=base_config.strategy.top_gainers_daily_min_return_pct,
    )

    enriched_paths: dict[str, Path] = {}
    selected_symbols: list[str] = []
    selection_rows: list[pd.DataFrame] = []
    for symbol, frame in enriched_frames.items():
        if frame.empty:
            continue
        if bool(frame["selected_day"].any()):
            selected_symbols.append(symbol)
        selection_rows.append(
            frame.reset_index()[
                [
                    "timestamp",
                    "symbol",
                    "selected_day",
                    "daily_rank",
                    "daily_return_pct",
                    "daily_close",
                    "daily_bb_upper",
                ]
            ]
        )
        if not bool(frame["selected_day"].any()):
            continue
        enriched_path = archive_cache_path(
            enriched_cache_dir,
            symbol,
            base_config.exchange.timeframe,
            start_month,
            end_month,
        )
        frame.reset_index().to_csv(enriched_path, index=False)
        enriched_paths[symbol] = enriched_path

    selection_frame = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()

    return {
        "universe": universe,
        "selection_frame": selection_frame,
        "selected_symbols": sorted(selected_symbols),
        "enriched_paths": enriched_paths,
        "skipped_daily": skipped_daily,
        "skipped_intraday": skipped_intraday,
        "cache_dir": cache_dir,
    }
