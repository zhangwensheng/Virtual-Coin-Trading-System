"""Deterministic signal calculations for the WR top-gainers short strategy.

The functions in this module deliberately have no network, account, or storage
dependencies.  Callers are responsible for market-data freshness and order
execution checks that are outside the signal definition.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd


_COLUMNS = ("open", "high", "low", "close", "volume")
DAILY_WR_PERIOD = 28
DAILY_WR_MIN = -30.0
MINUTE_WR_PERIOD = 28
PUMP_WR_MIN = -10.0
REVERSAL_WR_DROP_MIN = 20.0
BREAKDOWN_WR_DROP_MIN = 30.0
BREAKDOWN_WR_DROP_MAX = 50.0
MIN_VOLUME_RATIO = 3.0
MAX_BREAKDOWN_BODY_PCT = 0.7
MAX_BREAKDOWN_LOWER_WICK_FRACTION = 0.35
UNIVERSE_LIMIT = 7
LATE_UTC_START_HOUR = 14
BLOCKED_ENTRY_UTC_HOURS = frozenset({6, 7})
_STABLE_BASES = {
    "BUSD",
    "DAI",
    "FDUSD",
    "FRAX",
    "PYUSD",
    "TUSD",
    "USDC",
    "USDD",
    "USDP",
    "USDS",
    "USDT",
}
_NON_CRYPTO_BASES = {
    "AAPL",
    "AMZN",
    "DAX",
    "DJI",
    "DOW",
    "META",
    "MSFT",
    "NASDAQ",
    "NDX",
    "NFLX",
    "NVDA",
    "SP500",
    "SPX",
    "TSLA",
    "UK100",
    "US100",
    "US30",
    "US500",
    "XAG",
    "XAU",
}
_NON_CRYPTO_TYPES = {
    "COMMODITY",
    "EQUITY",
    "FOREX",
    "INDEX",
    "METAL",
    "STOCK",
}


def _as_utc(value: Any) -> pd.Timestamp | None:
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
        seconds = _finite_float(value)
        if seconds is None:
            return None
        try:
            return pd.Timestamp(seconds, unit="s", tz="UTC")
        except (TypeError, ValueError, OverflowError):
            return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        return None
    return timestamp.tz_convert("UTC")


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _gate_result(
    *,
    eligible: bool,
    reason: str,
    daily_open: float | None = None,
    daily_upper: float | None = None,
    daily_wr: float | None = None,
) -> dict[str, Any]:
    return {
        "eligible": eligible,
        "stage": "WAIT_PUMP" if eligible else "INELIGIBLE",
        "reason": reason,
        "daily_open": daily_open,
        "daily_upper": daily_upper,
        "daily_wr": daily_wr,
        "daily_wr_period": DAILY_WR_PERIOD,
    }


def _valid_utc_index(frame: Any) -> bool:
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        return False
    if frame.index.tz is None or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        return False
    try:
        offsets = {timestamp.utcoffset() for timestamp in frame.index}
    except (AttributeError, ValueError):
        return False
    return offsets == {pd.Timedelta(0).to_pytimedelta()} or len(frame.index) == 0


def daily_gate(daily: pd.DataFrame, now: Any) -> dict[str, Any]:
    """Evaluate the UTC daily Williams %R eligibility gate.

    The lookback range is based on the previous 28 completed daily bars.  The
    current still-forming daily close is allowed because it is the live price
    available at evaluation time; future daily rows are ignored.
    """

    timestamp = _as_utc(now)
    if timestamp is None or not _valid_utc_index(daily) or any(column not in daily for column in _COLUMNS):
        return _gate_result(eligible=False, reason="invalid_daily_data")

    utc_index = daily.index.tz_convert("UTC")
    day = timestamp.normalize()
    today_positions = np.flatnonzero(utc_index == day)
    if len(today_positions) != 1:
        return _gate_result(eligible=False, reason="missing_current_daily_open")

    historical_positions = np.flatnonzero(utc_index < day)
    if len(historical_positions) < DAILY_WR_PERIOD:
        return _gate_result(eligible=False, reason="insufficient_daily_history")
    historical_positions = historical_positions[-DAILY_WR_PERIOD:]
    history_index = utc_index[historical_positions]
    if len(history_index) > 1 and not np.all(np.diff(history_index.asi8) == pd.Timedelta(days=1).value):
        return _gate_result(eligible=False, reason="daily_data_not_continuous")
    if history_index[-1] != day - pd.Timedelta(days=1):
        return _gate_result(eligible=False, reason="daily_data_not_continuous")

    current = daily.iloc[int(today_positions[0])]
    current_open = _finite_float(current["open"])
    current_price = _finite_float(current["close"])
    history = daily.iloc[historical_positions][["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    values = history.to_numpy(dtype=float)
    if (
        current_open is None
        or current_price is None
        or current_open <= 0.0
        or current_price <= 0.0
        or values.shape != (DAILY_WR_PERIOD, 4)
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
        or (history["high"] < history[["open", "close", "low"]].max(axis=1)).any()
        or (history["low"] > history[["open", "close", "high"]].min(axis=1)).any()
    ):
        return _gate_result(eligible=False, reason="invalid_daily_data")

    highest = float(history["high"].max())
    lowest = float(history["low"].min())
    width = highest - lowest
    if not np.isfinite(width) or width <= 0.0:
        return _gate_result(eligible=False, reason="invalid_daily_data")

    daily_wr = float(-100.0 * (highest - current_price) / width)
    if not np.isfinite(daily_wr):
        return _gate_result(eligible=False, reason="invalid_daily_data")
    eligible = daily_wr >= DAILY_WR_MIN
    return _gate_result(
        eligible=eligible,
        reason="eligible" if eligible else "daily_wr_not_overbought",
        daily_open=current_open,
        daily_upper=None,
        daily_wr=daily_wr,
    )


def _looks_like_crypto_contract(symbol_info: dict[str, Any], now: pd.Timestamp) -> bool:
    if symbol_info.get("contractType") != "PERPETUAL" or symbol_info.get("status") != "TRADING":
        return False
    if symbol_info.get("quoteAsset") != "USDT":
        return False

    symbol = symbol_info.get("symbol")
    base = symbol_info.get("baseAsset")
    if not isinstance(symbol, str) or not symbol or not isinstance(base, str) or not base:
        return False
    if not all(character.isalnum() for character in base):
        return False

    normalized_base = base.upper()
    if normalized_base in _STABLE_BASES or normalized_base in _NON_CRYPTO_BASES:
        return False
    classifications = {
        str(symbol_info.get(field, "")).upper()
        for field in ("assetType", "underlyingType", "contractCategory")
    }
    if classifications & _NON_CRYPTO_TYPES:
        return False

    onboard_ms = _finite_float(symbol_info.get("onboardDate"))
    if onboard_ms is None:
        return False
    onboard = pd.Timestamp(onboard_ms, unit="ms", tz="UTC")
    return onboard <= now - pd.Timedelta(days=90)


def rank_universe(info: Any, tickers: Any, now: Any) -> list[dict[str, Any]]:
    """Return the liquid positive 24-hour gainers, ranked before daily gating."""

    timestamp = _as_utc(now)
    if timestamp is None or not isinstance(info, dict) or not isinstance(tickers, list):
        return []
    symbols = info.get("symbols")
    if not isinstance(symbols, list):
        return []

    allowed: set[str] = set()
    for row in symbols:
        if isinstance(row, dict) and _looks_like_crypto_contract(row, timestamp):
            allowed.add(str(row["symbol"]))

    candidates: dict[str, tuple[float, float]] = {}
    for ticker in tickers:
        if not isinstance(ticker, dict):
            continue
        symbol = ticker.get("symbol")
        if symbol not in allowed or symbol in candidates:
            continue
        change = _finite_float(ticker.get("priceChangePercent"))
        quote_volume = _finite_float(ticker.get("quoteVolume"))
        if change is None or quote_volume is None or change <= 0.0 or quote_volume < 20_000_000.0:
            continue
        candidates[str(symbol)] = (change, quote_volume)

    ordered = sorted(candidates.items(), key=lambda item: (-item[1][0], item[0]))[:UNIVERSE_LIMIT]
    return [
        {
            "symbol": symbol,
            "rank": rank,
            "price_change_percent": values[0],
            "quote_volume": values[1],
        }
        for rank, (symbol, values) in enumerate(ordered, start=1)
    ]


def _evaluation_result(
    symbol: str,
    stage: str,
    reason: str,
    daily_open: float | None,
    daily_upper: float | None,
    daily_wr: float | None = None,
    volume_ratio: float | None = None,
    minute_wr: float | None = None,
    wr_drop: float | None = None,
    signal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "symbol": str(symbol),
        "stage": stage,
        "reason": reason,
        "daily_open": daily_open,
        "daily_upper": daily_upper,
        "daily_wr": daily_wr,
        "minute_wr": minute_wr,
        "wr_drop": wr_drop,
        "volume_ratio": volume_ratio,
        "signal": signal,
    }


def _valid_closed_minutes(minutes: Any, now: pd.Timestamp) -> tuple[pd.DataFrame | None, str | None]:
    if not _valid_utc_index(minutes) or any(column not in minutes for column in _COLUMNS):
        return None, "invalid_minute_data"

    utc_index = minutes.index.tz_convert("UTC")
    closed_mask = utc_index + pd.Timedelta(minutes=1) <= now
    closed = minutes.loc[closed_mask, list(_COLUMNS)].copy()
    closed.index = utc_index[closed_mask]
    if len(closed) < MINUTE_WR_PERIOD:
        return None, "insufficient_minute_history"
    if len(closed.index) > 1 and not np.all(np.diff(closed.index.asi8) == pd.Timedelta(minutes=1).value):
        return None, "minute_data_not_continuous"
    expected_latest = now.floor("min") - pd.Timedelta(minutes=1)
    if closed.index[-1] != expected_latest:
        return None, "stale_minute_data"

    numeric = closed.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None, "invalid_minute_data"
    if (
        (numeric[["open", "high", "low", "close"]] <= 0.0).any().any()
        or
        (numeric["volume"] < 0.0).any()
        or (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any()
        or (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any()
    ):
        return None, "invalid_minute_data"
    return numeric, None


def _is_pump(row: pd.Series) -> bool:
    return bool(
        np.isfinite(row["wr"])
        and np.isfinite(row["volume_ratio"])
        and row["open"] < row["close"]
        and row["wr"] >= PUMP_WR_MIN
        and row["volume_ratio"] >= MIN_VOLUME_RATIO
    )


def _williams_r(frame: pd.DataFrame, period: int) -> pd.Series:
    highest = frame["high"].rolling(period, min_periods=period).max()
    lowest = frame["low"].rolling(period, min_periods=period).min()
    width = highest - lowest
    return -100.0 * (highest - frame["close"]) / width.replace(0.0, np.nan)


def _make_signal(
    symbol: str,
    frame: pd.DataFrame,
    true_range: pd.Series,
    pump_position: int,
    reversal_position: int,
    breakdown_position: int,
    pump_wr: float,
) -> dict[str, Any] | None:
    previous_ranges = true_range.iloc[breakdown_position - 14 : breakdown_position]
    if len(previous_ranges) != 14 or not np.isfinite(previous_ranges.to_numpy(dtype=float)).all():
        return None

    pump_time = frame.index[pump_position]
    reversal_time = frame.index[reversal_position]
    breakdown_time = frame.index[breakdown_position]
    signal_time = breakdown_time + pd.Timedelta(minutes=1)
    breakdown_wr = float(frame["wr"].iloc[breakdown_position])
    wr_drop = float(pump_wr - breakdown_wr)
    signal_key = "|".join(
        [
            "wr-short",
            str(symbol),
            str(int(pump_time.timestamp())),
            str(int(reversal_time.timestamp())),
            str(int(breakdown_time.timestamp())),
            f"{pump_wr:.6f}",
            f"{breakdown_wr:.6f}",
        ]
    )
    return {
        "signal_id": sha256(signal_key.encode("utf-8")).hexdigest(),
        "symbol": str(symbol),
        "signal_time": int(signal_time.timestamp()),
        "pump_time": int(pump_time.timestamp()),
        "reversal_time": int(reversal_time.timestamp()),
        "peak": float(frame["high"].iloc[pump_position : breakdown_position + 1].max()),
        "atr": float(previous_ranges.mean()),
        "pump_wr": pump_wr,
        "breakdown_wr": breakdown_wr,
        "wr_drop": wr_drop,
    }


def evaluate_signal(
    symbol: str,
    daily: pd.DataFrame,
    minutes: pd.DataFrame,
    now: Any,
    rank: int = 1,
) -> dict[str, Any]:
    """Evaluate one symbol's current deterministic P/R/B observation state."""

    timestamp = _as_utc(now)
    gate = daily_gate(daily, now)
    daily_open = gate["daily_open"]
    daily_upper = gate["daily_upper"]
    daily_wr = gate["daily_wr"]
    if timestamp is None:
        return _evaluation_result(str(symbol), "INELIGIBLE", "invalid_now", daily_open, daily_upper, daily_wr)
    if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= UNIVERSE_LIMIT:
        return _evaluation_result(str(symbol), "INELIGIBLE", "rank_outside_top_7", daily_open, daily_upper, daily_wr)
    if not gate["eligible"]:
        return _evaluation_result(str(symbol), "INELIGIBLE", gate["reason"], daily_open, daily_upper, daily_wr)

    frame, error = _valid_closed_minutes(minutes, timestamp)
    if error is not None or frame is None:
        return _evaluation_result(str(symbol), "INELIGIBLE", str(error), daily_open, daily_upper, daily_wr)

    frame["wr"] = _williams_r(frame, MINUTE_WR_PERIOD)
    previous_volume_mean = frame["volume"].shift(1).rolling(20, min_periods=20).mean()
    frame["volume_ratio"] = np.where(
        previous_volume_mean > 0.0,
        frame["volume"] / previous_volume_mean,
        np.nan,
    )
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)

    current_day = timestamp.normalize()
    state = "WAIT_PUMP"
    pump_position: int | None = None
    reversal_position: int | None = None
    pump_high: float | None = None
    reversal_low: float | None = None
    peak_through_reversal: float | None = None
    pump_ratio: float | None = None
    pump_wr: float | None = None
    candidate: tuple[int, dict[str, Any], float, float, float] | None = None
    late_blocked = False
    entry_hour_blocked = False

    def begin_pump(position: int) -> None:
        nonlocal state, pump_position, reversal_position, pump_high
        nonlocal reversal_low, peak_through_reversal, pump_ratio, pump_wr
        state = "WAIT_REVERSAL"
        pump_position = position
        reversal_position = None
        pump_high = float(frame["high"].iloc[position])
        reversal_low = None
        peak_through_reversal = None
        pump_ratio = float(frame["volume_ratio"].iloc[position])
        pump_wr = float(frame["wr"].iloc[position])

    def clear_observation() -> None:
        nonlocal state, pump_position, reversal_position, pump_high
        nonlocal reversal_low, peak_through_reversal, pump_ratio, pump_wr
        state = "WAIT_PUMP"
        pump_position = None
        reversal_position = None
        pump_high = None
        reversal_low = None
        peak_through_reversal = None
        pump_ratio = None
        pump_wr = None

    for position, (_, row) in enumerate(frame.iterrows()):
        row_day = frame.index[position].normalize()
        if row_day == current_day and frame.index[position].hour in BLOCKED_ENTRY_UTC_HOURS:
            entry_hour_blocked = True
            clear_observation()
            continue
        if row_day == current_day and frame.index[position].hour >= LATE_UTC_START_HOUR:
            late_blocked = True
            clear_observation()
            continue
        pump_now = row_day == current_day and _is_pump(row)

        if state == "WAIT_PUMP":
            if pump_now:
                begin_pump(position)
            continue

        if pump_position is None or row_day != frame.index[pump_position].normalize():
            clear_observation()
            if pump_now:
                begin_pump(position)
            continue

        if state == "WAIT_REVERSAL":
            distance = position - pump_position
            if distance > 5 or (pump_high is not None and row["high"] > pump_high):
                clear_observation()
                if pump_now:
                    begin_pump(position)
                continue
            wr_drop = float(pump_wr - row["wr"]) if pump_wr is not None and np.isfinite(row["wr"]) else np.nan
            if (
                distance >= 1
                and np.isfinite(wr_drop)
                and row["close"] < row["open"]
                and wr_drop >= REVERSAL_WR_DROP_MIN
            ):
                state = "WAIT_BREAKDOWN"
                reversal_position = position
                reversal_low = float(row["low"])
                peak_through_reversal = float(frame["high"].iloc[pump_position : position + 1].max())
            continue

        if reversal_position is None or reversal_low is None or peak_through_reversal is None:
            clear_observation()
            continue
        distance = position - reversal_position
        if distance > 3 or row["high"] > peak_through_reversal:
            clear_observation()
            if pump_now:
                begin_pump(position)
            continue
        breakdown_body_pct = float((row["open"] - row["close"]) / row["open"] * 100.0)
        breakdown_range = float(row["high"] - row["low"])
        breakdown_lower_wick_fraction = (
            float((min(row["open"], row["close"]) - row["low"]) / breakdown_range)
            if breakdown_range > 0.0
            else 0.0
        )
        current_wr_drop = float(pump_wr - row["wr"]) if pump_wr is not None and np.isfinite(row["wr"]) else np.nan
        if distance >= 1 and row["close"] < reversal_low and row["close"] < row["open"]:
            if (
                breakdown_body_pct <= MAX_BREAKDOWN_BODY_PCT
                and breakdown_lower_wick_fraction <= MAX_BREAKDOWN_LOWER_WICK_FRACTION
                and np.isfinite(current_wr_drop)
                and BREAKDOWN_WR_DROP_MIN <= current_wr_drop <= BREAKDOWN_WR_DROP_MAX
            ):
                signal = _make_signal(
                    str(symbol), frame, true_range, pump_position, reversal_position, position, float(pump_wr)
                )
                signal_time = (
                    pd.Timestamp(signal["signal_time"], unit="s", tz="UTC") if signal is not None else None
                )
                if (
                    signal is not None
                    and pump_ratio is not None
                    and signal_time is not None
                    and signal_time.hour not in BLOCKED_ENTRY_UTC_HOURS
                ):
                    candidate = (position, signal, pump_ratio, float(row["wr"]), float(current_wr_drop))
            clear_observation()

    if candidate is not None:
        breakdown_position, signal, signal_volume_ratio, signal_minute_wr, signal_wr_drop = candidate
        signal_closed_at = pd.Timestamp(signal["signal_time"], unit="s", tz="UTC")
        active_started_after_signal = (
            pump_position is not None and pump_position > breakdown_position
        )
        age = (timestamp - signal_closed_at).total_seconds()
        if 0.0 <= age <= 15.0:
            return _evaluation_result(
                str(symbol),
                "SIGNAL",
                "signal_ready",
                daily_open,
                daily_upper,
                daily_wr,
                signal_volume_ratio,
                signal_minute_wr,
                signal_wr_drop,
                signal,
            )
        if not active_started_after_signal:
            return _evaluation_result(
                str(symbol), "WAIT_PUMP", "signal_expired", daily_open, daily_upper, daily_wr
            )

    if state == "WAIT_REVERSAL":
        return _evaluation_result(
            str(symbol), "WAIT_REVERSAL", "waiting_for_wr_reversal", daily_open, daily_upper, daily_wr, pump_ratio,
            None if pump_wr is None else float(frame["wr"].iloc[-1]),
            None if pump_wr is None or not np.isfinite(frame["wr"].iloc[-1]) else float(pump_wr - frame["wr"].iloc[-1]),
        )
    if state == "WAIT_BREAKDOWN":
        return _evaluation_result(
            str(symbol), "WAIT_BREAKDOWN", "waiting_for_breakdown", daily_open, daily_upper, daily_wr, pump_ratio,
            None if not np.isfinite(frame["wr"].iloc[-1]) else float(frame["wr"].iloc[-1]),
            None if pump_wr is None or not np.isfinite(frame["wr"].iloc[-1]) else float(pump_wr - frame["wr"].iloc[-1]),
        )
    return _evaluation_result(
        str(symbol),
        "WAIT_PUMP",
        (
            "entry_hour_blocked"
            if entry_hour_blocked and timestamp.hour in BLOCKED_ENTRY_UTC_HOURS
            else "late_session_blocked"
            if late_blocked and timestamp.hour >= LATE_UTC_START_HOUR
            else "waiting_for_pump"
        ),
        daily_open,
        daily_upper,
        daily_wr,
        minute_wr=None if not np.isfinite(frame["wr"].iloc[-1]) else float(frame["wr"].iloc[-1]),
    )
