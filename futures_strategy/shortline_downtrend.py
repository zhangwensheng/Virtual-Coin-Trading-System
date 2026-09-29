from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from futures_strategy.indicators import atr, ema, rsi


@dataclass(frozen=True)
class ShortlineDowntrendSettings:
    min_entry_score: float = 74.0
    min_watch_score: float = 58.0
    exit_score: float = 45.0
    fast_ema: int = 9
    slow_ema: int = 21
    trend_ema: int = 55
    volume_window: int = 30
    trade_window: int = 30
    down_pressure_window: int = 15
    break_low_lookback: int = 8
    swing_lookback: int = 20
    min_volume_ratio: float = 1.25
    min_trade_ratio: float = 1.05
    min_sell_delta: float = 0.05
    min_down_volume_ratio: float = 0.56
    min_atr_pct: float = 0.0008
    max_atr_pct: float = 0.05
    max_vwap_extension_pct: float = 0.035
    min_rsi: float = 22.0
    max_rsi: float = 58.0
    min_drop_from_high: float = 0.006
    pump_fail_return: float = 0.025


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _session_vwap(frame: pd.DataFrame) -> pd.Series:
    session = frame.index.floor("D")
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    volume_sum = frame["volume"].groupby(session).cumsum().replace(0, np.nan)
    return ((typical * frame["volume"]).groupby(session).cumsum() / volume_sum).ffill()


def resample_to_5m(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    data = frame.sort_index()
    five = data.resample("5min").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
        trade_count=("trade_count", "sum"),
        taker_buy_quote=("taker_buy_quote", "sum"),
    )
    return five.dropna(subset=["open", "high", "low", "close"])


def prepare_shortline_downtrend_frame(
    frame: pd.DataFrame,
    settings: ShortlineDowntrendSettings | None = None,
) -> pd.DataFrame:
    settings = settings or ShortlineDowntrendSettings()
    if frame.empty:
        return pd.DataFrame()
    data = frame.copy().sort_index()
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")

    required = {"open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_quote"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"shortline frame missing columns: {sorted(missing)}")

    data["ema_fast"] = ema(data["close"], settings.fast_ema)
    data["ema_slow"] = ema(data["close"], settings.slow_ema)
    data["ema_trend"] = ema(data["close"], settings.trend_ema)
    data["rsi"] = rsi(data["close"], 14)
    data["atr"] = atr(data, 14)
    data["atr_pct"] = data["atr"] / data["close"]
    data["volume_ratio"] = _safe_ratio(data["volume"], data["volume"].rolling(settings.volume_window).mean()).fillna(0.0)
    data["trade_ratio"] = _safe_ratio(data["trade_count"], data["trade_count"].rolling(settings.trade_window).mean()).fillna(0.0)

    buy_pressure = _safe_ratio(data["taker_buy_quote"], data["quote_volume"]).fillna(0.5)
    data["delta_ratio"] = ((buy_pressure - 0.5) * 2.0).clip(-1.0, 1.0)
    down_volume = data["volume"].where(data["close"] < data["open"], 0.0)
    data["down_volume_ratio"] = _safe_ratio(
        down_volume.rolling(settings.down_pressure_window).sum(),
        data["volume"].rolling(settings.down_pressure_window).sum(),
    ).fillna(0.0)

    session = data.index.floor("D")
    day_open = data["open"].groupby(session).transform("first")
    day_high = data["high"].groupby(session).cummax()
    day_low = data["low"].groupby(session).cummin()
    day_range = (day_high - day_low).replace(0, np.nan)
    data["daily_return_pct"] = (data["close"] / day_open) - 1.0
    data["drop_from_day_high"] = (day_high / data["close"]) - 1.0
    data["day_close_location"] = ((data["close"] - day_low) / day_range).fillna(0.5)

    data["session_vwap"] = _session_vwap(data)
    data["vwap_gap"] = (data["close"] / data["session_vwap"]) - 1.0
    data["break_low"] = data["close"] < data["low"].rolling(settings.break_low_lookback).min().shift(1)
    data["swing_high"] = data["high"].rolling(settings.swing_lookback).max().shift(1)
    data["swing_low"] = data["low"].rolling(settings.swing_lookback).min().shift(1)

    five = resample_to_5m(data)
    if five.empty:
        for column in ["htf_close", "htf_ema_fast", "htf_ema_slow", "htf_volume_ratio", "htf_break_low"]:
            data[column] = np.nan
    else:
        five["htf_ema_fast"] = ema(five["close"], settings.fast_ema)
        five["htf_ema_slow"] = ema(five["close"], settings.slow_ema)
        five["htf_volume_ratio"] = _safe_ratio(five["volume"], five["volume"].rolling(20).mean()).fillna(0.0)
        five["htf_break_low"] = five["close"] < five["low"].rolling(4).min().shift(1)
        data["htf_close"] = five["close"].reindex(data.index, method="ffill")
        data["htf_ema_fast"] = five["htf_ema_fast"].reindex(data.index, method="ffill")
        data["htf_ema_slow"] = five["htf_ema_slow"].reindex(data.index, method="ffill")
        data["htf_volume_ratio"] = five["htf_volume_ratio"].reindex(data.index, method="ffill")
        data["htf_break_low"] = five["htf_break_low"].reindex(data.index, method="ffill").fillna(False)

    return data.replace([np.inf, -np.inf], np.nan).dropna(subset=["ema_fast", "ema_slow", "atr", "session_vwap"])


def build_market_context(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    bearish_count = 0
    details: dict[str, dict[str, Any]] = {}
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            continue
        row = frame.iloc[-1]
        close = _to_float(row.get("close"))
        bearish = bool(
            close < _to_float(row.get("ema_slow"))
            and _to_float(row.get("ema_fast")) < _to_float(row.get("ema_slow"))
            and _to_float(row.get("vwap_gap")) <= 0.001
        )
        if bearish:
            bearish_count += 1
        details[symbol] = {
            "bearish": bearish,
            "daily_return_pct": _to_float(row.get("daily_return_pct")),
            "vwap_gap": _to_float(row.get("vwap_gap")),
        }
    return {"bearish_count": bearish_count, "details": details}


def score_shortline_downtrend(
    frame: pd.DataFrame,
    *,
    symbol: str,
    market_context: dict[str, Any] | None = None,
    settings: ShortlineDowntrendSettings | None = None,
) -> dict[str, Any]:
    settings = settings or ShortlineDowntrendSettings()
    market_context = market_context or {}
    if frame.empty:
        return {"symbol": symbol, "score": 0.0, "short_signal": False, "failed_checks": ["empty_frame"]}

    row = frame.iloc[-1]
    score = 0.0
    components: dict[str, float] = {}

    price_score = 0.0
    if _to_float(row.get("close")) < _to_float(row.get("ema_fast")):
        price_score += 7.0
    if _to_float(row.get("ema_fast")) < _to_float(row.get("ema_slow")):
        price_score += 8.0
    if _to_float(row.get("ema_slow")) < _to_float(row.get("ema_trend")):
        price_score += 4.0
    if _to_float(row.get("close")) < _to_float(row.get("session_vwap")):
        price_score += 7.0
    if bool(row.get("break_low", False)):
        price_score += 8.0
    if _to_float(row.get("drop_from_day_high")) >= settings.min_drop_from_high:
        price_score += 4.0
    price_score = min(price_score, 38.0)
    components["price_structure"] = price_score
    score += price_score

    flow_score = 0.0
    if _to_float(row.get("volume_ratio")) >= settings.min_volume_ratio:
        flow_score += 8.0
    if _to_float(row.get("trade_ratio")) >= settings.min_trade_ratio:
        flow_score += 4.0
    if _to_float(row.get("delta_ratio")) <= -settings.min_sell_delta:
        flow_score += 8.0
    if _to_float(row.get("down_volume_ratio")) >= settings.min_down_volume_ratio:
        flow_score += 5.0
    flow_score = min(flow_score, 25.0)
    components["sell_pressure"] = flow_score
    score += flow_score

    htf_score = 0.0
    if _to_float(row.get("htf_close")) < _to_float(row.get("htf_ema_fast")):
        htf_score += 5.0
    if _to_float(row.get("htf_ema_fast")) < _to_float(row.get("htf_ema_slow")):
        htf_score += 5.0
    if bool(row.get("htf_break_low", False)):
        htf_score += 4.0
    if _to_float(row.get("day_close_location")) <= 0.45:
        htf_score += 3.0
    if (
        _to_float(row.get("daily_return_pct")) >= settings.pump_fail_return
        and _to_float(row.get("drop_from_day_high")) >= settings.min_drop_from_high * 1.5
    ):
        htf_score += 3.0
    htf_score = min(htf_score, 20.0)
    components["higher_timeframe"] = htf_score
    score += htf_score

    market_score = min(10.0, _to_float(market_context.get("bearish_count")) * 5.0)
    components["market_linkage"] = market_score
    score += market_score

    risk_score = 0.0
    atr_pct = _to_float(row.get("atr_pct"))
    if settings.min_atr_pct <= atr_pct <= settings.max_atr_pct:
        risk_score += 5.0
    if _to_float(row.get("vwap_gap")) >= -settings.max_vwap_extension_pct:
        risk_score += 3.0
    if settings.min_rsi <= _to_float(row.get("rsi")) <= settings.max_rsi:
        risk_score += 2.0
    components["risk_filter"] = risk_score
    score += risk_score

    structural_trigger = bool(row.get("break_low", False)) or (
        _to_float(row.get("drop_from_day_high")) >= settings.min_drop_from_high
        and _to_float(row.get("close")) < _to_float(row.get("ema_fast"))
        and _to_float(row.get("ema_fast")) < _to_float(row.get("ema_slow"))
    )
    checks = {
        "score_min": score >= settings.min_entry_score,
        "structural_trigger": structural_trigger,
        "close_below_vwap": _to_float(row.get("close")) < _to_float(row.get("session_vwap")),
        "sell_delta": _to_float(row.get("delta_ratio")) <= -settings.min_sell_delta,
        "volume_ok": _to_float(row.get("volume_ratio")) >= settings.min_volume_ratio,
        "down_pressure_ok": _to_float(row.get("down_volume_ratio")) >= settings.min_down_volume_ratio,
        "rsi_ok": settings.min_rsi <= _to_float(row.get("rsi")) <= settings.max_rsi,
        "atr_ok": settings.min_atr_pct <= atr_pct <= settings.max_atr_pct,
        "not_overextended": _to_float(row.get("vwap_gap")) >= -settings.max_vwap_extension_pct,
    }
    short_signal = all(checks.values())
    failed_checks = [key for key, passed in checks.items() if not passed]

    return {
        "symbol": symbol,
        "timestamp": frame.index[-1].isoformat(),
        "score": round(score, 2),
        "rank_score": round(score, 2),
        "short_signal": short_signal,
        "watch": score >= settings.min_watch_score,
        "failed_checks": failed_checks,
        "checks": checks,
        "components": components,
        "close": _to_float(row.get("close")),
        "rsi": round(_to_float(row.get("rsi")), 2),
        "daily_return_pct": _to_float(row.get("daily_return_pct")),
        "volume_ratio": round(_to_float(row.get("volume_ratio")), 3),
        "trade_ratio": round(_to_float(row.get("trade_ratio")), 3),
        "delta_ratio": round(_to_float(row.get("delta_ratio")), 3),
        "down_volume_ratio": round(_to_float(row.get("down_volume_ratio")), 3),
        "htf_volume_ratio": round(_to_float(row.get("htf_volume_ratio")), 3),
        "vwap_gap": _to_float(row.get("vwap_gap")),
        "vwap_gap_pct": round(_to_float(row.get("vwap_gap")) * 100.0, 3),
        "atr_pct": round(atr_pct * 100.0, 3),
        "drop_from_day_high": _to_float(row.get("drop_from_day_high")),
        "day_close_location": _to_float(row.get("day_close_location")),
        "market_bearish_count": _to_int_like(market_context.get("bearish_count", 0)),
    }


def _to_int_like(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
