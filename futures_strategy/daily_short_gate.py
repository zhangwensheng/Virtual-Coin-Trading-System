from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DailyShortGateSettings:
    lookback_months: int = 6
    top_symbols: int = 8
    min_symbol_trades: int = 2
    min_symbol_profit_factor: float = 1.05
    min_volume_ratio: float = 1.05
    min_sell_volume_ratio: float = 0.58
    min_down_hour_ratio: float = 0.50
    min_late_sell_ratio: float = 0.10
    max_close_location: float = 0.35
    max_day_return: float = 0.04
    min_atr_pct: float = 0.01
    max_atr_pct: float = 0.09
    hold_days: int = 2
    stop_atr: float = 1.1
    take_profit_atr: float = 1.4
    fee_rate: float = 0.0005


def prepare_daily_short_features(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame()

    data = hourly.copy().sort_index()
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"hourly frame missing columns: {sorted(missing)}")
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")

    data["hour_ret"] = (data["close"] / data["open"]) - 1.0
    data["down_volume"] = np.where(data["close"] < data["open"], data["volume"], 0.0)
    data["strong_down_volume"] = np.where(data["hour_ret"] <= -0.003, data["volume"], 0.0)
    data["down_hour"] = np.where(data["close"] < data["open"], 1.0, 0.0)
    data["late_down_volume"] = np.where((data.index.hour >= 16) & (data["close"] < data["open"]), data["volume"], 0.0)

    daily = data.resample("1D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        down_volume=("down_volume", "sum"),
        strong_down_volume=("strong_down_volume", "sum"),
        down_hours=("down_hour", "sum"),
        late_down_volume=("late_down_volume", "sum"),
        bar_count=("close", "count"),
    )
    daily = daily[daily["bar_count"] >= 20].dropna(subset=["open", "high", "low", "close"])
    if daily.empty:
        return daily

    daily["day_return"] = (daily["close"] / daily["open"]) - 1.0
    candle_range = (daily["high"] - daily["low"]).replace(0, np.nan)
    daily["close_location"] = (daily["close"] - daily["low"]) / candle_range
    daily["upper_wick_ratio"] = (daily["high"] - daily[["open", "close"]].max(axis=1)) / candle_range
    daily["sell_volume_ratio"] = daily["down_volume"] / daily["volume"].replace(0, np.nan)
    daily["strong_sell_volume_ratio"] = daily["strong_down_volume"] / daily["volume"].replace(0, np.nan)
    daily["down_hour_ratio"] = daily["down_hours"] / daily["bar_count"].replace(0, np.nan)
    daily["late_sell_ratio"] = daily["late_down_volume"] / daily["volume"].replace(0, np.nan)
    daily["volume_ratio"] = daily["volume"] / daily["volume"].shift(1).rolling(20).mean()
    daily["ema20"] = daily["close"].ewm(span=20, adjust=False).mean()
    daily["ema50"] = daily["close"].ewm(span=50, adjust=False).mean()

    prev_close = daily["close"].shift(1)
    true_range = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - prev_close).abs(),
            (daily["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    daily["atr"] = true_range.rolling(14).mean()
    daily["atr_pct"] = daily["atr"] / daily["close"]
    return daily.replace([np.inf, -np.inf], np.nan)


def daily_short_signal_mask(daily: pd.DataFrame, settings: DailyShortGateSettings) -> pd.Series:
    if daily.empty:
        return pd.Series(dtype=bool)
    mask = (
        (daily["day_return"] <= settings.max_day_return)
        & (daily["volume_ratio"] >= settings.min_volume_ratio)
        & (daily["sell_volume_ratio"] >= settings.min_sell_volume_ratio)
        & (daily["down_hour_ratio"] >= settings.min_down_hour_ratio)
        & (daily["late_sell_ratio"] >= settings.min_late_sell_ratio)
        & (daily["close_location"] <= settings.max_close_location)
        & (daily["close"] < daily["ema20"])
        & daily["atr_pct"].between(settings.min_atr_pct, settings.max_atr_pct)
    )
    return mask.fillna(False)


def simulate_symbol_short_trades(
    daily: pd.DataFrame,
    settings: DailyShortGateSettings,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    signal = daily_short_signal_mask(daily, settings)
    rows: list[dict[str, Any]] = []
    for signal_date in daily.index[signal]:
        entry_date = signal_date + pd.Timedelta(days=1)
        if entry_date < start or entry_date >= end:
            continue
        if entry_date not in daily.index:
            continue
        signal_row = daily.loc[signal_date]
        entry_price = float(daily.loc[entry_date, "open"])
        atr_value = float(signal_row["atr"])
        if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(atr_value) or atr_value <= 0:
            continue

        stop_price = entry_price + (atr_value * settings.stop_atr)
        tp_price = max(entry_price - (atr_value * settings.take_profit_atr), entry_price * 0.5)
        exit_price: float | None = None
        exit_reason = "time_exit"
        exit_date = entry_date
        for offset in range(settings.hold_days):
            current_date = entry_date + pd.Timedelta(days=offset)
            if current_date not in daily.index:
                break
            bar = daily.loc[current_date]
            exit_date = current_date
            if float(bar["high"]) >= stop_price:
                exit_price = stop_price
                exit_reason = "stop_loss"
                break
            if float(bar["low"]) <= tp_price:
                exit_price = tp_price
                exit_reason = "take_profit"
                break
            exit_price = float(bar["close"])

        if exit_price is None or not np.isfinite(exit_price):
            continue
        pnl_pct = ((entry_price - exit_price) / entry_price) - (settings.fee_rate * 2.0)
        rows.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "exit_reason": exit_reason,
            }
        )
    return pd.DataFrame(rows)


def score_symbols_for_gate(
    daily_by_symbol: dict[str, pd.DataFrame],
    settings: DailyShortGateSettings,
    *,
    target_trade_date: pd.Timestamp,
) -> pd.DataFrame:
    target_trade_date = _normalize_utc_day(target_trade_date)
    train_start = target_trade_date - pd.DateOffset(months=settings.lookback_months)
    train_end = target_trade_date - pd.Timedelta(days=max(settings.hold_days, 1))
    rows: list[dict[str, Any]] = []

    for symbol, daily in daily_by_symbol.items():
        trades = simulate_symbol_short_trades(daily, settings, start=train_start, end=train_end)
        if trades.empty:
            rows.append({"symbol": symbol, "trades": 0, "pnl_pct": 0.0, "profit_factor": 0.0, "win_rate": 0.0})
            continue
        winners = trades[trades["pnl_pct"] > 0]
        losers = trades[trades["pnl_pct"] <= 0]
        gross_win = float(winners["pnl_pct"].sum())
        gross_loss = float(losers["pnl_pct"].sum())
        profit_factor = gross_win / abs(gross_loss) if gross_loss < 0 else (99.0 if gross_win > 0 else 0.0)
        rows.append(
            {
                "symbol": symbol,
                "trades": int(len(trades)),
                "pnl_pct": float(trades["pnl_pct"].sum()),
                "profit_factor": float(profit_factor),
                "win_rate": float((trades["pnl_pct"] > 0).mean()),
            }
        )

    score = pd.DataFrame(rows)
    if score.empty:
        return score
    score["eligible"] = (
        (score["trades"] >= settings.min_symbol_trades)
        & (score["profit_factor"] >= settings.min_symbol_profit_factor)
        & (score["pnl_pct"] > 0)
    )
    score = score.sort_values(["eligible", "profit_factor", "pnl_pct", "trades"], ascending=[False, False, False, False])
    score["gate_rank"] = range(1, len(score) + 1)
    score["whitelisted"] = score["eligible"] & (score["gate_rank"] <= settings.top_symbols)
    return score.reset_index(drop=True)


def build_daily_short_gate(
    hourly_by_symbol: dict[str, pd.DataFrame],
    settings: DailyShortGateSettings | None = None,
    *,
    target_trade_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    settings = settings or DailyShortGateSettings()
    daily_by_symbol = {
        symbol: prepare_daily_short_features(frame)
        for symbol, frame in hourly_by_symbol.items()
        if not frame.empty
    }
    daily_by_symbol = {symbol: frame for symbol, frame in daily_by_symbol.items() if not frame.empty}
    if not daily_by_symbol:
        return pd.DataFrame(), {}

    if target_trade_date is None:
        latest_ts = max(frame.index.max() for frame in hourly_by_symbol.values() if not frame.empty)
        target_trade_date = latest_ts.floor("D")
    target_trade_date = _normalize_utc_day(target_trade_date)
    signal_date = target_trade_date - pd.Timedelta(days=1)
    score = score_symbols_for_gate(daily_by_symbol, settings, target_trade_date=target_trade_date)
    score_by_symbol = score.set_index("symbol").to_dict("index") if not score.empty else {}

    rows: list[dict[str, Any]] = []
    for symbol, daily in daily_by_symbol.items():
        signal = False
        signal_payload: dict[str, Any] = {}
        if signal_date in daily.index:
            mask = daily_short_signal_mask(daily, settings)
            signal = bool(mask.loc[signal_date])
            row = daily.loc[signal_date]
            signal_payload = {
                "signal_day_return": float(row.get("day_return", np.nan)),
                "signal_volume_ratio": float(row.get("volume_ratio", np.nan)),
                "signal_sell_volume_ratio": float(row.get("sell_volume_ratio", np.nan)),
                "signal_down_hour_ratio": float(row.get("down_hour_ratio", np.nan)),
                "signal_late_sell_ratio": float(row.get("late_sell_ratio", np.nan)),
                "signal_close_location": float(row.get("close_location", np.nan)),
                "signal_atr_pct": float(row.get("atr_pct", np.nan)),
            }
        symbol_score = score_by_symbol.get(symbol, {})
        whitelisted = bool(symbol_score.get("whitelisted", False))
        rows.append(
            {
                "symbol": symbol,
                "trade_date": target_trade_date,
                "signal_date": signal_date,
                "daily_short_signal": signal,
                "whitelisted": whitelisted,
                "daily_short_allowed": bool(signal and whitelisted),
                "gate_rank": int(symbol_score.get("gate_rank", 9999)),
                "score_trades": int(symbol_score.get("trades", 0)),
                "score_profit_factor": float(symbol_score.get("profit_factor", 0.0)),
                "score_pnl_pct": float(symbol_score.get("pnl_pct", 0.0)),
                "score_win_rate": float(symbol_score.get("win_rate", 0.0)),
                **signal_payload,
            }
        )

    gate = pd.DataFrame(rows).sort_values(["daily_short_allowed", "whitelisted", "gate_rank", "symbol"], ascending=[False, False, True, True])
    return gate.reset_index(drop=True), daily_by_symbol


def enrich_intraday_with_daily_short_gate(intraday: pd.DataFrame, symbol: str, gate: pd.DataFrame) -> pd.DataFrame:
    output = intraday.copy().sort_index()
    output["daily_short_allowed"] = False
    output["daily_short_gate_rank"] = 9999
    output["daily_short_profit_factor"] = 0.0
    if output.empty or gate.empty:
        return output
    rows = gate[gate["symbol"] == symbol]
    if rows.empty:
        return output
    lookup = rows.set_index("trade_date")[["daily_short_allowed", "gate_rank", "score_profit_factor"]]
    payload = pd.DataFrame({"trade_date": output.index.floor("D")}, index=output.index).join(lookup, on="trade_date")
    output["daily_short_allowed"] = payload["daily_short_allowed"].fillna(False).astype(bool)
    output["daily_short_gate_rank"] = payload["gate_rank"].fillna(9999).astype(int)
    output["daily_short_profit_factor"] = payload["score_profit_factor"].fillna(0.0).astype(float)
    return output


def _normalize_utc_day(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.floor("D")
