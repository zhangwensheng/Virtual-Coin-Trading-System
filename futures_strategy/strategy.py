from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from futures_strategy.config import StrategySettings
from futures_strategy.indicators import adx, atr, bollinger_bands, ema, rsi


def prepare_market_data(frame: pd.DataFrame, settings: StrategySettings) -> pd.DataFrame:
    if settings.strategy_kind == "funding_oi_bear_short":
        return prepare_funding_oi_bear_short_data(frame, settings)
    if settings.strategy_kind == "majors_bear_rally_short":
        return prepare_majors_bear_rally_short_data(frame, settings)
    if settings.strategy_kind == "majors_breakout_squeeze":
        return prepare_majors_breakout_squeeze_data(frame, settings)
    if settings.strategy_kind == "majors_ltf_trend":
        return prepare_majors_ltf_trend_data(frame, settings)
    if settings.strategy_kind == "top_gainers_boll_short_1m":
        return prepare_top_gainers_boll_short_data(frame, settings)
    if settings.strategy_kind == "top_gainers_pump_dump_1m":
        return prepare_top_gainers_pump_dump_data(frame, settings)
    if settings.strategy_kind == "reference_volume_reversal_1m":
        return prepare_reference_volume_reversal_data(frame, settings)
    if settings.strategy_kind == "orderflow_imbalance_1m":
        return prepare_orderflow_imbalance_data(frame, settings)
    if settings.strategy_kind == "smallcap_intraday_short":
        return prepare_smallcap_intraday_short_data(frame, settings)
    if settings.strategy_kind == "trend_pullback":
        return prepare_trend_pullback_data(frame, settings)
    raise ValueError(f"Unsupported strategy_kind: {settings.strategy_kind}")


def latest_signal_snapshot(frame: pd.DataFrame, settings: StrategySettings) -> dict[str, Any]:
    if frame.empty:
        return {}

    if settings.strategy_kind == "funding_oi_bear_short":
        return latest_funding_oi_bear_short_snapshot(frame, settings)
    if settings.strategy_kind == "majors_bear_rally_short":
        return latest_majors_bear_rally_short_snapshot(frame, settings)
    if settings.strategy_kind == "majors_breakout_squeeze":
        return latest_majors_breakout_snapshot(frame, settings)
    if settings.strategy_kind == "majors_ltf_trend":
        return latest_majors_ltf_snapshot(frame, settings)
    if settings.strategy_kind == "top_gainers_boll_short_1m":
        return latest_top_gainers_boll_snapshot(frame, settings)
    if settings.strategy_kind == "top_gainers_pump_dump_1m":
        return latest_top_gainers_pump_dump_snapshot(frame, settings)
    if settings.strategy_kind == "reference_volume_reversal_1m":
        return latest_reference_volume_snapshot(frame, settings)
    if settings.strategy_kind == "orderflow_imbalance_1m":
        return latest_orderflow_snapshot(frame, settings)
    if settings.strategy_kind == "smallcap_intraday_short":
        return latest_smallcap_short_snapshot(frame, settings)
    return latest_trend_snapshot(frame, settings)


def prepare_trend_pullback_data(frame: pd.DataFrame, settings: StrategySettings) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("K 线数据为空。")

    data = frame.copy().sort_index()
    data.index.name = "timestamp"
    data["ema_entry"] = ema(data["close"], settings.ltf_ema)
    data["rsi"] = rsi(data["close"], settings.rsi_period)
    data["atr"] = atr(data, settings.atr_period)
    data["atr_pct"] = data["atr"] / data["close"]
    data["volume_sma"] = data["volume"].rolling(settings.volume_sma).mean()
    data["volume_ratio"] = (data["volume"] / data["volume_sma"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    data["prev_high"] = data["high"].shift(1)
    data["prev_low"] = data["low"].shift(1)
    data["swing_low"] = data["low"].rolling(settings.swing_lookback).min().shift(1)
    data["swing_high"] = data["high"].rolling(settings.swing_lookback).max().shift(1)

    htf = resample_ohlcv(data, settings.higher_timeframe)
    htf["htf_close"] = htf["close"]
    htf["htf_ema_fast"] = ema(htf["close"], settings.htf_fast_ema)
    htf["htf_ema_slow"] = ema(htf["close"], settings.htf_slow_ema)
    htf["htf_adx"] = adx(htf, settings.htf_adx_period)

    merged = pd.merge_asof(
        data.reset_index().sort_values("timestamp"),
        htf[["htf_close", "htf_ema_fast", "htf_ema_slow", "htf_adx"]]
        .reset_index()
        .sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    volatility_filter = merged["atr_pct"].between(settings.min_atr_pct, settings.max_atr_pct)
    volume_filter = merged["volume_ratio"] >= settings.min_volume_ratio

    merged["long_regime"] = (
        (merged["htf_close"] > merged["htf_ema_fast"])
        & (merged["htf_ema_fast"] > merged["htf_ema_slow"])
        & (merged["htf_adx"] >= settings.htf_adx_threshold)
    )
    merged["short_regime"] = (
        (merged["htf_close"] < merged["htf_ema_fast"])
        & (merged["htf_ema_fast"] < merged["htf_ema_slow"])
        & (merged["htf_adx"] >= settings.htf_adx_threshold)
    )

    pullback_long = (
        (merged["low"] <= merged["ema_entry"] + (merged["atr"] * settings.pullback_atr_tolerance))
        & (merged["close"] > merged["ema_entry"])
    )
    pullback_short = (
        (merged["high"] >= merged["ema_entry"] - (merged["atr"] * settings.pullback_atr_tolerance))
        & (merged["close"] < merged["ema_entry"])
    )

    long_breakout = merged["close"] > merged["prev_high"]
    short_breakdown = merged["close"] < merged["prev_low"]
    bullish_bar = merged["close"] > merged["open"]
    bearish_bar = merged["close"] < merged["open"]

    merged["long_signal"] = (
        settings.allow_long
        & merged["long_regime"]
        & pullback_long
        & long_breakout
        & bullish_bar
        & merged["rsi"].between(settings.rsi_long_min, settings.rsi_long_max)
        & volume_filter
        & volatility_filter
    )
    merged["short_signal"] = (
        settings.allow_short
        & merged["short_regime"]
        & pullback_short
        & short_breakdown
        & bearish_bar
        & merged["rsi"].between(settings.rsi_short_min, settings.rsi_short_max)
        & volume_filter
        & volatility_filter
    )
    merged["exit_long_signal"] = merged["short_regime"] | (
        (merged["close"] < merged["ema_entry"]) & (merged["rsi"] < 45)
    )
    merged["exit_short_signal"] = merged["long_regime"] | (
        (merged["close"] > merged["ema_entry"]) & (merged["rsi"] > 55)
    )

    return merged.dropna()


def prepare_smallcap_intraday_short_data(frame: pd.DataFrame, settings: StrategySettings) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("K 线数据为空。")

    data = frame.copy().sort_index()
    data.index.name = "timestamp"
    data["ema_entry"] = ema(data["close"], settings.intraday_fast_ema)
    data["ema_slow"] = ema(data["close"], settings.intraday_slow_ema)
    data["rsi"] = rsi(data["close"], settings.rsi_period)
    data["atr"] = atr(data, settings.atr_period)
    data["atr_pct"] = data["atr"] / data["close"]
    data["volume_sma"] = data["volume"].rolling(settings.intraday_volume_window).mean()
    data["volume_ratio"] = (data["volume"] / data["volume_sma"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    data["notional"] = data["close"] * data["volume"]
    data["notional_sma"] = data["notional"].rolling(settings.intraday_notional_window).mean()
    data["notional_ratio"] = (data["notional"] / data["notional_sma"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    data["prev_low"] = data["low"].shift(1)
    data["prev_high"] = data["high"].shift(1)

    session = data.index.floor("D")
    grouped = data.groupby(session)
    data["day_open"] = grouped["open"].transform("first")
    data["session_high"] = grouped["high"].cummax()
    data["session_low"] = grouped["low"].cummin()
    data["bars_from_open"] = grouped.cumcount()

    typical_price = (data["high"] + data["low"] + data["close"]) / 3
    data["session_vwap"] = (
        (typical_price * data["volume"]).groupby(session).cumsum()
        / data["volume"].groupby(session).cumsum().replace(0, np.nan)
    )

    data["day_return"] = (data["close"] / data["day_open"]) - 1
    data["session_peak_return"] = (data["session_high"] / data["day_open"]) - 1
    data["vwap_gap"] = (data["close"] / data["session_vwap"]) - 1

    peak_window = max(settings.intraday_peak_lookback, settings.intraday_breakdown_lookback + 2)
    breakdown_window = max(settings.intraday_breakdown_lookback, 2)
    exhaustion_window = max(breakdown_window * 3, 8)

    data["pump_peak_high"] = data["high"].rolling(peak_window).max()
    data["pump_peak_return"] = (data["pump_peak_high"] / data["day_open"]) - 1
    data["distance_from_peak"] = (data["close"] / data["pump_peak_high"]) - 1
    data["peak_vwap_gap"] = (data["pump_peak_high"] / data["session_vwap"]) - 1

    bar_range = (data["high"] - data["low"]).replace(0, np.nan)
    data["upper_wick_ratio"] = ((data["high"] - np.maximum(data["open"], data["close"])) / bar_range).fillna(0)

    bearish_bar = data["close"] < data["open"]
    bullish_bar = data["close"] > data["open"]
    data["bearish_bar"] = bearish_bar
    data["bullish_bar"] = bullish_bar

    spike_window = max(peak_window, exhaustion_window, breakdown_window + 4)
    recent_volume_spike = (
        data["volume_ratio"].rolling(spike_window).max() >= settings.intraday_volume_spike_threshold
    ) | (
        data["notional_ratio"].rolling(spike_window).max() >= settings.intraday_volume_spike_threshold
    )

    data["exhaustion_bar"] = (
        (data["upper_wick_ratio"] >= settings.intraday_upper_wick_ratio)
        & (data["rsi"] >= settings.intraday_rsi_overbought)
        & (
            (data["vwap_gap"] >= settings.intraday_vwap_extension_threshold * 0.6)
            | (data["volume_ratio"] >= settings.intraday_volume_spike_threshold)
        )
    ) | (
        bearish_bar
        & (data["rsi"] >= settings.intraday_rsi_overbought)
        & (data["volume_ratio"] >= settings.intraday_volume_spike_threshold)
        & (data["close"] < data["high"] * (1 - settings.intraday_drop_from_peak_threshold * 0.5))
    )
    data["recent_exhaustion"] = (
        data["exhaustion_bar"].rolling(exhaustion_window).max().fillna(0).astype(bool)
    )

    recent_break_low = data["low"].rolling(breakdown_window).min().shift(1)
    data["recent_break_low"] = recent_break_low
    data["recent_low_break"] = data["close"] < recent_break_low

    pump_context = (
        (data["bars_from_open"] >= settings.intraday_min_bars_from_open)
        & (data["session_peak_return"] >= settings.intraday_day_return_threshold)
        & (data["pump_peak_return"] >= settings.intraday_day_return_threshold)
        & (data["peak_vwap_gap"] >= settings.intraday_vwap_extension_threshold)
        & recent_volume_spike.fillna(False)
    )

    ema_rollover = (
        (data["ema_entry"] < data["ema_entry"].shift(1))
        | ((data["close"] / data["ema_slow"]) - 1 <= 0.01)
    )

    breakdown_confirm = (
        bearish_bar
        & data["recent_low_break"].fillna(False)
        & (data["close"] < data["ema_entry"])
        & (data["distance_from_peak"] <= -settings.intraday_drop_from_peak_threshold)
        & (data["volume_ratio"] >= settings.intraday_sell_volume_ratio)
        & ema_rollover.fillna(False)
    )

    data["long_regime"] = False
    data["short_regime"] = pump_context.fillna(False)
    data["long_signal"] = False
    data["short_signal"] = (
        settings.allow_short
        & data["short_regime"]
        & data["recent_exhaustion"]
        & breakdown_confirm.fillna(False)
    )
    data["swing_high"] = data["pump_peak_high"]
    data["swing_low"] = data["session_low"]
    data["exit_long_signal"] = False
    data["exit_short_signal"] = (
        (
            (data["close"] <= data["session_vwap"] * (1 - settings.intraday_take_profit_vwap_buffer))
            & (data["rsi"] <= settings.intraday_rsi_take_profit)
        )
        | (
            bullish_bar
            & (data["close"] > data["ema_entry"])
            & (data["volume_ratio"] >= 1.0)
        )
    ).fillna(False)

    return data.dropna()


def prepare_top_gainers_boll_short_data(frame: pd.DataFrame, settings: StrategySettings) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("K 线数据为空。")

    required = {"selected_day", "daily_rank", "daily_return_pct", "daily_close", "daily_bb_upper"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "涨幅前五布林做空策略缺少日线筛选列: "
            f"{sorted(missing)}。请先用全市场筛选脚本生成 enriched CSV。"
        )

    data = frame.copy().sort_index()
    data.index.name = "timestamp"
    numeric_columns = [column for column in data.columns if column not in {"selected_day", "symbol"}]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["selected_day"] = data["selected_day"].astype(str).str.lower().isin({"true", "1", "yes"})
    data["ema_entry"] = ema(data["close"], 9)
    data["rsi"] = rsi(data["close"], settings.rsi_period)
    data["atr"] = atr(data, settings.atr_period)
    data["atr_pct"] = data["atr"] / data["close"]
    data["volume_sma"] = data["volume"].rolling(settings.top_gainers_volume_window).mean()
    data["volume_ratio"] = (data["volume"] / data["volume_sma"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    data["prev_low"] = data["low"].shift(1)
    data["prev_high"] = data["high"].shift(1)
    data["swing_high"] = data["high"].rolling(settings.top_gainers_peak_lookback).max().shift(1)
    data["swing_low"] = data["low"].rolling(settings.top_gainers_breakdown_lookback).min().shift(1)

    bb_mid, bb_upper, bb_lower = bollinger_bands(
        data["close"],
        settings.top_gainers_intraday_boll_window,
        settings.top_gainers_intraday_boll_std,
    )
    data["bb_mid"] = bb_mid
    data["bb_upper"] = bb_upper
    data["bb_lower"] = bb_lower

    htf = resample_ohlcv(data, "5m")
    htf_mid, htf_upper, htf_lower = bollinger_bands(
        htf["close"],
        settings.top_gainers_confirm_boll_window_5m,
        settings.top_gainers_confirm_boll_std_5m,
    )
    htf["htf_bb_mid"] = htf_mid
    htf["htf_bb_upper"] = htf_upper
    htf["htf_bb_lower"] = htf_lower
    htf["htf_open"] = htf["open"]
    htf["htf_high"] = htf["high"]
    htf["htf_close"] = htf["close"]
    htf["htf_volume_sma"] = htf["volume"].rolling(max(4, settings.top_gainers_confirm_boll_window_5m // 2)).mean()
    htf["htf_volume_ratio"] = (htf["volume"] / htf["htf_volume_sma"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    htf_range = (htf["high"] - htf["low"]).replace(0, np.nan)
    htf["htf_upper_wick_ratio"] = ((htf["high"] - np.maximum(htf["open"], htf["close"])) / htf_range).fillna(0)

    merged = pd.merge_asof(
        data.reset_index().sort_values("timestamp"),
        htf[
            [
                "htf_bb_mid",
                "htf_bb_upper",
                "htf_bb_lower",
                "htf_open",
                "htf_high",
                "htf_close",
                "htf_volume_ratio",
                "htf_upper_wick_ratio",
            ]
        ]
        .reset_index()
        .sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    band_buffer = max(0.0, settings.top_gainers_band_touch_buffer)
    bearish_bar = merged["close"] < merged["open"]
    volatility_filter = merged["atr_pct"].between(settings.min_atr_pct, settings.max_atr_pct)

    session = merged.index.floor("D")
    grouped = merged.groupby(session)
    merged["day_open"] = grouped["open"].transform("first")
    merged["session_high"] = grouped["high"].cummax()
    merged["session_low"] = grouped["low"].cummin()
    typical_price = (merged["high"] + merged["low"] + merged["close"]) / 3
    merged["session_vwap"] = (
        (typical_price * merged["volume"]).groupby(session).cumsum()
        / merged["volume"].groupby(session).cumsum().replace(0, np.nan)
    )
    merged["peak_vwap_gap"] = (merged["session_high"] / merged["session_vwap"]) - 1
    merged["distance_from_peak"] = (merged["close"] / merged["session_high"]) - 1
    bar_range = (merged["high"] - merged["low"]).replace(0, np.nan)
    merged["upper_wick_ratio"] = (
        (merged["high"] - np.maximum(merged["open"], merged["close"])) / bar_range
    ).fillna(0)

    daily_filter = (
        merged["selected_day"]
        & (merged["daily_rank"] <= settings.top_gainers_daily_top_n)
        & (merged["daily_return_pct"] >= settings.top_gainers_daily_min_return_pct)
        & (merged["daily_return_pct"] <= settings.top_gainers_daily_max_return_pct)
        & (merged["daily_close"] > merged["daily_bb_upper"])
    )
    current_upper_touch = merged["high"] >= merged["bb_upper"] * (1 - band_buffer)
    recent_upper_touch = current_upper_touch.rolling(settings.top_gainers_distribution_lookback).max().fillna(0).astype(bool)
    htf_upper_touch = (
        (merged["htf_high"] >= merged["htf_bb_upper"] * (1 - band_buffer))
        | (merged["htf_close"] >= merged["htf_bb_upper"] * (1 - band_buffer))
    )
    htf_distribution_bar = (
        (merged["htf_close"] < merged["htf_open"])
        & (merged["htf_volume_ratio"] >= max(1.2, settings.top_gainers_min_volume_ratio * 0.6))
        & htf_upper_touch.fillna(False)
        & (
            (merged["htf_upper_wick_ratio"] >= max(0.08, settings.top_gainers_min_upper_wick_ratio * 0.75))
            | (merged["htf_close"] <= merged["htf_bb_upper"])
        )
    )
    recent_htf_distribution = (
        htf_distribution_bar.rolling(max(2, settings.top_gainers_distribution_lookback // 2)).max().fillna(0).astype(bool)
    )
    recent_break_low = merged["low"].rolling(settings.top_gainers_breakdown_lookback).min().shift(1)
    merged["recent_break_low"] = recent_break_low
    distribution_bar = (
        bearish_bar
        & (merged["volume_ratio"] >= settings.top_gainers_min_volume_ratio)
        & current_upper_touch
        & (merged["peak_vwap_gap"] >= settings.top_gainers_min_vwap_extension_pct)
        & (
            (merged["upper_wick_ratio"] >= settings.top_gainers_min_upper_wick_ratio)
            | (merged["close"] <= merged["bb_upper"])
        )
        & htf_upper_touch.fillna(False)
        & recent_htf_distribution
    )
    recent_distribution = distribution_bar.rolling(settings.top_gainers_distribution_lookback).max().fillna(0).astype(bool)
    breakdown_confirm = (
        bearish_bar
        & recent_distribution
        & (merged["close"] < recent_break_low)
        & (merged["close"] < merged["ema_entry"])
        & (merged["distance_from_peak"] <= -settings.top_gainers_drop_from_peak_pct)
        & (merged["peak_vwap_gap"] >= settings.top_gainers_min_vwap_extension_pct)
    )

    rank_score = (settings.top_gainers_daily_top_n + 1 - merged["daily_rank"]).clip(lower=0)
    merged["factor_score"] = rank_score.fillna(0).astype(int)
    merged["risk_multiplier"] = (1.0 + (merged["factor_score"] * 0.15)).clip(lower=1.0, upper=2.0)
    merged["bounce_pct"] = merged["daily_return_pct"]
    merged["long_regime"] = False
    merged["short_regime"] = (
        daily_filter
        & recent_upper_touch
        & htf_upper_touch.fillna(False)
        & recent_htf_distribution
        & (merged["peak_vwap_gap"] >= settings.top_gainers_min_vwap_extension_pct)
    ).fillna(False)
    merged["long_signal"] = False
    merged["short_signal"] = (
        settings.allow_short
        & merged["short_regime"]
        & (merged["factor_score"] >= settings.top_gainers_min_daily_score)
        & breakdown_confirm.fillna(False)
        & volatility_filter
    )
    merged["exit_long_signal"] = False
    merged["exit_short_signal"] = (
        (merged["close"] <= merged["bb_mid"])
        | (merged["close"] <= merged["session_vwap"])
        | (merged["rsi"] <= settings.top_gainers_exit_rsi)
        | ((merged["close"] > merged["ema_entry"]) & (~bearish_bar))
        | (merged["close"] > merged["prev_high"])
        | (merged["close"] > merged["bb_upper"])
    ).fillna(False)

    return merged.dropna()


def prepare_top_gainers_pump_dump_data(frame: pd.DataFrame, settings: StrategySettings) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("K 线数据为空。")

    required = {"selected_day", "daily_rank", "daily_return_pct", "daily_close", "daily_bb_upper"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "涨幅前五泵拉策略缺少实时排名列: "
            f"{sorted(missing)}。请先用 realtime selection 生成 enriched CSV。"
        )

    data = frame.copy().sort_index()
    data.index.name = "timestamp"
    numeric_columns = [column for column in data.columns if column not in {"selected_day", "symbol"}]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["selected_day"] = data["selected_day"].astype(str).str.lower().isin({"true", "1", "yes"})
    data["ema_entry"] = ema(data["close"], 8)
    data["ema_slow"] = ema(data["close"], 21)
    data["rsi"] = rsi(data["close"], settings.rsi_period)
    data["atr"] = atr(data, settings.atr_period)
    data["atr_pct"] = data["atr"] / data["close"]
    data["volume_sma"] = data["volume"].rolling(settings.top_gainers_volume_window).mean()
    data["volume_ratio"] = (data["volume"] / data["volume_sma"]).replace([np.inf, -np.inf], np.nan).fillna(0)

    quote_volume = data["quote_volume"] if "quote_volume" in data.columns else data["close"] * data["volume"]
    taker_buy_quote = data["taker_buy_quote"] if "taker_buy_quote" in data.columns else quote_volume * 0.5
    data["buy_pressure_ratio"] = (taker_buy_quote / quote_volume.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.5)
    data["delta_ratio"] = ((data["buy_pressure_ratio"] - 0.5) * 2).clip(lower=-1.0, upper=1.0)
    if "trade_count" in data.columns:
        data["trade_sma"] = data["trade_count"].rolling(settings.top_gainers_volume_window).mean()
        data["trade_ratio"] = (data["trade_count"] / data["trade_sma"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    else:
        data["trade_ratio"] = data["volume_ratio"]

    data["prev_low"] = data["low"].shift(1)
    data["prev_high"] = data["high"].shift(1)
    breakout_window = max(2, settings.top_gainers_momentum_breakout_lookback)
    data["momentum_break_high"] = data["high"].rolling(breakout_window).max().shift(1)
    data["momentum_break_low"] = data["low"].rolling(breakout_window).min().shift(1)
    data["swing_low"] = data["low"].rolling(max(3, breakout_window * 2)).min().shift(1)
    data["swing_high"] = data["high"].rolling(settings.top_gainers_peak_lookback).max().shift(1)

    bb_mid, bb_upper, bb_lower = bollinger_bands(
        data["close"],
        settings.top_gainers_intraday_boll_window,
        settings.top_gainers_intraday_boll_std,
    )
    data["bb_mid"] = bb_mid
    data["bb_upper"] = bb_upper
    data["bb_lower"] = bb_lower

    htf = resample_ohlcv(data, "5m")
    htf_mid, htf_upper, htf_lower = bollinger_bands(
        htf["close"],
        settings.top_gainers_confirm_boll_window_5m,
        settings.top_gainers_confirm_boll_std_5m,
    )
    htf["htf_bb_mid"] = htf_mid
    htf["htf_bb_upper"] = htf_upper
    htf["htf_open"] = htf["open"]
    htf["htf_high"] = htf["high"]
    htf["htf_close"] = htf["close"]
    htf["htf_volume_sma"] = htf["volume"].rolling(max(4, settings.top_gainers_confirm_boll_window_5m // 2)).mean()
    htf["htf_volume_ratio"] = (htf["volume"] / htf["htf_volume_sma"]).replace([np.inf, -np.inf], np.nan).fillna(0)

    merged = pd.merge_asof(
        data.reset_index().sort_values("timestamp"),
        htf[["htf_bb_mid", "htf_bb_upper", "htf_open", "htf_high", "htf_close", "htf_volume_ratio"]]
        .reset_index()
        .sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    session = merged.index.floor("D")
    grouped = merged.groupby(session)
    merged["day_open"] = grouped["open"].transform("first")
    merged["bars_from_open"] = grouped.cumcount()
    merged["session_high"] = grouped["high"].cummax()
    merged["session_low"] = grouped["low"].cummin()
    typical_price = (merged["high"] + merged["low"] + merged["close"]) / 3
    merged["session_vwap"] = (
        (typical_price * merged["volume"]).groupby(session).cumsum()
        / merged["volume"].groupby(session).cumsum().replace(0, np.nan)
    )
    merged["vwap_gap"] = (merged["close"] / merged["session_vwap"]) - 1
    merged["peak_vwap_gap"] = (merged["session_high"] / merged["session_vwap"]) - 1
    merged["distance_from_peak"] = (merged["close"] / merged["session_high"]) - 1
    bar_range = (merged["high"] - merged["low"]).replace(0, np.nan)
    merged["upper_wick_ratio"] = (
        (merged["high"] - np.maximum(merged["open"], merged["close"])) / bar_range
    ).fillna(0)

    bullish_bar = merged["close"] > merged["open"]
    bearish_bar = merged["close"] < merged["open"]
    volatility_filter = merged["atr_pct"].between(settings.min_atr_pct, settings.max_atr_pct)
    rank_filter = (
        merged["selected_day"]
        & (merged["daily_rank"] <= settings.top_gainers_daily_top_n)
        & (merged["daily_return_pct"] >= settings.top_gainers_daily_min_return_pct)
        & (merged["daily_close"] > merged["daily_bb_upper"])
    )

    momentum_context = (
        rank_filter
        & (merged["close"] > merged["ema_entry"])
        & (merged["ema_entry"] > merged["ema_slow"])
        & (merged["close"] > merged["session_vwap"])
        & (merged["vwap_gap"] <= settings.top_gainers_momentum_max_vwap_extension_pct)
        & (merged["htf_close"] >= merged["htf_bb_mid"])
    )
    long_breakout = (
        bullish_bar
        & (merged["close"] > merged["momentum_break_high"])
        & (merged["volume_ratio"] >= settings.top_gainers_momentum_min_volume_ratio)
        & (merged["volume_ratio"] <= settings.top_gainers_momentum_max_volume_ratio)
        & (merged["trade_ratio"] >= settings.top_gainers_momentum_min_trade_ratio)
        & (merged["trade_ratio"] <= settings.top_gainers_momentum_max_trade_ratio)
        & (merged["delta_ratio"] >= settings.top_gainers_momentum_min_delta_ratio)
        & (merged["delta_ratio"] <= settings.top_gainers_momentum_max_delta_ratio)
        & (merged["rsi"] >= settings.top_gainers_momentum_min_rsi)
        & (merged["rsi"] <= settings.top_gainers_momentum_max_rsi)
        & (merged["htf_volume_ratio"] >= settings.top_gainers_momentum_min_htf_volume_ratio)
        & merged["bars_from_open"].between(
            settings.top_gainers_momentum_min_bars_from_open,
            settings.top_gainers_momentum_max_bars_from_open,
        )
    )

    band_buffer = max(0.0, settings.top_gainers_band_touch_buffer)
    upper_touch = merged["high"] >= merged["bb_upper"] * (1 - band_buffer)
    htf_upper_touch = (
        (merged["htf_high"] >= merged["htf_bb_upper"] * (1 - band_buffer))
        | (merged["htf_close"] >= merged["htf_bb_upper"] * (1 - band_buffer))
    )
    distribution_bar = (
        bearish_bar
        & upper_touch
        & htf_upper_touch.fillna(False)
        & (merged["volume_ratio"] >= settings.top_gainers_min_volume_ratio)
        & (merged["delta_ratio"] <= -settings.top_gainers_momentum_min_delta_ratio * 0.35)
        & (merged["upper_wick_ratio"] >= settings.top_gainers_min_upper_wick_ratio)
        & (merged["peak_vwap_gap"] >= settings.top_gainers_min_vwap_extension_pct)
    )
    recent_distribution = distribution_bar.rolling(settings.top_gainers_distribution_lookback).max().fillna(0).astype(bool)
    short_breakdown = (
        bearish_bar
        & recent_distribution
        & (merged["close"] < merged["momentum_break_low"])
        & (merged["close"] < merged["ema_entry"])
        & (merged["distance_from_peak"] <= -settings.top_gainers_drop_from_peak_pct)
    )

    rank_score = (settings.top_gainers_daily_top_n + 1 - merged["daily_rank"]).clip(lower=0)
    merged["factor_score"] = rank_score.fillna(0).astype(int)
    merged["risk_multiplier"] = (
        0.75
        + (merged["factor_score"] * 0.15)
        + (merged["volume_ratio"].clip(0, 5) * 0.08)
        + (merged["daily_return_pct"].clip(0, 0.5) * 1.2)
    ).clip(lower=0.75, upper=2.5)
    merged["bounce_pct"] = merged["daily_return_pct"]
    merged["long_regime"] = momentum_context.fillna(False)
    merged["short_regime"] = (rank_filter & recent_distribution).fillna(False)
    merged["long_signal"] = (
        settings.allow_long
        & merged["long_regime"]
        & long_breakout.fillna(False)
        & volatility_filter
    )
    merged["short_signal"] = (
        settings.allow_short
        & merged["short_regime"]
        & short_breakdown.fillna(False)
        & volatility_filter
    )
    merged["exit_long_signal"] = (
        distribution_bar
        | (merged["delta_ratio"] <= settings.top_gainers_momentum_exit_delta_ratio)
        | ((merged["close"] < merged["ema_entry"]) & bearish_bar)
        | (merged["close"] < merged["prev_low"])
    ).fillna(False)
    merged["exit_short_signal"] = (
        (merged["close"] <= merged["bb_mid"])
        | (merged["close"] <= merged["session_vwap"])
        | (merged["rsi"] <= settings.top_gainers_exit_rsi)
        | ((merged["close"] > merged["ema_entry"]) & bullish_bar)
    ).fillna(False)

    return merged.dropna()


def prepare_reference_volume_reversal_data(frame: pd.DataFrame, settings: StrategySettings) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("K 线数据为空。")

    data = frame.copy().sort_index()
    data.index.name = "timestamp"
    numeric_columns = [column for column in data.columns if column != "confirm"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    fast_ma = max(int(settings.reference_fast_ma), 2)
    slow_ma = max(int(settings.reference_slow_ma), fast_ma + 1)
    volume_window = max(int(settings.reference_volume_window), 3)
    breakout_lookback = max(int(settings.reference_breakout_lookback), 2)

    data["ema_entry"] = data["close"].rolling(fast_ma).mean()
    data["ema_slow"] = data["close"].rolling(slow_ma).mean()
    data["ma_fast_slope"] = data["ema_entry"].diff()
    data["rsi"] = rsi(data["close"], settings.rsi_period)
    data["atr"] = atr(data, settings.atr_period)
    data["atr_pct"] = data["atr"] / data["close"]
    data["volume_sma"] = data["volume"].rolling(volume_window).mean()
    data["volume_ratio"] = (data["volume"] / data["volume_sma"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    data["prev_high"] = data["high"].shift(1)
    data["prev_low"] = data["low"].shift(1)
    data["swing_high"] = data["high"].rolling(breakout_lookback).max().shift(1)
    data["swing_low"] = data["low"].rolling(breakout_lookback).min().shift(1)

    typical_price = (data["high"] + data["low"] + data["close"]) / 3
    session = data.index.floor("D")
    vwap_numerator = (typical_price * data["volume"]).groupby(session).cumsum()
    vwap_denominator = data["volume"].groupby(session).cumsum().replace(0, np.nan)
    data["session_vwap"] = vwap_numerator / vwap_denominator
    data["vwap_gap"] = (data["close"] / data["session_vwap"]) - 1

    bar_range = (data["high"] - data["low"]).replace(0, np.nan)
    data["body_pct"] = ((data["close"] - data["open"]).abs() / data["open"].replace(0, np.nan)).fillna(0)
    data["upper_wick_ratio"] = (
        (data["high"] - np.maximum(data["open"], data["close"])) / bar_range
    ).fillna(0)
    data["lower_wick_ratio"] = (
        (np.minimum(data["open"], data["close"]) - data["low"]) / bar_range
    ).fillna(0)

    if {"quote_volume", "taker_buy_quote"}.issubset(data.columns):
        data["buy_pressure_ratio"] = (
            data["taker_buy_quote"] / data["quote_volume"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    else:
        data["buy_pressure_ratio"] = 0.5
    data["sell_pressure_ratio"] = 1.0 - data["buy_pressure_ratio"]

    prev = data.shift(1)
    previous_bull_climax = (
        (prev["close"] > prev["open"])
        & (prev["volume_ratio"] >= settings.reference_min_volume_ratio)
        & (prev["body_pct"] >= settings.reference_min_body_pct)
        & (
            ((prev["close"] / prev["ema_entry"]) - 1 >= settings.reference_min_extension_pct)
            | (prev["vwap_gap"] >= settings.reference_min_extension_pct)
        )
        & (
            (prev["upper_wick_ratio"] >= settings.reference_min_rejection_wick)
            | (prev["buy_pressure_ratio"] >= settings.reference_min_pressure_ratio)
        )
    )
    previous_bear_climax = (
        (prev["close"] < prev["open"])
        & (prev["volume_ratio"] >= settings.reference_min_volume_ratio)
        & (prev["body_pct"] >= settings.reference_min_body_pct)
        & (
            (((prev["close"] / prev["ema_entry"]) - 1) <= -settings.reference_min_extension_pct)
            | (prev["vwap_gap"] <= -settings.reference_min_extension_pct)
        )
        & (
            (prev["lower_wick_ratio"] >= settings.reference_min_rejection_wick)
            | (prev["sell_pressure_ratio"] >= settings.reference_min_pressure_ratio)
        )
    )

    bearish_bar = data["close"] < data["open"]
    bullish_bar = data["close"] > data["open"]
    volume_confirm = data["volume_ratio"] >= settings.reference_confirm_volume_ratio
    volatility_filter = data["atr_pct"].between(settings.min_atr_pct, settings.max_atr_pct)

    data["long_regime"] = (
        (data["close"] > data["session_vwap"])
        & ((data["ema_entry"] > data["ema_slow"]) | (data["ma_fast_slope"] > 0))
    ).fillna(False)
    data["short_regime"] = (
        (data["close"] < data["session_vwap"])
        & ((data["ema_entry"] < data["ema_slow"]) | (data["ma_fast_slope"] < 0))
    ).fillna(False)
    data["long_context"] = previous_bear_climax.fillna(False)
    data["short_context"] = previous_bull_climax.fillna(False)

    data["long_signal"] = (
        settings.allow_long
        & data["long_context"]
        & bullish_bar
        & (data["close"] > data["prev_high"])
        & (data["close"] > data["ema_entry"])
        & volume_confirm
        & volatility_filter
    ).fillna(False)
    data["short_signal"] = (
        settings.allow_short
        & data["short_context"]
        & bearish_bar
        & (data["close"] < data["prev_low"])
        & (data["close"] < data["ema_entry"])
        & volume_confirm
        & volatility_filter
    ).fillna(False)

    data["exit_long_signal"] = (
        (data["close"] < data["ema_entry"])
        | (data["ema_entry"] < data["ema_slow"])
        | (data["close"] < data["prev_low"])
        | (data["rsi"] <= (100 - settings.reference_exit_rsi))
    ).fillna(False)
    data["exit_short_signal"] = (
        (data["close"] > data["ema_entry"])
        | (data["ema_entry"] > data["ema_slow"])
        | (data["close"] > data["prev_high"])
        | (data["rsi"] >= settings.reference_exit_rsi)
    ).fillna(False)

    return data.dropna()


def prepare_orderflow_imbalance_data(frame: pd.DataFrame, settings: StrategySettings) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("K 线数据为空。")

    required = {"quote_volume", "trade_count", "taker_buy_base", "taker_buy_quote"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "订单流策略缺少代理成交列: "
            f"{sorted(missing)}。请使用带 taker_buy / trade_count 的 Binance 归档 CSV。"
        )

    data = frame.copy().sort_index()
    data.index.name = "timestamp"

    numeric_columns = [
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
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["ema_entry"] = ema(data["close"], settings.orderflow_fast_ema)
    data["ema_slow"] = ema(data["close"], settings.orderflow_slow_ema)
    data["rsi"] = rsi(data["close"], settings.rsi_period)
    data["atr"] = atr(data, settings.atr_period)
    data["atr_pct"] = data["atr"] / data["close"]

    volume_window = max(int(settings.orderflow_volume_window), 5)
    trade_window = max(int(settings.orderflow_trade_window), 5)
    delta_window = max(int(settings.orderflow_delta_window), 3)
    breakout_lookback = max(int(settings.orderflow_breakout_lookback), 3)
    zscore_window = max(delta_window * 5, 20)

    data["quote_volume_sma"] = data["quote_volume"].rolling(volume_window).mean()
    data["quote_volume_ratio"] = (
        data["quote_volume"] / data["quote_volume_sma"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    data["trade_count_sma"] = data["trade_count"].rolling(trade_window).mean()
    data["trade_count_ratio"] = (
        data["trade_count"] / data["trade_count_sma"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    data["taker_sell_base"] = (data["volume"] - data["taker_buy_base"]).clip(lower=0)
    data["taker_sell_quote"] = (data["quote_volume"] - data["taker_buy_quote"]).clip(lower=0)
    data["delta_base"] = data["taker_buy_base"] - data["taker_sell_base"]
    data["delta_quote"] = data["taker_buy_quote"] - data["taker_sell_quote"]
    data["delta_ratio"] = (
        data["delta_quote"] / data["quote_volume"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    data["buy_pressure_ratio"] = (
        data["taker_buy_quote"] / data["quote_volume"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    data["sell_pressure_ratio"] = (
        data["taker_sell_quote"] / data["quote_volume"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    data["rolling_delta_quote"] = data["delta_quote"].rolling(delta_window).sum()
    data["rolling_quote_volume"] = data["quote_volume"].rolling(delta_window).sum()
    data["rolling_delta_ratio"] = (
        data["rolling_delta_quote"] / data["rolling_quote_volume"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    data["rolling_delta_mean"] = data["rolling_delta_quote"].rolling(zscore_window).mean()
    data["rolling_delta_std"] = data["rolling_delta_quote"].rolling(zscore_window).std()
    data["flow_zscore"] = (
        (data["rolling_delta_quote"] - data["rolling_delta_mean"])
        / data["rolling_delta_std"].replace(0, np.nan)
    )

    data["prev_high"] = data["high"].shift(1)
    data["prev_low"] = data["low"].shift(1)
    data["breakout_high"] = data["high"].rolling(breakout_lookback).max().shift(1)
    data["breakout_low"] = data["low"].rolling(breakout_lookback).min().shift(1)
    data["swing_low"] = data["low"].rolling(breakout_lookback).min().shift(1)
    data["swing_high"] = data["high"].rolling(breakout_lookback).max().shift(1)

    session = data.index.floor("D")
    typical_price = (data["high"] + data["low"] + data["close"]) / 3
    data["session_vwap"] = (
        (typical_price * data["quote_volume"]).groupby(session).cumsum()
        / data["quote_volume"].groupby(session).cumsum().replace(0, np.nan)
    )
    data["vwap_gap"] = (data["close"] / data["session_vwap"]) - 1
    data["ema_gap"] = ((data["close"] / data["ema_entry"]) - 1).abs()

    bullish_bar = data["close"] > data["open"]
    bearish_bar = data["close"] < data["open"]
    volume_filter = data["quote_volume_ratio"] >= settings.orderflow_min_volume_ratio
    activity_filter = data["trade_count_ratio"] >= settings.orderflow_min_trade_ratio
    volatility_filter = data["atr_pct"].between(settings.min_atr_pct, settings.max_atr_pct)
    not_overextended = data["ema_gap"] <= settings.orderflow_max_extension_pct

    positive_flow = (
        (data["delta_ratio"] >= settings.orderflow_min_delta_ratio)
        & (data["rolling_delta_ratio"] >= settings.orderflow_min_delta_ratio * 0.6)
        & (data["flow_zscore"] >= settings.orderflow_min_delta_zscore)
    )
    negative_flow = (
        (data["delta_ratio"] <= -settings.orderflow_min_delta_ratio)
        & (data["rolling_delta_ratio"] <= -(settings.orderflow_min_delta_ratio * 0.6))
        & (data["flow_zscore"] <= -settings.orderflow_min_delta_zscore)
    )

    data["long_regime"] = (
        (data["close"] > data["ema_entry"])
        & (data["ema_entry"] > data["ema_slow"])
        & (data["close"] > data["session_vwap"] * (1 + settings.orderflow_vwap_buffer))
    )
    data["short_regime"] = (
        (data["close"] < data["ema_entry"])
        & (data["ema_entry"] < data["ema_slow"])
        & (data["close"] < data["session_vwap"] * (1 - settings.orderflow_vwap_buffer))
    )

    data["long_signal"] = (
        settings.allow_long
        & data["long_regime"]
        & bullish_bar
        & (data["close"] > data["breakout_high"])
        & positive_flow.fillna(False)
        & volume_filter
        & activity_filter
        & volatility_filter
        & not_overextended.fillna(False)
    )
    data["short_signal"] = (
        settings.allow_short
        & data["short_regime"]
        & bearish_bar
        & (data["close"] < data["breakout_low"])
        & negative_flow.fillna(False)
        & volume_filter
        & activity_filter
        & volatility_filter
        & not_overextended.fillna(False)
    )

    data["exit_long_signal"] = (
        data["short_regime"]
        | (data["close"] < data["ema_entry"])
        | (data["close"] < data["session_vwap"])
        | (data["delta_ratio"] <= settings.orderflow_exit_delta_ratio)
        | (data["rolling_delta_ratio"] <= 0)
    ).fillna(False)
    data["exit_short_signal"] = (
        data["long_regime"]
        | (data["close"] > data["ema_entry"])
        | (data["close"] > data["session_vwap"])
        | (data["delta_ratio"] >= -settings.orderflow_exit_delta_ratio)
        | (data["rolling_delta_ratio"] >= 0)
    ).fillna(False)

    return data.dropna()


def prepare_majors_ltf_trend_data(frame: pd.DataFrame, settings: StrategySettings) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("K 线数据为空。")

    data = frame.copy().sort_index()
    data.index.name = "timestamp"
    data["ema_entry"] = ema(data["close"], settings.majors_fast_ema)
    data["ema_slow"] = ema(data["close"], settings.majors_slow_ema)
    data["rsi"] = rsi(data["close"], settings.rsi_period)
    data["atr"] = atr(data, settings.atr_period)
    data["atr_pct"] = data["atr"] / data["close"]
    data["volume_sma"] = data["volume"].rolling(settings.majors_volume_window).mean()
    data["volume_ratio"] = (data["volume"] / data["volume_sma"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    data["prev_high"] = data["high"].shift(1)
    data["prev_low"] = data["low"].shift(1)

    session = data.index.floor("D")
    grouped = data.groupby(session)
    data["bars_from_open"] = grouped.cumcount()
    data["session_high"] = grouped["high"].cummax()
    data["session_low"] = grouped["low"].cummin()

    typical_price = (data["high"] + data["low"] + data["close"]) / 3
    data["session_vwap"] = (
        (typical_price * data["volume"]).groupby(session).cumsum()
        / data["volume"].groupby(session).cumsum().replace(0, np.nan)
    )

    impulse_window = max(settings.majors_impulse_window, settings.majors_pullback_window + 2)
    pullback_window = max(settings.majors_pullback_window, 2)
    data["recent_high"] = data["high"].rolling(impulse_window).max().shift(1)
    data["recent_low"] = data["low"].rolling(impulse_window).min().shift(1)
    data["impulse_range_pct"] = (data["recent_high"] / data["recent_low"]) - 1
    data["recent_pullback_low"] = data["low"].rolling(pullback_window).min()
    data["recent_bounce_high"] = data["high"].rolling(pullback_window).max()
    data["retrace_from_high"] = (data["recent_pullback_low"] / data["recent_high"]) - 1
    data["bounce_from_low"] = (data["recent_bounce_high"] / data["recent_low"]) - 1

    data["recent_touch_long"] = (
        (data["low"] <= data["ema_entry"] + (data["atr"] * settings.majors_pullback_atr_tolerance))
        .rolling(pullback_window)
        .max()
        .fillna(0)
        .astype(bool)
    )
    data["recent_touch_short"] = (
        (data["high"] >= data["ema_entry"] - (data["atr"] * settings.majors_pullback_atr_tolerance))
        .rolling(pullback_window)
        .max()
        .fillna(0)
        .astype(bool)
    )

    htf = resample_ohlcv(data, settings.higher_timeframe)
    htf["htf_close"] = htf["close"]
    htf["htf_ema_fast"] = ema(htf["close"], settings.htf_fast_ema)
    htf["htf_ema_slow"] = ema(htf["close"], settings.htf_slow_ema)
    htf["htf_adx"] = adx(htf, settings.htf_adx_period)

    merged = pd.merge_asof(
        data.reset_index().sort_values("timestamp"),
        htf[["htf_close", "htf_ema_fast", "htf_ema_slow", "htf_adx"]]
        .reset_index()
        .sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    bullish_bar = merged["close"] > merged["open"]
    bearish_bar = merged["close"] < merged["open"]
    volume_filter = merged["volume_ratio"] >= settings.majors_volume_ratio_threshold
    volatility_filter = merged["atr_pct"].between(settings.min_atr_pct, settings.max_atr_pct)
    bars_filter = merged["bars_from_open"] >= settings.majors_min_bars_from_open

    retrace_ok_long = merged["retrace_from_high"].between(
        -settings.majors_max_retrace_from_extreme,
        -settings.majors_min_retrace_from_extreme,
    )
    bounce_ok_short = merged["bounce_from_low"].between(
        settings.majors_min_retrace_from_extreme,
        settings.majors_max_retrace_from_extreme,
    )
    impulse_ok = merged["impulse_range_pct"] >= settings.majors_impulse_threshold

    vwap_gap = (merged["close"] / merged["session_vwap"]) - 1
    not_extended_long = vwap_gap <= settings.majors_vwap_extension_cap
    not_extended_short = (-vwap_gap) <= settings.majors_vwap_extension_cap

    merged["long_regime"] = (
        (merged["htf_close"] > merged["htf_ema_fast"])
        & (merged["htf_ema_fast"] > merged["htf_ema_slow"])
        & (merged["htf_adx"] >= settings.htf_adx_threshold)
        & (merged["close"] > merged["session_vwap"])
        & (merged["ema_entry"] > merged["ema_slow"])
        & bars_filter
    )
    merged["short_regime"] = (
        (merged["htf_close"] < merged["htf_ema_fast"])
        & (merged["htf_ema_fast"] < merged["htf_ema_slow"])
        & (merged["htf_adx"] >= settings.htf_adx_threshold)
        & (merged["close"] < merged["session_vwap"])
        & (merged["ema_entry"] < merged["ema_slow"])
        & bars_filter
    )

    merged["long_signal"] = (
        settings.allow_long
        & merged["long_regime"]
        & impulse_ok.fillna(False)
        & merged["recent_touch_long"]
        & retrace_ok_long.fillna(False)
        & (merged["high"] > merged["prev_high"])
        & (merged["close"] > merged["ema_entry"])
        & bullish_bar
        & merged["rsi"].between(settings.majors_entry_rsi_low, settings.majors_entry_rsi_high)
        & volume_filter
        & volatility_filter
        & not_extended_long.fillna(False)
    )
    merged["short_signal"] = (
        settings.allow_short
        & merged["short_regime"]
        & impulse_ok.fillna(False)
        & merged["recent_touch_short"]
        & bounce_ok_short.fillna(False)
        & (merged["low"] < merged["prev_low"])
        & (merged["close"] < merged["ema_entry"])
        & bearish_bar
        & merged["rsi"].between(100 - settings.majors_entry_rsi_high, 100 - settings.majors_entry_rsi_low)
        & volume_filter
        & volatility_filter
        & not_extended_short.fillna(False)
    )

    merged["swing_low"] = merged["recent_pullback_low"].shift(1)
    merged["swing_high"] = merged["recent_bounce_high"].shift(1)
    merged["exit_long_signal"] = (
        merged["short_regime"]
        | ((merged["close"] < merged["session_vwap"]) & (merged["rsi"] < settings.majors_exit_rsi))
        | ((merged["close"] < merged["ema_slow"]) & (merged["rsi"] < settings.majors_exit_rsi))
    ).fillna(False)
    merged["exit_short_signal"] = (
        merged["long_regime"]
        | ((merged["close"] > merged["session_vwap"]) & (merged["rsi"] > (100 - settings.majors_exit_rsi)))
        | ((merged["close"] > merged["ema_slow"]) & (merged["rsi"] > (100 - settings.majors_exit_rsi)))
    ).fillna(False)

    return merged.dropna()


def prepare_majors_breakout_squeeze_data(frame: pd.DataFrame, settings: StrategySettings) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("K 线数据为空。")

    data = frame.copy().sort_index()
    data.index.name = "timestamp"
    data["ema_entry"] = ema(data["close"], settings.breakout_fast_ema)
    data["ema_slow"] = ema(data["close"], settings.breakout_slow_ema)
    data["rsi"] = rsi(data["close"], settings.rsi_period)
    data["atr"] = atr(data, settings.atr_period)
    data["atr_pct"] = data["atr"] / data["close"]
    data["volume_sma"] = data["volume"].rolling(settings.breakout_volume_window).mean()
    data["volume_ratio"] = (data["volume"] / data["volume_sma"]).replace([np.inf, -np.inf], np.nan).fillna(0)

    channel_window = max(settings.breakout_channel_window, settings.breakout_exit_window + 1)
    squeeze_window = max(settings.breakout_squeeze_window, settings.breakout_squeeze_lookback + 1)
    data["donchian_high"] = data["high"].rolling(channel_window).max().shift(1)
    data["donchian_low"] = data["low"].rolling(channel_window).min().shift(1)
    data["exit_high"] = data["high"].rolling(settings.breakout_exit_window).max().shift(1)
    data["exit_low"] = data["low"].rolling(settings.breakout_exit_window).min().shift(1)
    data["range_high"] = data["high"].rolling(squeeze_window).max()
    data["range_low"] = data["low"].rolling(squeeze_window).min()
    data["range_pct"] = (data["range_high"] - data["range_low"]) / data["close"]
    data["atr_pct_mean"] = data["atr_pct"].rolling(squeeze_window).mean()
    data["squeeze"] = (
        (data["range_pct"] <= settings.breakout_max_range_pct)
        | (data["atr_pct"] <= (data["atr_pct_mean"] * settings.breakout_atr_compression_mult))
    ).fillna(False)
    data["recent_squeeze"] = (
        data["squeeze"].rolling(settings.breakout_squeeze_lookback).max().fillna(0).astype(bool)
    )

    htf = resample_ohlcv(data, settings.higher_timeframe)
    htf["htf_close"] = htf["close"]
    htf["htf_ema_fast"] = ema(htf["close"], settings.htf_fast_ema)
    htf["htf_ema_slow"] = ema(htf["close"], settings.htf_slow_ema)
    htf["htf_adx"] = adx(htf, settings.htf_adx_period)

    merged = pd.merge_asof(
        data.reset_index().sort_values("timestamp"),
        htf[["htf_close", "htf_ema_fast", "htf_ema_slow", "htf_adx"]]
        .reset_index()
        .sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    volume_filter = merged["volume_ratio"] >= settings.breakout_min_volume_ratio
    volatility_filter = merged["atr_pct"].between(settings.min_atr_pct, settings.max_atr_pct)
    bullish_bar = merged["close"] > merged["open"]
    bearish_bar = merged["close"] < merged["open"]

    merged["long_regime"] = (
        (merged["htf_close"] > merged["htf_ema_fast"])
        & (merged["htf_ema_fast"] > merged["htf_ema_slow"])
        & (merged["htf_adx"] >= settings.htf_adx_threshold)
        & (merged["ema_entry"] > merged["ema_slow"])
    )
    merged["short_regime"] = (
        (merged["htf_close"] < merged["htf_ema_fast"])
        & (merged["htf_ema_fast"] < merged["htf_ema_slow"])
        & (merged["htf_adx"] >= settings.htf_adx_threshold)
        & (merged["ema_entry"] < merged["ema_slow"])
    )

    merged["long_signal"] = (
        settings.allow_long
        & merged["long_regime"]
        & merged["recent_squeeze"]
        & (merged["close"] > merged["donchian_high"])
        & bullish_bar
        & merged["rsi"].between(settings.breakout_rsi_long_min, settings.breakout_rsi_long_max)
        & volume_filter
        & volatility_filter
    )
    merged["short_signal"] = (
        settings.allow_short
        & merged["short_regime"]
        & merged["recent_squeeze"]
        & (merged["close"] < merged["donchian_low"])
        & bearish_bar
        & merged["rsi"].between(settings.breakout_rsi_short_min, settings.breakout_rsi_short_max)
        & volume_filter
        & volatility_filter
    )

    merged["swing_low"] = merged["donchian_low"]
    merged["swing_high"] = merged["donchian_high"]
    merged["exit_long_signal"] = (
        merged["short_regime"]
        | (merged["close"] < merged["exit_low"])
        | ((merged["close"] < merged["ema_entry"]) & (merged["rsi"] < 48))
    ).fillna(False)
    merged["exit_short_signal"] = (
        merged["long_regime"]
        | (merged["close"] > merged["exit_high"])
        | ((merged["close"] > merged["ema_entry"]) & (merged["rsi"] > 52))
    ).fillna(False)

    return merged.dropna()


def prepare_majors_bear_rally_short_data(frame: pd.DataFrame, settings: StrategySettings) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("K 线数据为空。")

    data = frame.copy().sort_index()
    data.index.name = "timestamp"
    data["ema_entry"] = ema(data["close"], settings.bear_fast_ema)
    data["ema_slow"] = ema(data["close"], settings.bear_slow_ema)
    data["rsi"] = rsi(data["close"], settings.rsi_period)
    data["atr"] = atr(data, settings.atr_period)
    data["atr_pct"] = data["atr"] / data["close"]
    data["volume_sma"] = data["volume"].rolling(settings.bear_volume_window).mean()
    data["volume_ratio"] = (data["volume"] / data["volume_sma"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    data["prev_low"] = data["low"].shift(1)
    data["prev_high"] = data["high"].shift(1)

    session = data.index.floor("D")
    grouped = data.groupby(session)
    data["bars_from_open"] = grouped.cumcount()
    data["session_high"] = grouped["high"].cummax()
    data["session_low"] = grouped["low"].cummin()
    data["day_open"] = grouped["open"].transform("first")

    typical_price = (data["high"] + data["low"] + data["close"]) / 3
    data["session_vwap"] = (
        (typical_price * data["volume"]).groupby(session).cumsum()
        / data["volume"].groupby(session).cumsum().replace(0, np.nan)
    )

    impulse_window = max(settings.bear_impulse_window, settings.bear_bounce_window + 4)
    breakdown_window = max(settings.bear_breakdown_lookback, 2)
    exit_window = max(settings.bear_exit_window, breakdown_window + 1)

    data["recent_impulse_low"] = data["low"].rolling(impulse_window).min()
    data["recent_bounce_high"] = data["high"].rolling(settings.bear_bounce_window).max()
    data["bounce_pct"] = (data["recent_bounce_high"] / data["recent_impulse_low"]) - 1
    data["recent_rsi_peak"] = data["rsi"].rolling(settings.bear_bounce_window).max()
    data["breakdown_low"] = data["low"].rolling(breakdown_window).min().shift(1)
    data["cover_high"] = data["high"].rolling(exit_window).max().shift(1)
    data["swing_high"] = data["recent_bounce_high"].shift(1)
    data["swing_low"] = data["recent_impulse_low"].shift(1)

    bar_range = (data["high"] - data["low"]).replace(0, np.nan)
    data["upper_wick_ratio"] = ((data["high"] - np.maximum(data["open"], data["close"])) / bar_range).fillna(0)
    data["vwap_gap"] = (data["close"] / data["session_vwap"]) - 1
    data["ema_gap"] = (data["high"] / data["ema_entry"]) - 1
    data["day_return"] = (data["close"] / data["day_open"]) - 1

    htf = resample_ohlcv(data, settings.higher_timeframe)
    htf["htf_close"] = htf["close"]
    htf["htf_ema_fast"] = ema(htf["close"], settings.htf_fast_ema)
    htf["htf_ema_slow"] = ema(htf["close"], settings.htf_slow_ema)
    htf["htf_adx"] = adx(htf, settings.htf_adx_period)

    merged = pd.merge_asof(
        data.reset_index().sort_values("timestamp"),
        htf[["htf_close", "htf_ema_fast", "htf_ema_slow", "htf_adx"]]
        .reset_index()
        .sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    bearish_bar = merged["close"] < merged["open"]
    bullish_bar = merged["close"] > merged["open"]
    volume_filter = merged["volume_ratio"] >= settings.bear_min_volume_ratio
    volatility_filter = merged["atr_pct"].between(settings.min_atr_pct, settings.max_atr_pct)
    bars_filter = merged["bars_from_open"] >= settings.bear_min_bars_from_open

    bounce_context = merged["bounce_pct"].between(settings.bear_min_bounce_pct, settings.bear_max_bounce_pct)
    rsi_context = merged["recent_rsi_peak"] >= settings.bear_min_rsi_peak
    retest_context = (
        (merged["high"] >= merged["session_vwap"] * (1 - settings.bear_vwap_reclaim_buffer))
        | (merged["ema_gap"] >= settings.bear_ema_extension_threshold)
        | (merged["high"] >= merged["prev_high"])
    )
    rejection_bar = (
        bearish_bar
        & (merged["close"] < merged["ema_entry"])
        & (
            (merged["upper_wick_ratio"] >= settings.bear_min_rejection_wick_ratio)
            | (merged["high"] >= merged["session_vwap"] * (1 - settings.bear_vwap_reclaim_buffer))
            | (merged["high"] >= merged["prev_high"])
        )
        & ((merged["close"] < merged["breakdown_low"]) | (merged["low"] < merged["prev_low"]))
    )

    merged["long_regime"] = (
        (merged["htf_close"] > merged["htf_ema_fast"])
        & (merged["htf_ema_fast"] > merged["htf_ema_slow"])
        & (merged["htf_adx"] >= settings.htf_adx_threshold)
        & (merged["ema_entry"] > merged["ema_slow"])
        & (merged["close"] > merged["session_vwap"])
        & bars_filter
    )
    merged["short_regime"] = (
        (merged["htf_close"] < merged["htf_ema_fast"])
        & (merged["htf_ema_fast"] < merged["htf_ema_slow"])
        & (merged["htf_adx"] >= settings.htf_adx_threshold)
        & (merged["ema_entry"] < merged["ema_slow"])
        & bars_filter
    )
    merged["long_signal"] = False
    merged["short_signal"] = (
        settings.allow_short
        & merged["short_regime"]
        & bounce_context.fillna(False)
        & rsi_context.fillna(False)
        & retest_context.fillna(False)
        & rejection_bar.fillna(False)
        & merged["rsi"].between(settings.bear_entry_rsi_min, settings.bear_entry_rsi_max)
        & volume_filter
        & volatility_filter
    )

    merged["exit_long_signal"] = False
    merged["exit_short_signal"] = (
        merged["long_regime"]
        | (
            bullish_bar
            & (
                (merged["close"] > merged["session_vwap"])
                | (merged["close"] > merged["ema_entry"])
                | (merged["close"] > merged["cover_high"])
            )
            & (merged["rsi"] > 54)
        )
        | (
            (merged["rsi"] <= settings.bear_take_profit_rsi)
            & bullish_bar
            & (merged["close"] <= merged["session_low"] * (1 + settings.bear_vwap_reclaim_buffer))
        )
    ).fillna(False)

    return merged.dropna()


def prepare_funding_oi_bear_short_data(frame: pd.DataFrame, settings: StrategySettings) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("K 线数据为空。")
    required = {"funding_rate", "oi_value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"增强版空头策略缺少因子列: {sorted(missing)}")

    data = frame.copy().sort_index()
    data.index.name = "timestamp"
    data["ema_entry"] = ema(data["close"], settings.ltf_ema)
    data["rsi"] = rsi(data["close"], settings.rsi_period)
    data["atr"] = atr(data, settings.atr_period)
    data["atr_pct"] = data["atr"] / data["close"]
    data["volume_sma"] = data["volume"].rolling(settings.volume_sma).mean()
    data["volume_ratio"] = (data["volume"] / data["volume_sma"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    data["prev_high"] = data["high"].shift(1)
    data["prev_low"] = data["low"].shift(1)
    data["swing_low"] = data["low"].rolling(settings.swing_lookback).min().shift(1)
    data["swing_high"] = data["high"].rolling(settings.swing_lookback).max().shift(1)
    data["funding_rate"] = pd.to_numeric(data["funding_rate"], errors="coerce").ffill()
    data["oi_value"] = pd.to_numeric(data["oi_value"], errors="coerce").ffill()
    if "oi_contracts" in data.columns:
        data["oi_contracts"] = pd.to_numeric(data["oi_contracts"], errors="coerce").ffill()

    funding_lookback = max(int(settings.enhanced_funding_lookback), 1)
    oi_lookback = max(int(settings.enhanced_oi_lookback), 1)
    price_lookback = max(int(settings.enhanced_price_lookback), 1)
    bounce_window = max(price_lookback, settings.swing_lookback)

    data["funding_change"] = data["funding_rate"] - data["funding_rate"].shift(funding_lookback)
    data["oi_value_change_pct"] = data["oi_value"].pct_change(oi_lookback)
    data["price_change_pct"] = data["close"].pct_change(price_lookback)
    data["bounce_high"] = data["high"].rolling(bounce_window).max().shift(1)
    data["bounce_low"] = data["low"].rolling(bounce_window).min().shift(1)
    data["bounce_pct"] = (data["bounce_high"] / data["bounce_low"]) - 1
    data["oi_value_zscore"] = (
        (data["oi_value"] - data["oi_value"].rolling(oi_lookback * 3).mean())
        / data["oi_value"].rolling(oi_lookback * 3).std().replace(0, np.nan)
    )

    htf = resample_ohlcv(data, settings.higher_timeframe)
    htf["htf_close"] = htf["close"]
    htf["htf_ema_fast"] = ema(htf["close"], settings.htf_fast_ema)
    htf["htf_ema_slow"] = ema(htf["close"], settings.htf_slow_ema)
    htf["htf_adx"] = adx(htf, settings.htf_adx_period)

    merged = pd.merge_asof(
        data.reset_index().sort_values("timestamp"),
        htf[["htf_close", "htf_ema_fast", "htf_ema_slow", "htf_adx"]]
        .reset_index()
        .sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    volatility_filter = merged["atr_pct"].between(settings.min_atr_pct, settings.max_atr_pct)
    volume_filter = merged["volume_ratio"] >= settings.min_volume_ratio
    bounce_filter = merged["bounce_pct"].between(
        settings.enhanced_min_price_bounce_pct,
        settings.enhanced_max_price_bounce_pct,
    )

    merged["long_regime"] = (
        (merged["htf_close"] > merged["htf_ema_fast"])
        & (merged["htf_ema_fast"] > merged["htf_ema_slow"])
        & (merged["htf_adx"] >= settings.htf_adx_threshold)
    )
    merged["short_regime"] = (
        (merged["htf_close"] < merged["htf_ema_fast"])
        & (merged["htf_ema_fast"] < merged["htf_ema_slow"])
        & (merged["htf_adx"] >= settings.htf_adx_threshold)
    )

    pullback_short = (
        (merged["high"] >= merged["ema_entry"] - (merged["atr"] * settings.pullback_atr_tolerance))
        & (merged["close"] < merged["ema_entry"])
    )
    short_breakdown = merged["close"] < merged["prev_low"]
    bearish_bar = merged["close"] < merged["open"]

    funding_level_ok = merged["funding_rate"] >= settings.enhanced_min_funding_rate
    funding_change_ok = merged["funding_change"] >= settings.enhanced_min_funding_change
    oi_change_ok = merged["oi_value_change_pct"] >= settings.enhanced_min_oi_change_pct
    bounce_ok = bounce_filter.fillna(False)
    factor_score = (
        funding_level_ok.fillna(False).astype(int)
        + funding_change_ok.fillna(False).astype(int)
        + oi_change_ok.fillna(False).astype(int)
        + bounce_ok.astype(int)
    )
    merged["factor_score"] = factor_score
    merged["risk_multiplier"] = (
        1.0 + (merged["factor_score"] * settings.enhanced_risk_step)
    ).clip(upper=settings.enhanced_max_risk_multiplier)

    crowded_long_filter = merged["factor_score"] >= int(settings.enhanced_entry_min_score)

    merged["long_signal"] = False
    merged["short_signal"] = (
        settings.allow_short
        & merged["short_regime"]
        & pullback_short
        & short_breakdown
        & bearish_bar
        & merged["rsi"].between(settings.rsi_short_min, settings.rsi_short_max)
        & crowded_long_filter.fillna(False)
        & volume_filter
        & volatility_filter
    )
    merged["exit_long_signal"] = False
    merged["exit_short_signal"] = (
        merged["long_regime"]
        | ((merged["close"] > merged["ema_entry"]) & (merged["rsi"] > 55))
        | (merged["funding_rate"] <= settings.enhanced_exit_funding_rate)
        | (merged["oi_value_change_pct"] <= settings.enhanced_exit_oi_change_pct)
    ).fillna(False)

    return merged.dropna()


def latest_trend_snapshot(frame: pd.DataFrame, settings: StrategySettings) -> dict[str, Any]:
    row = frame.iloc[-1]
    if row["long_signal"]:
        action = "LONG_SETUP"
        stop = suggested_stop(row, "long", settings)
    elif row["short_signal"]:
        action = "SHORT_SETUP"
        stop = suggested_stop(row, "short", settings)
    elif row["long_regime"]:
        action = "LONG_BIAS_WAIT_PULLBACK"
        stop = None
    elif row["short_regime"]:
        action = "SHORT_BIAS_WAIT_PULLBACK"
        stop = None
    else:
        action = "NO_TRADE_ZONE"
        stop = None

    return {
        "strategy_kind": settings.strategy_kind,
        "timestamp": frame.index[-1].isoformat(),
        "action": action,
        "close": round(float(row["close"]), 4),
        "ema_entry": round(float(row["ema_entry"]), 4),
        "atr": round(float(row["atr"]), 4),
        "rsi": round(float(row["rsi"]), 2),
        "htf_adx": round(float(row["htf_adx"]), 2),
        "stop_hint": None if stop is None else round(float(stop), 4),
    }


def latest_smallcap_short_snapshot(frame: pd.DataFrame, settings: StrategySettings) -> dict[str, Any]:
    row = frame.iloc[-1]
    if row["short_signal"]:
        action = "SMALLCAP_PUMP_SHORT_SETUP"
        stop = suggested_stop(row, "short", settings)
    elif row["short_regime"]:
        action = "SMALLCAP_PUMP_WATCH"
        stop = None
    else:
        action = "NO_TRADE_ZONE"
        stop = None

    return {
        "strategy_kind": settings.strategy_kind,
        "timestamp": frame.index[-1].isoformat(),
        "action": action,
        "close": round(float(row["close"]), 6),
        "ema_entry": round(float(row["ema_entry"]), 6),
        "session_vwap": round(float(row["session_vwap"]), 6),
        "atr": round(float(row["atr"]), 6),
        "rsi": round(float(row["rsi"]), 2),
        "day_return_pct": round(float(row["day_return"] * 100), 2),
        "session_peak_return_pct": round(float(row["session_peak_return"] * 100), 2),
        "volume_ratio": round(float(row["volume_ratio"]), 2),
        "vwap_gap_pct": round(float(row["vwap_gap"] * 100), 2),
        "stop_hint": None if stop is None else round(float(stop), 6),
    }


def latest_top_gainers_boll_snapshot(frame: pd.DataFrame, settings: StrategySettings) -> dict[str, Any]:
    row = frame.iloc[-1]
    if row["short_signal"]:
        action = "TOP_GAINERS_BOLL_SHORT_SETUP"
        stop = suggested_stop(row, "short", settings)
    elif row["short_regime"]:
        action = "TOP_GAINERS_BOLL_SHORT_WATCH"
        stop = None
    else:
        action = "NO_TRADE_ZONE"
        stop = None

    return {
        "strategy_kind": settings.strategy_kind,
        "timestamp": frame.index[-1].isoformat(),
        "action": action,
        "close": round(float(row["close"]), 4),
        "bb_upper": round(float(row["bb_upper"]), 4),
        "bb_mid": round(float(row["bb_mid"]), 4),
        "htf_bb_upper": round(float(row["htf_bb_upper"]), 4),
        "atr": round(float(row["atr"]), 4),
        "rsi": round(float(row["rsi"]), 2),
        "volume_ratio": round(float(row["volume_ratio"]), 2),
        "daily_return_pct": round(float(row["daily_return_pct"] * 100), 2),
        "daily_rank": int(row["daily_rank"]),
        "factor_score": int(row["factor_score"]),
        "stop_hint": None if stop is None else round(float(stop), 4),
    }


def latest_top_gainers_pump_dump_snapshot(frame: pd.DataFrame, settings: StrategySettings) -> dict[str, Any]:
    row = frame.iloc[-1]
    if row["long_signal"]:
        action = "TOP_GAINERS_PUMP_LONG_SETUP"
        stop = suggested_stop(row, "long", settings)
    elif row["short_signal"]:
        action = "TOP_GAINERS_DUMP_SHORT_SETUP"
        stop = suggested_stop(row, "short", settings)
    elif row["long_regime"]:
        action = "TOP_GAINERS_PUMP_WATCH"
        stop = None
    elif row["short_regime"]:
        action = "TOP_GAINERS_DUMP_WATCH"
        stop = None
    else:
        action = "NO_TRADE_ZONE"
        stop = None

    return {
        "strategy_kind": settings.strategy_kind,
        "timestamp": frame.index[-1].isoformat(),
        "action": action,
        "close": round(float(row["close"]), 4),
        "daily_return_pct": round(float(row["daily_return_pct"] * 100), 2),
        "daily_rank": int(row["daily_rank"]),
        "volume_ratio": round(float(row["volume_ratio"]), 2),
        "trade_ratio": round(float(row.get("trade_ratio", 0.0)), 2),
        "delta_ratio": round(float(row.get("delta_ratio", 0.0)), 3),
        "vwap_gap_pct": round(float(row.get("vwap_gap", 0.0) * 100), 2),
        "factor_score": int(row["factor_score"]),
        "stop_hint": None if stop is None else round(float(stop), 4),
    }


def latest_reference_volume_snapshot(frame: pd.DataFrame, settings: StrategySettings) -> dict[str, Any]:
    row = frame.iloc[-1]
    if row["long_signal"]:
        action = "REFERENCE_1M_LONG_SETUP"
        stop = suggested_stop(row, "long", settings)
    elif row["short_signal"]:
        action = "REFERENCE_1M_SHORT_SETUP"
        stop = suggested_stop(row, "short", settings)
    elif row["long_context"]:
        action = "REFERENCE_1M_LONG_WATCH"
        stop = None
    elif row["short_context"]:
        action = "REFERENCE_1M_SHORT_WATCH"
        stop = None
    else:
        action = "NO_TRADE_ZONE"
        stop = None

    return {
        "strategy_kind": settings.strategy_kind,
        "timestamp": frame.index[-1].isoformat(),
        "action": action,
        "close": round(float(row["close"]), 4),
        "ema_entry": round(float(row["ema_entry"]), 4),
        "ema_slow": round(float(row["ema_slow"]), 4),
        "session_vwap": round(float(row["session_vwap"]), 4),
        "atr": round(float(row["atr"]), 4),
        "rsi": round(float(row["rsi"]), 2),
        "volume_ratio": round(float(row["volume_ratio"]), 2),
        "body_pct": round(float(row["body_pct"] * 100), 3),
        "vwap_gap_pct": round(float(row["vwap_gap"] * 100), 3),
        "buy_pressure_ratio": round(float(row["buy_pressure_ratio"]), 3),
        "stop_hint": None if stop is None else round(float(stop), 4),
    }


def latest_orderflow_snapshot(frame: pd.DataFrame, settings: StrategySettings) -> dict[str, Any]:
    row = frame.iloc[-1]
    if row["long_signal"]:
        action = "ORDERFLOW_LONG_SETUP"
        stop = suggested_stop(row, "long", settings)
    elif row["short_signal"]:
        action = "ORDERFLOW_SHORT_SETUP"
        stop = suggested_stop(row, "short", settings)
    elif row["long_regime"]:
        action = "ORDERFLOW_LONG_BIAS"
        stop = None
    elif row["short_regime"]:
        action = "ORDERFLOW_SHORT_BIAS"
        stop = None
    else:
        action = "NO_TRADE_ZONE"
        stop = None

    return {
        "strategy_kind": settings.strategy_kind,
        "timestamp": frame.index[-1].isoformat(),
        "action": action,
        "close": round(float(row["close"]), 4),
        "ema_entry": round(float(row["ema_entry"]), 4),
        "ema_slow": round(float(row["ema_slow"]), 4),
        "session_vwap": round(float(row["session_vwap"]), 4),
        "atr": round(float(row["atr"]), 4),
        "rsi": round(float(row["rsi"]), 2),
        "quote_volume_ratio": round(float(row["quote_volume_ratio"]), 2),
        "trade_count_ratio": round(float(row["trade_count_ratio"]), 2),
        "delta_ratio": round(float(row["delta_ratio"]), 3),
        "rolling_delta_ratio": round(float(row["rolling_delta_ratio"]), 3),
        "flow_zscore": round(float(row["flow_zscore"]), 2),
        "stop_hint": None if stop is None else round(float(stop), 4),
    }


def latest_majors_ltf_snapshot(frame: pd.DataFrame, settings: StrategySettings) -> dict[str, Any]:
    row = frame.iloc[-1]
    if row["long_signal"]:
        action = "MAJORS_LTF_LONG_SETUP"
        stop = suggested_stop(row, "long", settings)
    elif row["short_signal"]:
        action = "MAJORS_LTF_SHORT_SETUP"
        stop = suggested_stop(row, "short", settings)
    elif row["long_regime"]:
        action = "MAJORS_LTF_LONG_BIAS"
        stop = None
    elif row["short_regime"]:
        action = "MAJORS_LTF_SHORT_BIAS"
        stop = None
    else:
        action = "NO_TRADE_ZONE"
        stop = None

    return {
        "strategy_kind": settings.strategy_kind,
        "timestamp": frame.index[-1].isoformat(),
        "action": action,
        "close": round(float(row["close"]), 4),
        "ema_entry": round(float(row["ema_entry"]), 4),
        "ema_slow": round(float(row["ema_slow"]), 4),
        "session_vwap": round(float(row["session_vwap"]), 4),
        "atr": round(float(row["atr"]), 4),
        "rsi": round(float(row["rsi"]), 2),
        "htf_adx": round(float(row["htf_adx"]), 2),
        "volume_ratio": round(float(row["volume_ratio"]), 2),
        "impulse_range_pct": round(float(row["impulse_range_pct"] * 100), 2),
        "stop_hint": None if stop is None else round(float(stop), 4),
    }


def latest_majors_breakout_snapshot(frame: pd.DataFrame, settings: StrategySettings) -> dict[str, Any]:
    row = frame.iloc[-1]
    if row["long_signal"]:
        action = "MAJORS_BREAKOUT_LONG_SETUP"
        stop = suggested_stop(row, "long", settings)
    elif row["short_signal"]:
        action = "MAJORS_BREAKOUT_SHORT_SETUP"
        stop = suggested_stop(row, "short", settings)
    elif row["recent_squeeze"]:
        action = "MAJORS_BREAKOUT_SQUEEZE_WATCH"
        stop = None
    else:
        action = "NO_TRADE_ZONE"
        stop = None

    return {
        "strategy_kind": settings.strategy_kind,
        "timestamp": frame.index[-1].isoformat(),
        "action": action,
        "close": round(float(row["close"]), 4),
        "ema_entry": round(float(row["ema_entry"]), 4),
        "ema_slow": round(float(row["ema_slow"]), 4),
        "atr": round(float(row["atr"]), 4),
        "rsi": round(float(row["rsi"]), 2),
        "htf_adx": round(float(row["htf_adx"]), 2),
        "volume_ratio": round(float(row["volume_ratio"]), 2),
        "range_pct": round(float(row["range_pct"] * 100), 2),
        "stop_hint": None if stop is None else round(float(stop), 4),
    }


def latest_majors_bear_rally_short_snapshot(frame: pd.DataFrame, settings: StrategySettings) -> dict[str, Any]:
    row = frame.iloc[-1]
    if row["short_signal"]:
        action = "MAJORS_BEAR_RALLY_SHORT_SETUP"
        stop = suggested_stop(row, "short", settings)
    elif row["short_regime"]:
        action = "MAJORS_BEAR_RALLY_SHORT_BIAS"
        stop = None
    else:
        action = "NO_TRADE_ZONE"
        stop = None

    return {
        "strategy_kind": settings.strategy_kind,
        "timestamp": frame.index[-1].isoformat(),
        "action": action,
        "close": round(float(row["close"]), 4),
        "ema_entry": round(float(row["ema_entry"]), 4),
        "ema_slow": round(float(row["ema_slow"]), 4),
        "session_vwap": round(float(row["session_vwap"]), 4),
        "atr": round(float(row["atr"]), 4),
        "rsi": round(float(row["rsi"]), 2),
        "htf_adx": round(float(row["htf_adx"]), 2),
        "volume_ratio": round(float(row["volume_ratio"]), 2),
        "bounce_pct": round(float(row["bounce_pct"] * 100), 2),
        "recent_rsi_peak": round(float(row["recent_rsi_peak"]), 2),
        "stop_hint": None if stop is None else round(float(stop), 4),
    }


def latest_funding_oi_bear_short_snapshot(frame: pd.DataFrame, settings: StrategySettings) -> dict[str, Any]:
    row = frame.iloc[-1]
    if row["short_signal"]:
        action = "FUNDING_OI_BEAR_SHORT_SETUP"
        stop = suggested_stop(row, "short", settings)
    elif row["short_regime"]:
        action = "FUNDING_OI_BEAR_SHORT_BIAS"
        stop = None
    else:
        action = "NO_TRADE_ZONE"
        stop = None

    return {
        "strategy_kind": settings.strategy_kind,
        "timestamp": frame.index[-1].isoformat(),
        "action": action,
        "close": round(float(row["close"]), 4),
        "ema_entry": round(float(row["ema_entry"]), 4),
        "atr": round(float(row["atr"]), 4),
        "rsi": round(float(row["rsi"]), 2),
        "htf_adx": round(float(row["htf_adx"]), 2),
        "funding_rate": round(float(row["funding_rate"]), 8),
        "funding_change": round(float(row["funding_change"]), 8),
        "oi_value_change_pct": round(float(row["oi_value_change_pct"] * 100), 2),
        "price_change_pct": round(float(row["price_change_pct"] * 100), 2),
        "bounce_pct": round(float(row["bounce_pct"] * 100), 2),
        "factor_score": int(row["factor_score"]),
        "risk_multiplier": round(float(row["risk_multiplier"]), 2),
        "stop_hint": None if stop is None else round(float(stop), 4),
    }


def suggested_stop(row: pd.Series, side: str, settings: StrategySettings) -> float | None:
    atr_stop = row["close"] - (row["atr"] * settings.atr_stop_mult) if side == "long" else row["close"] + (
        row["atr"] * settings.atr_stop_mult
    )
    if side == "long":
        if pd.notna(row["swing_low"]):
            return min(float(row["swing_low"]), float(atr_stop))
        return float(atr_stop)
    if pd.notna(row["swing_high"]):
        return max(float(row["swing_high"]), float(atr_stop))
    return float(atr_stop)


def resample_ohlcv(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = to_pandas_rule(timeframe)
    resampled = frame.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()
    resampled.index.name = "timestamp"
    return resampled


def to_pandas_rule(timeframe: str) -> str:
    mapping = {
        "m": "min",
        "h": "h",
        "d": "D",
        "w": "W",
    }
    suffix = timeframe[-1].lower()
    if suffix not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return f"{timeframe[:-1]}{mapping[suffix]}"
