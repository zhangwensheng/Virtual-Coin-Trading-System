from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TRAIN_END = pd.Timestamp("2026-03-01", tz="UTC")
OOS_START = pd.Timestamp("2026-03-01", tz="UTC")


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    min_return: float
    max_return: float
    min_volume_ratio: float
    min_sell_volume_ratio: float
    min_down_hour_ratio: float
    min_late_sell_ratio: float
    max_close_location: float
    min_upper_wick: float
    daily_position: str
    market_filter: str
    hold_days: int
    stop_atr: float
    tp_atr: float


@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    signal_date: pd.Timestamp
    entry_price: float
    qty: float
    stop_price: float
    tp_price: float
    entry_fee: float
    variant: str
    family: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research daily short direction from daily shape + intraday sell pressure.")
    parser.add_argument("--train-dir", default="data/binance_universal_short_2y_1h")
    parser.add_argument("--oos-dir", default="data/binance_universal_short_oos_1h")
    parser.add_argument("--output-dir", default="outputs/daily_short_pressure_research")
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.006)
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--max-notional-fraction", type=float, default=0.12)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    return parser.parse_args()


def symbol_from_path(path: Path) -> str:
    return path.name.split("_1h_")[0].upper()


def load_hourly_data(train_dir: Path, oos_dir: Path) -> dict[str, pd.DataFrame]:
    by_symbol: dict[str, list[pd.DataFrame]] = {}
    for directory in [train_dir, oos_dir]:
        if not directory.exists():
            continue
        for path in directory.glob("*_1h_*.csv"):
            symbol = symbol_from_path(path)
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.set_index("timestamp").sort_index()
            for column in ["open", "high", "low", "close", "volume"]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
            by_symbol.setdefault(symbol, []).append(frame[["open", "high", "low", "close", "volume"]])

    merged: dict[str, pd.DataFrame] = {}
    for symbol, frames in by_symbol.items():
        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        if len(combined) >= 24 * 60:
            merged[symbol] = combined
    return merged


def prepare_daily_features(hourly: pd.DataFrame) -> pd.DataFrame:
    data = hourly.copy()
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

    daily["day_return"] = (daily["close"] / daily["open"]) - 1.0
    daily["range_pct"] = (daily["high"] / daily["low"]) - 1.0
    candle_range = (daily["high"] - daily["low"]).replace(0, np.nan)
    daily["close_location"] = (daily["close"] - daily["low"]) / candle_range
    daily["upper_wick_ratio"] = (daily["high"] - daily[["open", "close"]].max(axis=1)) / candle_range
    daily["lower_wick_ratio"] = (daily[["open", "close"]].min(axis=1) - daily["low"]) / candle_range
    daily["sell_volume_ratio"] = daily["down_volume"] / daily["volume"].replace(0, np.nan)
    daily["strong_sell_volume_ratio"] = daily["strong_down_volume"] / daily["volume"].replace(0, np.nan)
    daily["down_hour_ratio"] = daily["down_hours"] / daily["bar_count"].replace(0, np.nan)
    daily["late_sell_ratio"] = daily["late_down_volume"] / daily["volume"].replace(0, np.nan)
    daily["volume_ratio"] = daily["volume"] / daily["volume"].shift(1).rolling(20).mean()
    daily["ema20"] = daily["close"].ewm(span=20, adjust=False).mean()
    daily["ema50"] = daily["close"].ewm(span=50, adjust=False).mean()

    prev_close = daily["close"].shift(1)
    bb_mid = prev_close.rolling(20).mean()
    bb_std = prev_close.rolling(20).std(ddof=0)
    daily["bb_mid_prev"] = bb_mid
    daily["bb_upper_prev"] = bb_mid + (bb_std * 2.0)
    daily["bb_lower_prev"] = bb_mid - (bb_std * 2.0)
    daily["bb_z_prev"] = (daily["close"] - daily["bb_mid_prev"]) / (bb_std.replace(0, np.nan))

    prev = daily["close"].shift(1)
    tr = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - prev).abs(),
            (daily["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    daily["atr"] = tr.rolling(14).mean()
    daily["atr_pct"] = daily["atr"] / daily["close"]

    daily["future_short_1d"] = (daily["open"].shift(-1) - daily["close"].shift(-1)) / daily["open"].shift(-1)
    daily["future_short_3d"] = (daily["open"].shift(-1) - daily["close"].shift(-3)) / daily["open"].shift(-1)
    return daily.replace([np.inf, -np.inf], np.nan).dropna(subset=["volume_ratio", "atr", "bb_upper_prev"])


def build_variants() -> list[Variant]:
    variants: list[Variant] = []
    market_filters = ["none", "btc_below_ema20", "btc_below_ema50", "btc_ema_bear"]
    for min_return in [0.015, 0.03, 0.05]:
        for min_volume_ratio in [1.05, 1.25]:
            for min_sell in [0.50, 0.54, 0.58]:
                for close_location in [0.45, 0.60]:
                    for market_filter in ["none", "btc_below_ema20"]:
                        name = (
                            f"pump_dist_r{min_return:.3f}_v{min_volume_ratio:.2f}_s{min_sell:.2f}"
                            f"_cl{close_location:.2f}_m{market_filter}"
                        )
                        variants.append(
                            Variant(
                                name=name,
                                family="pump_distribution",
                                min_return=min_return,
                                max_return=0.45,
                                min_volume_ratio=min_volume_ratio,
                                min_sell_volume_ratio=min_sell,
                                min_down_hour_ratio=0.46,
                                min_late_sell_ratio=0.12,
                                max_close_location=close_location,
                                min_upper_wick=0.12,
                                daily_position="high_above_upper",
                                market_filter=market_filter,
                                hold_days=3,
                                stop_atr=1.0,
                                tp_atr=1.6,
                            )
                        )

    for min_sell in [0.52, 0.56, 0.60]:
        for wick in [0.18, 0.28]:
            for market_filter in ["none", "btc_below_ema20"]:
                variants.append(
                    Variant(
                        name=f"failed_upper_s{min_sell:.2f}_w{wick:.2f}_m{market_filter}",
                        family="failed_upper_breakout",
                        min_return=0.005,
                        max_return=0.35,
                        min_volume_ratio=1.0,
                        min_sell_volume_ratio=min_sell,
                        min_down_hour_ratio=0.46,
                        min_late_sell_ratio=0.10,
                        max_close_location=0.58,
                        min_upper_wick=wick,
                        daily_position="failed_upper",
                        market_filter=market_filter,
                        hold_days=3,
                        stop_atr=0.9,
                        tp_atr=1.5,
                    )
                )

    for min_sell in [0.54, 0.58, 0.62]:
        for close_location in [0.35, 0.45]:
            for market_filter in market_filters:
                variants.append(
                    Variant(
                        name=f"bear_cont_s{min_sell:.2f}_cl{close_location:.2f}_m{market_filter}",
                        family="bear_continuation",
                        min_return=-0.99,
                        max_return=0.04,
                        min_volume_ratio=1.05,
                        min_sell_volume_ratio=min_sell,
                        min_down_hour_ratio=0.50,
                        min_late_sell_ratio=0.10,
                        max_close_location=close_location,
                        min_upper_wick=0.0,
                        daily_position="below_ema20",
                        market_filter=market_filter,
                        hold_days=2,
                        stop_atr=1.1,
                        tp_atr=1.4,
                    )
                )
    return variants


def market_condition(daily: pd.DataFrame, market_daily: pd.DataFrame | None, variant: Variant) -> pd.Series:
    if variant.market_filter == "none" or market_daily is None or market_daily.empty:
        return pd.Series(True, index=daily.index)
    market = market_daily.reindex(daily.index).ffill()
    if variant.market_filter == "btc_below_ema20":
        return (market["close"] < market["ema20"]).fillna(False)
    if variant.market_filter == "btc_below_ema50":
        return (market["close"] < market["ema50"]).fillna(False)
    if variant.market_filter == "btc_ema_bear":
        return (market["ema20"] < market["ema50"]).fillna(False)
    raise ValueError(f"Unsupported market_filter: {variant.market_filter}")


def signal_mask(daily: pd.DataFrame, variant: Variant, market_daily: pd.DataFrame | None = None) -> pd.Series:
    base = (
        (daily["day_return"] >= variant.min_return)
        & (daily["day_return"] <= variant.max_return)
        & (daily["volume_ratio"] >= variant.min_volume_ratio)
        & (daily["sell_volume_ratio"] >= variant.min_sell_volume_ratio)
        & (daily["down_hour_ratio"] >= variant.min_down_hour_ratio)
        & (daily["late_sell_ratio"] >= variant.min_late_sell_ratio)
        & (daily["close_location"] <= variant.max_close_location)
        & (daily["upper_wick_ratio"] >= variant.min_upper_wick)
        & (daily["atr_pct"].between(0.008, 0.18))
    )

    if variant.daily_position == "high_above_upper":
        position = (daily["high"] >= daily["bb_upper_prev"]) & (daily["close"] > daily["bb_mid_prev"])
    elif variant.daily_position == "failed_upper":
        position = (daily["high"] >= daily["bb_upper_prev"]) & (
            (daily["close"] < daily["bb_upper_prev"]) | (daily["close_location"] <= 0.55)
        )
    elif variant.daily_position == "below_ema20":
        position = (daily["close"] < daily["ema20"]) & (daily["close"] < daily["open"])
    else:
        raise ValueError(f"Unsupported daily_position: {variant.daily_position}")
    return (base & position & market_condition(daily, market_daily, variant)).fillna(False)


def build_signal_events(
    daily_by_symbol: dict[str, pd.DataFrame],
    variants: list[Variant],
) -> dict[str, pd.DataFrame]:
    events_by_variant: dict[str, list[dict[str, Any]]] = {variant.name: [] for variant in variants}
    market_daily = daily_by_symbol.get("BTCUSDT")
    for symbol, daily in daily_by_symbol.items():
        for variant in variants:
            mask = signal_mask(daily, variant, market_daily)
            signal_days = daily.index[mask]
            for signal_date in signal_days:
                loc = daily.index.get_loc(signal_date)
                if not isinstance(loc, int) or loc + 1 >= len(daily):
                    continue
                entry_date = daily.index[loc + 1]
                row = daily.loc[signal_date]
                next_row = daily.iloc[loc + 1]
                if not np.isfinite(float(row["atr"])) or not np.isfinite(float(next_row["open"])):
                    continue
                score = (
                    max(0.0, float(row["day_return"])) * 100.0
                    + max(0.0, float(row["volume_ratio"]) - 1.0) * 8.0
                    + max(0.0, float(row["sell_volume_ratio"]) - 0.5) * 30.0
                    + float(row["upper_wick_ratio"]) * 10.0
                    + max(0.0, 1.0 - float(row["close_location"])) * 4.0
                )
                events_by_variant[variant.name].append(
                    {
                        "variant": variant.name,
                        "family": variant.family,
                        "symbol": symbol,
                        "signal_date": signal_date,
                        "entry_date": entry_date,
                        "score": score,
                        "entry_open": float(next_row["open"]),
                        "atr": float(row["atr"]),
                        "day_return": float(row["day_return"]),
                        "volume_ratio": float(row["volume_ratio"]),
                        "sell_volume_ratio": float(row["sell_volume_ratio"]),
                        "down_hour_ratio": float(row["down_hour_ratio"]),
                        "late_sell_ratio": float(row["late_sell_ratio"]),
                        "close_location": float(row["close_location"]),
                        "upper_wick_ratio": float(row["upper_wick_ratio"]),
                        "bb_z_prev": float(row["bb_z_prev"]),
                        "future_short_1d": float(row.get("future_short_1d", np.nan)),
                        "future_short_3d": float(row.get("future_short_3d", np.nan)),
                    }
                )
    return {name: pd.DataFrame(rows) for name, rows in events_by_variant.items()}


def slippage_price(side: str, price: float, is_entry: bool, slippage_bps: float) -> float:
    slip = slippage_bps / 10_000.0
    if side != "short":
        raise ValueError("This research script is short-only.")
    return price * (1 - slip) if is_entry else price * (1 + slip)


def backtest_portfolio(
    events: pd.DataFrame,
    daily_by_symbol: dict[str, pd.DataFrame],
    variant: Variant,
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    initial_capital: float,
    risk_per_trade: float,
    leverage: float,
    max_notional_fraction: float,
    max_positions: int,
    fee_rate: float,
    slippage_bps: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if events.empty:
        empty_equity = pd.DataFrame(columns=["timestamp", "equity", "drawdown_pct"]).set_index("timestamp")
        return summarize_trades(pd.DataFrame(), empty_equity, initial_capital), pd.DataFrame(), empty_equity

    events = events.copy()
    if start is not None:
        events = events[events["entry_date"] >= start]
    if end is not None:
        events = events[events["entry_date"] < end]
    if events.empty:
        empty_equity = pd.DataFrame(columns=["timestamp", "equity", "drawdown_pct"]).set_index("timestamp")
        return summarize_trades(pd.DataFrame(), empty_equity, initial_capital), pd.DataFrame(), empty_equity

    grouped_events = {date: frame.sort_values("score", ascending=False) for date, frame in events.groupby("entry_date")}
    all_dates = sorted(set(pd.concat([frame.index.to_series() for frame in daily_by_symbol.values()]).unique()))
    min_date = events["entry_date"].min()
    max_date = events["entry_date"].max() + pd.Timedelta(days=variant.hold_days + 2)
    dates = [date for date in all_dates if date >= min_date and date <= max_date]

    cash = initial_capital
    peak_equity = initial_capital
    positions: list[Position] = []
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    for current_date in dates:
        today_events = grouped_events.get(current_date)
        if today_events is not None:
            for _, event in today_events.iterrows():
                if len(positions) >= max_positions:
                    break
                symbol = str(event["symbol"])
                if any(position.symbol == symbol for position in positions):
                    continue
                daily = daily_by_symbol.get(symbol)
                if daily is None or current_date not in daily.index:
                    continue
                open_price = float(daily.loc[current_date, "open"])
                if not math.isfinite(open_price) or open_price <= 0:
                    continue
                entry = slippage_price("short", open_price, is_entry=True, slippage_bps=slippage_bps)
                stop_distance = max(float(event["atr"]) * variant.stop_atr, entry * 0.006)
                stop_price = entry + stop_distance
                tp_price = max(entry - (float(event["atr"]) * variant.tp_atr), entry * 0.75)
                risk_amount = cash * risk_per_trade
                risk_qty = risk_amount / stop_distance
                max_notional = cash * leverage * max_notional_fraction
                qty = min(risk_qty, max_notional / entry)
                if qty <= 0:
                    continue
                entry_fee = entry * qty * fee_rate
                cash -= entry_fee
                positions.append(
                    Position(
                        symbol=symbol,
                        entry_date=current_date,
                        signal_date=event["signal_date"],
                        entry_price=entry,
                        qty=qty,
                        stop_price=stop_price,
                        tp_price=tp_price,
                        entry_fee=entry_fee,
                        variant=variant.name,
                        family=variant.family,
                    )
                )

        still_open: list[Position] = []
        for position in positions:
            daily = daily_by_symbol.get(position.symbol)
            if daily is None or current_date not in daily.index:
                still_open.append(position)
                continue
            bar = daily.loc[current_date]
            exit_reason = ""
            exit_price: float | None = None
            if float(bar["high"]) >= position.stop_price:
                exit_reason = "stop_loss"
                exit_price = slippage_price("short", position.stop_price, is_entry=False, slippage_bps=slippage_bps)
            elif float(bar["low"]) <= position.tp_price:
                exit_reason = "take_profit"
                exit_price = slippage_price("short", position.tp_price, is_entry=False, slippage_bps=slippage_bps)
            elif (current_date - position.entry_date).days + 1 >= variant.hold_days:
                exit_reason = "time_exit"
                exit_price = slippage_price("short", float(bar["close"]), is_entry=False, slippage_bps=slippage_bps)

            if exit_price is None:
                still_open.append(position)
                continue

            gross = (position.entry_price - exit_price) * position.qty
            exit_fee = exit_price * position.qty * fee_rate
            pnl = gross - exit_fee - position.entry_fee
            cash += gross - exit_fee
            trades.append(
                {
                    "variant": position.variant,
                    "family": position.family,
                    "symbol": position.symbol,
                    "signal_date": position.signal_date.isoformat(),
                    "entry_date": position.entry_date.isoformat(),
                    "exit_date": current_date.isoformat(),
                    "entry_price": round(position.entry_price, 8),
                    "exit_price": round(exit_price, 8),
                    "qty": round(position.qty, 8),
                    "pnl": round(pnl, 4),
                    "pnl_pct_entry_notional": round(pnl / (position.entry_price * position.qty) * 100.0, 4),
                    "bars_held": (current_date - position.entry_date).days + 1,
                    "exit_reason": exit_reason,
                }
            )
        positions = still_open

        equity = cash
        for position in positions:
            daily = daily_by_symbol.get(position.symbol)
            if daily is None or current_date not in daily.index:
                continue
            mark = float(daily.loc[current_date, "close"])
            equity += (position.entry_price - mark) * position.qty
        peak_equity = max(peak_equity, equity)
        equity_rows.append({"timestamp": current_date, "equity": equity, "drawdown_pct": (equity / peak_equity) - 1.0})

    trades_frame = pd.DataFrame(trades)
    equity_frame = pd.DataFrame(equity_rows).set_index("timestamp") if equity_rows else pd.DataFrame()
    summary = summarize_trades(trades_frame, equity_frame, initial_capital)
    return summary, trades_frame, equity_frame


def summarize_trades(trades: pd.DataFrame, equity: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    if equity.empty:
        final_equity = initial_capital
        max_dd = 0.0
    else:
        final_equity = float(equity["equity"].iloc[-1])
        max_dd = float(equity["drawdown_pct"].min() * 100.0)
    if trades.empty:
        return {
            "return_pct": round(((final_equity / initial_capital) - 1.0) * 100.0, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
            "final_equity": round(final_equity, 2),
        }
    winners = trades[trades["pnl"] > 0]
    losers = trades[trades["pnl"] <= 0]
    gross_win = float(winners["pnl"].sum())
    gross_loss = float(losers["pnl"].sum())
    profit_factor = gross_win / abs(gross_loss) if gross_loss else 99.0
    return {
        "return_pct": round(((final_equity / initial_capital) - 1.0) * 100.0, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": int(len(trades)),
        "win_rate_pct": round((len(winners) / len(trades)) * 100.0, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_trade_pnl": round(float(trades["pnl"].mean()), 4),
        "final_equity": round(final_equity, 2),
    }


def monthly_stats(equity: pd.DataFrame) -> dict[str, Any]:
    if equity.empty:
        return {"active_months": 0, "positive_months": 0, "positive_month_ratio": 0.0, "worst_month_pct": 0.0}
    monthly_last = equity["equity"].resample("ME").last().dropna()
    monthly_ret = monthly_last.pct_change().dropna() * 100.0
    if monthly_ret.empty:
        total_ret = (monthly_last.iloc[-1] / equity["equity"].iloc[0] - 1.0) * 100.0
        monthly_ret = pd.Series([total_ret], index=[monthly_last.index[-1]])
    positive = monthly_ret[monthly_ret > 0]
    return {
        "active_months": int(len(monthly_ret)),
        "positive_months": int(len(positive)),
        "positive_month_ratio": round(len(positive) / len(monthly_ret) * 100.0, 2),
        "worst_month_pct": round(float(monthly_ret.min()), 2),
    }


def direction_stats(events: pd.DataFrame) -> dict[str, Any]:
    if events.empty:
        return {
            "signal_count": 0,
            "future_1d_win_pct": 0.0,
            "avg_future_short_1d_pct": 0.0,
            "future_3d_win_pct": 0.0,
            "avg_future_short_3d_pct": 0.0,
            "symbols_with_signals": 0,
        }
    valid_1d = events["future_short_1d"].replace([np.inf, -np.inf], np.nan).dropna()
    valid_3d = events["future_short_3d"].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "signal_count": int(len(events)),
        "future_1d_win_pct": round(float((valid_1d > 0).mean() * 100.0), 2) if len(valid_1d) else 0.0,
        "avg_future_short_1d_pct": round(float(valid_1d.mean() * 100.0), 3) if len(valid_1d) else 0.0,
        "future_3d_win_pct": round(float((valid_3d > 0).mean() * 100.0), 2) if len(valid_3d) else 0.0,
        "avg_future_short_3d_pct": round(float(valid_3d.mean() * 100.0), 3) if len(valid_3d) else 0.0,
        "symbols_with_signals": int(events["symbol"].nunique()),
    }


def symbol_stats(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"profitable_symbols": 0, "losing_symbols": 0}
    pnl_by_symbol = trades.groupby("symbol")["pnl"].sum()
    return {
        "profitable_symbols": int((pnl_by_symbol > 0).sum()),
        "losing_symbols": int((pnl_by_symbol <= 0).sum()),
    }


def robust_score(row: pd.Series) -> float:
    if int(row["train_total_trades"]) < 25 or int(row["oos_total_trades"]) < 3:
        return -999.0
    dd_penalty = abs(float(row["all_max_drawdown_pct"])) + 1.0
    return (
        float(row["train_return_pct"]) * 0.25
        + float(row["oos_return_pct"]) * 0.9
        + float(row["all_profit_factor"]) * 8.0
        + float(row["positive_month_ratio"]) * 0.12
        + float(row["future_3d_win_pct"]) * 0.08
        - dd_penalty * 0.65
    )


def render_report(output_dir: Path, summary: pd.DataFrame, best_variant: str | None) -> None:
    lines = [
        "# Daily Short Pressure Research",
        "",
        "Signal timing: use finished daily candle plus hourly sell-pressure aggregates, enter short at next daily open.",
        "Execution model: short-only, ATR stop/take-profit, conservative daily OHLC assumption where stop is hit before take-profit if both occur on the same day.",
        "",
    ]
    if best_variant is not None:
        best = summary[summary["variant"] == best_variant].iloc[0]
        lines.extend(
            [
                f"Best robust variant: `{best_variant}`",
                "",
                f"- Train return: {best['train_return_pct']}%, trades: {best['train_total_trades']}, PF: {best['train_profit_factor']}",
                f"- OOS return: {best['oos_return_pct']}%, trades: {best['oos_total_trades']}, PF: {best['oos_profit_factor']}",
                f"- All return: {best['all_return_pct']}%, max DD: {best['all_max_drawdown_pct']}%, win rate: {best['all_win_rate_pct']}%",
                f"- Direction check: 3d short win {best['future_3d_win_pct']}%, avg 3d short move {best['avg_future_short_3d_pct']}%",
                "",
            ]
        )
    lines.append("## Top 15")
    top = summary.sort_values("robust_score", ascending=False).head(15)
    lines.append(top.to_markdown(index=False))
    lines.append("")
    lines.append("## Notes")
    lines.append("- `future_short_*` is pure direction correlation, before trade management.")
    lines.append("- Portfolio results are more conservative because fees, slippage, stops, overlap limits and max positions are applied.")
    lines.append("- If OOS trades are too few, treat the result as a hypothesis, not a production strategy.")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hourly_by_symbol = load_hourly_data(Path(args.train_dir), Path(args.oos_dir))
    if not hourly_by_symbol:
        raise SystemExit("No hourly data found.")

    daily_by_symbol = {symbol: prepare_daily_features(frame) for symbol, frame in hourly_by_symbol.items()}
    daily_by_symbol = {symbol: frame for symbol, frame in daily_by_symbol.items() if len(frame) >= 80}
    variants = build_variants()
    events_by_variant = build_signal_events(daily_by_symbol, variants)
    variant_lookup = {variant.name: variant for variant in variants}

    summary_rows: list[dict[str, Any]] = []
    best_payload: tuple[str, pd.DataFrame, pd.DataFrame] | None = None
    best_score = -9999.0

    for variant in variants:
        events = events_by_variant[variant.name]
        all_summary, all_trades, all_equity = backtest_portfolio(
            events,
            daily_by_symbol,
            variant,
            start=None,
            end=None,
            initial_capital=args.initial_capital,
            risk_per_trade=args.risk_per_trade,
            leverage=args.leverage,
            max_notional_fraction=args.max_notional_fraction,
            max_positions=args.max_positions,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
        )
        train_summary, train_trades, _ = backtest_portfolio(
            events,
            daily_by_symbol,
            variant,
            start=None,
            end=TRAIN_END,
            initial_capital=args.initial_capital,
            risk_per_trade=args.risk_per_trade,
            leverage=args.leverage,
            max_notional_fraction=args.max_notional_fraction,
            max_positions=args.max_positions,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
        )
        oos_summary, oos_trades, _ = backtest_portfolio(
            events,
            daily_by_symbol,
            variant,
            start=OOS_START,
            end=None,
            initial_capital=args.initial_capital,
            risk_per_trade=args.risk_per_trade,
            leverage=args.leverage,
            max_notional_fraction=args.max_notional_fraction,
            max_positions=args.max_positions,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
        )

        row = {
            "variant": variant.name,
            **asdict(variant),
            **{f"all_{key}": value for key, value in all_summary.items()},
            **{f"train_{key}": value for key, value in train_summary.items()},
            **{f"oos_{key}": value for key, value in oos_summary.items()},
            **monthly_stats(all_equity),
            **direction_stats(events),
            **symbol_stats(all_trades),
        }
        row["robust_score"] = robust_score(pd.Series(row))
        summary_rows.append(row)
        if row["robust_score"] > best_score:
            best_score = float(row["robust_score"])
            best_payload = (variant.name, all_trades, all_equity)

        events.to_csv(output_dir / f"signals_{variant.name}.csv", index=False)
        if not all_trades.empty:
            all_trades.to_csv(output_dir / f"trades_{variant.name}.csv", index=False)

    summary = pd.DataFrame(summary_rows).sort_values("robust_score", ascending=False)
    summary.to_csv(output_dir / "variant_summary.csv", index=False)

    if best_payload is not None:
        best_name, trades, equity = best_payload
        if not trades.empty:
            trades.to_csv(output_dir / "best_trades.csv", index=False)
        if not equity.empty:
            equity.to_csv(output_dir / "best_equity.csv")
        render_report(output_dir, summary, best_name)
    else:
        render_report(output_dir, summary, None)

    print(f"Loaded symbols: {len(daily_by_symbol)}")
    print(f"Variants tested: {len(variants)}")
    print(f"Output: {output_dir}")
    print(summary.head(12)[[
        "variant",
        "family",
        "robust_score",
        "all_return_pct",
        "all_max_drawdown_pct",
        "all_total_trades",
        "all_win_rate_pct",
        "all_profit_factor",
        "train_return_pct",
        "oos_return_pct",
        "future_3d_win_pct",
        "avg_future_short_3d_pct",
        "positive_month_ratio",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
