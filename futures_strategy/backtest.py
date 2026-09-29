from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from futures_strategy.config import AppConfig


@dataclass
class BacktestResult:
    summary: dict
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


@dataclass
class Position:
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    remaining_qty: float
    stop_price: float
    initial_stop: float
    tp1_price: float
    initial_risk: float
    entry_fee: float
    realized_pnl: float
    highest_price: float
    lowest_price: float
    risk_multiplier: float = 1.0
    bars_held: int = 0
    partial_taken: bool = False


def run_backtest(frame: pd.DataFrame, config: AppConfig) -> BacktestResult:
    if frame.empty:
        raise ValueError("No prepared market data to backtest.")

    risk = config.risk
    strategy = config.strategy

    cash = risk.initial_capital
    peak_equity = cash
    cooldown = 0
    consecutive_losses = 0
    position: Position | None = None
    trades: list[dict] = []
    equity_rows: list[dict] = [
        {
            "timestamp": frame.index[0],
            "equity": cash,
            "drawdown_pct": 0.0,
        }
    ]

    for i in range(1, len(frame)):
        current_time = frame.index[i]
        row = frame.iloc[i]
        prev = frame.iloc[i - 1]

        if cooldown > 0:
            cooldown -= 1

        if position is not None:
            exit_fill = None
            position.bars_held += 1

            if strategy.flat_on_new_day and current_time.normalize() > position.entry_time.normalize():
                price = apply_slippage(position.side, float(row["open"]), is_entry=False, slippage_bps=risk.slippage_bps)
                exit_fill = ("new_day_exit", price)
            elif should_signal_exit(position.side, prev):
                price = apply_slippage(position.side, float(row["open"]), is_entry=False, slippage_bps=risk.slippage_bps)
                exit_fill = ("signal_exit", price)
            elif position.bars_held >= strategy.max_bars_in_trade:
                price = apply_slippage(position.side, float(row["open"]), is_entry=False, slippage_bps=risk.slippage_bps)
                exit_fill = ("time_exit", price)
            else:
                exit_fill, partial_cash_delta = intrabar_exit(position, row, config)
                cash += partial_cash_delta

            if exit_fill is not None:
                reason, price = exit_fill
                cash, trade = close_position(position, cash, current_time, price, reason, risk.fee_rate)
                trades.append(trade)
                if trade["pnl"] <= 0:
                    consecutive_losses += 1
                    if consecutive_losses >= risk.max_consecutive_losses:
                        cooldown = risk.cooldown_bars
                        consecutive_losses = 0
                else:
                    consecutive_losses = 0
                position = None
            else:
                update_trailing_stop(position, row, strategy.trail_atr_mult)

        if position is None and cooldown == 0:
            signal_side = entry_signal(prev)
            if signal_side is not None:
                position = open_position(signal_side, current_time, row, prev, cash, config)
                if position is not None:
                    cash -= position.entry_fee

        equity = cash + unrealized_pnl(position, float(row["close"])) if position else cash
        peak_equity = max(peak_equity, equity)
        equity_rows.append(
            {
                "timestamp": current_time,
                "equity": equity,
                "drawdown_pct": (equity / peak_equity) - 1,
            }
        )

    if position is not None:
        last_time = frame.index[-1]
        last_close = float(frame.iloc[-1]["close"])
        exit_price = apply_slippage(position.side, last_close, is_entry=False, slippage_bps=risk.slippage_bps)
        cash, trade = close_position(position, cash, last_time, exit_price, "end_of_test", risk.fee_rate)
        trades.append(trade)
        equity_rows[-1]["equity"] = cash
        peak_equity = max(peak_equity, cash)
        equity_rows[-1]["drawdown_pct"] = (cash / peak_equity) - 1

    trades_frame = pd.DataFrame(trades)
    equity_frame = pd.DataFrame(equity_rows).set_index("timestamp")
    summary = summarize(trades_frame, equity_frame, risk.initial_capital)
    return BacktestResult(summary=summary, trades=trades_frame, equity_curve=equity_frame)


def entry_signal(row: pd.Series) -> str | None:
    if bool(row.get("long_signal", False)):
        return "long"
    if bool(row.get("short_signal", False)):
        return "short"
    return None


def open_position(
    side: str,
    current_time: pd.Timestamp,
    current_row: pd.Series,
    signal_row: pd.Series,
    cash: float,
    config: AppConfig,
) -> Position | None:
    strategy = config.strategy
    risk = config.risk

    raw_entry = float(current_row["open"])
    entry_price = apply_slippage(side, raw_entry, is_entry=True, slippage_bps=risk.slippage_bps)
    atr_value = float(signal_row["atr"])
    if not np.isfinite(atr_value) or atr_value <= 0:
        return None

    if side == "long":
        swing_stop = float(signal_row["swing_low"]) if pd.notna(signal_row["swing_low"]) else np.nan
        atr_stop = entry_price - (atr_value * strategy.atr_stop_mult)
        stop_price = min(swing_stop, atr_stop) if np.isfinite(swing_stop) else atr_stop
    else:
        swing_stop = float(signal_row["swing_high"]) if pd.notna(signal_row["swing_high"]) else np.nan
        atr_stop = entry_price + (atr_value * strategy.atr_stop_mult)
        stop_price = max(swing_stop, atr_stop) if np.isfinite(swing_stop) else atr_stop

    risk_distance = abs(entry_price - stop_price)
    if not np.isfinite(risk_distance) or risk_distance <= 0:
        return None

    risk_multiplier = resolve_risk_multiplier(signal_row)
    risk_amount = cash * risk.risk_per_trade * risk_multiplier
    raw_qty = risk_amount / risk_distance
    max_notional = cash * risk.leverage * risk.max_notional_fraction
    capped_qty = min(raw_qty, max_notional / entry_price)
    if capped_qty <= 0:
        return None

    entry_fee = entry_price * capped_qty * risk.fee_rate
    tp1_price = entry_price + (risk_distance * strategy.partial_rr) if side == "long" else entry_price - (
        risk_distance * strategy.partial_rr
    )

    return Position(
        side=side,
        entry_time=current_time,
        entry_price=entry_price,
        qty=capped_qty,
        remaining_qty=capped_qty,
        stop_price=stop_price,
        initial_stop=stop_price,
        tp1_price=tp1_price,
        initial_risk=risk_distance * capped_qty,
        entry_fee=entry_fee,
        realized_pnl=-entry_fee,
        highest_price=entry_price,
        lowest_price=entry_price,
        risk_multiplier=risk_multiplier,
    )


def resolve_risk_multiplier(signal_row: pd.Series) -> float:
    raw_value = signal_row.get("risk_multiplier", 1.0)
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(parsed):
        return 1.0
    return max(0.25, min(parsed, 5.0))


def intrabar_exit(position: Position, row: pd.Series, config: AppConfig) -> tuple[tuple[str, float] | None, float]:
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    strategy = config.strategy
    risk = config.risk
    partial_cash_delta = 0.0

    if position.side == "long":
        if low <= position.stop_price:
            price = apply_slippage("long", position.stop_price, is_entry=False, slippage_bps=risk.slippage_bps)
            return ("stop_loss" if not position.partial_taken else "trail_stop", price), partial_cash_delta
        if (not position.partial_taken) and high >= position.tp1_price:
            partial_qty = position.qty * strategy.partial_close_ratio
            partial_price = apply_slippage("long", position.tp1_price, is_entry=False, slippage_bps=risk.slippage_bps)
            partial_cash_delta = realize_partial(position, partial_qty, partial_price, risk.fee_rate)
            position.partial_taken = True
            position.stop_price = max(position.stop_price, position.entry_price)
        position.highest_price = max(position.highest_price, high, close)
    else:
        if high >= position.stop_price:
            price = apply_slippage("short", position.stop_price, is_entry=False, slippage_bps=risk.slippage_bps)
            return ("stop_loss" if not position.partial_taken else "trail_stop", price), partial_cash_delta
        if (not position.partial_taken) and low <= position.tp1_price:
            partial_qty = position.qty * strategy.partial_close_ratio
            partial_price = apply_slippage("short", position.tp1_price, is_entry=False, slippage_bps=risk.slippage_bps)
            partial_cash_delta = realize_partial(position, partial_qty, partial_price, risk.fee_rate)
            position.partial_taken = True
            position.stop_price = min(position.stop_price, position.entry_price)
        position.lowest_price = min(position.lowest_price, low, close)

    return None, partial_cash_delta


def realize_partial(position: Position, qty: float, exit_price: float, fee_rate: float) -> float:
    if qty <= 0 or qty > position.remaining_qty:
        return 0.0
    gross = trade_gross_pnl(position.side, position.entry_price, exit_price, qty)
    fee = exit_price * qty * fee_rate
    position.remaining_qty -= qty
    position.realized_pnl += gross - fee
    return gross - fee


def update_trailing_stop(position: Position, row: pd.Series, trail_atr_mult: float) -> None:
    atr_value = float(row["atr"])
    if not np.isfinite(atr_value) or atr_value <= 0 or not position.partial_taken:
        return

    if position.side == "long":
        trailing = position.highest_price - (atr_value * trail_atr_mult)
        position.stop_price = max(position.stop_price, trailing, position.entry_price)
    else:
        trailing = position.lowest_price + (atr_value * trail_atr_mult)
        position.stop_price = min(position.stop_price, trailing, position.entry_price)


def should_signal_exit(side: str, prev: pd.Series) -> bool:
    custom_key = "exit_long_signal" if side == "long" else "exit_short_signal"
    if custom_key in prev.index:
        custom_value = prev.get(custom_key)
        if pd.notna(custom_value):
            return bool(custom_value)

    if side == "long":
        return bool(prev["short_regime"] or ((prev["close"] < prev["ema_entry"]) and (prev["rsi"] < 45)))
    return bool(prev["long_regime"] or ((prev["close"] > prev["ema_entry"]) and (prev["rsi"] > 55)))


def close_position(
    position: Position,
    cash: float,
    exit_time: pd.Timestamp,
    exit_price: float,
    reason: str,
    fee_rate: float,
) -> tuple[float, dict]:
    gross = trade_gross_pnl(position.side, position.entry_price, exit_price, position.remaining_qty)
    fee = exit_price * position.remaining_qty * fee_rate
    trade_pnl = position.realized_pnl + gross - fee
    cash += gross - fee

    trade = {
        "entry_time": position.entry_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "side": position.side,
        "qty": round(position.qty, 8),
        "entry_price": round(position.entry_price, 4),
        "exit_price": round(exit_price, 4),
        "pnl": round(trade_pnl, 4),
        "pnl_pct_on_equity_risk": round((trade_pnl / position.initial_risk) * 100, 2) if position.initial_risk else 0.0,
        "bars_held": position.bars_held,
        "risk_multiplier": round(float(position.risk_multiplier), 2),
        "scaled_out": position.partial_taken,
        "exit_reason": reason,
    }
    return cash, trade


def unrealized_pnl(position: Position | None, mark_price: float) -> float:
    if position is None:
        return 0.0
    return trade_gross_pnl(position.side, position.entry_price, mark_price, position.remaining_qty)


def trade_gross_pnl(side: str, entry_price: float, exit_price: float, qty: float) -> float:
    if side == "long":
        return (exit_price - entry_price) * qty
    return (entry_price - exit_price) * qty


def apply_slippage(side: str, price: float, is_entry: bool, slippage_bps: float) -> float:
    slippage = slippage_bps / 10_000
    if is_entry:
        return price * (1 + slippage) if side == "long" else price * (1 - slippage)
    return price * (1 - slippage) if side == "long" else price * (1 + slippage)


def summarize(trades: pd.DataFrame, equity_curve: pd.DataFrame, initial_capital: float) -> dict:
    final_equity = float(equity_curve["equity"].iloc[-1])
    return_pct = ((final_equity / initial_capital) - 1) * 100
    max_drawdown_pct = float(equity_curve["drawdown_pct"].min() * 100)

    if trades.empty:
        return {
            "initial_capital": round(initial_capital, 2),
            "final_equity": round(final_equity, 2),
            "net_profit": round(final_equity - initial_capital, 2),
            "return_pct": round(return_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
        }

    winners = trades[trades["pnl"] > 0]
    losers = trades[trades["pnl"] <= 0]
    gross_win = winners["pnl"].sum()
    gross_loss = losers["pnl"].sum()
    profit_factor = gross_win / abs(gross_loss) if gross_loss else float("inf")

    return {
        "initial_capital": round(initial_capital, 2),
        "final_equity": round(final_equity, 2),
        "net_profit": round(final_equity - initial_capital, 2),
        "return_pct": round(return_pct, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "total_trades": int(len(trades)),
        "win_rate_pct": round((len(winners) / len(trades)) * 100, 2),
        "profit_factor": round(float(profit_factor), 2) if np.isfinite(profit_factor) else "inf",
        "avg_trade_pnl": round(float(trades["pnl"].mean()), 2),
        "best_trade": round(float(trades["pnl"].max()), 2),
        "worst_trade": round(float(trades["pnl"].min()), 2),
    }
