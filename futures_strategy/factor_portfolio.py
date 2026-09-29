from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from futures_strategy.backtest import (
    Position,
    close_position,
    entry_signal,
    intrabar_exit,
    open_position,
    summarize,
    unrealized_pnl,
    update_trailing_stop,
)
from futures_strategy.config import AppConfig, load_config
from futures_strategy.data import load_market_data
from futures_strategy.strategy import prepare_market_data


@dataclass
class PortfolioBacktestResult:
    summary: dict[str, Any]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    symbol_summary: pd.DataFrame


def load_prepared_symbol_frames(config_paths: list[str | Path]) -> tuple[dict[str, AppConfig], dict[str, pd.DataFrame]]:
    configs: dict[str, AppConfig] = {}
    prepared_frames: dict[str, pd.DataFrame] = {}

    for raw_path in config_paths:
        config_path = Path(raw_path)
        config = load_config(config_path)
        symbol = config.exchange.symbol
        raw = load_market_data(config.exchange)
        prepared = prepare_market_data(raw, config.strategy)
        if prepared.empty:
            continue
        configs[symbol] = config
        prepared_frames[symbol] = prepared

    if not prepared_frames:
        raise ValueError("No prepared symbol frames available for portfolio backtest.")
    return configs, prepared_frames


def override_portfolio_risk(configs: dict[str, AppConfig], risk_per_trade: float | None = None) -> dict[str, AppConfig]:
    adjusted: dict[str, AppConfig] = {}
    for symbol, config in configs.items():
        cloned = deepcopy(config)
        if risk_per_trade is not None:
            cloned.risk.risk_per_trade = risk_per_trade
        adjusted[symbol] = cloned
    return adjusted


def common_index(prepared_frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    indexes = [frame.index for frame in prepared_frames.values() if not frame.empty]
    if not indexes:
        raise ValueError("No indexes available for portfolio backtest.")
    common = indexes[0]
    for index in indexes[1:]:
        common = common.intersection(index)
    if len(common) < 3:
        raise ValueError("Not enough overlapping timestamps for portfolio backtest.")
    return common.sort_values()


def candidate_rank_tuple(signal_row: pd.Series) -> tuple[float, ...]:
    return (
        float(signal_row.get("factor_score", 0.0)),
        float(signal_row.get("risk_multiplier", 1.0)),
        float(signal_row.get("oi_value_change_pct", 0.0)),
        float(signal_row.get("funding_rate", 0.0)),
        float(signal_row.get("bounce_pct", 0.0)),
        float(signal_row.get("volume_ratio", 0.0)),
    )


def candidate_rank_score(signal_row: pd.Series) -> float:
    factor_score, risk_multiplier, oi_change, funding_rate, bounce_pct, volume_ratio = candidate_rank_tuple(signal_row)
    return (
        (factor_score * 100.0)
        + (risk_multiplier * 10.0)
        + (oi_change * 1_500.0)
        + (funding_rate * 100_000.0)
        + (bounce_pct * 1_000.0)
        + (max(0.0, volume_ratio - 1.0) * 5.0)
    )


def per_position_config(config: AppConfig, max_positions: int, portfolio_notional_fraction: float) -> AppConfig:
    adjusted = deepcopy(config)
    capped_fraction = portfolio_notional_fraction / max(1, max_positions)
    adjusted.risk.max_notional_fraction = min(adjusted.risk.max_notional_fraction, capped_fraction)
    return adjusted


def build_symbol_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "total_trades",
                "net_profit",
                "win_rate_pct",
                "profit_factor",
                "avg_trade_pnl",
                "best_trade",
                "worst_trade",
            ]
        )

    rows: list[dict[str, Any]] = []
    for symbol, group in trades.groupby("symbol", sort=False):
        winners = group[group["pnl"] > 0]
        losers = group[group["pnl"] <= 0]
        gross_win = winners["pnl"].sum()
        gross_loss = losers["pnl"].sum()
        profit_factor = gross_win / abs(gross_loss) if gross_loss else float("inf")
        rows.append(
            {
                "symbol": symbol,
                "total_trades": int(len(group)),
                "net_profit": round(float(group["pnl"].sum()), 2),
                "win_rate_pct": round((len(winners) / len(group)) * 100, 2),
                "profit_factor": round(float(profit_factor), 2) if profit_factor != float("inf") else "inf",
                "avg_trade_pnl": round(float(group["pnl"].mean()), 2),
                "best_trade": round(float(group["pnl"].max()), 2),
                "worst_trade": round(float(group["pnl"].min()), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("net_profit", ascending=False).reset_index(drop=True)


def build_entry_candidate_index(
    prepared_frames: dict[str, pd.DataFrame],
) -> dict[pd.Timestamp, list[tuple[tuple[float, ...], float, str, str, pd.Series]]]:
    candidates_by_time: dict[pd.Timestamp, list[tuple[tuple[float, ...], float, str, str, pd.Series]]] = {}
    for symbol, frame in prepared_frames.items():
        long_signal = frame.get("long_signal")
        short_signal = frame.get("short_signal")
        if long_signal is None and short_signal is None:
            continue

        signal_mask = pd.Series(False, index=frame.index)
        if long_signal is not None:
            signal_mask |= long_signal.fillna(False).astype(bool)
        if short_signal is not None:
            signal_mask |= short_signal.fillna(False).astype(bool)

        for signal_time in frame.index[signal_mask]:
            row = frame.loc[signal_time]
            signal_side = entry_signal(row)
            if signal_side is None:
                continue
            candidates_by_time.setdefault(signal_time, []).append(
                (
                    candidate_rank_tuple(row),
                    candidate_rank_score(row),
                    symbol,
                    signal_side,
                    row,
                )
            )

    for rows in candidates_by_time.values():
        rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates_by_time


def run_factor_portfolio_backtest(
    configs: dict[str, AppConfig],
    prepared_frames: dict[str, pd.DataFrame],
    max_positions: int = 2,
    portfolio_notional_fraction: float = 0.9,
    symbol_daily_max_losses: int | None = None,
    symbol_daily_loss_limit_pct: float | None = None,
    portfolio_daily_loss_limit_pct: float | None = None,
) -> PortfolioBacktestResult:
    index = common_index(prepared_frames)
    base_config = next(iter(configs.values()))
    initial_capital = float(base_config.risk.initial_capital)

    cash = initial_capital
    peak_equity = cash
    active_positions: dict[str, dict[str, Any]] = {}
    cooldowns = {symbol: 0 for symbol in prepared_frames}
    consecutive_losses = {symbol: 0 for symbol in prepared_frames}
    symbol_day_stats = {
        symbol: {"day": index[0].normalize(), "losses": 0, "pnl": 0.0}
        for symbol in prepared_frames
    }
    portfolio_day = index[0].normalize()
    portfolio_day_start_equity = cash
    portfolio_day_blocked = False
    trades: list[dict[str, Any]] = []
    max_concurrent_positions = 0
    entry_candidates_by_time = build_entry_candidate_index(prepared_frames)
    equity_rows: list[dict[str, Any]] = [
        {
            "timestamp": index[0],
            "equity": cash,
            "drawdown_pct": 0.0,
            "open_positions": 0,
        }
    ]

    for i in range(1, len(index)):
        current_time = index[i]
        prev_time = index[i - 1]
        current_day = current_time.normalize()

        if current_day > portfolio_day:
            portfolio_day = current_day
            portfolio_day_start_equity = cash + sum(
                unrealized_pnl(managed["position"], float(prepared_frames[symbol].loc[current_time]["close"]))
                for symbol, managed in active_positions.items()
            )
            portfolio_day_blocked = False
            for stats in symbol_day_stats.values():
                stats["day"] = current_day
                stats["losses"] = 0
                stats["pnl"] = 0.0

        for symbol in cooldowns:
            if cooldowns[symbol] > 0:
                cooldowns[symbol] -= 1

        for symbol in list(active_positions.keys()):
            frame = prepared_frames[symbol]
            current_row = frame.loc[current_time]
            prev_row = frame.loc[prev_time]
            managed = active_positions[symbol]
            position: Position = managed["position"]
            config = managed["config"]

            exit_fill = None
            position.bars_held += 1

            if config.strategy.flat_on_new_day and current_time.normalize() > position.entry_time.normalize():
                price = float(current_row["open"])
                exit_fill = ("new_day_exit", price)
            elif position.bars_held >= config.strategy.max_bars_in_trade:
                price = float(current_row["open"])
                exit_fill = ("time_exit", price)
            else:
                exit_fill, partial_cash_delta = intrabar_exit(position, current_row, config)
                cash += partial_cash_delta

            exit_key = "exit_long_signal" if position.side == "long" else "exit_short_signal"
            custom_exit = prev_row.get(exit_key, False)
            if exit_fill is None and pd.notna(custom_exit) and bool(custom_exit):
                price = float(current_row["open"])
                exit_fill = ("signal_exit", price)

            if exit_fill is not None:
                reason, price = exit_fill
                cash, trade = close_position(position, cash, current_time, price, reason, config.risk.fee_rate)
                trade["symbol"] = symbol
                trade["entry_rank_score"] = round(float(managed["entry_rank_score"]), 4)
                trade["entry_factor_score"] = int(managed["entry_factor_score"])
                trades.append(trade)
                symbol_day_stats[symbol]["pnl"] += float(trade["pnl"])
                if trade["pnl"] <= 0:
                    symbol_day_stats[symbol]["losses"] += 1
                    consecutive_losses[symbol] += 1
                    if consecutive_losses[symbol] >= config.risk.max_consecutive_losses:
                        cooldowns[symbol] = config.risk.cooldown_bars
                        consecutive_losses[symbol] = 0
                else:
                    consecutive_losses[symbol] = 0
                del active_positions[symbol]
            else:
                update_trailing_stop(position, current_row, config.strategy.trail_atr_mult)

        pre_entry_unrealized = 0.0
        for symbol, managed in active_positions.items():
            close_price = float(prepared_frames[symbol].loc[current_time]["close"])
            pre_entry_unrealized += unrealized_pnl(managed["position"], close_price)
        pre_entry_equity = cash + pre_entry_unrealized
        if (
            portfolio_daily_loss_limit_pct is not None
            and portfolio_daily_loss_limit_pct > 0
            and pre_entry_equity <= portfolio_day_start_equity * (1 - portfolio_daily_loss_limit_pct)
        ):
            portfolio_day_blocked = True

        slots = max(0, max_positions - len(active_positions))
        if slots > 0 and not portfolio_day_blocked:
            candidates: list[tuple[tuple[float, ...], float, str, str, pd.Series, pd.Series]] = []
            for rank_tuple, rank_score, symbol, signal_side, prev_row in entry_candidates_by_time.get(prev_time, []):
                if symbol in active_positions or cooldowns[symbol] > 0:
                    continue
                stats = symbol_day_stats[symbol]
                if symbol_daily_max_losses is not None and symbol_daily_max_losses >= 0:
                    if stats["losses"] >= symbol_daily_max_losses:
                        continue
                if symbol_daily_loss_limit_pct is not None and symbol_daily_loss_limit_pct > 0:
                    if stats["pnl"] <= -(initial_capital * symbol_daily_loss_limit_pct):
                        continue
                current_row = prepared_frames[symbol].loc[current_time]
                candidates.append((rank_tuple, rank_score, symbol, signal_side, current_row, prev_row))

            for _, rank_score, symbol, signal_side, current_row, prev_row in candidates[:slots]:
                config = per_position_config(configs[symbol], max_positions=max_positions, portfolio_notional_fraction=portfolio_notional_fraction)
                position = open_position(signal_side, current_time, current_row, prev_row, cash, config)
                if position is None:
                    continue
                cash -= position.entry_fee
                active_positions[symbol] = {
                    "position": position,
                    "config": config,
                    "entry_rank_score": rank_score,
                    "entry_factor_score": int(prev_row.get("factor_score", 0)),
                }

        max_concurrent_positions = max(max_concurrent_positions, len(active_positions))
        unrealized = 0.0
        for symbol, managed in active_positions.items():
            close_price = float(prepared_frames[symbol].loc[current_time]["close"])
            unrealized += unrealized_pnl(managed["position"], close_price)
        equity = cash + unrealized
        peak_equity = max(peak_equity, equity)
        equity_rows.append(
            {
                "timestamp": current_time,
                "equity": equity,
                "drawdown_pct": (equity / peak_equity) - 1,
                "open_positions": len(active_positions),
            }
        )

    if active_positions:
        last_time = index[-1]
        for symbol in list(active_positions.keys()):
            managed = active_positions[symbol]
            position: Position = managed["position"]
            frame = prepared_frames[symbol]
            exit_price = float(frame.loc[last_time]["close"])
            cash, trade = close_position(position, cash, last_time, exit_price, "end_of_test", managed["config"].risk.fee_rate)
            trade["symbol"] = symbol
            trade["entry_rank_score"] = round(float(managed["entry_rank_score"]), 4)
            trade["entry_factor_score"] = int(managed["entry_factor_score"])
            trades.append(trade)
            del active_positions[symbol]
        peak_equity = max(peak_equity, cash)
        equity_rows[-1]["equity"] = cash
        equity_rows[-1]["drawdown_pct"] = (cash / peak_equity) - 1
        equity_rows[-1]["open_positions"] = 0

    trades_frame = pd.DataFrame(trades)
    equity_frame = pd.DataFrame(equity_rows).set_index("timestamp")
    summary = summarize(trades_frame, equity_frame, initial_capital)
    summary["symbols_traded"] = sorted(trades_frame["symbol"].unique().tolist()) if not trades_frame.empty else []
    summary["num_symbols_traded"] = len(summary["symbols_traded"])
    summary["max_concurrent_positions"] = int(max_concurrent_positions)
    if symbol_daily_max_losses is not None:
        summary["symbol_daily_max_losses"] = int(symbol_daily_max_losses)
    if symbol_daily_loss_limit_pct is not None:
        summary["symbol_daily_loss_limit_pct"] = round(float(symbol_daily_loss_limit_pct) * 100, 4)
    if portfolio_daily_loss_limit_pct is not None:
        summary["portfolio_daily_loss_limit_pct"] = round(float(portfolio_daily_loss_limit_pct) * 100, 4)
    symbol_summary = build_symbol_summary(trades_frame)
    return PortfolioBacktestResult(
        summary=summary,
        trades=trades_frame,
        equity_curve=equity_frame,
        symbol_summary=symbol_summary,
    )
