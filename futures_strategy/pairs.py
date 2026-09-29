from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from futures_strategy.config import ExchangeSettings
from futures_strategy.data import load_market_data


@dataclass
class PairMarketSettings:
    leg_a: ExchangeSettings = field(default_factory=ExchangeSettings)
    leg_b: ExchangeSettings = field(
        default_factory=lambda: ExchangeSettings(symbol="ETH/USDT:USDT", timeframe="5m")
    )


@dataclass
class PairStrategySettings:
    beta_window: int = 288
    zscore_window: int = 288
    entry_z: float = 2.2
    exit_z: float = 0.4
    stop_z: float = 4.0
    min_correlation: float = 0.75
    correlation_exit_buffer: float = 0.1
    reversion_confirm: bool = True
    max_holding_bars: int = 96
    cooldown_bars: int = 12
    hedge_ratio_min: float = 0.5
    hedge_ratio_max: float = 2.0


@dataclass
class PairRiskSettings:
    initial_capital: float = 10_000.0
    leverage: float = 2.0
    gross_exposure_fraction: float = 1.0
    fee_rate: float = 0.0005
    slippage_bps: float = 2.0


@dataclass
class PairOutputSettings:
    output_dir: str = "outputs/pairs_default"


@dataclass
class PairConfig:
    pair: PairMarketSettings = field(default_factory=PairMarketSettings)
    strategy: PairStrategySettings = field(default_factory=PairStrategySettings)
    risk: PairRiskSettings = field(default_factory=PairRiskSettings)
    output: PairOutputSettings = field(default_factory=PairOutputSettings)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "PairConfig":
        payload = payload or {}
        pair_payload = payload.get("pair", {})
        return cls(
            pair=PairMarketSettings(
                leg_a=ExchangeSettings(**pair_payload.get("leg_a", {})),
                leg_b=ExchangeSettings(**pair_payload.get("leg_b", {})),
            ),
            strategy=PairStrategySettings(**payload.get("strategy", {})),
            risk=PairRiskSettings(**payload.get("risk", {})),
            output=PairOutputSettings(**payload.get("output", {})),
        )


@dataclass
class PairPosition:
    side: str
    entry_time: pd.Timestamp
    entry_price_a: float
    entry_price_b: float
    qty_a: float
    qty_b: float
    hedge_ratio: float
    entry_fee: float
    realized_pnl: float
    entry_zscore: float
    bars_held: int = 0


@dataclass
class PairBacktestResult:
    summary: dict[str, Any]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    prepared: pd.DataFrame


def load_pair_config(path: str | Path) -> PairConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return PairConfig.from_dict(payload)


def load_pair_market_data(settings: PairMarketSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame_a = load_market_data(settings.leg_a)
    frame_b = load_market_data(settings.leg_b)
    return frame_a, frame_b


def prepare_pair_data(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    settings: PairStrategySettings,
) -> pd.DataFrame:
    if frame_a.empty or frame_b.empty:
        raise ValueError("Pair data cannot be empty.")

    data_a = frame_a.rename(columns={column: f"{column}_a" for column in frame_a.columns})
    data_b = frame_b.rename(columns={column: f"{column}_b" for column in frame_b.columns})
    merged = data_a.join(data_b, how="inner").sort_index()
    if merged.empty:
        raise ValueError("No overlapping timestamps between the two legs.")

    merged["log_close_a"] = np.log(merged["close_a"])
    merged["log_close_b"] = np.log(merged["close_b"])
    merged["ret_a"] = merged["close_a"].pct_change()
    merged["ret_b"] = merged["close_b"].pct_change()

    raw_beta = (
        merged["log_close_a"].rolling(settings.beta_window).cov(merged["log_close_b"])
        / merged["log_close_b"].rolling(settings.beta_window).var().replace(0, np.nan)
    )
    merged["hedge_ratio"] = (
        raw_beta.clip(lower=settings.hedge_ratio_min, upper=settings.hedge_ratio_max).shift(1)
    )
    merged["correlation"] = merged["ret_a"].rolling(settings.beta_window).corr(merged["ret_b"]).shift(1)
    merged["spread"] = merged["log_close_a"] - (merged["hedge_ratio"] * merged["log_close_b"])
    merged["spread_mean"] = merged["spread"].rolling(settings.zscore_window).mean().shift(1)
    merged["spread_std"] = merged["spread"].rolling(settings.zscore_window).std(ddof=0).shift(1)
    merged["zscore"] = (merged["spread"] - merged["spread_mean"]) / merged["spread_std"].replace(0, np.nan)
    merged["abs_zscore"] = merged["zscore"].abs()
    merged["zscore_change"] = merged["zscore"].diff()

    if settings.reversion_confirm:
        short_signal = merged["zscore"] >= settings.entry_z
        short_signal &= merged["zscore_change"] < 0
        long_signal = merged["zscore"] <= -settings.entry_z
        long_signal &= merged["zscore_change"] > 0
    else:
        short_signal = merged["zscore"] >= settings.entry_z
        long_signal = merged["zscore"] <= -settings.entry_z

    corr_filter = merged["correlation"] >= settings.min_correlation
    merged["entry_short_a_long_b"] = short_signal & corr_filter
    merged["entry_long_a_short_b"] = long_signal & corr_filter
    merged["exit_signal"] = merged["abs_zscore"] <= settings.exit_z
    merged["stop_signal"] = merged["abs_zscore"] >= settings.stop_z
    merged["corr_break_signal"] = merged["correlation"] < (settings.min_correlation - settings.correlation_exit_buffer)
    return merged.dropna()


def latest_pair_snapshot(frame: pd.DataFrame, settings: PairStrategySettings) -> dict[str, Any]:
    if frame.empty:
        return {}
    row = frame.iloc[-1]
    if bool(row["entry_short_a_long_b"]):
        action = "SHORT_A_LONG_B_SETUP"
    elif bool(row["entry_long_a_short_b"]):
        action = "LONG_A_SHORT_B_SETUP"
    elif abs(float(row["zscore"])) >= settings.entry_z * 0.8:
        action = "SPREAD_WIDENING_WATCH"
    else:
        action = "NO_SETUP"
    return {
        "timestamp": frame.index[-1].isoformat(),
        "action": action,
        "zscore": round(float(row["zscore"]), 4),
        "abs_zscore": round(float(row["abs_zscore"]), 4),
        "correlation": round(float(row["correlation"]), 4),
        "hedge_ratio": round(float(row["hedge_ratio"]), 4),
        "close_a": round(float(row["close_a"]), 4),
        "close_b": round(float(row["close_b"]), 4),
    }


def run_pair_backtest(frame: pd.DataFrame, config: PairConfig) -> PairBacktestResult:
    if frame.empty:
        raise ValueError("No prepared pair data to backtest.")

    strategy = config.strategy
    risk = config.risk
    cash = risk.initial_capital
    peak_equity = cash
    cooldown = 0
    position: PairPosition | None = None
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = [
        {
            "timestamp": frame.index[0],
            "equity": cash,
            "drawdown_pct": 0.0,
        }
    ]

    for i in range(2, len(frame)):
        current_time = frame.index[i]
        row = frame.iloc[i]
        prev = frame.iloc[i - 1]

        if cooldown > 0:
            cooldown -= 1

        if position is not None:
            position.bars_held += 1
            exit_reason = None
            if bool(prev["exit_signal"]):
                exit_reason = "mean_reversion_exit"
            elif bool(prev["stop_signal"]):
                exit_reason = "spread_stop"
            elif bool(prev["corr_break_signal"]):
                exit_reason = "correlation_break"
            elif position.bars_held >= strategy.max_holding_bars:
                exit_reason = "time_exit"

            if exit_reason is not None:
                exit_price_a = apply_leg_slippage(
                    leg_side(position.side, "a"),
                    float(row["open_a"]),
                    is_entry=False,
                    slippage_bps=risk.slippage_bps,
                )
                exit_price_b = apply_leg_slippage(
                    leg_side(position.side, "b"),
                    float(row["open_b"]),
                    is_entry=False,
                    slippage_bps=risk.slippage_bps,
                )
                cash, trade = close_pair_position(
                    position=position,
                    cash=cash,
                    exit_time=current_time,
                    exit_price_a=exit_price_a,
                    exit_price_b=exit_price_b,
                    exit_zscore=float(prev["zscore"]),
                    exit_reason=exit_reason,
                    fee_rate=risk.fee_rate,
                )
                trades.append(trade)
                position = None
                cooldown = strategy.cooldown_bars

        if position is None and cooldown == 0:
            signal_side = entry_signal(prev)
            if signal_side is not None:
                position = open_pair_position(signal_side, current_time, row, prev, cash, config)
                if position is not None:
                    cash -= position.entry_fee

        equity = cash + unrealized_pair_pnl(position, float(row["close_a"]), float(row["close_b"])) if position else cash
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
        last_row = frame.iloc[-1]
        exit_price_a = apply_leg_slippage(leg_side(position.side, "a"), float(last_row["close_a"]), False, risk.slippage_bps)
        exit_price_b = apply_leg_slippage(leg_side(position.side, "b"), float(last_row["close_b"]), False, risk.slippage_bps)
        cash, trade = close_pair_position(
            position=position,
            cash=cash,
            exit_time=last_time,
            exit_price_a=exit_price_a,
            exit_price_b=exit_price_b,
            exit_zscore=float(last_row["zscore"]),
            exit_reason="end_of_test",
            fee_rate=risk.fee_rate,
        )
        trades.append(trade)
        equity_rows[-1]["equity"] = cash
        peak_equity = max(peak_equity, cash)
        equity_rows[-1]["drawdown_pct"] = (cash / peak_equity) - 1

    trades_frame = pd.DataFrame(trades)
    equity_frame = pd.DataFrame(equity_rows).set_index("timestamp")
    summary = summarize_pair_results(trades_frame, equity_frame, risk.initial_capital)
    return PairBacktestResult(summary=summary, trades=trades_frame, equity_curve=equity_frame, prepared=frame)


def entry_signal(row: pd.Series) -> str | None:
    if bool(row.get("entry_short_a_long_b", False)):
        return "short_a_long_b"
    if bool(row.get("entry_long_a_short_b", False)):
        return "long_a_short_b"
    return None


def open_pair_position(
    side: str,
    current_time: pd.Timestamp,
    row: pd.Series,
    signal_row: pd.Series,
    cash: float,
    config: PairConfig,
) -> PairPosition | None:
    risk = config.risk
    hedge_ratio = float(signal_row["hedge_ratio"])
    if not np.isfinite(hedge_ratio) or hedge_ratio <= 0:
        return None

    gross_notional = cash * risk.leverage * risk.gross_exposure_fraction
    if gross_notional <= 0:
        return None

    weight_b = hedge_ratio / (1 + hedge_ratio)
    weight_a = 1 - weight_b
    notional_a = gross_notional * weight_a
    notional_b = gross_notional * weight_b

    entry_price_a = apply_leg_slippage(leg_side(side, "a"), float(row["open_a"]), True, risk.slippage_bps)
    entry_price_b = apply_leg_slippage(leg_side(side, "b"), float(row["open_b"]), True, risk.slippage_bps)
    qty_a = notional_a / entry_price_a
    qty_b = notional_b / entry_price_b
    if qty_a <= 0 or qty_b <= 0:
        return None

    entry_fee = ((entry_price_a * qty_a) + (entry_price_b * qty_b)) * risk.fee_rate
    return PairPosition(
        side=side,
        entry_time=current_time,
        entry_price_a=entry_price_a,
        entry_price_b=entry_price_b,
        qty_a=qty_a,
        qty_b=qty_b,
        hedge_ratio=hedge_ratio,
        entry_fee=entry_fee,
        realized_pnl=-entry_fee,
        entry_zscore=float(signal_row["zscore"]),
    )


def close_pair_position(
    position: PairPosition,
    cash: float,
    exit_time: pd.Timestamp,
    exit_price_a: float,
    exit_price_b: float,
    exit_zscore: float,
    exit_reason: str,
    fee_rate: float,
) -> tuple[float, dict[str, Any]]:
    gross = pair_leg_pnl(position.side, position.entry_price_a, position.entry_price_b, exit_price_a, exit_price_b, position.qty_a, position.qty_b)
    exit_fee = ((exit_price_a * position.qty_a) + (exit_price_b * position.qty_b)) * fee_rate
    trade_pnl = position.realized_pnl + gross - exit_fee
    cash += gross - exit_fee

    trade = {
        "entry_time": position.entry_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "side": position.side,
        "qty_a": round(position.qty_a, 8),
        "qty_b": round(position.qty_b, 8),
        "entry_price_a": round(position.entry_price_a, 4),
        "entry_price_b": round(position.entry_price_b, 4),
        "exit_price_a": round(exit_price_a, 4),
        "exit_price_b": round(exit_price_b, 4),
        "entry_zscore": round(position.entry_zscore, 4),
        "exit_zscore": round(exit_zscore, 4),
        "bars_held": position.bars_held,
        "hedge_ratio": round(position.hedge_ratio, 4),
        "pnl": round(trade_pnl, 4),
        "exit_reason": exit_reason,
    }
    return cash, trade


def unrealized_pair_pnl(position: PairPosition | None, price_a: float, price_b: float) -> float:
    if position is None:
        return 0.0
    return pair_leg_pnl(
        position.side,
        position.entry_price_a,
        position.entry_price_b,
        price_a,
        price_b,
        position.qty_a,
        position.qty_b,
    )


def pair_leg_pnl(
    side: str,
    entry_price_a: float,
    entry_price_b: float,
    exit_price_a: float,
    exit_price_b: float,
    qty_a: float,
    qty_b: float,
) -> float:
    if side == "short_a_long_b":
        pnl_a = (entry_price_a - exit_price_a) * qty_a
        pnl_b = (exit_price_b - entry_price_b) * qty_b
        return pnl_a + pnl_b
    pnl_a = (exit_price_a - entry_price_a) * qty_a
    pnl_b = (entry_price_b - exit_price_b) * qty_b
    return pnl_a + pnl_b


def leg_side(pair_side: str, leg: str) -> str:
    if pair_side == "short_a_long_b":
        return "short" if leg == "a" else "long"
    return "long" if leg == "a" else "short"


def apply_leg_slippage(side: str, price: float, is_entry: bool, slippage_bps: float) -> float:
    slippage = slippage_bps / 10_000
    if is_entry:
        return price * (1 + slippage) if side == "long" else price * (1 - slippage)
    return price * (1 - slippage) if side == "long" else price * (1 + slippage)


def summarize_pair_results(trades: pd.DataFrame, equity_curve: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
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
            "avg_bars_held": 0.0,
        }

    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
    win_rate = (trades["pnl"] > 0).mean() * 100

    return {
        "initial_capital": round(initial_capital, 2),
        "final_equity": round(final_equity, 2),
        "net_profit": round(final_equity - initial_capital, 2),
        "return_pct": round(return_pct, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "total_trades": int(len(trades)),
        "win_rate_pct": round(float(win_rate), 2),
        "profit_factor": "inf" if np.isinf(profit_factor) else round(float(profit_factor), 2),
        "avg_trade_pnl": round(float(trades["pnl"].mean()), 2),
        "avg_bars_held": round(float(trades["bars_held"].mean()), 2),
    }


def config_as_dict(config: PairConfig) -> dict[str, Any]:
    return asdict(config)

