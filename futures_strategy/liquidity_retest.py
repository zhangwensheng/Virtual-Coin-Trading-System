from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from futures_strategy.indicators import atr


Side = Literal["long", "short"]
ZoneSide = Literal["resistance", "support"]
TradeTransformMode = Literal["original", "mirror_original_longs_short_only"]


@dataclass(frozen=True)
class LiquidityRetestSettings:
    universe_size: int = 5
    quote_volume_lookback_hours: int = 24
    stablecoin_bases: tuple[str, ...] = ("USDC", "FDUSD", "USDP", "TUSD", "DAI", "USDE", "USD1", "BFUSD")
    pivot_left_bars: int = 3
    pivot_right_bars: int = 3
    zone_lookback_4h: int = 120
    zone_lookback_1h: int = 120
    zone_merge_atr: float = 0.5
    zone_min_touches: int = 2
    breakout_buffer_atr: float = 0.15
    breakout_body_atr: float = 0.4
    breakout_volume_window: int = 20
    breakout_min_quote_volume_ratio: float = 1.2
    retest_tolerance_atr: float = 0.2
    retest_max_5m_bars: int = 24
    confirm_volume_window: int = 20
    confirm_min_quote_volume_ratio: float = 1.1
    risk_per_trade: float = 0.005
    leverage: float = 3.0
    min_margin_usdt: float = 1.0
    max_stop_pct: float = 0.03
    fee_rate: float = 0.0005
    slippage_bps: float = 2.0
    max_positions: int = 2
    max_symbol_entries_per_day: int = 3
    max_account_entries_per_day: int = 6
    stop_cooldown_minutes: int = 60
    partial_take_profit_r: float = 1.5
    partial_close_ratio: float = 0.5
    trailing_atr: float = 2.0
    max_hold_5m_bars: int = 72
    daily_reset_timezone: str = "Asia/Shanghai"
    trade_transform_mode: TradeTransformMode = "original"
    max_drawdown_pct: float = 0.20


@dataclass(frozen=True)
class Zone:
    side: ZoneSide
    timeframe: str
    low: float
    high: float
    touches: int
    score: float
    last_touch_time: pd.Timestamp

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0


@dataclass
class SetupState:
    state: str = "IDLE"
    side: Side | None = None
    zone: Zone | None = None
    breakout_time: pd.Timestamp | None = None
    breakout_price: float | None = None
    breakout_atr_distance: float = 0.0
    breakout_body_atr: float = 0.0
    breakout_quote_volume_ratio: float = 0.0
    reject_time: pd.Timestamp | None = None
    reject_high: float | None = None
    reject_low: float | None = None
    retest_depth_atr: float = 0.0
    wait_bars: int = 0


@dataclass(frozen=True)
class SignalCandidate:
    signal_id: str
    symbol: str
    side: Side
    signal_time: pd.Timestamp
    zone: Zone
    planned_entry: float
    planned_stop: float
    atr_5m: float
    universe_rank: int | None = None
    quote_volume_24h: float | None = None
    breakout_atr_distance: float = 0.0
    breakout_body_atr: float = 0.0
    breakout_quote_volume_ratio: float = 0.0
    retest_depth_atr: float = 0.0
    retest_wait_bars: int = 0
    confirm_quote_volume_ratio: float = 0.0
    planned_tp1: float | None = None
    transform_mode: str = "original"
    original_side: Side | None = None
    original_stop: float | None = None
    original_tp1: float | None = None


@dataclass(frozen=True)
class SignalTransformResult:
    candidate: SignalCandidate | None
    status: Literal["kept", "filtered"]
    reason: str | None = None


@dataclass(frozen=True)
class RiskDecision:
    status: Literal["accepted", "blocked"]
    reason: str | None
    qty: float = 0.0
    notional_usdt: float = 0.0
    margin_usdt: float = 0.0
    risk_usdt: float = 0.0


@dataclass
class SimulatedPosition:
    trade_id: str
    signal_id: str
    symbol: str
    side: Side
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    remaining_qty: float
    notional_usdt: float
    margin_usdt: float
    initial_stop: float
    stop_price: float
    tp1_price: float
    risk_per_unit: float
    fee_usdt: float
    realized_pnl: float
    transform_mode: str = "original"
    original_side: Side | None = None
    original_stop: float | None = None
    original_tp1: float | None = None
    bars_held: int = 0
    scaled_out: bool = False
    highest_price: float = 0.0
    lowest_price: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0


@dataclass
class LiquidityRetestBacktestResult:
    summary: dict[str, Any]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    signals: pd.DataFrame


def normalize_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def beijing_timestamp(timestamp: pd.Timestamp | str) -> str:
    return normalize_timestamp(timestamp).tz_convert("Asia/Shanghai").isoformat()


class EntryCounters:
    def __init__(self, timezone: str = "Asia/Shanghai", entries: list[dict[str, str]] | None = None) -> None:
        self.timezone = timezone
        self.entries: list[dict[str, str]] = list(entries or [])

    def record_entry(self, symbol: str, timestamp: pd.Timestamp | str) -> None:
        ts = normalize_timestamp(timestamp)
        self.entries.append({"symbol": symbol.upper(), "timestamp": ts.isoformat(), "day": self._day_key(ts)})

    def symbol_count(self, symbol: str, timestamp: pd.Timestamp | str) -> int:
        day = self._day_key(timestamp)
        symbol = symbol.upper()
        return sum(1 for entry in self.entries if entry["symbol"] == symbol and entry["day"] == day)

    def account_count(self, timestamp: pd.Timestamp | str) -> int:
        day = self._day_key(timestamp)
        return sum(1 for entry in self.entries if entry["day"] == day)

    def to_dict(self) -> dict[str, Any]:
        return {"timezone": self.timezone, "entries": self.entries}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "EntryCounters":
        payload = payload or {}
        return cls(timezone=str(payload.get("timezone", "Asia/Shanghai")), entries=list(payload.get("entries", [])))

    def _day_key(self, timestamp: pd.Timestamp | str) -> str:
        return normalize_timestamp(timestamp).tz_convert(self.timezone).date().isoformat()


class RiskGate:
    def __init__(self, settings: LiquidityRetestSettings) -> None:
        self.settings = settings

    def evaluate(
        self,
        candidate: SignalCandidate,
        *,
        equity: float,
        open_positions: dict[str, Any],
        counters: EntryCounters,
        cooldowns: dict[str, pd.Timestamp | str],
    ) -> RiskDecision:
        if not _valid_candidate(candidate) or equity <= 0:
            return RiskDecision("blocked", "BLOCKED_INVALID_DATA")
        if candidate.universe_rank is not None and candidate.universe_rank > self.settings.universe_size:
            return RiskDecision("blocked", "BLOCKED_UNIVERSE")
        if candidate.symbol in open_positions:
            return RiskDecision("blocked", "BLOCKED_DUPLICATE_POSITION")
        if len(open_positions) >= self.settings.max_positions:
            return RiskDecision("blocked", "BLOCKED_MAX_POSITIONS")
        cooldown_until = cooldowns.get(candidate.symbol)
        if cooldown_until is not None and normalize_timestamp(cooldown_until) > candidate.signal_time:
            return RiskDecision("blocked", "BLOCKED_STOP_COOLDOWN")
        if counters.symbol_count(candidate.symbol, candidate.signal_time) >= self.settings.max_symbol_entries_per_day:
            return RiskDecision("blocked", "BLOCKED_SYMBOL_DAILY_LIMIT")
        if counters.account_count(candidate.signal_time) >= self.settings.max_account_entries_per_day:
            return RiskDecision("blocked", "BLOCKED_ACCOUNT_DAILY_LIMIT")

        risk_distance = _risk_distance(candidate.side, candidate.planned_entry, candidate.planned_stop)
        if risk_distance <= 0 or risk_distance > candidate.planned_entry * self.settings.max_stop_pct:
            return RiskDecision("blocked", "BLOCKED_INVALID_STOP")
        risk_usdt = float(equity) * self.settings.risk_per_trade
        qty = risk_usdt / risk_distance
        notional = qty * candidate.planned_entry
        margin = notional / self.settings.leverage if self.settings.leverage > 0 else 0.0
        if qty <= 0 or not math.isfinite(qty):
            return RiskDecision("blocked", "BLOCKED_RISK_SIZE", qty=qty, notional_usdt=notional, margin_usdt=margin, risk_usdt=risk_usdt)
        if margin < self.settings.min_margin_usdt:
            return RiskDecision("blocked", "BLOCKED_MIN_MARGIN", qty=qty, notional_usdt=notional, margin_usdt=margin, risk_usdt=risk_usdt)
        return RiskDecision("accepted", None, qty=qty, notional_usdt=notional, margin_usdt=margin, risk_usdt=risk_usdt)


class EventLogger:
    SIGNAL_FIELDS = [
        "signal_id",
        "status",
        "block_reason",
        "symbol",
        "side",
        "transform_mode",
        "original_side",
        "signal_time",
        "fill_time",
        "universe_rank",
        "quote_volume_24h",
        "zone_timeframe",
        "zone_low",
        "zone_high",
        "zone_score",
        "zone_touches",
        "breakout_atr_distance",
        "breakout_body_atr",
        "breakout_quote_volume_ratio",
        "retest_depth_atr",
        "retest_wait_bars",
        "confirm_quote_volume_ratio",
        "daily_symbol_entries",
        "daily_account_entries",
        "open_positions",
        "cooldown_until",
        "planned_entry",
        "planned_stop",
        "planned_tp1",
        "original_stop",
        "original_tp1",
        "risk_usdt",
        "notional_usdt",
        "margin_usdt",
        "leverage",
    ]
    TRADE_FIELDS = [
        "trade_id",
        "signal_id",
        "symbol",
        "side",
        "transform_mode",
        "original_side",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "qty",
        "notional_usdt",
        "margin_usdt",
        "initial_stop",
        "final_stop",
        "partial_tp_price",
        "original_stop",
        "original_tp1",
        "fee_usdt",
        "slippage_bps",
        "pnl",
        "r_multiple",
        "mfe",
        "mae",
        "bars_held",
        "scaled_out",
        "exit_reason",
    ]
    EQUITY_FIELDS = ["timestamp", "equity", "drawdown_pct", "open_positions"]

    def __init__(self, output_dir: str | Path, *, run_id: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def log_event(self, payload: dict[str, Any]) -> None:
        event = dict(payload)
        if "timestamp" in event:
            event.setdefault("timestamp_bj", beijing_timestamp(event["timestamp"]))
        event.setdefault("run_id", self.run_id)
        path = self.output_dir / "events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            handle.flush()

    def log_signal(self, row: dict[str, Any]) -> None:
        self._append_csv(self.output_dir / "signals.csv", self.SIGNAL_FIELDS, row)

    def log_trade(self, row: dict[str, Any]) -> None:
        self._append_csv(self.output_dir / "trades.csv", self.TRADE_FIELDS, row)

    def log_equity(self, row: dict[str, Any]) -> None:
        self._append_csv(self.output_dir / "equity_curve.csv", self.EQUITY_FIELDS, row)

    def write_daily_summary(self, trades: pd.DataFrame) -> None:
        columns = ["trade_day", "entries", "net_profit", "win_rate_pct", "profit_factor", "avg_r", "max_drawdown_pct", "symbols"]
        if trades.empty:
            pd.DataFrame(columns=columns).to_csv(self.output_dir / "daily_summary.csv", index=False)
            return
        data = trades.copy()
        data["trade_day"] = pd.to_datetime(data["entry_time"], utc=True).dt.tz_convert("Asia/Shanghai").dt.date.astype(str)
        rows: list[dict[str, Any]] = []
        for trade_day, group in data.groupby("trade_day", sort=True):
            pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
            winners = pnl[pnl > 0]
            losers = pnl[pnl <= 0]
            gross_loss = float(losers.sum())
            profit_factor = float(winners.sum()) / abs(gross_loss) if gross_loss < 0 else (99.0 if float(winners.sum()) > 0 else 0.0)
            rows.append(
                {
                    "trade_day": trade_day,
                    "entries": int(len(group)),
                    "net_profit": round(float(pnl.sum()), 8),
                    "win_rate_pct": round(float((pnl > 0).mean() * 100), 4),
                    "profit_factor": round(profit_factor, 4),
                    "avg_r": round(float(pd.to_numeric(group.get("r_multiple", 0.0), errors="coerce").fillna(0.0).mean()), 4),
                    "max_drawdown_pct": "",
                    "symbols": ",".join(sorted(group["symbol"].astype(str).unique().tolist())),
                }
            )
        pd.DataFrame(rows, columns=columns).to_csv(self.output_dir / "daily_summary.csv", index=False)

    def write_summary(self, payload: dict[str, Any]) -> None:
        summary = dict(payload)
        summary.setdefault("run_id", self.run_id)
        (self.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def ensure_output_files(self) -> None:
        for filename, fields in {
            "signals.csv": self.SIGNAL_FIELDS,
            "trades.csv": self.TRADE_FIELDS,
            "equity_curve.csv": self.EQUITY_FIELDS,
        }.items():
            path = self.output_dir / filename
            if not path.exists():
                self._append_csv(path, fields, {})
        daily_path = self.output_dir / "daily_summary.csv"
        if not daily_path.exists():
            self.write_daily_summary(pd.DataFrame())
        events_path = self.output_dir / "events.jsonl"
        if not events_path.exists():
            events_path.write_text("", encoding="utf-8")

    @staticmethod
    def _append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            if row:
                writer.writerow({field: row.get(field, "") for field in fields})
            handle.flush()


class LiquidityRetestPortfolioSimulator:
    def __init__(
        self,
        settings: LiquidityRetestSettings,
        *,
        logger: EventLogger,
        initial_equity: float = 10_000.0,
        counters: EntryCounters | None = None,
        cooldowns: dict[str, pd.Timestamp | str] | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger
        self.initial_equity = float(initial_equity)
        self.cash = float(initial_equity)
        self.peak_equity = float(initial_equity)
        self.last_marked_equity = float(initial_equity)
        self.positions: dict[str, SimulatedPosition] = {}
        self.counters = counters or EntryCounters(settings.daily_reset_timezone)
        self.cooldowns: dict[str, pd.Timestamp | str] = dict(cooldowns or {})
        self.risk_gate = RiskGate(settings)
        self.trades: list[dict[str, Any]] = []
        self.equity_rows: list[dict[str, Any]] = []

    @property
    def equity(self) -> float:
        return self.cash

    @property
    def current_drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        reference_equity = min(self.cash, self.last_marked_equity)
        return max(0.0, 1.0 - (reference_equity / self.peak_equity))

    @property
    def entries_enabled(self) -> bool:
        return self.current_drawdown_pct < self.settings.max_drawdown_pct

    def open_from_candidate(self, candidate: SignalCandidate, *, fill_price: float, fill_time: pd.Timestamp | str) -> bool:
        fill_time = normalize_timestamp(fill_time)
        fill_candidate = replace(candidate, planned_entry=float(fill_price), signal_time=fill_time)
        if not self.entries_enabled:
            decision = RiskDecision("blocked", "BLOCKED_DRAWDOWN_LIMIT")
            self.logger.log_signal(signal_row(fill_candidate, decision, self.counters, len(self.positions), None, self.settings))
            self.logger.log_event(
                {
                    "event": "RISK_BLOCK",
                    "timestamp": fill_time.isoformat(),
                    "symbol": fill_candidate.symbol,
                    "side": fill_candidate.side,
                    "reason": decision.reason,
                    "drawdown_pct": self.current_drawdown_pct,
                }
            )
            return False
        decision = self.risk_gate.evaluate(
            fill_candidate,
            equity=self.equity,
            open_positions=self.positions,
            counters=self.counters,
            cooldowns=self.cooldowns,
        )
        self.logger.log_signal(signal_row(fill_candidate, decision, self.counters, len(self.positions), fill_time if decision.status == "accepted" else None, self.settings))
        if decision.status != "accepted":
            self.logger.log_event({"event": "RISK_BLOCK", "timestamp": fill_time.isoformat(), "symbol": candidate.symbol, "side": candidate.side, "reason": decision.reason})
            return False
        fee = decision.notional_usdt * self.settings.fee_rate
        self.cash -= fee
        trade_id = f"trade-{candidate.signal_id}"
        tp1_price = (
            float(fill_candidate.planned_tp1)
            if fill_candidate.planned_tp1 is not None
            else _tp1_price(fill_candidate.side, float(fill_price), fill_candidate.planned_stop, self.settings.partial_take_profit_r)
        )
        position = SimulatedPosition(
            trade_id=trade_id,
            signal_id=candidate.signal_id,
            symbol=candidate.symbol,
            side=fill_candidate.side,
            entry_time=fill_time,
            entry_price=float(fill_price),
            qty=decision.qty,
            remaining_qty=decision.qty,
            notional_usdt=decision.notional_usdt,
            margin_usdt=decision.margin_usdt,
            initial_stop=fill_candidate.planned_stop,
            stop_price=fill_candidate.planned_stop,
            tp1_price=tp1_price,
            risk_per_unit=abs(float(fill_price) - fill_candidate.planned_stop),
            fee_usdt=fee,
            realized_pnl=-fee,
            transform_mode=fill_candidate.transform_mode,
            original_side=fill_candidate.original_side,
            original_stop=fill_candidate.original_stop,
            original_tp1=fill_candidate.original_tp1,
            highest_price=float(fill_price),
            lowest_price=float(fill_price),
        )
        self.positions[candidate.symbol] = position
        self.counters.record_entry(candidate.symbol, fill_time)
        self.logger.log_event({"event": "OPEN", "timestamp": fill_time.isoformat(), "symbol": candidate.symbol, "side": candidate.side, "entry_price": fill_price})
        return True

    def manage_positions(self, timestamp: pd.Timestamp | str, rows_by_symbol: dict[str, pd.Series]) -> None:
        timestamp = normalize_timestamp(timestamp)
        for symbol, position in list(self.positions.items()):
            row = rows_by_symbol.get(symbol)
            if row is None:
                continue
            position.bars_held += 1
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            atr_value = float(row.get("atr", 0.0) or 0.0)
            self._update_excursion(position, high, low)

            exit_price: float | None = None
            exit_reason = ""
            if position.side == "long":
                if low <= position.stop_price:
                    exit_price = position.stop_price
                    exit_reason = "stop_loss" if not position.scaled_out else "trail_stop"
                elif (not position.scaled_out) and high >= position.tp1_price:
                    self._partial_exit(position, position.tp1_price)
                    position.stop_price = max(position.stop_price, _breakeven_price(position))
                    position.scaled_out = True
                    position.highest_price = max(position.highest_price, high, close)
                    if atr_value > 0:
                        position.stop_price = max(position.stop_price, position.highest_price - (atr_value * self.settings.trailing_atr))
            else:
                if high >= position.stop_price:
                    exit_price = position.stop_price
                    exit_reason = "stop_loss" if not position.scaled_out else "trail_stop"
                elif (not position.scaled_out) and low <= position.tp1_price:
                    self._partial_exit(position, position.tp1_price)
                    position.stop_price = min(position.stop_price, _breakeven_price(position))
                    position.scaled_out = True
                    position.lowest_price = min(position.lowest_price, low, close)
                    if atr_value > 0:
                        position.stop_price = min(position.stop_price, position.lowest_price + (atr_value * self.settings.trailing_atr))

            if exit_price is None and position.scaled_out and atr_value > 0:
                if position.side == "long":
                    position.highest_price = max(position.highest_price, high, close)
                    position.stop_price = max(position.stop_price, position.highest_price - (atr_value * self.settings.trailing_atr))
                    if low <= position.stop_price:
                        exit_price = position.stop_price
                        exit_reason = "trail_stop"
                else:
                    position.lowest_price = min(position.lowest_price, low, close)
                    position.stop_price = min(position.stop_price, position.lowest_price + (atr_value * self.settings.trailing_atr))
                    if high >= position.stop_price:
                        exit_price = position.stop_price
                        exit_reason = "trail_stop"
            if exit_price is None and position.bars_held >= self.settings.max_hold_5m_bars:
                exit_price = close
                exit_reason = "time_exit"
            if exit_price is not None:
                self._close_position(position, timestamp, exit_price, exit_reason)

    def close_all(self, timestamp: pd.Timestamp | str, marks: dict[str, float], *, reason: str) -> None:
        timestamp = normalize_timestamp(timestamp)
        for symbol, position in list(self.positions.items()):
            self._close_position(position, timestamp, float(marks.get(symbol, position.entry_price)), reason)

    def mark_equity(self, timestamp: pd.Timestamp | str, marks: dict[str, float]) -> None:
        timestamp = normalize_timestamp(timestamp)
        unrealized = sum(_gross_pnl(pos.side, pos.entry_price, float(marks.get(symbol, pos.entry_price)), pos.remaining_qty) for symbol, pos in self.positions.items())
        equity = self.cash + unrealized
        self.peak_equity = max(self.peak_equity, equity)
        self.last_marked_equity = equity
        row = {"timestamp": timestamp.isoformat(), "equity": equity, "drawdown_pct": (equity / self.peak_equity) - 1, "open_positions": len(self.positions)}
        self.equity_rows.append(row)
        self.logger.log_equity(row)

    def summary(self) -> dict[str, Any]:
        trades = pd.DataFrame(self.trades)
        final_equity = self.cash
        if trades.empty:
            return {"initial_capital": self.initial_equity, "final_equity": round(final_equity, 8), "net_profit": round(final_equity - self.initial_equity, 8), "total_trades": 0}
        pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
        winners = pnl[pnl > 0]
        losers = pnl[pnl <= 0]
        gross_loss = float(losers.sum())
        profit_factor = float(winners.sum()) / abs(gross_loss) if gross_loss < 0 else (99.0 if float(winners.sum()) > 0 else 0.0)
        return {
            "initial_capital": self.initial_equity,
            "final_equity": round(final_equity, 8),
            "net_profit": round(final_equity - self.initial_equity, 8),
            "return_pct": round(((final_equity / self.initial_equity) - 1.0) * 100.0, 4),
            "total_trades": int(len(trades)),
            "win_rate_pct": round(float((pnl > 0).mean() * 100.0), 4),
            "profit_factor": round(profit_factor, 4),
        }

    def _partial_exit(self, position: SimulatedPosition, exit_price: float) -> None:
        qty = min(position.remaining_qty, position.qty * self.settings.partial_close_ratio)
        gross = _gross_pnl(position.side, position.entry_price, exit_price, qty)
        fee = exit_price * qty * self.settings.fee_rate
        position.remaining_qty -= qty
        position.realized_pnl += gross - fee
        position.fee_usdt += fee
        self.cash += gross - fee
        self.logger.log_event({"event": "PARTIAL_TAKE_PROFIT", "timestamp": position.entry_time.isoformat(), "symbol": position.symbol, "side": position.side, "price": exit_price, "qty": qty})

    def _close_position(self, position: SimulatedPosition, timestamp: pd.Timestamp, exit_price: float, reason: str) -> None:
        gross = _gross_pnl(position.side, position.entry_price, exit_price, position.remaining_qty)
        fee = exit_price * position.remaining_qty * self.settings.fee_rate
        pnl = position.realized_pnl + gross - fee
        self.cash += gross - fee
        position.fee_usdt += fee
        trade = {
            "trade_id": position.trade_id,
            "signal_id": position.signal_id,
            "symbol": position.symbol,
            "side": position.side,
            "transform_mode": position.transform_mode,
            "original_side": position.original_side or "",
            "entry_time": position.entry_time.isoformat(),
            "exit_time": timestamp.isoformat(),
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "qty": position.qty,
            "notional_usdt": position.notional_usdt,
            "margin_usdt": position.margin_usdt,
            "initial_stop": position.initial_stop,
            "final_stop": position.stop_price,
            "partial_tp_price": position.tp1_price,
            "original_stop": position.original_stop if position.original_stop is not None else "",
            "original_tp1": position.original_tp1 if position.original_tp1 is not None else "",
            "fee_usdt": position.fee_usdt,
            "slippage_bps": self.settings.slippage_bps,
            "pnl": pnl,
            "r_multiple": pnl / (position.risk_per_unit * position.qty) if position.risk_per_unit > 0 and position.qty > 0 else 0.0,
            "mfe": position.mfe,
            "mae": position.mae,
            "bars_held": position.bars_held,
            "scaled_out": position.scaled_out,
            "exit_reason": reason,
        }
        self.trades.append(trade)
        self.logger.log_trade(trade)
        self.logger.log_event({"event": "CLOSE", "timestamp": timestamp.isoformat(), "symbol": position.symbol, "side": position.side, "exit_price": exit_price, "reason": reason})
        if reason == "stop_loss":
            self.cooldowns[position.symbol] = (timestamp + pd.Timedelta(minutes=self.settings.stop_cooldown_minutes)).isoformat()
        del self.positions[position.symbol]

    @staticmethod
    def _update_excursion(position: SimulatedPosition, high: float, low: float) -> None:
        if position.side == "long":
            position.highest_price = max(position.highest_price, high)
            position.lowest_price = min(position.lowest_price, low)
            position.mfe = max(position.mfe, high - position.entry_price)
            position.mae = min(position.mae, low - position.entry_price)
        else:
            position.highest_price = max(position.highest_price, high)
            position.lowest_price = min(position.lowest_price, low)
            position.mfe = max(position.mfe, position.entry_price - low)
            position.mae = min(position.mae, position.entry_price - high)


def timeframe_to_rule(timeframe: str) -> str:
    suffix = timeframe[-1].lower()
    mapping = {"m": "min", "h": "h", "d": "D"}
    if suffix not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return f"{timeframe[:-1]}{mapping[suffix]}"


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def ensure_utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy().sort_index()
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")
    return data


def closed_resample_ohlcv(frame: pd.DataFrame, timeframe: str, as_of: pd.Timestamp | str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    data = ensure_utc_index(frame)
    cutoff = normalize_timestamp(as_of).floor(timeframe_to_rule(timeframe))
    visible = data[data.index < cutoff]
    if visible.empty:
        return pd.DataFrame()

    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "quote_volume" in visible.columns:
        agg["quote_volume"] = "sum"
    resampled = visible.resample(timeframe_to_rule(timeframe)).agg(agg).dropna(subset=["open", "high", "low", "close"])
    resampled.index.name = "timestamp"
    return resampled


def select_top_quote_volume_symbols(
    frames_by_symbol: dict[str, pd.DataFrame],
    as_of: pd.Timestamp | str,
    settings: LiquidityRetestSettings,
) -> tuple[list[str], pd.DataFrame]:
    end = normalize_timestamp(as_of)
    start = end - pd.Timedelta(hours=settings.quote_volume_lookback_hours)
    rows: list[dict[str, Any]] = []
    stablecoin_bases = {item.upper() for item in settings.stablecoin_bases}

    for raw_symbol, frame in frames_by_symbol.items():
        symbol = raw_symbol.upper()
        base = symbol[:-4] if symbol.endswith("USDT") else symbol.split("/")[0].upper()
        if base in stablecoin_bases:
            continue
        if frame.empty:
            continue
        data = ensure_utc_index(frame)
        visible = data[(data.index >= start) & (data.index < end)]
        if visible.empty:
            continue
        if "quote_volume" in visible.columns:
            quote_volume = float(pd.to_numeric(visible["quote_volume"], errors="coerce").fillna(0.0).sum())
        else:
            quote_volume = float((visible["close"] * visible["volume"]).fillna(0.0).sum())
        if not math.isfinite(quote_volume) or quote_volume <= 0:
            continue
        rows.append({"symbol": symbol, "quote_volume_24h": quote_volume})

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return [], pd.DataFrame(columns=["symbol", "quote_volume_24h", "universe_rank"])
    ranking = ranking.sort_values(["quote_volume_24h", "symbol"], ascending=[False, True]).reset_index(drop=True)
    ranking["universe_rank"] = np.arange(1, len(ranking) + 1)
    selected = ranking.head(max(1, settings.universe_size))["symbol"].tolist()
    return selected, ranking


def detect_zones(
    frame: pd.DataFrame,
    timeframe: str,
    as_of: pd.Timestamp | str,
    settings: LiquidityRetestSettings,
) -> list[Zone]:
    if frame.empty:
        return []
    data = ensure_utc_index(frame)
    cutoff = normalize_timestamp(as_of).floor(timeframe_to_rule(timeframe))
    data = data[data.index < cutoff].copy()
    lookback = settings.zone_lookback_4h if timeframe == "4h" else settings.zone_lookback_1h
    data = data.tail(max(lookback, settings.pivot_left_bars + settings.pivot_right_bars + 1))
    left = max(1, settings.pivot_left_bars)
    right = max(1, settings.pivot_right_bars)
    if len(data) < left + right + 1:
        return []

    atr_values = atr(data, 14)
    fallback_atr = float((data["high"] - data["low"]).replace(0, np.nan).dropna().mean() or 0.0)
    merge_width = float((atr_values.dropna().iloc[-1] if not atr_values.dropna().empty else fallback_atr) * settings.zone_merge_atr)
    if not math.isfinite(merge_width) or merge_width <= 0:
        merge_width = max(float(data["close"].iloc[-1]) * 0.001, 1e-9)

    resistance_points: list[tuple[float, pd.Timestamp, int]] = []
    support_points: list[tuple[float, pd.Timestamp, int]] = []
    highs = data["high"].to_numpy(dtype=float)
    lows = data["low"].to_numpy(dtype=float)
    timestamps = data.index.to_list()

    for index in range(left, len(data) - right):
        high_window = highs[index - left : index + right + 1]
        low_window = lows[index - left : index + right + 1]
        if highs[index] == np.max(high_window) and highs[index] > np.max(highs[index + 1 : index + right + 1]):
            resistance_points.append((float(highs[index]), timestamps[index], index))
        if lows[index] == np.min(low_window) and lows[index] < np.min(lows[index + 1 : index + right + 1]):
            support_points.append((float(lows[index]), timestamps[index], index))

    zones: list[Zone] = []
    zones.extend(_cluster_points(resistance_points, "resistance", timeframe, merge_width, len(data), settings))
    zones.extend(_cluster_points(support_points, "support", timeframe, merge_width, len(data), settings))
    return sorted(zones, key=lambda item: (-item.score, -item.last_touch_time.value, item.midpoint))


def _cluster_points(
    points: list[tuple[float, pd.Timestamp, int]],
    side: ZoneSide,
    timeframe: str,
    merge_width: float,
    total_bars: int,
    settings: LiquidityRetestSettings,
) -> list[Zone]:
    if not points:
        return []
    points = sorted(points, key=lambda item: item[0])
    clusters: list[list[tuple[float, pd.Timestamp, int]]] = []
    for point in points:
        if not clusters:
            clusters.append([point])
            continue
        current_mid = float(np.mean([item[0] for item in clusters[-1]]))
        if abs(point[0] - current_mid) <= merge_width:
            clusters[-1].append(point)
        else:
            clusters.append([point])

    zones: list[Zone] = []
    weight = 2.0 if timeframe == "4h" else 1.0
    lookback = settings.zone_lookback_4h if timeframe == "4h" else settings.zone_lookback_1h
    for cluster in clusters:
        spaced: list[tuple[float, pd.Timestamp, int]] = []
        for point in sorted(cluster, key=lambda item: item[2]):
            if not spaced or point[2] - spaced[-1][2] >= 3:
                spaced.append(point)
        if len(spaced) < settings.zone_min_touches:
            continue
        prices = [item[0] for item in spaced]
        last_index = max(item[2] for item in spaced)
        bars_since = max(0, total_bars - 1 - last_index)
        recency = max(0.0, 1.0 - (bars_since / max(1, lookback)))
        zones.append(
            Zone(
                side=side,
                timeframe=timeframe,
                low=float(min(prices)),
                high=float(max(prices)),
                touches=len(spaced),
                score=weight + min(len(spaced), 5) + recency,
                last_touch_time=max(item[1] for item in spaced),
            )
        )
    return zones


class LiquidityRetestSignalEngine:
    def __init__(self, settings: LiquidityRetestSettings | None = None) -> None:
        self.settings = settings or LiquidityRetestSettings()
        self.states: dict[tuple[str, Side], SetupState] = {}
        self.last_1h_breakout_scan: dict[str, pd.Timestamp] = {}
        self.emitted_signal_ids: set[str] = set()

    def on_bar(
        self,
        symbol: str,
        history_5m: pd.DataFrame,
        top_symbols: set[str] | list[str],
        as_of: pd.Timestamp | str,
        universe_row: dict[str, Any] | None = None,
    ) -> list[SignalCandidate]:
        symbol = symbol.upper()
        as_of_ts = normalize_timestamp(as_of)
        top_set = {item.upper() for item in top_symbols}
        if symbol not in top_set:
            for side in ("long", "short"):
                self.states.pop((symbol, side), None)
            return []
        if history_5m.empty:
            return []

        data = ensure_utc_index(history_5m)
        candidates = self._advance_states(symbol, data, as_of_ts, universe_row or {})
        self._scan_new_breakout(symbol, data, as_of_ts)
        return candidates

    def _advance_states(
        self,
        symbol: str,
        data: pd.DataFrame,
        as_of: pd.Timestamp,
        universe_row: dict[str, Any],
    ) -> list[SignalCandidate]:
        row = data.iloc[-1]
        current_time = data.index[-1]
        atr_5m = _latest_atr(data)
        output: list[SignalCandidate] = []

        for key, state in list(self.states.items()):
            if key[0] != symbol or state.zone is None or state.side is None:
                continue
            state.wait_bars += 1
            if state.state == "WAIT_RETEST" and state.wait_bars > self.settings.retest_max_5m_bars:
                del self.states[key]
                continue
            zone = state.zone
            side = state.side
            tolerance = atr_5m * self.settings.retest_tolerance_atr
            if state.state == "WAIT_RETEST":
                if side == "long":
                    touched = float(row["low"]) <= zone.high + tolerance
                    accepted = float(row["close"]) > zone.high
                    depth = max(0.0, (zone.high - float(row["low"])) / atr_5m) if atr_5m > 0 else 0.0
                else:
                    touched = float(row["high"]) >= zone.low - tolerance
                    accepted = float(row["close"]) < zone.low
                    depth = max(0.0, (float(row["high"]) - zone.low) / atr_5m) if atr_5m > 0 else 0.0
                if touched and accepted:
                    state.state = "WAIT_CONFIRM"
                    state.reject_time = current_time
                    state.reject_high = float(row["high"])
                    state.reject_low = float(row["low"])
                    state.retest_depth_atr = depth
                    state.wait_bars = 0
                continue

            if state.state != "WAIT_CONFIRM":
                continue
            if state.reject_high is None or state.reject_low is None:
                del self.states[key]
                continue
            quote_ratio = _quote_volume_ratio(data, self.settings.confirm_volume_window)
            if side == "long":
                confirmed = float(row["close"]) > state.reject_high
                stop = min(state.reject_low, zone.low) - (atr_5m * 0.3)
            else:
                confirmed = float(row["close"]) < state.reject_low
                stop = max(state.reject_high, zone.high) + (atr_5m * 0.3)
            if not confirmed or quote_ratio < self.settings.confirm_min_quote_volume_ratio:
                continue
            signal_id = f"{symbol}-{side}-{current_time.isoformat()}"
            if signal_id in self.emitted_signal_ids:
                continue
            self.emitted_signal_ids.add(signal_id)
            output.append(
                SignalCandidate(
                    signal_id=signal_id,
                    symbol=symbol,
                    side=side,
                    signal_time=current_time,
                    zone=zone,
                    planned_entry=float(row["close"]),
                    planned_stop=float(stop),
                    atr_5m=atr_5m,
                    universe_rank=_optional_int(universe_row.get("universe_rank")),
                    quote_volume_24h=_optional_float(universe_row.get("quote_volume_24h")),
                    breakout_atr_distance=state.breakout_atr_distance,
                    breakout_body_atr=state.breakout_body_atr,
                    breakout_quote_volume_ratio=state.breakout_quote_volume_ratio,
                    retest_depth_atr=state.retest_depth_atr,
                    retest_wait_bars=state.wait_bars,
                    confirm_quote_volume_ratio=quote_ratio,
                )
            )
            del self.states[key]
        return output

    def _scan_new_breakout(self, symbol: str, data_5m: pd.DataFrame, as_of: pd.Timestamp) -> None:
        expected_latest_time = as_of.floor("h") - pd.Timedelta(hours=1)
        if self.last_1h_breakout_scan.get(symbol) == expected_latest_time:
            return
        h1 = closed_resample_ohlcv(data_5m, "1h", as_of)
        if len(h1) < self.settings.pivot_left_bars + self.settings.pivot_right_bars + 3:
            return
        latest_time = h1.index[-1]
        if self.last_1h_breakout_scan.get(symbol) == latest_time:
            return
        self.last_1h_breakout_scan[symbol] = latest_time
        latest = h1.iloc[-1]
        prior = h1.iloc[:-1]
        if prior.empty:
            return

        zones_1h = detect_zones(prior, "1h", latest_time, self.settings)
        if not zones_1h:
            return
        h1_atr = _latest_atr(h1)
        if h1_atr <= 0:
            return
        body_atr = abs(float(latest["close"]) - float(latest["open"])) / h1_atr
        quote_ratio = _quote_volume_ratio(h1, self.settings.breakout_volume_window)
        if body_atr < self.settings.breakout_body_atr or quote_ratio < self.settings.breakout_min_quote_volume_ratio:
            return

        resistance_zones = [zone for zone in zones_1h if zone.side == "resistance"]
        support_zones = [zone for zone in zones_1h if zone.side == "support"]
        for zone in resistance_zones:
            distance_atr = (float(latest["close"]) - zone.high) / h1_atr
            if distance_atr >= self.settings.breakout_buffer_atr:
                self.states[(symbol, "long")] = SetupState(
                    state="WAIT_RETEST",
                    side="long",
                    zone=zone,
                    breakout_time=latest_time,
                    breakout_price=float(latest["close"]),
                    breakout_atr_distance=distance_atr,
                    breakout_body_atr=body_atr,
                    breakout_quote_volume_ratio=quote_ratio,
                )
                break
        for zone in support_zones:
            distance_atr = (zone.low - float(latest["close"])) / h1_atr
            if distance_atr >= self.settings.breakout_buffer_atr:
                self.states[(symbol, "short")] = SetupState(
                    state="WAIT_RETEST",
                    side="short",
                    zone=zone,
                    breakout_time=latest_time,
                    breakout_price=float(latest["close"]),
                    breakout_atr_distance=distance_atr,
                    breakout_body_atr=body_atr,
                    breakout_quote_volume_ratio=quote_ratio,
                )
                break


def _latest_atr(frame: pd.DataFrame) -> float:
    values = atr(frame, 14).dropna()
    if not values.empty and math.isfinite(float(values.iloc[-1])) and float(values.iloc[-1]) > 0:
        return float(values.iloc[-1])
    fallback = (frame["high"] - frame["low"]).replace(0, np.nan).dropna()
    if not fallback.empty:
        return float(fallback.mean())
    return 0.0


def _quote_volume_ratio(frame: pd.DataFrame, window: int) -> float:
    if frame.empty:
        return 0.0
    quote_volume = frame["quote_volume"] if "quote_volume" in frame.columns else frame["close"] * frame["volume"]
    current = float(quote_volume.iloc[-1])
    previous = quote_volume.iloc[:-1].tail(max(1, window))
    if previous.empty:
        return 0.0
    mean = float(previous.mean())
    if not math.isfinite(mean) or mean <= 0:
        return 0.0
    return current / mean


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _valid_candidate(candidate: SignalCandidate) -> bool:
    return (
        bool(candidate.symbol)
        and candidate.side in {"long", "short"}
        and math.isfinite(float(candidate.planned_entry))
        and math.isfinite(float(candidate.planned_stop))
        and float(candidate.planned_entry) > 0
    )


def _risk_distance(side: Side, entry: float, stop: float) -> float:
    if side == "long":
        return entry - stop
    return stop - entry


def _tp1_price(side: Side, entry: float, stop: float, rr: float) -> float:
    distance = abs(entry - stop)
    if side == "long":
        return entry + (distance * rr)
    return entry - (distance * rr)


def transform_candidate(candidate: SignalCandidate, settings: LiquidityRetestSettings) -> SignalTransformResult:
    original_tp1 = candidate.planned_tp1
    if original_tp1 is None:
        original_tp1 = _tp1_price(
            candidate.side,
            candidate.planned_entry,
            candidate.planned_stop,
            settings.partial_take_profit_r,
        )
    if settings.trade_transform_mode == "original":
        return SignalTransformResult(replace(candidate, planned_tp1=float(original_tp1)), "kept")
    if settings.trade_transform_mode != "mirror_original_longs_short_only":
        raise ValueError(f"Unsupported trade transform mode: {settings.trade_transform_mode}")
    if candidate.side != "long":
        return SignalTransformResult(None, "filtered", "FILTERED_SHORT_ONLY_SOURCE_SIDE")
    mirrored = replace(
        candidate,
        side="short",
        planned_stop=float(original_tp1),
        planned_tp1=float(candidate.planned_stop),
        transform_mode=settings.trade_transform_mode,
        original_side=candidate.side,
        original_stop=float(candidate.planned_stop),
        original_tp1=float(original_tp1),
    )
    return SignalTransformResult(mirrored, "kept")


def process_raw_candidate(
    raw: SignalCandidate,
    settings: LiquidityRetestSettings,
    logger: EventLogger,
) -> SignalCandidate | None:
    result = transform_candidate(raw, settings)
    if result.candidate is None:
        logger.log_event(
            {
                "event": "SIGNAL_FILTERED",
                "timestamp": raw.signal_time.isoformat(),
                "symbol": raw.symbol,
                "side": raw.side,
                "reason": result.reason,
                "transform_mode": settings.trade_transform_mode,
            }
        )
    return result.candidate


def _gross_pnl(side: Side, entry: float, exit_price: float, qty: float) -> float:
    if side == "long":
        return (exit_price - entry) * qty
    return (entry - exit_price) * qty


def _breakeven_price(position: SimulatedPosition) -> float:
    if position.remaining_qty <= 0:
        return position.entry_price
    fee_per_unit = position.fee_usdt / max(position.qty, 1e-12)
    if position.side == "long":
        return position.entry_price + fee_per_unit
    return position.entry_price - fee_per_unit


def signal_row(
    candidate: SignalCandidate,
    decision: RiskDecision,
    counters: EntryCounters,
    open_positions: int,
    fill_time: pd.Timestamp | None,
    settings: LiquidityRetestSettings,
) -> dict[str, Any]:
    cooldown_until = ""
    return {
        "signal_id": candidate.signal_id,
        "status": decision.status,
        "block_reason": decision.reason or "",
        "symbol": candidate.symbol,
        "side": candidate.side,
        "transform_mode": candidate.transform_mode,
        "original_side": candidate.original_side or "",
        "signal_time": candidate.signal_time.isoformat(),
        "fill_time": "" if fill_time is None else fill_time.isoformat(),
        "universe_rank": candidate.universe_rank or "",
        "quote_volume_24h": candidate.quote_volume_24h or "",
        "zone_timeframe": candidate.zone.timeframe,
        "zone_low": candidate.zone.low,
        "zone_high": candidate.zone.high,
        "zone_score": candidate.zone.score,
        "zone_touches": candidate.zone.touches,
        "breakout_atr_distance": candidate.breakout_atr_distance,
        "breakout_body_atr": candidate.breakout_body_atr,
        "breakout_quote_volume_ratio": candidate.breakout_quote_volume_ratio,
        "retest_depth_atr": candidate.retest_depth_atr,
        "retest_wait_bars": candidate.retest_wait_bars,
        "confirm_quote_volume_ratio": candidate.confirm_quote_volume_ratio,
        "daily_symbol_entries": counters.symbol_count(candidate.symbol, candidate.signal_time),
        "daily_account_entries": counters.account_count(candidate.signal_time),
        "open_positions": open_positions,
        "cooldown_until": cooldown_until,
        "planned_entry": candidate.planned_entry,
        "planned_stop": candidate.planned_stop,
        "planned_tp1": candidate.planned_tp1 if candidate.planned_tp1 is not None else "",
        "original_stop": candidate.original_stop if candidate.original_stop is not None else "",
        "original_tp1": candidate.original_tp1 if candidate.original_tp1 is not None else "",
        "risk_usdt": decision.risk_usdt,
        "notional_usdt": decision.notional_usdt,
        "margin_usdt": decision.margin_usdt,
        "leverage": settings.leverage,
    }
