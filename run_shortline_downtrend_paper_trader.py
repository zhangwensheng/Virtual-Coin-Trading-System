from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from futures_strategy.shortline_downtrend import (
    ShortlineDowntrendSettings,
    build_market_context,
    prepare_shortline_downtrend_frame,
    score_shortline_downtrend,
)


BINANCE_FAPI = "https://fapi.binance.com"
INTERVAL_MS = 60_000
DEFAULT_ANCHORS = ["BTCUSDT", "ETHUSDT"]


@dataclass
class ShortlinePaperRuntime:
    output_dir: str = "outputs/shortline_downtrend_score_paper_10x_1u"
    poll_sec: int = 60
    scan_limit: int = 80
    leverage: int = 10
    margin_usdt: float = 1.0
    max_positions: int = 1
    initial_balance: float = 100.0
    daily_loss_limit_usdt: float = 1.0
    max_consecutive_losses: int = 2
    max_bars_in_trade: int = 60
    stop_atr_mult: float = 1.05
    stop_min_pct: float = 0.004
    stop_max_pct: float = 0.035
    partial_rr: float = 1.05
    trail_atr_mult: float = 1.0
    fee_rate: float = 0.0004
    slippage_bps: float = 1.0
    once: bool = False


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def request_json(session: requests.Session, path: str, params: dict[str, Any] | None = None) -> Any:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = session.get(f"{BINANCE_FAPI}{path}", params=params, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"Binance request failed: {path} {params or {}}") from last_error


def klines_to_frame(rows: list[list[Any]], *, server_time_ms: int | None = None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    for column in ["open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_quote", "close_time"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if server_time_ms is not None:
        frame = frame[frame["close_time"] <= server_time_ms]
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame = frame.set_index("timestamp").sort_index()
    return frame[["open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_quote"]].dropna(
        subset=["open", "high", "low", "close", "volume"]
    )


def fetch_1m_frame(session: requests.Session, symbol: str, *, server_time_ms: int, hours: int = 20) -> pd.DataFrame:
    end_time = pd.to_datetime(server_time_ms, unit="ms", utc=True)
    start = (end_time - pd.Timedelta(hours=hours)).value // 1_000_000
    rows = request_json(
        session,
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": "1m", "startTime": int(start), "limit": 1200},
    )
    return klines_to_frame(rows, server_time_ms=server_time_ms)


def fetch_active_symbols(session: requests.Session) -> set[str]:
    payload = request_json(session, "/fapi/v1/exchangeInfo")
    symbols: set[str] = set()
    for row in payload.get("symbols", []):
        if (
            row.get("contractType") == "PERPETUAL"
            and row.get("quoteAsset") == "USDT"
            and row.get("status") == "TRADING"
        ):
            symbols.add(str(row.get("symbol", "")).upper())
    return symbols


def fetch_scan_symbols(
    session: requests.Session,
    active_symbols: set[str],
    *,
    scan_limit: int,
    open_symbols: set[str],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    tickers = request_json(session, "/fapi/v1/ticker/24hr")
    rows: list[dict[str, Any]] = []
    ticker_by_symbol: dict[str, dict[str, Any]] = {}
    for row in tickers:
        symbol = str(row.get("symbol", "")).upper()
        if symbol not in active_symbols:
            continue
        quote_volume = _to_float(row.get("quoteVolume"))
        change_pct = _to_float(row.get("priceChangePercent"))
        if quote_volume <= 0:
            continue
        normalized = {**row, "quoteVolume": quote_volume, "priceChangePercent": change_pct}
        ticker_by_symbol[symbol] = normalized
        rows.append(normalized)

    by_volume = sorted(rows, key=lambda item: _to_float(item.get("quoteVolume")), reverse=True)
    by_gain = sorted(rows, key=lambda item: _to_float(item.get("priceChangePercent")), reverse=True)
    by_loss = sorted(rows, key=lambda item: _to_float(item.get("priceChangePercent")))
    by_abs_move = sorted(rows, key=lambda item: abs(_to_float(item.get("priceChangePercent"))), reverse=True)

    selected: list[str] = []
    for source, count in [
        (by_volume, max(20, int(scan_limit * 0.55))),
        (by_gain, max(10, int(scan_limit * 0.25))),
        (by_loss, max(8, int(scan_limit * 0.15))),
        (by_abs_move, max(10, int(scan_limit * 0.20))),
    ]:
        for row in source[:count]:
            symbol = str(row.get("symbol", "")).upper()
            if symbol and symbol not in selected:
                selected.append(symbol)
            if len(selected) >= scan_limit:
                break
        if len(selected) >= scan_limit:
            break

    for symbol in [*DEFAULT_ANCHORS, *sorted(open_symbols)]:
        if symbol in active_symbols and symbol not in selected:
            selected.append(symbol)
    return selected, ticker_by_symbol


def load_state(path: Path, initial_balance: float) -> dict[str, Any]:
    if not path.exists():
        return {
            "positions": {},
            "realized_pnl": 0.0,
            "initial_balance": initial_balance,
            "day_key": "",
            "day_start_realized": 0.0,
            "consecutive_losses": 0,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("positions", {})
    payload.setdefault("realized_pnl", 0.0)
    payload.setdefault("initial_balance", initial_balance)
    payload.setdefault("day_key", "")
    payload.setdefault("day_start_realized", float(payload.get("realized_pnl", 0.0)))
    payload.setdefault("consecutive_losses", 0)
    return payload


def reset_day_if_needed(state: dict[str, Any], now: pd.Timestamp) -> None:
    day_key = now.strftime("%Y-%m-%d")
    if state.get("day_key") != day_key:
        state["day_key"] = day_key
        state["day_start_realized"] = float(state.get("realized_pnl", 0.0))
        state["consecutive_losses"] = 0


def apply_slippage_short(price: float, *, is_entry: bool, slippage_bps: float) -> float:
    slip = slippage_bps / 10_000.0
    return price * (1 - slip) if is_entry else price * (1 + slip)


def position_unrealized(position: dict[str, Any], mark: float) -> float:
    return (float(position["entry_price"]) - mark) * float(position["remaining_qty"])


def record_closed_trade(state: dict[str, Any], pnl: float) -> None:
    state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + pnl
    if pnl < 0:
        state["consecutive_losses"] = int(state.get("consecutive_losses", 0)) + 1
    elif pnl > 0:
        state["consecutive_losses"] = 0


def close_position(
    state: dict[str, Any],
    symbol: str,
    position: dict[str, Any],
    *,
    timestamp: pd.Timestamp,
    exit_price: float,
    reason: str,
    runtime: ShortlinePaperRuntime,
    trades_path: Path,
) -> dict[str, Any]:
    remaining_qty = float(position["remaining_qty"])
    gross = (float(position["entry_price"]) - exit_price) * remaining_qty
    exit_fee = exit_price * remaining_qty * runtime.fee_rate
    pnl = float(position["realized_pnl"]) + gross - exit_fee
    event = {
        "event": "CLOSE",
        "timestamp": timestamp.isoformat(),
        "symbol": symbol,
        "side": "short",
        "entry_time": position["entry_time"],
        "entry_price": position["entry_price"],
        "exit_price": exit_price,
        "qty": position["qty"],
        "remaining_qty": remaining_qty,
        "notional_usdt": position["notional_usdt"],
        "margin_usdt": position["margin_usdt"],
        "leverage": position["leverage"],
        "pnl": pnl,
        "reason": reason,
    }
    append_jsonl(trades_path, event)
    record_closed_trade(state, pnl)
    state["positions"].pop(symbol, None)
    return event


def manage_positions(
    state: dict[str, Any],
    prepared_frames: dict[str, pd.DataFrame],
    runtime: ShortlinePaperRuntime,
    score_settings: ShortlineDowntrendSettings,
    *,
    trades_path: Path,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for symbol, position in list(state["positions"].items()):
        frame = prepared_frames.get(symbol)
        if frame is None or frame.empty:
            continue
        row = frame.iloc[-1]
        bar_time = frame.index[-1]
        last_bar = position.get("last_bar_time")
        if last_bar and pd.Timestamp(last_bar) >= bar_time:
            continue
        position["last_bar_time"] = bar_time.isoformat()
        position["bars_held"] = int(position.get("bars_held", 0)) + 1
        position["lowest_price"] = min(
            float(position.get("lowest_price", position["entry_price"])),
            float(row["low"]),
            float(row["close"]),
        )

        exit_price = None
        reason = ""
        if float(row["high"]) >= float(position["stop_price"]):
            exit_price = apply_slippage_short(float(position["stop_price"]), is_entry=False, slippage_bps=runtime.slippage_bps)
            reason = "stop_loss" if not position.get("partial_taken") else "trail_stop"
        elif not position.get("partial_taken") and float(row["low"]) <= float(position["tp_price"]):
            partial_qty = float(position["qty"]) * 0.5
            partial_qty = min(partial_qty, float(position["remaining_qty"]))
            partial_price = apply_slippage_short(float(position["tp_price"]), is_entry=False, slippage_bps=runtime.slippage_bps)
            gross = (float(position["entry_price"]) - partial_price) * partial_qty
            fee = partial_price * partial_qty * runtime.fee_rate
            position["remaining_qty"] = float(position["remaining_qty"]) - partial_qty
            position["realized_pnl"] = float(position["realized_pnl"]) + gross - fee
            position["partial_taken"] = True
            position["stop_price"] = min(float(position["stop_price"]), float(position["entry_price"]))
            event = {
                "event": "PARTIAL_TAKE_PROFIT",
                "timestamp": bar_time.isoformat(),
                "symbol": symbol,
                "side": "short",
                "qty": partial_qty,
                "price": partial_price,
                "realized_pnl": position["realized_pnl"],
            }
            append_jsonl(trades_path, event)
            events.append(event)

        if position.get("partial_taken") and exit_price is None:
            trail_stop = float(position["lowest_price"]) + (float(row["atr"]) * runtime.trail_atr_mult)
            position["stop_price"] = min(float(position["stop_price"]), trail_stop, float(position["entry_price"]))

        if exit_price is None and (
            bool(row.get("exit_short_signal", False))
            or float(row.get("shortline_score", 0.0)) <= score_settings.exit_score
        ):
            exit_price = apply_slippage_short(float(row["close"]), is_entry=False, slippage_bps=runtime.slippage_bps)
            reason = "signal_exit"
        if exit_price is None and int(position.get("bars_held", 0)) >= runtime.max_bars_in_trade:
            exit_price = apply_slippage_short(float(row["close"]), is_entry=False, slippage_bps=runtime.slippage_bps)
            reason = "time_exit"
        if exit_price is None and pd.Timestamp(position["entry_time"]).floor("D") < bar_time.floor("D"):
            exit_price = apply_slippage_short(float(row["close"]), is_entry=False, slippage_bps=runtime.slippage_bps)
            reason = "new_day_exit"

        if exit_price is not None:
            events.append(
                close_position(
                    state,
                    symbol,
                    position,
                    timestamp=bar_time,
                    exit_price=float(exit_price),
                    reason=reason,
                    runtime=runtime,
                    trades_path=trades_path,
                )
            )
    return events


def open_short_position(
    state: dict[str, Any],
    symbol: str,
    row: pd.Series,
    score_row: dict[str, Any],
    runtime: ShortlinePaperRuntime,
    *,
    trades_path: Path,
) -> dict[str, Any] | None:
    entry_raw = float(row["close"])
    if entry_raw <= 0:
        return None
    entry_price = apply_slippage_short(entry_raw, is_entry=True, slippage_bps=runtime.slippage_bps)
    atr_value = float(row.get("atr", 0.0))
    swing_high = _to_float(row.get("swing_high"), default=float("nan"))
    atr_stop = entry_price + max(atr_value * runtime.stop_atr_mult, entry_price * runtime.stop_min_pct)
    stop_price = max(swing_high, atr_stop) if math.isfinite(swing_high) else atr_stop
    stop_distance_pct = (stop_price / entry_price) - 1.0
    if stop_distance_pct <= 0 or stop_distance_pct > runtime.stop_max_pct:
        return None

    notional = float(runtime.margin_usdt) * float(runtime.leverage)
    qty = notional / entry_price
    entry_fee = notional * runtime.fee_rate
    risk_distance = stop_price - entry_price
    tp_price = max(entry_price - (risk_distance * runtime.partial_rr), entry_price * 0.5)
    event = {
        "event": "OPEN",
        "timestamp": row.name.isoformat(),
        "symbol": symbol,
        "side": "short",
        "entry_price": entry_price,
        "qty": qty,
        "notional_usdt": notional,
        "margin_usdt": runtime.margin_usdt,
        "leverage": runtime.leverage,
        "stop_price": float(stop_price),
        "tp_price": float(tp_price),
        "rank_score": score_row.get("rank_score"),
        "volume_ratio": score_row.get("volume_ratio"),
        "delta_ratio": score_row.get("delta_ratio"),
        "vwap_gap": score_row.get("vwap_gap"),
    }
    state["positions"][symbol] = {
        "symbol": symbol,
        "side": "short",
        "entry_time": row.name.isoformat(),
        "entry_price": entry_price,
        "qty": qty,
        "remaining_qty": qty,
        "notional_usdt": notional,
        "margin_usdt": runtime.margin_usdt,
        "leverage": runtime.leverage,
        "stop_price": float(stop_price),
        "tp_price": float(tp_price),
        "lowest_price": entry_price,
        "bars_held": 0,
        "partial_taken": False,
        "realized_pnl": -entry_fee,
        "last_bar_time": row.name.isoformat(),
    }
    append_jsonl(trades_path, event)
    return event


def is_open_allowed(state: dict[str, Any], runtime: ShortlinePaperRuntime) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    day_realized = float(state.get("realized_pnl", 0.0)) - float(state.get("day_start_realized", 0.0))
    if day_realized <= -abs(runtime.daily_loss_limit_usdt):
        reasons.append("daily_loss_limit")
    if int(state.get("consecutive_losses", 0)) >= runtime.max_consecutive_losses:
        reasons.append("max_consecutive_losses")
    return not reasons, reasons


def run_cycle(
    session: requests.Session,
    runtime: ShortlinePaperRuntime,
    score_settings: ShortlineDowntrendSettings,
    state: dict[str, Any],
    *,
    output_dir: Path,
    active_symbols_cache: dict[str, Any],
) -> dict[str, Any]:
    server_time_ms = int(request_json(session, "/fapi/v1/time")["serverTime"])
    now = pd.to_datetime(server_time_ms, unit="ms", utc=True)
    reset_day_if_needed(state, now)

    if "symbols" not in active_symbols_cache or time.time() - float(active_symbols_cache.get("updated_at", 0.0)) > 3600:
        active_symbols_cache["symbols"] = fetch_active_symbols(session)
        active_symbols_cache["updated_at"] = time.time()
    active_symbols = set(active_symbols_cache["symbols"])
    scan_symbols, ticker_by_symbol = fetch_scan_symbols(
        session,
        active_symbols,
        scan_limit=runtime.scan_limit,
        open_symbols=set(state["positions"].keys()),
    )

    raw_frames: dict[str, pd.DataFrame] = {}
    prepared_frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for symbol in scan_symbols:
        try:
            raw = fetch_1m_frame(session, symbol, server_time_ms=server_time_ms)
            if len(raw) < 120:
                continue
            prepared = prepare_shortline_downtrend_frame(raw, score_settings)
        except Exception as exc:  # noqa: BLE001
            errors[symbol] = str(exc)
            continue
        if prepared.empty:
            continue
        raw_frames[symbol] = raw
        prepared_frames[symbol] = prepared
        time.sleep(0.015)

    market_context = build_market_context({symbol: prepared_frames[symbol] for symbol in DEFAULT_ANCHORS if symbol in prepared_frames})
    score_rows: list[dict[str, Any]] = []
    for symbol, frame in prepared_frames.items():
        score_row = score_shortline_downtrend(
            frame,
            symbol=symbol,
            market_context=market_context,
            settings=score_settings,
        )
        latest_index = prepared_frames[symbol].index[-1]
        latest_row = prepared_frames[symbol].iloc[-1]
        prepared_frames[symbol].loc[latest_index, "shortline_score"] = float(score_row.get("rank_score", 0.0))
        prepared_frames[symbol].loc[latest_index, "exit_short_signal"] = bool(
            (float(latest_row.get("close", 0.0)) > float(latest_row.get("ema_fast", 0.0)))
            or (float(latest_row.get("delta_ratio", 0.0)) >= 0.12)
            or (float(latest_row.get("close", 0.0)) > float(latest_row.get("session_vwap", 0.0)))
            or (float(latest_row.get("rsi", 50.0)) <= 20.0)
        )
        ticker = ticker_by_symbol.get(symbol, {})
        score_row["quote_volume"] = _to_float(ticker.get("quoteVolume"))
        score_row["price_change_24h_pct"] = _to_float(ticker.get("priceChangePercent"))
        score_rows.append(score_row)

    score_rows.sort(key=lambda row: float(row.get("rank_score", 0.0)), reverse=True)
    rank_by_symbol = {str(row["symbol"]): index for index, row in enumerate(score_rows, start=1)}
    for row in score_rows:
        row["daily_rank"] = rank_by_symbol[str(row["symbol"])]

    trades_path = output_dir / "trades.jsonl"
    events = manage_positions(state, prepared_frames, runtime, score_settings, trades_path=trades_path)
    can_open, risk_reasons = is_open_allowed(state, runtime)
    slots = max(0, runtime.max_positions - len(state["positions"]))
    candidates = [row for row in score_rows if bool(row.get("short_signal", False))]
    opened: list[dict[str, Any]] = []
    if can_open and slots > 0:
        for candidate in candidates:
            if slots <= 0:
                break
            symbol = str(candidate["symbol"])
            if symbol in state["positions"]:
                continue
            frame = prepared_frames.get(symbol)
            if frame is None or frame.empty:
                continue
            event = open_short_position(state, symbol, frame.iloc[-1], candidate, runtime, trades_path=trades_path)
            if event is not None:
                opened.append(event)
                slots -= 1
    elif candidates:
        events.append({"event": "RISK_GUARD_BLOCKED_OPEN", "timestamp": now.isoformat(), "reasons": risk_reasons})

    marks: dict[str, float] = {}
    unrealized = 0.0
    for symbol, position in state["positions"].items():
        frame = prepared_frames.get(symbol)
        if frame is None or frame.empty:
            continue
        mark = float(frame.iloc[-1]["close"])
        marks[symbol] = mark
        unrealized += position_unrealized(position, mark)

    equity = float(state.get("initial_balance", runtime.initial_balance)) + float(state.get("realized_pnl", 0.0)) + unrealized
    top_ranked = [
        {
            "daily_rank": int(row.get("daily_rank", 999)),
            "symbol": row.get("symbol"),
            "daily_return_pct": row.get("daily_return_pct"),
            "daily_close": row.get("close"),
            "daily_upper_break": bool(row.get("short_signal", False)),
            "selected_day": bool(row.get("watch", False)),
            "rank_score": row.get("rank_score"),
            "delta_ratio": row.get("delta_ratio"),
            "volume_ratio": row.get("volume_ratio"),
        }
        for row in score_rows[:40]
    ]
    diagnostic_rows = [
        {
            **row,
            "daily_return_pct": round(float(row.get("daily_return_pct", 0.0)) * 100.0, 2),
            "failed_checks": row.get("failed_checks", []),
        }
        for row in score_rows[:35]
    ]
    candidate_rows = [
        {
            **row,
            "display_symbol": row.get("symbol"),
        }
        for row in candidates[:20]
    ]

    day_realized = float(state.get("realized_pnl", 0.0)) - float(state.get("day_start_realized", 0.0))
    status = {
        "mode": "PAPER",
        "cycle_time": now.isoformat(),
        "strategy": "shortline_downtrend_score_1m",
        "leverage": runtime.leverage,
        "margin_usdt_per_trade": runtime.margin_usdt,
        "notional_usdt_per_trade": runtime.margin_usdt * runtime.leverage,
        "scanned_symbols": len(scan_symbols),
        "prepared_symbols": len(prepared_frames),
        "candidate_count": len(candidates),
        "top_ranked": top_ranked,
        "candidates": candidate_rows,
        "diagnostics": diagnostic_rows,
        "opened": opened,
        "position_events": events,
        "positions": state["positions"],
        "marks": marks,
        "realized_pnl": float(state.get("realized_pnl", 0.0)),
        "unrealized_pnl": unrealized,
        "paper_equity": equity,
        "risk_guard": {
            "can_open": can_open,
            "reasons": risk_reasons,
            "day_realized_pnl": day_realized,
            "daily_loss_limit_usdt": runtime.daily_loss_limit_usdt,
            "consecutive_losses": int(state.get("consecutive_losses", 0)),
            "max_consecutive_losses": runtime.max_consecutive_losses,
        },
        "market_context": market_context,
        "errors": errors,
    }
    append_jsonl(output_dir / "signals.jsonl", status)
    write_json(output_dir / "latest_status.json", status)
    write_json(output_dir / "positions.json", {"positions": state["positions"], "marks": marks})
    append_csv(
        output_dir / "equity.csv",
        {
            "timestamp": now.isoformat(),
            "paper_equity": round(equity, 8),
            "realized_pnl": round(float(state.get("realized_pnl", 0.0)), 8),
            "unrealized_pnl": round(unrealized, 8),
            "open_positions": len(state["positions"]),
            "candidate_count": len(candidates),
            "prepared_symbols": len(prepared_frames),
        },
    )
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shortline downtrend score paper trader.")
    parser.add_argument("--output-dir", default="outputs/shortline_downtrend_score_paper_10x_1u")
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--scan-limit", type=int, default=80)
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--margin-usdt", type=float, default=1.0)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--initial-balance", type=float, default=100.0)
    parser.add_argument("--min-entry-score", type=float, default=74.0)
    parser.add_argument("--min-watch-score", type=float, default=58.0)
    parser.add_argument("--daily-loss-limit-usdt", type=float, default=1.0)
    parser.add_argument("--max-consecutive-losses", type=int, default=2)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = ShortlinePaperRuntime(
        output_dir=args.output_dir,
        poll_sec=max(10, args.poll_sec),
        scan_limit=max(20, args.scan_limit),
        leverage=max(1, args.leverage),
        margin_usdt=max(0.1, args.margin_usdt),
        max_positions=max(1, args.max_positions),
        initial_balance=max(1.0, args.initial_balance),
        daily_loss_limit_usdt=max(0.1, args.daily_loss_limit_usdt),
        max_consecutive_losses=max(1, args.max_consecutive_losses),
        once=bool(args.once),
    )
    score_settings = ShortlineDowntrendSettings(
        min_entry_score=float(args.min_entry_score),
        min_watch_score=float(args.min_watch_score),
    )
    output_dir = Path(runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "runtime.json", {"runtime": asdict(runtime), "score_settings": asdict(score_settings)})
    state_path = output_dir / "state.json"
    state = load_state(state_path, runtime.initial_balance)
    session = requests.Session()
    active_symbols_cache: dict[str, Any] = {}

    while True:
        try:
            status = run_cycle(
                session,
                runtime,
                score_settings,
                state,
                output_dir=output_dir,
                active_symbols_cache=active_symbols_cache,
            )
            write_json(
                state_path,
                {
                    "positions": state["positions"],
                    "realized_pnl": state.get("realized_pnl", 0.0),
                    "initial_balance": state.get("initial_balance", runtime.initial_balance),
                    "day_key": state.get("day_key", ""),
                    "day_start_realized": state.get("day_start_realized", 0.0),
                    "consecutive_losses": state.get("consecutive_losses", 0),
                },
            )
            print(
                json.dumps(
                    {
                        "cycle_time": status["cycle_time"],
                        "candidates": status["candidate_count"],
                        "positions": len(status["positions"]),
                        "equity": round(status["paper_equity"], 6),
                        "top": [
                            {
                                "symbol": row.get("symbol"),
                                "score": row.get("rank_score"),
                                "signal": row.get("daily_upper_break"),
                            }
                            for row in status["top_ranked"][:5]
                        ],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            error = {"timestamp": utc_now().isoformat(), "error": str(exc)}
            append_jsonl(output_dir / "errors.jsonl", error)
            write_json(output_dir / "latest_error.json", error)
            print(json.dumps(error, ensure_ascii=False), flush=True)
        if runtime.once:
            break
        time.sleep(runtime.poll_sec)


if __name__ == "__main__":
    main()
