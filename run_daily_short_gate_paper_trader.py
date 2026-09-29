from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from futures_strategy.daily_short_gate import DailyShortGateSettings, build_daily_short_gate, enrich_intraday_with_daily_short_gate
from futures_strategy.indicators import atr, ema, rsi
from run_daily_short_gate_scan import DEFAULT_UNIVERSE, fetch_1h_history, request_json


INTERVAL_MS = 60_000
BINANCE_FAPI = "https://fapi.binance.com"


@dataclass
class DailyShortPaperRuntime:
    output_dir: str
    poll_sec: int = 60
    gate_refresh_min: int = 60
    lookback_days: int = 230
    leverage: int = 10
    margin_usdt: float = 1.0
    max_positions: int = 1
    initial_balance: float = 100.0
    min_volume_ratio: float = 1.25
    min_sell_delta: float = 0.05
    max_vwap_extension_pct: float = 0.03
    once: bool = False


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def append_equity_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def apply_slippage_short(price: float, *, is_entry: bool, slippage_bps: float = 1.0) -> float:
    slip = slippage_bps / 10_000.0
    return price * (1 - slip) if is_entry else price * (1 + slip)


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


def fetch_1m_frame(session: requests.Session, symbol: str, *, server_time_ms: int) -> pd.DataFrame:
    end_time = pd.to_datetime(server_time_ms, unit="ms", utc=True)
    start = (end_time - pd.Timedelta(hours=18)).value // 1_000_000
    rows = request_json(
        session,
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": "1m", "startTime": int(start), "limit": 1200},
    )
    return klines_to_frame(rows, server_time_ms=server_time_ms)


def load_state(path: Path, initial_balance: float) -> dict[str, Any]:
    if not path.exists():
        return {"positions": {}, "realized_pnl": 0.0, "initial_balance": initial_balance}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("positions", {})
    payload.setdefault("realized_pnl", 0.0)
    payload.setdefault("initial_balance", initial_balance)
    return payload


def prepare_intraday_short_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    gate: pd.DataFrame,
    runtime: DailyShortPaperRuntime,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    data = enrich_intraday_with_daily_short_gate(frame, symbol, gate).copy()
    data["ema_fast"] = ema(data["close"], 9)
    data["ema_slow"] = ema(data["close"], 21)
    data["rsi"] = rsi(data["close"], 14)
    data["atr"] = atr(data, 14)
    data["atr_pct"] = data["atr"] / data["close"]
    data["volume_sma"] = data["volume"].rolling(30).mean()
    data["volume_ratio"] = (data["volume"] / data["volume_sma"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    quote_volume = data["quote_volume"].replace(0, np.nan)
    buy_pressure = (data["taker_buy_quote"] / quote_volume).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    data["delta_ratio"] = ((buy_pressure - 0.5) * 2.0).clip(-1.0, 1.0)
    session = data.index.floor("D")
    typical = (data["high"] + data["low"] + data["close"]) / 3.0
    data["session_vwap"] = (typical * data["volume"]).groupby(session).cumsum() / data["volume"].groupby(session).cumsum().replace(0, np.nan)
    data["vwap_gap"] = (data["close"] / data["session_vwap"]) - 1.0
    data["break_low"] = data["close"] < data["low"].rolling(6).min().shift(1)
    data["swing_high"] = data["high"].rolling(18).max().shift(1)
    data["swing_low"] = data["low"].rolling(18).min().shift(1)
    bearish_bar = data["close"] < data["open"]
    data["short_signal"] = (
        data["daily_short_allowed"]
        & bearish_bar
        & (data["close"] < data["ema_fast"])
        & (data["ema_fast"] < data["ema_slow"])
        & (data["close"] < data["session_vwap"])
        & (data["vwap_gap"] >= -runtime.max_vwap_extension_pct)
        & data["break_low"].fillna(False)
        & (data["volume_ratio"] >= runtime.min_volume_ratio)
        & (data["delta_ratio"] <= -runtime.min_sell_delta)
        & data["rsi"].between(22, 56)
        & data["atr_pct"].between(0.0008, 0.05)
    ).fillna(False)
    data["exit_short_signal"] = (
        (data["close"] > data["ema_fast"])
        | (data["delta_ratio"] >= 0.12)
        | (data["rsi"] <= 24)
        | (data["close"] > data["session_vwap"])
    ).fillna(False)
    return data.dropna()


def position_unrealized(position: dict[str, Any], mark: float) -> float:
    return (float(position["entry_price"]) - mark) * float(position["remaining_qty"])


def close_position(
    state: dict[str, Any],
    symbol: str,
    position: dict[str, Any],
    *,
    timestamp: pd.Timestamp,
    exit_price: float,
    reason: str,
    trades_path: Path,
    fee_rate: float = 0.0004,
) -> dict[str, Any]:
    remaining_qty = float(position["remaining_qty"])
    gross = (float(position["entry_price"]) - exit_price) * remaining_qty
    exit_fee = exit_price * remaining_qty * fee_rate
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
    state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + pnl
    state["positions"].pop(symbol, None)
    return event


def manage_positions(
    state: dict[str, Any],
    prepared_frames: dict[str, pd.DataFrame],
    runtime: DailyShortPaperRuntime,
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
        position["lowest_price"] = min(float(position.get("lowest_price", position["entry_price"])), float(row["low"]), float(row["close"]))

        exit_price = None
        reason = ""
        if float(row["high"]) >= float(position["stop_price"]):
            exit_price = apply_slippage_short(float(position["stop_price"]), is_entry=False)
            reason = "stop_loss" if not position.get("partial_taken") else "trail_stop"
        elif not position.get("partial_taken") and float(row["low"]) <= float(position["tp_price"]):
            partial_qty = float(position["qty"]) * 0.5
            partial_qty = min(partial_qty, float(position["remaining_qty"]))
            partial_price = apply_slippage_short(float(position["tp_price"]), is_entry=False)
            gross = (float(position["entry_price"]) - partial_price) * partial_qty
            fee = partial_price * partial_qty * 0.0004
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
            trail_stop = float(position["lowest_price"]) + (float(row["atr"]) * 1.1)
            position["stop_price"] = min(float(position["stop_price"]), trail_stop, float(position["entry_price"]))

        if exit_price is None and bool(row.get("exit_short_signal", False)):
            exit_price = apply_slippage_short(float(row["close"]), is_entry=False)
            reason = "signal_exit"
        if exit_price is None and int(position.get("bars_held", 0)) >= 90:
            exit_price = apply_slippage_short(float(row["close"]), is_entry=False)
            reason = "time_exit"
        if exit_price is None and pd.Timestamp(position["entry_time"]).floor("D") < bar_time.floor("D"):
            exit_price = apply_slippage_short(float(row["close"]), is_entry=False)
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
                    trades_path=trades_path,
                )
            )
    return events


def open_short_position(
    state: dict[str, Any],
    symbol: str,
    row: pd.Series,
    runtime: DailyShortPaperRuntime,
    *,
    trades_path: Path,
) -> dict[str, Any] | None:
    entry_raw = float(row["close"])
    if entry_raw <= 0:
        return None
    entry_price = apply_slippage_short(entry_raw, is_entry=True)
    atr_value = float(row.get("atr", 0.0))
    swing_high = float(row.get("swing_high", np.nan))
    atr_stop = entry_price + max(atr_value * 1.1, entry_price * 0.004)
    stop_price = max(swing_high, atr_stop) if math.isfinite(swing_high) else atr_stop
    risk_distance = stop_price - entry_price
    if not math.isfinite(risk_distance) or risk_distance <= 0:
        return None
    notional = float(runtime.margin_usdt) * float(runtime.leverage)
    qty = notional / entry_price
    entry_fee = notional * 0.0004
    tp_price = max(entry_price - (risk_distance * 1.2), entry_price * 0.5)
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
        "daily_short_gate_rank": int(row.get("daily_short_gate_rank", 9999)),
        "daily_short_profit_factor": float(row.get("daily_short_profit_factor", 0.0)),
        "volume_ratio": float(row.get("volume_ratio", 0.0)),
        "delta_ratio": float(row.get("delta_ratio", 0.0)),
        "vwap_gap": float(row.get("vwap_gap", 0.0)),
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


def build_gate(
    session: requests.Session,
    symbols: list[str],
    runtime: DailyShortPaperRuntime,
    gate_settings: DailyShortGateSettings,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    hourly_by_symbol: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for symbol in symbols:
        try:
            frame = fetch_1h_history(session, symbol, lookback_days=runtime.lookback_days, sleep_sec=0.05)
            if not frame.empty:
                hourly_by_symbol[symbol] = frame
        except Exception as exc:  # noqa: BLE001
            errors[symbol] = str(exc)
        time.sleep(0.03)
    gate, _ = build_daily_short_gate(hourly_by_symbol, gate_settings)
    allowed = gate.loc[gate["daily_short_allowed"], "symbol"].tolist() if not gate.empty else []
    return gate, allowed, errors


def build_diagnostic(symbol: str, row: pd.Series) -> dict[str, Any]:
    checks = {
        "daily_short_allowed": bool(row.get("daily_short_allowed", False)),
        "bearish_bar": bool(row["close"] < row["open"]),
        "close_lt_ema_fast": bool(row["close"] < row["ema_fast"]),
        "ema_stack_short": bool(row["ema_fast"] < row["ema_slow"]),
        "close_lt_vwap": bool(row["close"] < row["session_vwap"]),
        "not_overextended": bool(row.get("vwap_gap", -9.0) >= -0.03),
        "break_low": bool(row.get("break_low", False)),
        "volume_ok": bool(row.get("volume_ratio", 0.0) >= 1.25),
        "delta_ok": bool(row.get("delta_ratio", 0.0) <= -0.05),
        "rsi_ok": bool(22 <= row.get("rsi", 0.0) <= 56),
        "atr_ok": bool(0.0008 <= row.get("atr_pct", 0.0) <= 0.05),
        "short_signal": bool(row.get("short_signal", False)),
    }
    return {
        "symbol": symbol,
        "close": float(row["close"]),
        "rsi": round(float(row.get("rsi", 0.0)), 2),
        "volume_ratio": round(float(row.get("volume_ratio", 0.0)), 2),
        "delta_ratio": round(float(row.get("delta_ratio", 0.0)), 3),
        "vwap_gap_pct": round(float(row.get("vwap_gap", 0.0)) * 100.0, 2),
        "daily_short_gate_rank": int(row.get("daily_short_gate_rank", 9999)),
        "failed_checks": [key for key, ok in checks.items() if not ok],
        "checks": checks,
    }


def run_cycle(
    session: requests.Session,
    runtime: DailyShortPaperRuntime,
    state: dict[str, Any],
    *,
    symbols: list[str],
    gate_settings: DailyShortGateSettings,
    output_dir: Path,
    gate_cache: dict[str, Any],
) -> dict[str, Any]:
    now = time.time()
    if "gate" not in gate_cache or now - float(gate_cache.get("updated_at", 0.0)) >= runtime.gate_refresh_min * 60:
        gate, allowed, errors = build_gate(session, symbols, runtime, gate_settings)
        gate_cache.update({"gate": gate, "allowed": allowed, "errors": errors, "updated_at": now})

    gate = gate_cache.get("gate", pd.DataFrame())
    allowed = list(gate_cache.get("allowed", []))
    open_symbols = sorted(state["positions"].keys())
    scan_symbols = list(dict.fromkeys([*allowed, *open_symbols]))
    server_time_ms = int(request_json(session, "/fapi/v1/time")["serverTime"])
    prepared_frames: dict[str, pd.DataFrame] = {}
    candidates: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for symbol in scan_symbols:
        try:
            raw = fetch_1m_frame(session, symbol, server_time_ms=server_time_ms)
            prepared = prepare_intraday_short_frame(raw, symbol=symbol, gate=gate, runtime=runtime)
        except Exception:
            continue
        if prepared.empty:
            continue
        prepared_frames[symbol] = prepared
        row = prepared.iloc[-1]
        diagnostics.append(build_diagnostic(symbol, row))
        if bool(row.get("short_signal", False)):
            candidates.append(
                {
                    "symbol": symbol,
                    "timestamp": prepared.index[-1].isoformat(),
                    "close": float(row["close"]),
                    "daily_short_gate_rank": int(row.get("daily_short_gate_rank", 9999)),
                    "daily_short_profit_factor": float(row.get("daily_short_profit_factor", 0.0)),
                    "volume_ratio": float(row.get("volume_ratio", 0.0)),
                    "delta_ratio": float(row.get("delta_ratio", 0.0)),
                    "vwap_gap": float(row.get("vwap_gap", 0.0)),
                }
            )
    candidates.sort(key=lambda item: (item["daily_short_profit_factor"], item["volume_ratio"]), reverse=True)

    trades_path = output_dir / "trades.jsonl"
    events = manage_positions(state, prepared_frames, runtime, trades_path=trades_path)
    slots = max(0, runtime.max_positions - len(state["positions"]))
    opened: list[dict[str, Any]] = []
    for candidate in candidates:
        if slots <= 0:
            break
        symbol = candidate["symbol"]
        if symbol in state["positions"]:
            continue
        event = open_short_position(state, symbol, prepared_frames[symbol].iloc[-1], runtime, trades_path=trades_path)
        if event is not None:
            opened.append(event)
            slots -= 1

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
    cycle_time = pd.to_datetime(server_time_ms, unit="ms", utc=True).isoformat()
    gate_rows = gate.to_dict("records") if isinstance(gate, pd.DataFrame) and not gate.empty else []
    status = {
        "mode": "PAPER",
        "cycle_time": cycle_time,
        "strategy": "daily_short_gate_1m_short",
        "leverage": runtime.leverage,
        "margin_usdt_per_trade": runtime.margin_usdt,
        "notional_usdt_per_trade": runtime.margin_usdt * runtime.leverage,
        "scanned_symbols": len(scan_symbols),
        "prepared_symbols": len(prepared_frames),
        "candidate_count": len(candidates),
        "allowed_symbols": allowed,
        "gate_rows": gate_rows,
        "gate_errors": gate_cache.get("errors", {}),
        "candidates": candidates[:20],
        "diagnostics": diagnostics[:30],
        "opened": opened,
        "position_events": events,
        "positions": state["positions"],
        "marks": marks,
        "realized_pnl": float(state.get("realized_pnl", 0.0)),
        "unrealized_pnl": unrealized,
        "paper_equity": equity,
    }
    append_jsonl(output_dir / "signals.jsonl", status)
    write_json(output_dir / "latest_status.json", status)
    write_json(output_dir / "positions.json", {"positions": state["positions"], "marks": marks})
    append_equity_csv(
        output_dir / "equity.csv",
        {
            "timestamp": cycle_time,
            "paper_equity": round(equity, 8),
            "realized_pnl": round(float(state.get("realized_pnl", 0.0)), 8),
            "unrealized_pnl": round(unrealized, 8),
            "open_positions": len(state["positions"]),
            "candidate_count": len(candidates),
            "prepared_symbols": len(prepared_frames),
        },
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily short gate + 1m paper trader.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--output-dir", default="outputs/daily_short_gate_1m_paper_10x_1u")
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--gate-refresh-min", type=int, default=60)
    parser.add_argument("--lookback-days", type=int, default=230)
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--margin-usdt", type=float, default=1.0)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--initial-balance", type=float, default=100.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    runtime = DailyShortPaperRuntime(
        output_dir=args.output_dir,
        poll_sec=max(10, args.poll_sec),
        gate_refresh_min=max(5, args.gate_refresh_min),
        lookback_days=max(80, args.lookback_days),
        leverage=max(1, args.leverage),
        margin_usdt=max(0.1, args.margin_usdt),
        max_positions=max(1, args.max_positions),
        initial_balance=max(1.0, args.initial_balance),
        once=bool(args.once),
    )
    gate_settings = DailyShortGateSettings()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    output_dir = Path(runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "runtime.json", {"runtime": asdict(runtime), "gate_settings": asdict(gate_settings), "symbols": symbols})
    state_path = output_dir / "state.json"
    state = load_state(state_path, runtime.initial_balance)
    session = requests.Session()
    gate_cache: dict[str, Any] = {}

    while True:
        try:
            status = run_cycle(
                session,
                runtime,
                state,
                symbols=symbols,
                gate_settings=gate_settings,
                output_dir=output_dir,
                gate_cache=gate_cache,
            )
            save_payload = {
                "positions": state["positions"],
                "realized_pnl": state.get("realized_pnl", 0.0),
                "initial_balance": state.get("initial_balance", runtime.initial_balance),
            }
            write_json(state_path, save_payload)
            print(
                json.dumps(
                    {
                        "cycle_time": status["cycle_time"],
                        "allowed": status["allowed_symbols"],
                        "candidates": status["candidate_count"],
                        "positions": len(status["positions"]),
                        "equity": round(status["paper_equity"], 6),
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as exc:  # noqa: BLE001
            error = {"timestamp": utc_now().isoformat(), "error": str(exc)}
            append_jsonl(output_dir / "errors.jsonl", error)
            write_json(output_dir / "latest_error.json", error)
            print(json.dumps(error, ensure_ascii=False))
        if runtime.once:
            break
        time.sleep(runtime.poll_sec)


if __name__ == "__main__":
    main()
