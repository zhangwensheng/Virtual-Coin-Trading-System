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

from futures_strategy.binance_account_client import (
    BINANCE_FUTURES_BASE_URL,
    BinanceApiError,
    BinanceCredentials,
    BinanceFuturesAccountClient,
    summarize_income,
)
from futures_strategy.binance_secure_store import load_dashboard_settings
from futures_strategy.daily_short_gate import DailyShortGateSettings
from run_daily_short_gate_paper_trader import (
    DEFAULT_UNIVERSE,
    append_jsonl,
    build_diagnostic,
    build_gate,
    fetch_1m_frame,
    prepare_intraday_short_frame,
    utc_now,
    write_json,
)


LIVE_CONFIRM_TEXT = "I_UNDERSTAND_REAL_MONEY"


@dataclass
class DailyShortLiveRuntime:
    output_dir: str = "outputs/daily_short_gate_1m_live_10x_1u"
    poll_sec: int = 60
    gate_refresh_min: int = 60
    lookback_days: int = 230
    leverage: int = 10
    margin_usdt: float = 1.0
    max_positions: int = 1
    min_volume_ratio: float = 1.25
    min_sell_delta: float = 0.05
    max_vwap_extension_pct: float = 0.03
    working_type: str = "MARK_PRICE"
    min_notional_buffer: float = 1.02
    max_notional_multiplier: float = 1.35
    once: bool = False


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def client_order_id(prefix: str, symbol: str) -> str:
    suffix = str(pd.Timestamp.utcnow().value)[-10:]
    return f"DSG{prefix}_{symbol}_{suffix}"[:36]


def load_live_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "positions": {},
            "last_signal_timestamps": {},
            "one_way_mode_confirmed": False,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("positions", {})
    payload.setdefault("last_signal_timestamps", {})
    payload.setdefault("one_way_mode_confirmed", False)
    return payload


def save_live_state(path: Path, state: dict[str, Any]) -> None:
    write_json(
        path,
        {
            "positions": state.get("positions", {}),
            "last_signal_timestamps": state.get("last_signal_timestamps", {}),
            "one_way_mode_confirmed": bool(state.get("one_way_mode_confirmed", False)),
        },
    )


def fetch_account_summary(client: BinanceFuturesAccountClient) -> dict[str, Any]:
    account = client.account_info()
    incomes = client.income_history(limit=100)
    income_summary = summarize_income(incomes)
    return {
        "total_wallet_balance": _to_float(account.get("totalWalletBalance")),
        "available_balance": _to_float(account.get("availableBalance")),
        "total_margin_balance": _to_float(account.get("totalMarginBalance")),
        "total_unrealized_profit": _to_float(account.get("totalUnrealizedProfit")),
        "total_initial_margin": _to_float(account.get("totalInitialMargin")),
        "total_maint_margin": _to_float(account.get("totalMaintMargin")),
        **income_summary,
    }


def active_positions(client: BinanceFuturesAccountClient) -> dict[str, dict[str, Any]]:
    rows = client.position_risk()
    return {
        str(row.get("symbol", "")): row
        for row in rows
        if abs(_to_float(row.get("positionAmt"))) > 1e-12
    }


def open_orders_by_symbol(client: BinanceFuturesAccountClient, symbols: list[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for symbol in symbols:
        try:
            result[symbol] = client.open_orders(symbol)
        except BinanceApiError:
            result[symbol] = []
    return result


def cancel_order_if_present(
    client: BinanceFuturesAccountClient,
    symbol: str,
    order_id: int | None,
    messages: list[str],
) -> None:
    if not order_id:
        return
    try:
        client.cancel_order(symbol, order_id=order_id)
        messages.append(f"[{symbol}] 已撤销保护单 #{order_id}")
    except BinanceApiError as exc:
        if "-2011" not in str(exc) and "Unknown order" not in str(exc):
            messages.append(f"[{symbol}] 撤保护单失败 #{order_id}: {exc}")


def cancel_managed_orders(
    client: BinanceFuturesAccountClient,
    position: dict[str, Any],
    messages: list[str],
) -> None:
    symbol = str(position.get("symbol", ""))
    cancel_order_if_present(client, symbol, _to_int(position.get("stop_order_id")), messages)
    cancel_order_if_present(client, symbol, _to_int(position.get("tp_order_id")), messages)
    position["stop_order_id"] = None
    position["tp_order_id"] = None


def place_stop_order(
    client: BinanceFuturesAccountClient,
    *,
    symbol: str,
    quantity: float,
    stop_price: float,
    runtime: DailyShortLiveRuntime,
) -> dict[str, Any]:
    return client.place_order(
        symbol=symbol,
        side="BUY",
        type="STOP_MARKET",
        quantity=quantity,
        stopPrice=stop_price,
        reduceOnly="true",
        workingType=runtime.working_type,
        newClientOrderId=client_order_id("ST", symbol),
    )


def place_take_profit_order(
    client: BinanceFuturesAccountClient,
    *,
    symbol: str,
    quantity: float,
    stop_price: float,
    runtime: DailyShortLiveRuntime,
) -> dict[str, Any]:
    return client.place_order(
        symbol=symbol,
        side="BUY",
        type="TAKE_PROFIT_MARKET",
        quantity=quantity,
        stopPrice=stop_price,
        reduceOnly="true",
        workingType=runtime.working_type,
        newClientOrderId=client_order_id("TP", symbol),
    )


def live_close_short(
    client: BinanceFuturesAccountClient,
    symbol: str,
    quantity: float,
    *,
    reason: str,
    trades_path: Path,
    messages: list[str],
) -> dict[str, Any]:
    qty = client.quantize_quantity(symbol, abs(quantity), market=True)
    response = client.close_position_market(
        symbol,
        qty,
        position_amt=-abs(qty),
        position_side="BOTH",
        client_order_id=client_order_id("CL", symbol),
    )
    event = {
        "event": "LIVE_CLOSE",
        "timestamp": utc_now().isoformat(),
        "symbol": symbol,
        "side": "short",
        "qty": qty,
        "price": _to_float(response.get("avgPrice")),
        "order_id": response.get("orderId"),
        "reason": reason,
        "raw_status": response.get("status"),
    }
    append_jsonl(trades_path, event)
    messages.append(f"[{symbol}] 已实盘市价平空 qty={qty} reason={reason}")
    return event


def build_live_order_plan(
    client: BinanceFuturesAccountClient,
    symbol: str,
    row: pd.Series,
    runtime: DailyShortLiveRuntime,
) -> dict[str, Any] | None:
    entry_estimate = _to_float(row.get("close"))
    atr_value = _to_float(row.get("atr"))
    if entry_estimate <= 0 or atr_value <= 0:
        return None

    target_notional = runtime.margin_usdt * runtime.leverage
    raw_qty = target_notional / entry_estimate
    qty = client.quantize_quantity(symbol, raw_qty, market=True)
    if qty <= 0:
        return None

    min_notional = client.min_notional(symbol)
    if min_notional > 0 and qty * entry_estimate < min_notional:
        qty = client.quantize_quantity(symbol, (min_notional * runtime.min_notional_buffer) / entry_estimate, market=True)

    estimated_notional = qty * entry_estimate
    if estimated_notional <= 0:
        return None
    max_allowed_notional = max(target_notional * runtime.max_notional_multiplier, min_notional * runtime.min_notional_buffer)
    if estimated_notional > max_allowed_notional:
        return None

    swing_high = _to_float(row.get("swing_high"), default=float("nan"))
    atr_stop = entry_estimate + max(atr_value * 1.1, entry_estimate * 0.004)
    stop_raw = max(swing_high, atr_stop) if math.isfinite(swing_high) else atr_stop
    stop_price = client.quantize_price(symbol, stop_raw, rounding="up")
    if stop_price <= entry_estimate:
        stop_price = client.quantize_price(symbol, entry_estimate * 1.004, rounding="up")

    risk_distance = stop_price - entry_estimate
    if risk_distance <= 0:
        return None
    tp_price = client.quantize_price(symbol, max(entry_estimate - (risk_distance * 1.2), entry_estimate * 0.5), rounding="down")
    tp_qty = client.quantize_quantity(symbol, qty * 0.5, market=True)
    if tp_qty <= 0 or tp_qty >= qty or tp_price >= entry_estimate:
        tp_price = None
        tp_qty = 0.0

    return {
        "symbol": symbol,
        "qty": qty,
        "estimated_entry": entry_estimate,
        "estimated_notional": estimated_notional,
        "stop_price": stop_price,
        "tp_price": tp_price,
        "tp_qty": tp_qty,
    }


def enter_live_short(
    client: BinanceFuturesAccountClient,
    symbol: str,
    row: pd.Series,
    runtime: DailyShortLiveRuntime,
    *,
    state: dict[str, Any],
    trades_path: Path,
    messages: list[str],
) -> dict[str, Any] | None:
    plan = build_live_order_plan(client, symbol, row, runtime)
    if plan is None:
        messages.append(f"[{symbol}] 信号出现，但数量/最小名义价值/止损约束未通过，跳过。")
        state["last_signal_timestamps"][symbol] = row.name.isoformat()
        return None

    client.change_margin_type(symbol, "ISOLATED")
    client.change_initial_leverage(symbol, runtime.leverage)
    entry_order = client.place_order(
        symbol=symbol,
        side="SELL",
        type="MARKET",
        quantity=plan["qty"],
        newOrderRespType="RESULT",
        newClientOrderId=client_order_id("SE", symbol),
    )
    entry_qty = abs(_to_float(entry_order.get("executedQty"), plan["qty"]))
    if entry_qty <= 0:
        entry_qty = plan["qty"]
    entry_price = _to_float(entry_order.get("avgPrice"))
    if entry_price <= 0:
        cum_quote = _to_float(entry_order.get("cumQuote"))
        entry_price = cum_quote / entry_qty if cum_quote > 0 and entry_qty > 0 else plan["estimated_entry"]

    atr_value = _to_float(row.get("atr"))
    stop_price = client.quantize_price(symbol, max(plan["stop_price"], entry_price + max(atr_value * 1.1, entry_price * 0.004)), rounding="up")
    if stop_price <= entry_price:
        stop_price = client.quantize_price(symbol, entry_price * 1.004, rounding="up")

    risk_distance = stop_price - entry_price
    tp_price = client.quantize_price(symbol, max(entry_price - (risk_distance * 1.2), entry_price * 0.5), rounding="down")
    tp_qty = client.quantize_quantity(symbol, entry_qty * 0.5, market=True)
    if tp_qty <= 0 or tp_qty >= entry_qty or tp_price >= entry_price:
        tp_price = None
        tp_qty = 0.0

    try:
        stop_order = place_stop_order(
            client,
            symbol=symbol,
            quantity=entry_qty,
            stop_price=stop_price,
            runtime=runtime,
        )
    except Exception:
        live_close_short(client, symbol, entry_qty, reason="stop_order_failed", trades_path=trades_path, messages=messages)
        raise

    tp_order = None
    if tp_price is not None and tp_qty > 0:
        try:
            tp_order = place_take_profit_order(
                client,
                symbol=symbol,
                quantity=tp_qty,
                stop_price=tp_price,
                runtime=runtime,
            )
        except BinanceApiError as exc:
            messages.append(f"[{symbol}] 实盘止盈单挂单失败，继续保留止损: {exc}")

    position = {
        "symbol": symbol,
        "side": "short",
        "entry_time": utc_now().isoformat(),
        "signal_timestamp": row.name.isoformat(),
        "entry_price": entry_price,
        "qty": entry_qty,
        "remaining_qty": entry_qty,
        "notional_usdt": entry_price * entry_qty,
        "margin_usdt": (entry_price * entry_qty) / runtime.leverage,
        "leverage": runtime.leverage,
        "stop_price": stop_price,
        "tp_price": "" if tp_price is None else tp_price,
        "stop_order_id": _to_int(stop_order.get("orderId")),
        "tp_order_id": None if tp_order is None else _to_int(tp_order.get("orderId")),
        "entry_order_id": entry_order.get("orderId"),
        "lowest_price": entry_price,
        "bars_held": 0,
        "partial_taken": False,
        "realized_pnl": 0.0,
        "last_bar_time": row.name.isoformat(),
    }
    state["positions"][symbol] = position
    state["last_signal_timestamps"][symbol] = row.name.isoformat()

    event = {
        "event": "LIVE_OPEN",
        "timestamp": position["entry_time"],
        "symbol": symbol,
        "side": "short",
        "entry_price": entry_price,
        "qty": entry_qty,
        "notional_usdt": position["notional_usdt"],
        "margin_usdt": position["margin_usdt"],
        "leverage": runtime.leverage,
        "stop_price": stop_price,
        "tp_price": tp_price,
        "entry_order_id": entry_order.get("orderId"),
        "stop_order_id": position["stop_order_id"],
        "tp_order_id": position["tp_order_id"],
        "daily_short_gate_rank": _to_int(row.get("daily_short_gate_rank"), 9999),
        "daily_short_profit_factor": _to_float(row.get("daily_short_profit_factor")),
        "volume_ratio": _to_float(row.get("volume_ratio")),
        "delta_ratio": _to_float(row.get("delta_ratio")),
        "vwap_gap": _to_float(row.get("vwap_gap")),
    }
    append_jsonl(trades_path, event)
    messages.append(f"[{symbol}] 已实盘开空 qty={entry_qty} entry={entry_price:.8g} stop={stop_price:.8g}")
    return event


def replace_stop_order(
    client: BinanceFuturesAccountClient,
    position: dict[str, Any],
    current_qty: float,
    new_stop: float,
    runtime: DailyShortLiveRuntime,
    messages: list[str],
) -> None:
    symbol = str(position["symbol"])
    cancel_order_if_present(client, symbol, _to_int(position.get("stop_order_id")), messages)
    stop_order = place_stop_order(
        client,
        symbol=symbol,
        quantity=current_qty,
        stop_price=new_stop,
        runtime=runtime,
    )
    position["stop_order_id"] = _to_int(stop_order.get("orderId"))
    position["stop_price"] = new_stop
    messages.append(f"[{symbol}] 已更新实盘止损到 {new_stop:.8g}")


def ensure_live_protection(
    client: BinanceFuturesAccountClient,
    position: dict[str, Any],
    current_qty: float,
    open_order_ids: set[int],
    runtime: DailyShortLiveRuntime,
    messages: list[str],
) -> None:
    symbol = str(position["symbol"])
    if _to_int(position.get("stop_order_id")) not in open_order_ids:
        stop_order = place_stop_order(
            client,
            symbol=symbol,
            quantity=current_qty,
            stop_price=_to_float(position.get("stop_price")),
            runtime=runtime,
        )
        position["stop_order_id"] = _to_int(stop_order.get("orderId"))
        messages.append(f"[{symbol}] 已补挂实盘止损单 #{position['stop_order_id']}")

    tp_price = _to_float(position.get("tp_price"))
    if (not position.get("partial_taken")) and tp_price > 0 and _to_int(position.get("tp_order_id")) not in open_order_ids:
        tp_qty = client.quantize_quantity(symbol, current_qty * 0.5, market=True)
        if 0 < tp_qty < current_qty:
            tp_order = place_take_profit_order(
                client,
                symbol=symbol,
                quantity=tp_qty,
                stop_price=tp_price,
                runtime=runtime,
            )
            position["tp_order_id"] = _to_int(tp_order.get("orderId"))
            messages.append(f"[{symbol}] 已补挂实盘止盈单 #{position['tp_order_id']}")


def manage_live_positions(
    client: BinanceFuturesAccountClient,
    state: dict[str, Any],
    prepared_frames: dict[str, pd.DataFrame],
    runtime: DailyShortLiveRuntime,
    *,
    positions: dict[str, dict[str, Any]],
    open_orders: dict[str, list[dict[str, Any]]],
    trades_path: Path,
    messages: list[str],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for symbol, position in list(state.get("positions", {}).items()):
        live_position = positions.get(symbol)
        if live_position is None or abs(_to_float(live_position.get("positionAmt"))) <= 1e-12:
            cancel_managed_orders(client, position, messages)
            event = {
                "event": "LIVE_POSITION_CLOSED_DETECTED",
                "timestamp": utc_now().isoformat(),
                "symbol": symbol,
                "side": "short",
                "reason": "exchange_position_missing",
            }
            append_jsonl(trades_path, event)
            events.append(event)
            state["positions"].pop(symbol, None)
            continue

        live_qty_signed = _to_float(live_position.get("positionAmt"))
        if live_qty_signed >= 0:
            messages.append(f"[{symbol}] 检测到非空头仓位，策略不会管理该仓位。")
            continue

        current_qty = abs(live_qty_signed)
        current_qty = client.quantize_quantity(symbol, current_qty, market=True)
        position["remaining_qty"] = current_qty
        order_ids = {_to_int(order.get("orderId")) for order in open_orders.get(symbol, [])}
        ensure_live_protection(client, position, current_qty, order_ids, runtime, messages)

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
            _to_float(position.get("lowest_price"), _to_float(position.get("entry_price"))),
            _to_float(row.get("low")),
            _to_float(row.get("close")),
        )

        entry_qty = max(_to_float(position.get("qty")), current_qty)
        if (not position.get("partial_taken")) and current_qty < entry_qty * 0.7:
            position["partial_taken"] = True
            cancel_order_if_present(client, symbol, _to_int(position.get("tp_order_id")), messages)
            position["tp_order_id"] = None
            breakeven_stop = client.quantize_price(symbol, min(_to_float(position.get("stop_price")), _to_float(position.get("entry_price"))), rounding="up")
            replace_stop_order(client, position, current_qty, breakeven_stop, runtime, messages)
            events.append(
                {
                    "event": "LIVE_PARTIAL_DETECTED",
                    "timestamp": bar_time.isoformat(),
                    "symbol": symbol,
                    "side": "short",
                    "remaining_qty": current_qty,
                    "stop_price": breakeven_stop,
                }
            )

        if position.get("partial_taken"):
            atr_value = _to_float(row.get("atr"))
            if atr_value > 0:
                trailing_stop = _to_float(position.get("lowest_price")) + atr_value * 1.1
                new_stop = client.quantize_price(
                    symbol,
                    min(_to_float(position.get("stop_price")), trailing_stop, _to_float(position.get("entry_price"))),
                    rounding="up",
                )
                if new_stop > 0 and new_stop < _to_float(position.get("stop_price")):
                    replace_stop_order(client, position, current_qty, new_stop, runtime, messages)

        exit_reason = ""
        if bool(row.get("exit_short_signal", False)):
            exit_reason = "signal_exit"
        elif int(position.get("bars_held", 0)) >= 90:
            exit_reason = "time_exit"
        elif pd.Timestamp(position["entry_time"]).floor("D") < bar_time.floor("D"):
            exit_reason = "new_day_exit"

        if exit_reason:
            event = live_close_short(
                client,
                symbol,
                current_qty,
                reason=exit_reason,
                trades_path=trades_path,
                messages=messages,
            )
            cancel_managed_orders(client, position, messages)
            state["positions"].pop(symbol, None)
            events.append(event)
    return events


def run_live_cycle(
    client: BinanceFuturesAccountClient,
    public_session: requests.Session,
    runtime: DailyShortLiveRuntime,
    state: dict[str, Any],
    *,
    symbols: list[str],
    gate_settings: DailyShortGateSettings,
    output_dir: Path,
    gate_cache: dict[str, Any],
) -> dict[str, Any]:
    messages: list[str] = []
    if not state.get("one_way_mode_confirmed"):
        client.change_position_mode(False)
        state["one_way_mode_confirmed"] = True
        messages.append("已确认 Binance Futures 单向持仓模式。")

    now = time.time()
    if "gate" not in gate_cache or now - float(gate_cache.get("updated_at", 0.0)) >= runtime.gate_refresh_min * 60:
        gate, allowed, errors = build_gate(public_session, symbols, runtime, gate_settings)
        gate_cache.update({"gate": gate, "allowed": allowed, "errors": errors, "updated_at": now})

    gate = gate_cache.get("gate", pd.DataFrame())
    allowed = list(gate_cache.get("allowed", []))
    open_symbols = sorted(state.get("positions", {}).keys())
    scan_symbols = list(dict.fromkeys([*allowed, *open_symbols]))

    server_time_ms = int(public_session.get("https://fapi.binance.com/fapi/v1/time", timeout=15).json()["serverTime"])
    prepared_frames: dict[str, pd.DataFrame] = {}
    candidates: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for symbol in scan_symbols:
        try:
            raw = fetch_1m_frame(public_session, symbol, server_time_ms=server_time_ms)
            prepared = prepare_intraday_short_frame(raw, symbol=symbol, gate=gate, runtime=runtime)
        except Exception as exc:  # noqa: BLE001
            messages.append(f"[{symbol}] 1m 准备失败: {exc}")
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
                    "close": _to_float(row.get("close")),
                    "daily_short_gate_rank": _to_int(row.get("daily_short_gate_rank"), 9999),
                    "daily_short_profit_factor": _to_float(row.get("daily_short_profit_factor")),
                    "volume_ratio": _to_float(row.get("volume_ratio")),
                    "delta_ratio": _to_float(row.get("delta_ratio")),
                    "vwap_gap": _to_float(row.get("vwap_gap")),
                }
            )
    candidates.sort(key=lambda item: (item["daily_short_profit_factor"], item["volume_ratio"]), reverse=True)

    managed_symbols = sorted(state.get("positions", {}).keys())
    exchange_positions = active_positions(client)
    tracked_order_symbols = sorted(set([*managed_symbols, *scan_symbols]))
    open_orders = open_orders_by_symbol(client, tracked_order_symbols)
    trades_path = output_dir / "trades.jsonl"
    position_events = manage_live_positions(
        client,
        state,
        prepared_frames,
        runtime,
        positions=exchange_positions,
        open_orders=open_orders,
        trades_path=trades_path,
        messages=messages,
    )

    exchange_positions = active_positions(client)
    existing_strategy_symbols = set(state.get("positions", {}).keys())
    unmanaged_symbols = sorted(symbol for symbol in exchange_positions if symbol not in existing_strategy_symbols)
    slots = max(0, runtime.max_positions - len(existing_strategy_symbols))
    opened: list[dict[str, Any]] = []
    for candidate in candidates:
        if slots <= 0:
            break
        symbol = str(candidate["symbol"])
        if symbol in state.get("positions", {}):
            continue
        if symbol in exchange_positions:
            messages.append(f"[{symbol}] 交易所已有仓位，跳过开仓。")
            continue
        if str(state.get("last_signal_timestamps", {}).get(symbol, "")) == str(candidate["timestamp"]):
            continue
        event = enter_live_short(
            client,
            symbol,
            prepared_frames[symbol].iloc[-1],
            runtime,
            state=state,
            trades_path=trades_path,
            messages=messages,
        )
        state["last_signal_timestamps"][symbol] = str(candidate["timestamp"])
        if event is not None:
            opened.append(event)
            slots -= 1

    exchange_positions = active_positions(client)
    marks: dict[str, float] = {}
    display_positions: dict[str, dict[str, Any]] = {}
    for symbol, position in state.get("positions", {}).items():
        live_position = exchange_positions.get(symbol, {})
        mark = _to_float(live_position.get("markPrice"))
        if mark <= 0 and symbol in prepared_frames:
            mark = _to_float(prepared_frames[symbol].iloc[-1].get("close"))
        marks[symbol] = mark
        display_positions[symbol] = {
            **position,
            "remaining_qty": abs(_to_float(live_position.get("positionAmt"), position.get("remaining_qty", 0.0))),
            "mark_price": mark,
            "unrealized_pnl": _to_float(live_position.get("unRealizedProfit")),
            "liquidation_price": _to_float(live_position.get("liquidationPrice")),
            "margin_type": live_position.get("marginType", "isolated"),
        }

    account_summary = fetch_account_summary(client)
    cycle_time = pd.to_datetime(server_time_ms, unit="ms", utc=True).isoformat()
    status = {
        "mode": "LIVE",
        "cycle_time": cycle_time,
        "strategy": "daily_short_gate_1m_live_short",
        "leverage": runtime.leverage,
        "margin_usdt_per_trade": runtime.margin_usdt,
        "notional_usdt_per_trade": runtime.margin_usdt * runtime.leverage,
        "scanned_symbols": len(scan_symbols),
        "prepared_symbols": len(prepared_frames),
        "candidate_count": len(candidates),
        "allowed_symbols": allowed,
        "gate_rows": gate.to_dict("records") if isinstance(gate, pd.DataFrame) and not gate.empty else [],
        "gate_errors": gate_cache.get("errors", {}),
        "candidates": candidates[:20],
        "diagnostics": diagnostics[:30],
        "opened": opened,
        "position_events": position_events,
        "positions": display_positions,
        "marks": marks,
        "messages": messages[-50:],
        "unmanaged_exchange_positions": unmanaged_symbols,
        "account_summary": account_summary,
        "realized_pnl": account_summary.get("realized_pnl_24h", 0.0),
        "unrealized_pnl": account_summary.get("total_unrealized_profit", 0.0),
        "paper_equity": account_summary.get("total_margin_balance", 0.0),
        "live_equity": account_summary.get("total_margin_balance", 0.0),
    }
    write_json(output_dir / "latest_status.json", status)
    write_json(output_dir / "positions.json", {"positions": display_positions, "marks": marks})
    append_jsonl(output_dir / "signals.jsonl", status)
    append_csv(
        output_dir / "equity.csv",
        {
            "timestamp": cycle_time,
            "paper_equity": round(_to_float(status.get("paper_equity")), 8),
            "live_equity": round(_to_float(status.get("live_equity")), 8),
            "realized_pnl": round(_to_float(status.get("realized_pnl")), 8),
            "unrealized_pnl": round(_to_float(status.get("unrealized_pnl")), 8),
            "open_positions": len(display_positions),
            "candidate_count": len(candidates),
            "prepared_symbols": len(prepared_frames),
        },
    )
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LIVE Binance Futures daily short gate + 1m short trader.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--output-dir", default="outputs/daily_short_gate_1m_live_10x_1u")
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--gate-refresh-min", type=int, default=60)
    parser.add_argument("--lookback-days", type=int, default=230)
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--margin-usdt", type=float, default=1.0)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--working-type", default="MARK_PRICE")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--live-confirm", default="", help=f"Must equal {LIVE_CONFIRM_TEXT} to enable real orders.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.live_confirm != LIVE_CONFIRM_TEXT:
        raise SystemExit(f"Refusing to start LIVE trader without --live-confirm {LIVE_CONFIRM_TEXT}")

    settings = load_dashboard_settings()
    if not settings.api_key or not settings.api_secret:
        raise SystemExit("No Binance API key/secret saved in dashboard settings.")
    if settings.use_testnet:
        raise SystemExit("Dashboard settings are still TESTNET. Switch account mode to LIVE first.")

    runtime = DailyShortLiveRuntime(
        output_dir=args.output_dir,
        poll_sec=max(10, args.poll_sec),
        gate_refresh_min=max(5, args.gate_refresh_min),
        lookback_days=max(80, args.lookback_days),
        leverage=max(1, min(20, args.leverage)),
        margin_usdt=max(0.1, args.margin_usdt),
        max_positions=max(1, min(3, args.max_positions)),
        working_type=str(args.working_type or "MARK_PRICE").upper(),
        once=bool(args.once),
    )
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    output_dir = Path(runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_settings = DailyShortGateSettings()
    write_json(output_dir / "runtime.json", {"runtime": asdict(runtime), "gate_settings": asdict(gate_settings), "symbols": symbols})

    state_path = output_dir / "state.json"
    state = load_live_state(state_path)
    client = BinanceFuturesAccountClient(
        BinanceCredentials(settings.api_key, settings.api_secret),
        base_url=BINANCE_FUTURES_BASE_URL,
    )
    public_session = requests.Session()
    gate_cache: dict[str, Any] = {}

    while True:
        try:
            status = run_live_cycle(
                client,
                public_session,
                runtime,
                state,
                symbols=symbols,
                gate_settings=gate_settings,
                output_dir=output_dir,
                gate_cache=gate_cache,
            )
            save_live_state(state_path, state)
            print(
                json.dumps(
                    {
                        "cycle_time": status["cycle_time"],
                        "mode": status["mode"],
                        "allowed": status["allowed_symbols"],
                        "candidates": status["candidate_count"],
                        "positions": len(status["positions"]),
                        "equity": round(_to_float(status["live_equity"]), 6),
                        "opened": len(status["opened"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            error = {"timestamp": utc_now().isoformat(), "mode": "LIVE", "error": str(exc)}
            append_jsonl(output_dir / "errors.jsonl", error)
            write_json(output_dir / "latest_error.json", error)
            print(json.dumps(error, ensure_ascii=False), flush=True)
        if runtime.once:
            break
        time.sleep(runtime.poll_sec)


if __name__ == "__main__":
    main()
