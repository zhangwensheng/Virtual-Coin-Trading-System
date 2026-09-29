from __future__ import annotations

import argparse
import csv
import json
import math
import re
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
from futures_strategy.shortline_downtrend import (
    ShortlineDowntrendSettings,
    build_market_context,
    prepare_shortline_downtrend_frame,
    score_shortline_downtrend,
)
from run_shortline_downtrend_paper_trader import (
    DEFAULT_ANCHORS,
    fetch_1m_frame,
    fetch_active_symbols,
    fetch_scan_symbols,
    request_json,
    utc_now,
)


LIVE_CONFIRM_TEXT = "I_UNDERSTAND_REAL_MONEY"
LIVE_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]+USDT$")


@dataclass
class ShortlineLiveRuntime:
    output_dir: str = "outputs/shortline_downtrend_score_live_10x_1u"
    trade_side: str = "LONG"
    poll_sec: int = 60
    scan_limit: int = 80
    leverage: int = 10
    margin_usdt: float = 1.0
    max_positions: int = 1
    daily_loss_limit_usdt: float = 1.0
    max_bars_in_trade: int = 0
    stop_atr_mult: float = 1.05
    stop_min_pct: float = 0.004
    stop_max_pct: float = 0.035
    partial_rr: float = 1.05
    take_profit_roe_pct: float = 10.0
    stop_loss_roe_pct: float = 20.0
    exit_on_new_day: bool = False
    trail_atr_mult: float = 1.0
    working_type: str = "MARK_PRICE"
    min_notional_buffer: float = 1.02
    max_notional_multiplier: float = 1.35
    once: bool = False


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def is_live_symbol_allowed(symbol: str) -> bool:
    return bool(LIVE_SYMBOL_PATTERN.fullmatch(symbol))


def client_order_id(prefix: str, symbol: str) -> str:
    safe_symbol = re.sub(r"[^A-Za-z0-9]", "", symbol)[:16] or "SYMBOL"
    suffix = str(pd.Timestamp.utcnow().value)[-10:]
    return f"SLD{prefix}_{safe_symbol}_{suffix}"[:36]


def protection_order_id(order: dict[str, Any]) -> int:
    return _to_int(order.get("algoId") or order.get("orderId"))


def is_unknown_order_error(exc: BinanceApiError) -> bool:
    message = str(exc)
    return "-2011" in message or "Unknown order" in message or "-4045" in message


def normalized_trade_side(runtime: ShortlineLiveRuntime | dict[str, Any] | str) -> str:
    if isinstance(runtime, str):
        value = runtime
    elif isinstance(runtime, dict):
        value = str(runtime.get("side") or runtime.get("trade_side") or "SHORT")
    else:
        value = runtime.trade_side
    value = str(value or "SHORT").upper()
    return "LONG" if value == "LONG" else "SHORT"


def entry_order_side(runtime: ShortlineLiveRuntime) -> str:
    return "BUY" if normalized_trade_side(runtime) == "LONG" else "SELL"


def close_order_side(runtime: ShortlineLiveRuntime | dict[str, Any]) -> str:
    return "SELL" if normalized_trade_side(runtime) == "LONG" else "BUY"


def signed_position_amt(quantity: float, runtime: ShortlineLiveRuntime | dict[str, Any]) -> float:
    qty = abs(float(quantity))
    return qty if normalized_trade_side(runtime) == "LONG" else -qty


def position_matches_runtime(position_amt: float, runtime: ShortlineLiveRuntime) -> bool:
    return position_amt > 0 if normalized_trade_side(runtime) == "LONG" else position_amt < 0


def side_label(runtime: ShortlineLiveRuntime | dict[str, Any]) -> str:
    return normalized_trade_side(runtime).lower()


def roe_pct_to_price_fraction(roe_pct: float, leverage: int) -> float:
    return max(0.0, float(roe_pct)) / max(1, int(leverage)) / 100.0


def build_roe_exit_prices(
    client: BinanceFuturesAccountClient,
    symbol: str,
    entry_price: float,
    runtime: ShortlineLiveRuntime,
) -> tuple[float, float]:
    tp_pct = roe_pct_to_price_fraction(runtime.take_profit_roe_pct, runtime.leverage)
    stop_pct = roe_pct_to_price_fraction(runtime.stop_loss_roe_pct, runtime.leverage)
    if entry_price <= 0 or tp_pct <= 0 or stop_pct <= 0 or stop_pct > runtime.stop_max_pct:
        return 0.0, 0.0
    if normalized_trade_side(runtime) == "LONG":
        tp_price = client.quantize_price(symbol, entry_price * (1.0 + tp_pct), rounding="up")
        stop_price = client.quantize_price(symbol, entry_price * (1.0 - stop_pct), rounding="down")
        valid = stop_price > 0 and stop_price < entry_price < tp_price
    else:
        tp_price = client.quantize_price(symbol, entry_price * (1.0 - tp_pct), rounding="down")
        stop_price = client.quantize_price(symbol, entry_price * (1.0 + stop_pct), rounding="up")
        valid = tp_price > 0 and tp_price < entry_price < stop_price
    if not valid:
        return 0.0, 0.0
    return stop_price, tp_price


def score_long_momentum(
    row: pd.Series,
    *,
    settings: ShortlineDowntrendSettings | None = None,
) -> dict[str, Any]:
    settings = settings or ShortlineDowntrendSettings()
    score = 0.0
    components: dict[str, float] = {}

    price_score = 0.0
    if _to_float(row.get("close")) > _to_float(row.get("ema_fast")):
        price_score += 8.0
    if _to_float(row.get("ema_fast")) > _to_float(row.get("ema_slow")):
        price_score += 8.0
    if _to_float(row.get("ema_slow")) > _to_float(row.get("ema_trend")):
        price_score += 5.0
    if _to_float(row.get("close")) > _to_float(row.get("session_vwap")):
        price_score += 8.0
    if _to_float(row.get("day_close_location")) >= 0.65:
        price_score += 5.0
    price_score = min(price_score, 38.0)
    components["price_structure"] = price_score
    score += price_score

    flow_score = 0.0
    if _to_float(row.get("volume_ratio")) >= settings.min_volume_ratio:
        flow_score += 8.0
    if _to_float(row.get("trade_ratio")) >= settings.min_trade_ratio:
        flow_score += 4.0
    if _to_float(row.get("delta_ratio")) >= settings.min_sell_delta:
        flow_score += 10.0
    if _to_float(row.get("day_close_location")) >= 0.6:
        flow_score += 3.0
    flow_score = min(flow_score, 25.0)
    components["buy_pressure"] = flow_score
    score += flow_score

    htf_score = 0.0
    if _to_float(row.get("htf_close")) > _to_float(row.get("htf_ema_fast")):
        htf_score += 5.0
    if _to_float(row.get("htf_ema_fast")) > _to_float(row.get("htf_ema_slow")):
        htf_score += 5.0
    if _to_float(row.get("vwap_gap")) >= 0:
        htf_score += 4.0
    if _to_float(row.get("daily_return_pct")) >= 0:
        htf_score += 3.0
    htf_score = min(htf_score, 17.0)
    components["higher_timeframe"] = htf_score
    score += htf_score

    risk_score = 0.0
    atr_pct = _to_float(row.get("atr_pct"))
    if settings.min_atr_pct <= atr_pct <= settings.max_atr_pct:
        risk_score += 5.0
    if _to_float(row.get("vwap_gap")) <= settings.max_vwap_extension_pct:
        risk_score += 3.0
    if 35.0 <= _to_float(row.get("rsi")) <= 72.0:
        risk_score += 4.0
    risk_score = min(risk_score, 12.0)
    components["risk_filter"] = risk_score
    score += risk_score

    checks = {
        "score_min": score >= settings.min_entry_score,
        "close_above_vwap": _to_float(row.get("close")) > _to_float(row.get("session_vwap")),
        "buy_delta": _to_float(row.get("delta_ratio")) >= settings.min_sell_delta,
        "volume_ok": _to_float(row.get("volume_ratio")) >= settings.min_volume_ratio,
        "trend_ok": _to_float(row.get("ema_fast")) >= _to_float(row.get("ema_slow")),
        "rsi_ok": 35.0 <= _to_float(row.get("rsi")) <= 72.0,
        "atr_ok": settings.min_atr_pct <= atr_pct <= settings.max_atr_pct,
        "not_overextended": _to_float(row.get("vwap_gap")) <= settings.max_vwap_extension_pct,
    }
    return {
        "rank_score": round(score, 2),
        "long_signal": all(checks.values()),
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "checks": checks,
        "components": components,
    }


def fetch_account_summary(client: BinanceFuturesAccountClient) -> dict[str, Any]:
    account = client.account_info()
    incomes = client.income_history(limit=100)
    return {
        "total_wallet_balance": _to_float(account.get("totalWalletBalance")),
        "available_balance": _to_float(account.get("availableBalance")),
        "total_margin_balance": _to_float(account.get("totalMarginBalance")),
        "total_unrealized_profit": _to_float(account.get("totalUnrealizedProfit")),
        "total_initial_margin": _to_float(account.get("totalInitialMargin")),
        "total_maint_margin": _to_float(account.get("totalMaintMargin")),
        **summarize_income(incomes),
    }


def retry_call(label: str, func: Any, *args: Any, attempts: int = 3, sleep_sec: float = 1.0, **kwargs: Any) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(sleep_sec * (attempt + 1))
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last_error}") from last_error


def active_positions(client: BinanceFuturesAccountClient) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("symbol", "")): row
        for row in client.position_risk()
        if abs(_to_float(row.get("positionAmt"))) > 1e-12
    }


def open_orders_by_symbol(client: BinanceFuturesAccountClient, symbols: list[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for symbol in symbols:
        orders: list[dict[str, Any]] = []
        try:
            orders.extend(client.open_orders(symbol))
        except BinanceApiError:
            pass
        try:
            orders.extend(client.open_algo_orders(symbol))
        except BinanceApiError:
            pass
        result[symbol] = orders
    return result


def symbols_requiring_order_checks(
    managed_symbols: list[str],
    exchange_positions: dict[str, dict[str, Any]],
) -> list[str]:
    return sorted(set(managed_symbols) | set(exchange_positions.keys()))


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "positions": {},
            "last_signal_timestamps": {},
            "one_way_mode_confirmed": False,
            "consecutive_losses": 0,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("positions", {})
    payload.setdefault("last_signal_timestamps", {})
    payload.setdefault("one_way_mode_confirmed", False)
    payload.setdefault("consecutive_losses", 0)
    return payload


def save_state(path: Path, state: dict[str, Any]) -> None:
    write_json(
        path,
        {
            "positions": state.get("positions", {}),
            "last_signal_timestamps": state.get("last_signal_timestamps", {}),
            "one_way_mode_confirmed": bool(state.get("one_way_mode_confirmed", False)),
            "consecutive_losses": int(state.get("consecutive_losses", 0)),
        },
    )


def place_stop_order(
    client: BinanceFuturesAccountClient,
    *,
    symbol: str,
    quantity: float,
    stop_price: float,
    runtime: ShortlineLiveRuntime,
) -> dict[str, Any]:
    return client.place_close_position_algo_order(
        symbol=symbol,
        side=close_order_side(runtime),
        order_type="STOP_MARKET",
        trigger_price=stop_price,
        working_type=runtime.working_type,
        client_algo_id=client_order_id("ST", symbol),
    )


def place_take_profit_order(
    client: BinanceFuturesAccountClient,
    *,
    symbol: str,
    quantity: float,
    stop_price: float,
    runtime: ShortlineLiveRuntime,
) -> dict[str, Any]:
    return client.place_close_position_algo_order(
        symbol=symbol,
        side=close_order_side(runtime),
        order_type="TAKE_PROFIT_MARKET",
        trigger_price=stop_price,
        working_type=runtime.working_type,
        client_algo_id=client_order_id("TP", symbol),
    )


def cancel_order_if_present(
    client: BinanceFuturesAccountClient,
    symbol: str,
    order_id: int | None,
    messages: list[str],
) -> None:
    if not order_id:
        return
    try:
        client.cancel_algo_order(algo_id=order_id)
        messages.append(f"[{symbol}] 已撤销保护单 #{order_id}")
    except BinanceApiError as exc:
        if not is_unknown_order_error(exc):
            messages.append(f"[{symbol}] 撤保护单失败 #{order_id}: {exc}")
            return
        try:
            client.cancel_order(symbol, order_id=order_id)
            messages.append(f"[{symbol}] 已撤销保护单 #{order_id}")
        except BinanceApiError as fallback_exc:
            if not is_unknown_order_error(fallback_exc):
                messages.append(f"[{symbol}] 撤保护单失败 #{order_id}: {fallback_exc}")


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


def live_close_position(
    client: BinanceFuturesAccountClient,
    symbol: str,
    quantity: float,
    *,
    runtime_or_position: ShortlineLiveRuntime | dict[str, Any],
    reason: str,
    trades_path: Path,
    messages: list[str],
) -> dict[str, Any]:
    qty = client.quantize_quantity(symbol, abs(quantity), market=True)
    response = client.close_position_market(
        symbol,
        qty,
        position_amt=signed_position_amt(qty, runtime_or_position),
        position_side="BOTH",
        client_order_id=client_order_id("CL", symbol),
    )
    event = {
        "event": "LIVE_CLOSE",
        "timestamp": utc_now().isoformat(),
        "symbol": symbol,
        "side": side_label(runtime_or_position),
        "qty": qty,
        "price": _to_float(response.get("avgPrice")),
        "order_id": response.get("orderId"),
        "reason": reason,
        "raw_status": response.get("status"),
    }
    append_jsonl(trades_path, event)
    messages.append(f"[{symbol}] 已实盘市价平{side_label(runtime_or_position)} qty={qty} reason={reason}")
    return event


def build_order_plan(
    client: BinanceFuturesAccountClient,
    symbol: str,
    row: pd.Series,
    runtime: ShortlineLiveRuntime,
) -> dict[str, Any] | None:
    entry_estimate = _to_float(row.get("close"))
    atr_value = _to_float(row.get("atr"))
    if entry_estimate <= 0 or atr_value <= 0:
        return None

    target_notional = runtime.margin_usdt * runtime.leverage
    raw_qty = target_notional / entry_estimate
    qty = client.quantize_quantity(symbol, raw_qty, market=True)
    min_notional = client.min_notional(symbol)
    if min_notional > 0 and qty * entry_estimate < min_notional:
        qty = client.quantize_quantity(symbol, (min_notional * runtime.min_notional_buffer) / entry_estimate, market=True)

    estimated_notional = qty * entry_estimate
    max_allowed_notional = max(target_notional * runtime.max_notional_multiplier, min_notional * runtime.min_notional_buffer)
    if qty <= 0 or estimated_notional <= 0 or estimated_notional > max_allowed_notional:
        return None

    stop_price, tp_price = build_roe_exit_prices(client, symbol, entry_estimate, runtime)
    if stop_price <= 0 or tp_price <= 0:
        return None
    tp_qty = qty

    return {
        "symbol": symbol,
        "qty": qty,
        "estimated_entry": entry_estimate,
        "estimated_notional": estimated_notional,
        "stop_price": stop_price,
        "tp_price": tp_price,
        "tp_qty": tp_qty,
    }


def enter_live_position(
    client: BinanceFuturesAccountClient,
    symbol: str,
    row: pd.Series,
    score_row: dict[str, Any],
    runtime: ShortlineLiveRuntime,
    *,
    state: dict[str, Any],
    trades_path: Path,
    messages: list[str],
) -> dict[str, Any] | None:
    plan = build_order_plan(client, symbol, row, runtime)
    if plan is None:
        messages.append(f"[{symbol}] 信号出现，但最小名义价值/止损距离/数量约束未通过，跳过。")
        state["last_signal_timestamps"][symbol] = row.name.isoformat()
        return None

    client.change_margin_type(symbol, "ISOLATED")
    client.change_initial_leverage(symbol, runtime.leverage)
    entry_order = client.place_order(
        symbol=symbol,
        side=entry_order_side(runtime),
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

    stop_price, tp_price = build_roe_exit_prices(client, symbol, entry_price, runtime)
    if stop_price <= 0 or tp_price <= 0:
        live_close_position(
            client,
            symbol,
            entry_qty,
            runtime_or_position=runtime,
            reason="stop_distance_invalid",
            trades_path=trades_path,
            messages=messages,
        )
        return None
    tp_qty = entry_qty

    try:
        stop_order = place_stop_order(client, symbol=symbol, quantity=entry_qty, stop_price=stop_price, runtime=runtime)
    except Exception:
        live_close_position(
            client,
            symbol,
            entry_qty,
            runtime_or_position=runtime,
            reason="stop_order_failed",
            trades_path=trades_path,
            messages=messages,
        )
        raise

    tp_order = None
    if tp_price is not None and tp_qty > 0:
        try:
            tp_order = place_take_profit_order(client, symbol=symbol, quantity=tp_qty, stop_price=tp_price, runtime=runtime)
        except BinanceApiError as exc:
            messages.append(f"[{symbol}] 止盈单失败，保留止损继续管理: {exc}")

    position = {
        "symbol": symbol,
        "side": side_label(runtime),
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
        "stop_order_id": protection_order_id(stop_order),
        "tp_order_id": None if tp_order is None else protection_order_id(tp_order),
        "entry_order_id": entry_order.get("orderId"),
        "lowest_price": entry_price,
        "highest_price": entry_price,
        "bars_held": 0,
        "partial_taken": False,
        "last_bar_time": row.name.isoformat(),
    }
    state["positions"][symbol] = position
    state["last_signal_timestamps"][symbol] = row.name.isoformat()

    event = {
        "event": "LIVE_OPEN",
        "timestamp": position["entry_time"],
        "symbol": symbol,
        "side": side_label(runtime),
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
        "rank_score": score_row.get("rank_score"),
        "volume_ratio": score_row.get("volume_ratio"),
        "delta_ratio": score_row.get("delta_ratio"),
        "vwap_gap": score_row.get("vwap_gap"),
    }
    append_jsonl(trades_path, event)
    messages.append(
        f"[{symbol}] 已实盘开{side_label(runtime)} qty={entry_qty} entry={entry_price:.8g} stop={stop_price:.8g} tp={tp_price}"
    )
    return event


def ensure_protection(
    client: BinanceFuturesAccountClient,
    position: dict[str, Any],
    current_qty: float,
    open_order_ids: set[int],
    runtime: ShortlineLiveRuntime,
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
        position["stop_order_id"] = protection_order_id(stop_order)
        messages.append(f"[{symbol}] 已补挂止损单 #{position['stop_order_id']}")

    tp_price = _to_float(position.get("tp_price"))
    if (not position.get("partial_taken")) and tp_price > 0 and _to_int(position.get("tp_order_id")) not in open_order_ids:
        tp_order = place_take_profit_order(client, symbol=symbol, quantity=current_qty, stop_price=tp_price, runtime=runtime)
        position["tp_order_id"] = protection_order_id(tp_order)
        messages.append(f"[{symbol}] 已补挂止盈单 #{position['tp_order_id']}")


def replace_stop_order(
    client: BinanceFuturesAccountClient,
    position: dict[str, Any],
    current_qty: float,
    new_stop: float,
    runtime: ShortlineLiveRuntime,
    messages: list[str],
) -> None:
    symbol = str(position["symbol"])
    cancel_order_if_present(client, symbol, _to_int(position.get("stop_order_id")), messages)
    stop_order = place_stop_order(client, symbol=symbol, quantity=current_qty, stop_price=new_stop, runtime=runtime)
    position["stop_order_id"] = protection_order_id(stop_order)
    position["stop_price"] = new_stop
    messages.append(f"[{symbol}] 已更新移动止损到 {new_stop:.8g}")


def manage_positions(
    client: BinanceFuturesAccountClient,
    state: dict[str, Any],
    prepared_frames: dict[str, pd.DataFrame],
    runtime: ShortlineLiveRuntime,
    score_settings: ShortlineDowntrendSettings,
    *,
    exchange_positions: dict[str, dict[str, Any]],
    open_orders: dict[str, list[dict[str, Any]]],
    trades_path: Path,
    messages: list[str],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for symbol, position in list(state.get("positions", {}).items()):
        live_position = exchange_positions.get(symbol)
        if live_position is None or abs(_to_float(live_position.get("positionAmt"))) <= 1e-12:
            cancel_managed_orders(client, position, messages)
            event = {
                "event": "LIVE_POSITION_CLOSED_DETECTED",
                "timestamp": utc_now().isoformat(),
                "symbol": symbol,
                "side": side_label(position),
                "reason": "exchange_position_missing",
            }
            append_jsonl(trades_path, event)
            events.append(event)
            state["positions"].pop(symbol, None)
            continue

        position_amt = _to_float(live_position.get("positionAmt"))
        if not position_matches_runtime(position_amt, runtime):
            messages.append(f"[{symbol}] 检测到非{side_label(runtime)}仓位，策略不会管理。")
            continue

        current_qty = client.quantize_quantity(symbol, abs(position_amt), market=True)
        position["remaining_qty"] = current_qty
        order_ids = {protection_order_id(order) for order in open_orders.get(symbol, [])}
        order_ids.discard(0)
        ensure_protection(client, position, current_qty, order_ids, runtime, messages)

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
        position["highest_price"] = max(
            _to_float(position.get("highest_price"), _to_float(position.get("entry_price"))),
            _to_float(row.get("high")),
            _to_float(row.get("close")),
        )

        entry_qty = max(_to_float(position.get("qty")), current_qty)
        if (not position.get("partial_taken")) and current_qty < entry_qty * 0.7:
            position["partial_taken"] = True
            cancel_order_if_present(client, symbol, _to_int(position.get("tp_order_id")), messages)
            position["tp_order_id"] = None
            if normalized_trade_side(runtime) == "LONG":
                breakeven = client.quantize_price(
                    symbol,
                    max(_to_float(position.get("stop_price")), _to_float(position.get("entry_price"))),
                    rounding="down",
                )
            else:
                breakeven = client.quantize_price(
                    symbol,
                    min(_to_float(position.get("stop_price")), _to_float(position.get("entry_price"))),
                    rounding="up",
                )
            replace_stop_order(client, position, current_qty, breakeven, runtime, messages)
            events.append(
                {
                    "event": "LIVE_PARTIAL_DETECTED",
                    "timestamp": bar_time.isoformat(),
                    "symbol": symbol,
                    "side": side_label(runtime),
                    "remaining_qty": current_qty,
                    "stop_price": breakeven,
                }
            )

        if position.get("partial_taken"):
            atr_value = _to_float(row.get("atr"))
            if atr_value > 0:
                if normalized_trade_side(runtime) == "LONG":
                    trailing = _to_float(position.get("highest_price")) - atr_value * runtime.trail_atr_mult
                    new_stop = client.quantize_price(
                        symbol,
                        max(_to_float(position.get("stop_price")), trailing, _to_float(position.get("entry_price"))),
                        rounding="down",
                    )
                    should_replace = new_stop > _to_float(position.get("stop_price"))
                else:
                    trailing = _to_float(position.get("lowest_price")) + atr_value * runtime.trail_atr_mult
                    new_stop = client.quantize_price(
                        symbol,
                        min(_to_float(position.get("stop_price")), trailing, _to_float(position.get("entry_price"))),
                        rounding="up",
                    )
                    should_replace = new_stop < _to_float(position.get("stop_price"))
                if new_stop > 0 and should_replace:
                    replace_stop_order(client, position, current_qty, new_stop, runtime, messages)

        exit_reason = ""
        if runtime.max_bars_in_trade > 0 and int(position.get("bars_held", 0)) >= runtime.max_bars_in_trade:
            exit_reason = "time_exit"
        elif runtime.exit_on_new_day and pd.Timestamp(position["entry_time"]).floor("D") < bar_time.floor("D"):
            exit_reason = "new_day_exit"

        if exit_reason:
            event = live_close_position(
                client,
                symbol,
                current_qty,
                runtime_or_position=runtime,
                reason=exit_reason,
                trades_path=trades_path,
                messages=messages,
            )
            cancel_managed_orders(client, position, messages)
            state["positions"].pop(symbol, None)
            events.append(event)
    return events


def risk_guard_allows_open(
    account_summary: dict[str, Any],
    runtime: ShortlineLiveRuntime,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    realized_24h = _to_float(account_summary.get("realized_pnl_24h"))
    if realized_24h <= -abs(runtime.daily_loss_limit_usdt):
        reasons.append("daily_loss_limit_24h")
    available = _to_float(account_summary.get("available_balance"))
    if available < runtime.margin_usdt * 0.8:
        reasons.append("available_balance_low")
    return not reasons, reasons


def run_cycle(
    client: BinanceFuturesAccountClient,
    public_session: requests.Session,
    runtime: ShortlineLiveRuntime,
    score_settings: ShortlineDowntrendSettings,
    state: dict[str, Any],
    *,
    output_dir: Path,
    active_symbols_cache: dict[str, Any],
) -> dict[str, Any]:
    messages: list[str] = []
    if not state.get("one_way_mode_confirmed"):
        retry_call("change_position_mode", client.change_position_mode, False)
        state["one_way_mode_confirmed"] = True
        messages.append("已确认 Binance Futures 单向持仓模式。")

    server_time_ms = int(request_json(public_session, "/fapi/v1/time")["serverTime"])
    now = pd.to_datetime(server_time_ms, unit="ms", utc=True)
    if "symbols" not in active_symbols_cache or time.time() - float(active_symbols_cache.get("updated_at", 0.0)) > 3600:
        active_symbols_cache["symbols"] = fetch_active_symbols(public_session)
        active_symbols_cache["updated_at"] = time.time()
    active_symbols = {symbol for symbol in active_symbols_cache["symbols"] if is_live_symbol_allowed(symbol)}

    scan_symbols, ticker_by_symbol = fetch_scan_symbols(
        public_session,
        active_symbols,
        scan_limit=runtime.scan_limit,
        open_symbols=set(state.get("positions", {}).keys()),
    )
    scan_symbols = [symbol for symbol in scan_symbols if is_live_symbol_allowed(symbol)]

    prepared_frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for symbol in scan_symbols:
        try:
            raw = fetch_1m_frame(public_session, symbol, server_time_ms=server_time_ms)
            if len(raw) < 120:
                continue
            prepared = prepare_shortline_downtrend_frame(raw, score_settings)
        except Exception as exc:  # noqa: BLE001
            errors[symbol] = str(exc)
            continue
        if prepared.empty:
            continue
        prepared_frames[symbol] = prepared
        time.sleep(0.015)

    market_context = build_market_context({symbol: prepared_frames[symbol] for symbol in DEFAULT_ANCHORS if symbol in prepared_frames})
    score_rows: list[dict[str, Any]] = []
    for symbol, frame in prepared_frames.items():
        if normalized_trade_side(runtime) == "LONG":
            score_row = score_long_momentum(frame.iloc[-1], settings=score_settings)
            score_row.update(
                {
                    "symbol": symbol,
                    "timestamp": frame.index[-1].isoformat(),
                    "score": score_row["rank_score"],
                    "watch": float(score_row["rank_score"]) >= score_settings.min_watch_score,
                    "short_signal": False,
                }
            )
        else:
            score_row = score_shortline_downtrend(frame, symbol=symbol, market_context=market_context, settings=score_settings)
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
    for index, row in enumerate(score_rows, start=1):
        row["daily_rank"] = index

    trades_path = output_dir / "trades.jsonl"
    managed_symbols = sorted(state.get("positions", {}).keys())
    exchange_positions = active_positions(client)
    tracked_symbols = symbols_requiring_order_checks(managed_symbols, exchange_positions)
    open_orders = open_orders_by_symbol(client, tracked_symbols)
    position_events = manage_positions(
        client,
        state,
        prepared_frames,
        runtime,
        score_settings,
        exchange_positions=exchange_positions,
        open_orders=open_orders,
        trades_path=trades_path,
        messages=messages,
    )

    account_summary = fetch_account_summary(client)
    can_open, risk_reasons = risk_guard_allows_open(account_summary, runtime)
    exchange_positions = active_positions(client)
    strategy_symbols = set(state.get("positions", {}).keys())
    unmanaged_symbols = sorted(symbol for symbol in exchange_positions if symbol not in strategy_symbols)
    signal_key = "long_signal" if normalized_trade_side(runtime) == "LONG" else "short_signal"
    candidates = [row for row in score_rows if bool(row.get(signal_key, False))]
    opened: list[dict[str, Any]] = []
    slots = max(0, runtime.max_positions - len(strategy_symbols))
    if can_open and slots > 0:
        for candidate in candidates:
            if slots <= 0:
                break
            symbol = str(candidate["symbol"])
            if symbol in state.get("positions", {}):
                continue
            if symbol in exchange_positions:
                messages.append(f"[{symbol}] 交易所已有仓位，跳过。")
                continue
            if str(state.get("last_signal_timestamps", {}).get(symbol, "")) == str(candidate["timestamp"]):
                continue
            frame = prepared_frames.get(symbol)
            if frame is None or frame.empty:
                continue
            event = enter_live_position(
                client,
                symbol,
                frame.iloc[-1],
                candidate,
                runtime,
                state=state,
                trades_path=trades_path,
                messages=messages,
            )
            state["last_signal_timestamps"][symbol] = str(candidate["timestamp"])
            if event is not None:
                opened.append(event)
                slots -= 1
    elif candidates:
        messages.append(f"风险守卫阻止开仓: {','.join(risk_reasons)}")

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

    top_ranked = [
        {
            "daily_rank": int(row.get("daily_rank", 999)),
            "symbol": row.get("symbol"),
            "daily_return_pct": row.get("daily_return_pct"),
            "daily_close": row.get("close"),
            "daily_upper_break": bool(row.get(signal_key, False)),
            "selected_day": bool(row.get("watch", False)),
            "rank_score": row.get("rank_score"),
            "delta_ratio": row.get("delta_ratio"),
            "volume_ratio": row.get("volume_ratio"),
        }
        for row in score_rows[:40]
    ]
    diagnostics = [
        {
            **row,
            "daily_return_pct": round(_to_float(row.get("daily_return_pct")) * 100.0, 2),
            "failed_checks": row.get("failed_checks", []),
        }
        for row in score_rows[:35]
    ]
    status = {
        "mode": "LIVE",
        "cycle_time": now.isoformat(),
        "strategy": f"shortline_{normalized_trade_side(runtime).lower()}_score_1m_live",
        "trade_side": normalized_trade_side(runtime),
        "leverage": runtime.leverage,
        "margin_usdt_per_trade": runtime.margin_usdt,
        "notional_usdt_per_trade": runtime.margin_usdt * runtime.leverage,
        "scanned_symbols": len(scan_symbols),
        "prepared_symbols": len(prepared_frames),
        "candidate_count": len(candidates),
        "top_ranked": top_ranked,
        "candidates": [{**row, "display_symbol": row.get("symbol")} for row in candidates[:20]],
        "diagnostics": diagnostics,
        "opened": opened,
        "position_events": position_events,
        "positions": display_positions,
        "marks": marks,
        "messages": messages[-50:],
        "unmanaged_exchange_positions": unmanaged_symbols,
        "account_summary": account_summary,
        "risk_guard": {
            "can_open": can_open,
            "reasons": risk_reasons,
            "realized_pnl_24h": account_summary.get("realized_pnl_24h", 0.0),
            "daily_loss_limit_usdt": runtime.daily_loss_limit_usdt,
            "available_balance": account_summary.get("available_balance", 0.0),
        },
        "market_context": market_context,
        "errors": errors,
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
            "timestamp": now.isoformat(),
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
    parser = argparse.ArgumentParser(description="Run LIVE shortline downtrend score trader.")
    parser.add_argument("--output-dir", default="outputs/shortline_downtrend_score_live_10x_1u")
    parser.add_argument("--trade-side", choices=["LONG", "SHORT"], default="LONG")
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--scan-limit", type=int, default=80)
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--margin-usdt", type=float, default=1.0)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--min-entry-score", type=float, default=74.0)
    parser.add_argument("--min-watch-score", type=float, default=58.0)
    parser.add_argument("--daily-loss-limit-usdt", type=float, default=1.0)
    parser.add_argument("--take-profit-roe-pct", type=float, default=10.0)
    parser.add_argument("--stop-loss-roe-pct", type=float, default=20.0)
    parser.add_argument("--exit-on-new-day", action="store_true")
    parser.add_argument("--working-type", default="MARK_PRICE")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--live-confirm", default="", help=f"Must equal {LIVE_CONFIRM_TEXT} to enable real orders.")
    return parser.parse_args()


def build_runtime_from_args(args: argparse.Namespace) -> ShortlineLiveRuntime:
    return ShortlineLiveRuntime(
        output_dir=args.output_dir,
        trade_side=str(args.trade_side or "LONG").upper(),
        poll_sec=max(10, args.poll_sec),
        scan_limit=max(20, args.scan_limit),
        leverage=max(1, min(20, args.leverage)),
        margin_usdt=max(0.1, args.margin_usdt),
        max_positions=max(1, min(5, args.max_positions)),
        daily_loss_limit_usdt=max(0.1, args.daily_loss_limit_usdt),
        take_profit_roe_pct=max(0.1, args.take_profit_roe_pct),
        stop_loss_roe_pct=max(0.1, args.stop_loss_roe_pct),
        exit_on_new_day=bool(args.exit_on_new_day),
        working_type=str(args.working_type or "MARK_PRICE").upper(),
        once=bool(args.once),
    )


def main() -> None:
    args = parse_args()
    if args.live_confirm != LIVE_CONFIRM_TEXT:
        raise SystemExit(f"Refusing to start LIVE trader without --live-confirm {LIVE_CONFIRM_TEXT}")

    settings = load_dashboard_settings()
    if not settings.api_key or not settings.api_secret:
        raise SystemExit("No Binance API key/secret saved in dashboard settings.")
    if settings.use_testnet:
        raise SystemExit("Dashboard settings are TESTNET. Switch to LIVE before running this script.")

    runtime = build_runtime_from_args(args)
    score_settings = ShortlineDowntrendSettings(
        min_entry_score=float(args.min_entry_score),
        min_watch_score=float(args.min_watch_score),
    )
    output_dir = Path(runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "runtime.json", {"runtime": asdict(runtime), "score_settings": asdict(score_settings)})

    client = BinanceFuturesAccountClient(
        BinanceCredentials(settings.api_key, settings.api_secret),
        base_url=BINANCE_FUTURES_BASE_URL,
    )
    public_session = requests.Session()
    state_path = output_dir / "state.json"
    state = load_state(state_path)
    active_symbols_cache: dict[str, Any] = {}

    while True:
        try:
            status = run_cycle(
                client,
                public_session,
                runtime,
                score_settings,
                state,
                output_dir=output_dir,
                active_symbols_cache=active_symbols_cache,
            )
            save_state(state_path, state)
            print(
                json.dumps(
                    {
                        "cycle_time": status["cycle_time"],
                        "mode": status["mode"],
                        "candidates": status["candidate_count"],
                        "positions": len(status["positions"]),
                        "equity": round(_to_float(status["live_equity"]), 6),
                        "opened": len(status["opened"]),
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
            error = {"timestamp": utc_now().isoformat(), "mode": "LIVE", "error": str(exc)}
            append_jsonl(output_dir / "errors.jsonl", error)
            write_json(output_dir / "latest_error.json", error)
            print(json.dumps(error, ensure_ascii=False), flush=True)
        if runtime.once:
            break
        time.sleep(runtime.poll_sec)


if __name__ == "__main__":
    main()
