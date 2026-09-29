from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from futures_strategy.backtest import resolve_risk_multiplier
from futures_strategy.binance_account_client import BinanceApiError, BinanceFuturesAccountClient
from futures_strategy.config import AppConfig, load_config
from futures_strategy.data import load_market_data
from futures_strategy.exchanges import to_linear_symbol
from futures_strategy.factor_daily_scan import parse_config_paths, rank_scan_results, refresh_factor_csv
from futures_strategy.factor_portfolio import candidate_rank_score
from futures_strategy.strategy import latest_signal_snapshot, prepare_market_data


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


def _is_unknown_order_error(exc: Exception) -> bool:
    message = str(exc)
    return "-2011" in message or "Unknown order" in message or "-4045" in message


def _protection_order_id(order: dict[str, Any]) -> int:
    return _to_int(order.get("algoId") or order.get("orderId"))


def _bars_held_since(index: pd.DatetimeIndex, entry_time: str) -> int:
    if len(index) == 0:
        return 0
    entry_ts = pd.Timestamp(entry_time)
    if entry_ts.tzinfo is None:
        entry_ts = entry_ts.tz_localize("UTC")
    return max(0, int((index >= entry_ts).sum()) - 1)


def _client_order_id(prefix: str, symbol: str, suffix: str) -> str:
    compact = symbol.replace("/", "").replace(":", "").replace("-", "")
    return f"{prefix}_{compact}_{suffix}"[:36]


def utc_now() -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        return now.tz_localize("UTC")
    return now.tz_convert("UTC")


@dataclass
class AutoTraderRuntimeConfig:
    config_dir: str = ""
    configs: str = ""
    cache_dir: str = "data/binance_factor_bundle"
    state_path: str = "outputs/factor_auto_trader/auto_trader_state.json"
    poll_sec: int = 60
    top_n: int = 1
    days_back: int = 27
    leverage: int = 20
    margin_type: str = "ISOLATED"
    risk_per_trade: float | None = 0.03
    max_positions: int = 1
    working_type: str = "MARK_PRICE"
    use_testnet: bool = True

    @property
    def mode_label(self) -> str:
        return "TESTNET" if self.use_testnet else "LIVE"

    def resolved_config_paths(self) -> list[Path]:
        return parse_config_paths(self.config_dir or None, self.configs or None)


@dataclass
class ManagedTradeState:
    native_symbol: str
    display_symbol: str
    config_path: str
    signal_timestamp: str
    entry_time: str
    entry_price: float
    entry_qty: float
    stop_price: float
    tp_price: float | None = None
    stop_order_id: int | None = None
    tp_order_id: int | None = None
    partial_taken: bool = False
    last_status: str = "OPEN"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ManagedTradeState":
        return cls(
            native_symbol=str(payload.get("native_symbol", "")),
            display_symbol=str(payload.get("display_symbol", "")),
            config_path=str(payload.get("config_path", "")),
            signal_timestamp=str(payload.get("signal_timestamp", "")),
            entry_time=str(payload.get("entry_time", "")),
            entry_price=_to_float(payload.get("entry_price")),
            entry_qty=_to_float(payload.get("entry_qty")),
            stop_price=_to_float(payload.get("stop_price")),
            tp_price=None if payload.get("tp_price") in (None, "") else _to_float(payload.get("tp_price")),
            stop_order_id=None if payload.get("stop_order_id") in (None, "") else _to_int(payload.get("stop_order_id")),
            tp_order_id=None if payload.get("tp_order_id") in (None, "") else _to_int(payload.get("tp_order_id")),
            partial_taken=bool(payload.get("partial_taken", False)),
            last_status=str(payload.get("last_status", "OPEN")),
        )


@dataclass
class ShortOrderPlan:
    native_symbol: str
    display_symbol: str
    qty: float
    entry_estimate: float
    stop_price: float
    tp_price: float | None
    tp_qty: float
    rank_score: float
    factor_score: int
    risk_multiplier: float


@dataclass
class SymbolContext:
    config_path: Path
    config: AppConfig
    prepared: pd.DataFrame
    native_symbol: str
    display_symbol: str
    scan_row: dict[str, Any]


@dataclass
class AutoTraderCycleResult:
    cycle_time: str
    mode_label: str
    status: str
    messages: list[str]
    candidate_rows: list[dict[str, Any]]
    managed_rows: list[dict[str, Any]]


def load_auto_trader_state(path: str | Path) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return {
            "managed_trades": {},
            "last_signal_timestamps": {},
            "one_way_mode_confirmed": False,
            "manual_flat_cooldowns": {},
        }
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload.setdefault("managed_trades", {})
    payload.setdefault("last_signal_timestamps", {})
    payload.setdefault("one_way_mode_confirmed", False)
    payload.setdefault("manual_flat_cooldowns", {})
    return payload


def save_auto_trader_state(path: str | Path, state: dict[str, Any]) -> Path:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state_path


def apply_manual_flat_cooldown(
    path: str | Path,
    symbol: str,
    *,
    cooldown_hours: int = 4,
) -> Path:
    state = load_auto_trader_state(path)
    state["managed_trades"].pop(symbol, None)
    cooldown_until = (utc_now() + pd.Timedelta(hours=max(1, cooldown_hours))).isoformat()
    state["manual_flat_cooldowns"][symbol] = cooldown_until
    return save_auto_trader_state(path, state)


def build_symbol_context(
    config_path: Path,
    *,
    cache_dir: Path,
    days_back: int,
    session: requests.Session,
) -> SymbolContext:
    config = load_config(config_path)
    existing_csv = Path(config.exchange.csv_path) if config.exchange.csv_path else None
    try:
        csv_path = refresh_factor_csv(
            config,
            output_dir=cache_dir,
            days_back=days_back,
            end_time=None,
            session=session,
        )
        config.exchange.csv_path = str(csv_path)
    except Exception:
        if existing_csv is None or not existing_csv.exists():
            raise
        config.exchange.csv_path = str(existing_csv)

    raw = load_market_data(config.exchange)
    prepared = prepare_market_data(raw, config.strategy)
    if prepared.empty:
        raise ValueError(f"No prepared rows for {config.exchange.symbol}")

    signal = latest_signal_snapshot(prepared, config.strategy)
    latest_row = prepared.iloc[-1]
    action = str(signal.get("action", "UNKNOWN"))
    scan_row = {
        "symbol": config.exchange.symbol,
        "native_symbol": to_linear_symbol(config.exchange.symbol),
        "config_path": str(config_path),
        "timestamp": str(signal.get("timestamp", prepared.index[-1].isoformat())),
        "action": action,
        "action_priority": 3 if action == "FUNDING_OI_BEAR_SHORT_SETUP" else 2 if action == "FUNDING_OI_BEAR_SHORT_BIAS" else 1,
        "ready_to_short": action == "FUNDING_OI_BEAR_SHORT_SETUP",
        "has_short_bias": action in {"FUNDING_OI_BEAR_SHORT_SETUP", "FUNDING_OI_BEAR_SHORT_BIAS"},
        "rank_score": round(float(candidate_rank_score(latest_row)), 4),
        "factor_score": int(signal.get("factor_score", 0)),
        "risk_multiplier": float(signal.get("risk_multiplier", 1.0)),
        "close": float(signal.get("close", latest_row["close"])),
        "ema_entry": float(signal.get("ema_entry", latest_row.get("ema_entry", latest_row["close"]))),
        "atr": float(signal.get("atr", latest_row.get("atr", 0.0))),
        "rsi": float(signal.get("rsi", latest_row.get("rsi", 0.0))),
        "htf_adx": float(signal.get("htf_adx", latest_row.get("htf_adx", 0.0))),
        "funding_rate": float(signal.get("funding_rate", latest_row.get("funding_rate", 0.0))),
        "funding_change": float(signal.get("funding_change", latest_row.get("funding_change", 0.0))),
        "oi_value_change_pct": float(signal.get("oi_value_change_pct", 0.0)),
        "price_change_pct": float(signal.get("price_change_pct", 0.0)),
        "bounce_pct": float(signal.get("bounce_pct", 0.0)),
        "stop_hint": signal.get("stop_hint"),
    }
    return SymbolContext(
        config_path=config_path,
        config=config,
        prepared=prepared,
        native_symbol=to_linear_symbol(config.exchange.symbol),
        display_symbol=config.exchange.symbol,
        scan_row=scan_row,
    )


def load_symbol_contexts(runtime: AutoTraderRuntimeConfig) -> tuple[pd.DataFrame, dict[str, SymbolContext]]:
    session = requests.Session()
    cache_dir = Path(runtime.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    contexts: dict[str, SymbolContext] = {}
    rows: list[dict[str, Any]] = []
    for config_path in runtime.resolved_config_paths():
        try:
            context = build_symbol_context(
                config_path,
                cache_dir=cache_dir,
                days_back=runtime.days_back,
                session=session,
            )
        except Exception:
            continue
        contexts[context.native_symbol] = context
        rows.append(context.scan_row)
    ranked = rank_scan_results(pd.DataFrame(rows))
    return ranked, contexts


def build_short_order_plan(
    client: BinanceFuturesAccountClient,
    context: SymbolContext,
    available_balance: float,
    runtime: AutoTraderRuntimeConfig,
) -> ShortOrderPlan | None:
    row = context.scan_row
    entry_estimate = _to_float(row.get("close"))
    atr_value = _to_float(row.get("atr"))
    if entry_estimate <= 0 or atr_value <= 0:
        return None

    stop_hint = _to_float(row.get("stop_hint"), default=0.0)
    if stop_hint <= entry_estimate:
        stop_hint = entry_estimate + max(atr_value * context.config.strategy.atr_stop_mult, entry_estimate * 0.0015)
    stop_price = client.quantize_price(context.native_symbol, stop_hint, rounding="up")
    if stop_price <= entry_estimate:
        stop_price = client.quantize_price(
            context.native_symbol,
            entry_estimate + max(atr_value * 0.5, entry_estimate * 0.0015),
            rounding="up",
        )

    risk_distance = stop_price - entry_estimate
    if risk_distance <= 0:
        return None

    risk_multiplier = resolve_risk_multiplier(pd.Series({"risk_multiplier": row.get("risk_multiplier", 1.0)}))
    risk_per_trade = runtime.risk_per_trade if runtime.risk_per_trade is not None else context.config.risk.risk_per_trade
    risk_amount = available_balance * risk_per_trade * risk_multiplier
    max_notional_fraction = min(context.config.risk.max_notional_fraction, 0.95 / max(1, runtime.max_positions))
    max_notional = available_balance * runtime.leverage * max_notional_fraction
    raw_qty = min(risk_amount / risk_distance, max_notional / entry_estimate)
    qty = client.quantize_quantity(context.native_symbol, raw_qty, market=True)
    if qty <= 0:
        return None

    min_notional = client.min_notional(context.native_symbol)
    if min_notional > 0 and (qty * entry_estimate) < min_notional:
        return None

    tp_price = client.quantize_price(
        context.native_symbol,
        entry_estimate - (risk_distance * context.config.strategy.partial_rr),
        rounding="down",
    )
    tp_qty = client.quantize_quantity(
        context.native_symbol,
        qty * context.config.strategy.partial_close_ratio,
        market=True,
    )
    if tp_qty >= qty or tp_qty <= 0:
        tp_price = None
        tp_qty = 0.0

    return ShortOrderPlan(
        native_symbol=context.native_symbol,
        display_symbol=context.display_symbol,
        qty=qty,
        entry_estimate=entry_estimate,
        stop_price=stop_price,
        tp_price=tp_price,
        tp_qty=tp_qty,
        rank_score=_to_float(row.get("rank_score")),
        factor_score=_to_int(row.get("factor_score")),
        risk_multiplier=risk_multiplier,
    )


def place_stop_order(
    client: BinanceFuturesAccountClient,
    *,
    symbol: str,
    quantity: float,
    stop_price: float,
    working_type: str,
    client_order_id: str,
) -> dict[str, Any]:
    return client.place_close_position_algo_order(
        symbol=symbol,
        side="BUY",
        order_type="STOP_MARKET",
        trigger_price=stop_price,
        working_type=working_type,
        client_algo_id=client_order_id,
    )


def place_take_profit_order(
    client: BinanceFuturesAccountClient,
    *,
    symbol: str,
    quantity: float,
    stop_price: float,
    working_type: str,
    client_order_id: str,
) -> dict[str, Any]:
    return client.place_close_position_algo_order(
        symbol=symbol,
        side="BUY",
        order_type="TAKE_PROFIT_MARKET",
        trigger_price=stop_price,
        working_type=working_type,
        client_algo_id=client_order_id,
    )


def cancel_managed_orders(
    client: BinanceFuturesAccountClient,
    trade: ManagedTradeState,
    messages: list[str],
) -> None:
    for order_id in [trade.stop_order_id, trade.tp_order_id]:
        if order_id is None:
            continue
        try:
            client.cancel_algo_order(algo_id=order_id)
            messages.append(f"[{trade.native_symbol}] 已撤销保护单 #{order_id}")
        except BinanceApiError as exc:
            if not _is_unknown_order_error(exc):
                messages.append(f"[{trade.native_symbol}] 撤单失败 #{order_id}: {exc}")
                continue
            try:
                client.cancel_order(trade.native_symbol, order_id=order_id)
                messages.append(f"[{trade.native_symbol}] 已撤销保护单 #{order_id}")
            except BinanceApiError as fallback_exc:
                if not _is_unknown_order_error(fallback_exc):
                    messages.append(f"[{trade.native_symbol}] 撤单失败 #{order_id}: {fallback_exc}")
    trade.stop_order_id = None
    trade.tp_order_id = None


def enter_short_trade(
    client: BinanceFuturesAccountClient,
    context: SymbolContext,
    plan: ShortOrderPlan,
    runtime: AutoTraderRuntimeConfig,
    state: dict[str, Any],
    messages: list[str],
) -> ManagedTradeState:
    suffix = str(pd.Timestamp.utcnow().value)[-10:]
    client.change_margin_type(plan.native_symbol, runtime.margin_type)
    client.change_initial_leverage(plan.native_symbol, runtime.leverage)

    entry_order = client.place_order(
        symbol=plan.native_symbol,
        side="SELL",
        type="MARKET",
        quantity=plan.qty,
        newOrderRespType="RESULT",
        newClientOrderId=_client_order_id("AUTOSE", plan.native_symbol, suffix),
    )
    entry_qty = abs(_to_float(entry_order.get("executedQty"), plan.qty))
    if entry_qty <= 0:
        entry_qty = plan.qty
    entry_price = _to_float(entry_order.get("avgPrice"))
    if entry_price <= 0:
        cum_quote = _to_float(entry_order.get("cumQuote"))
        entry_price = (cum_quote / entry_qty) if cum_quote > 0 and entry_qty > 0 else plan.entry_estimate

    stop_price = client.quantize_price(
        plan.native_symbol,
        max(plan.stop_price, entry_price + max(_to_float(context.scan_row.get("atr")) * 0.4, entry_price * 0.001)),
        rounding="up",
    )
    if stop_price <= entry_price:
        stop_price = client.quantize_price(plan.native_symbol, entry_price * 1.002, rounding="up")

    tp_price = None
    tp_qty = 0.0
    if plan.tp_price is not None:
        tp_price = client.quantize_price(
            plan.native_symbol,
            entry_price - ((stop_price - entry_price) * context.config.strategy.partial_rr),
            rounding="down",
        )
        tp_qty = client.quantize_quantity(
            plan.native_symbol,
            entry_qty * context.config.strategy.partial_close_ratio,
            market=True,
        )
        if tp_price >= entry_price or tp_qty <= 0 or tp_qty >= entry_qty:
            tp_price = None
            tp_qty = 0.0

    try:
        stop_order = place_stop_order(
            client,
            symbol=plan.native_symbol,
            quantity=entry_qty,
            stop_price=stop_price,
            working_type=runtime.working_type,
            client_order_id=_client_order_id("AUTOST", plan.native_symbol, suffix),
        )
    except Exception:
        close_market_short(client, plan.native_symbol, entry_qty, messages, reason="stop_order_failed")
        raise

    tp_order = None
    if tp_price is not None and tp_qty > 0:
        try:
            tp_order = place_take_profit_order(
                client,
                symbol=plan.native_symbol,
                quantity=tp_qty,
                stop_price=tp_price,
                working_type=runtime.working_type,
                client_order_id=_client_order_id("AUTOTP", plan.native_symbol, suffix),
            )
        except BinanceApiError as exc:
            messages.append(f"[{plan.native_symbol}] 分批止盈单下单失败，继续仅保留止损单: {exc}")

    trade = ManagedTradeState(
        native_symbol=plan.native_symbol,
        display_symbol=plan.display_symbol,
        config_path=str(context.config_path),
        signal_timestamp=str(context.scan_row["timestamp"]),
        entry_time=utc_now().isoformat(),
        entry_price=entry_price,
        entry_qty=entry_qty,
        stop_price=stop_price,
        tp_price=tp_price,
        stop_order_id=_protection_order_id(stop_order) if stop_order else None,
        tp_order_id=_protection_order_id(tp_order) if tp_order else None,
        partial_taken=False,
        last_status="OPEN",
    )
    state["managed_trades"][plan.native_symbol] = asdict(trade)
    state["last_signal_timestamps"][plan.native_symbol] = trade.signal_timestamp
    messages.append(
        f"[{plan.native_symbol}] 已开空 qty={entry_qty:.6f} entry={entry_price:.6f} stop={stop_price:.6f}"
    )
    return trade


def close_market_short(
    client: BinanceFuturesAccountClient,
    symbol: str,
    quantity: float,
    messages: list[str],
    *,
    reason: str,
) -> None:
    client.close_position_market(
        symbol,
        quantity,
        position_amt=-abs(quantity),
        position_side="BOTH",
        client_order_id=_client_order_id("AUTOCL", symbol, str(pd.Timestamp.utcnow().value)[-10:]),
    )
    messages.append(f"[{symbol}] 已市价平空，原因: {reason}")


def ensure_protective_orders(
    client: BinanceFuturesAccountClient,
    trade: ManagedTradeState,
    open_order_ids: set[int],
    current_qty: float,
    runtime: AutoTraderRuntimeConfig,
    messages: list[str],
    *,
    tp_ratio: float = 0.5,
) -> ManagedTradeState:
    suffix = str(pd.Timestamp.utcnow().value)[-10:]
    if trade.stop_order_id not in open_order_ids:
        stop_order = place_stop_order(
            client,
            symbol=trade.native_symbol,
            quantity=current_qty,
            stop_price=trade.stop_price,
            working_type=runtime.working_type,
            client_order_id=_client_order_id("AUTOST", trade.native_symbol, suffix),
        )
        trade.stop_order_id = _protection_order_id(stop_order)
        messages.append(f"[{trade.native_symbol}] 已补挂止损单 #{trade.stop_order_id}")

    if (not trade.partial_taken) and trade.tp_price and trade.tp_order_id not in open_order_ids:
        tp_qty = client.quantize_quantity(
            trade.native_symbol,
            current_qty * tp_ratio,
            market=True,
        )
        if tp_qty > 0 and tp_qty < current_qty:
            tp_order = place_take_profit_order(
                client,
                symbol=trade.native_symbol,
                quantity=tp_qty,
                stop_price=trade.tp_price,
                working_type=runtime.working_type,
                client_order_id=_client_order_id("AUTOTP", trade.native_symbol, suffix),
            )
            trade.tp_order_id = _protection_order_id(tp_order)
            messages.append(f"[{trade.native_symbol}] 已补挂止盈单 #{trade.tp_order_id}")
    return trade


def update_trailing_stop_if_needed(
    client: BinanceFuturesAccountClient,
    trade: ManagedTradeState,
    context: SymbolContext,
    current_qty: float,
    runtime: AutoTraderRuntimeConfig,
    messages: list[str],
) -> ManagedTradeState:
    frame = context.prepared
    subset = frame.loc[frame.index >= pd.Timestamp(trade.signal_timestamp)]
    if subset.empty:
        subset = frame
    latest_row = subset.iloc[-1]
    atr_value = _to_float(latest_row.get("atr"))
    if atr_value <= 0:
        return trade

    lowest_price = _to_float(subset["low"].min())
    if lowest_price <= 0:
        return trade

    trailing_stop = min(
        trade.stop_price,
        trade.entry_price,
        client.quantize_price(
            trade.native_symbol,
            lowest_price + (atr_value * context.config.strategy.trail_atr_mult),
            rounding="up",
        ),
    )
    if trailing_stop >= trade.stop_price:
        return trade

    cancel_managed_orders(client, trade, messages)
    stop_order = place_stop_order(
        client,
        symbol=trade.native_symbol,
        quantity=current_qty,
        stop_price=trailing_stop,
        working_type=runtime.working_type,
        client_order_id=_client_order_id("AUTOST", trade.native_symbol, str(pd.Timestamp.utcnow().value)[-10:]),
    )
    trade.stop_price = trailing_stop
    trade.stop_order_id = _protection_order_id(stop_order)
    trade.tp_order_id = None
    messages.append(f"[{trade.native_symbol}] 已更新移动止损到 {trailing_stop:.6f}")
    return trade


def build_managed_row(
    trade: ManagedTradeState,
    position: dict[str, Any] | None,
) -> dict[str, Any]:
    current_qty = abs(_to_float(position.get("positionAmt"))) if position else 0.0
    return {
        "native_symbol": trade.native_symbol,
        "symbol": trade.display_symbol,
        "entry_time": trade.entry_time,
        "signal_timestamp": trade.signal_timestamp,
        "entry_price": round(trade.entry_price, 6),
        "qty": round(current_qty or trade.entry_qty, 6),
        "mark_price": round(_to_float(position.get("markPrice")) if position else 0.0, 6),
        "unrealized_pnl": round(_to_float(position.get("unRealizedProfit")) if position else 0.0, 4),
        "stop_price": round(trade.stop_price, 6),
        "tp_price": "" if trade.tp_price is None else round(trade.tp_price, 6),
        "partial_taken": "YES" if trade.partial_taken else "NO",
        "status": trade.last_status,
    }


def execute_auto_trader_cycle(
    client: BinanceFuturesAccountClient,
    runtime: AutoTraderRuntimeConfig,
) -> AutoTraderCycleResult:
    messages: list[str] = []
    state = load_auto_trader_state(runtime.state_path)
    now_utc = utc_now()
    for symbol, until in list(state.get("manual_flat_cooldowns", {}).items()):
        try:
            cooldown_until = pd.Timestamp(until)
            if cooldown_until.tzinfo is None:
                cooldown_until = cooldown_until.tz_localize("UTC")
        except Exception:  # noqa: BLE001
            state["manual_flat_cooldowns"].pop(symbol, None)
            continue
        if cooldown_until <= now_utc:
            state["manual_flat_cooldowns"].pop(symbol, None)

    session = requests.Session()
    cache_dir = Path(runtime.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    contexts: dict[str, SymbolContext] = {}
    context_rows: list[dict[str, Any]] = []
    for config_path in runtime.resolved_config_paths():
        try:
            context = build_symbol_context(
                config_path,
                cache_dir=cache_dir,
                days_back=runtime.days_back,
                session=session,
            )
        except Exception as exc:  # noqa: BLE001
            messages.append(f"[{Path(config_path).name}] 配置加载失败，已跳过: {exc}")
            continue
        contexts[context.native_symbol] = context
        context_rows.append(context.scan_row)
    ranked_rows = rank_scan_results(pd.DataFrame(context_rows))

    if not state.get("one_way_mode_confirmed"):
        client.change_position_mode(False)
        state["one_way_mode_confirmed"] = True
        messages.append("已确认账户使用单向持仓模式(BOTH)。")

    account = client.account_info()
    available_balance = _to_float(account.get("availableBalance"))
    positions = {
        row.get("symbol", ""): row
        for row in client.position_risk()
        if abs(_to_float(row.get("positionAmt"))) > 1e-12
    }
    open_orders: dict[str, list[dict[str, Any]]] = {}
    for symbol in contexts:
        symbol_orders: list[dict[str, Any]] = []
        try:
            symbol_orders.extend(client.open_orders(symbol))
        except BinanceApiError:
            pass
        try:
            symbol_orders.extend(client.open_algo_orders(symbol))
        except BinanceApiError:
            pass
        open_orders[symbol] = symbol_orders

    managed_rows: list[dict[str, Any]] = []
    active_managed = 0

    for native_symbol, payload in list(state["managed_trades"].items()):
        trade = ManagedTradeState.from_dict(payload)
        context = contexts.get(native_symbol)
        position = positions.get(native_symbol)

        if position is None:
            cancel_managed_orders(client, trade, messages)
            state["managed_trades"].pop(native_symbol, None)
            messages.append(f"[{native_symbol}] 检测到仓位已关闭，状态已清理。")
            continue

        current_qty = abs(_to_float(position.get("positionAmt")))
        if current_qty <= 1e-12:
            cancel_managed_orders(client, trade, messages)
            state["managed_trades"].pop(native_symbol, None)
            continue

        order_ids = {_protection_order_id(item) for item in open_orders.get(native_symbol, [])}
        order_ids.discard(0)
        tp_ratio = context.config.strategy.partial_close_ratio if context is not None else 0.5
        trade = ensure_protective_orders(
            client,
            trade,
            order_ids,
            current_qty,
            runtime,
            messages,
            tp_ratio=tp_ratio,
        )

        if context is not None:
            partial_threshold = trade.entry_qty * max(0.2, 1.0 - (context.config.strategy.partial_close_ratio * 0.7))
            if (not trade.partial_taken) and current_qty < partial_threshold:
                trade.partial_taken = True
                trade.last_status = "PARTIAL"
                messages.append(f"[{native_symbol}] 检测到已部分止盈，开始启用移动止损。")

            if trade.partial_taken:
                trade = update_trailing_stop_if_needed(client, trade, context, current_qty, runtime, messages)

            latest_row = context.prepared.iloc[-1]
            exit_signal = bool(latest_row.get("exit_short_signal", False))
            bars_held = _bars_held_since(context.prepared.index, trade.signal_timestamp)
            if exit_signal or bars_held >= context.config.strategy.max_bars_in_trade:
                cancel_managed_orders(client, trade, messages)
                close_market_short(
                    client,
                    trade.native_symbol,
                    current_qty,
                    messages,
                    reason="signal_exit" if exit_signal else "time_exit",
                )
                state["managed_trades"].pop(native_symbol, None)
                continue

        trade.last_status = "MANAGED"
        state["managed_trades"][native_symbol] = asdict(trade)
        managed_rows.append(build_managed_row(trade, position))
        active_managed += 1

    slots = max(0, runtime.max_positions - active_managed)
    candidate_frame = ranked_rows.copy()
    if not candidate_frame.empty:
        candidate_frame = candidate_frame[candidate_frame["ready_to_short"]].head(max(runtime.top_n, slots))

    for _, row in candidate_frame.iterrows():
        if slots <= 0:
            break
        native_symbol = str(row["native_symbol"])
        if native_symbol in state["managed_trades"]:
            continue
        if native_symbol in positions:
            messages.append(f"[{native_symbol}] 已有仓位，跳过自动开空。")
            continue
        cooldown_until = state.get("manual_flat_cooldowns", {}).get(native_symbol)
        if cooldown_until:
            messages.append(f"[{native_symbol}] 手动平仓冷静期中，跳过自动开空，直到 {cooldown_until}。")
            continue
        if open_orders.get(native_symbol):
            messages.append(f"[{native_symbol}] 已有挂单，跳过自动开空。")
            continue
        if str(state["last_signal_timestamps"].get(native_symbol, "")) == str(row["timestamp"]):
            continue

        context = contexts[native_symbol]
        plan = build_short_order_plan(client, context, available_balance, runtime)
        if plan is None:
            messages.append(f"[{native_symbol}] 因风险或最小名义价值限制，跳过本次信号。")
            state["last_signal_timestamps"][native_symbol] = str(row["timestamp"])
            continue

        enter_short_trade(client, context, plan, runtime, state, messages)
        slots -= 1
        available_balance = _to_float(client.account_info().get("availableBalance"), available_balance)

    save_auto_trader_state(runtime.state_path, state)

    candidate_rows = ranked_rows.to_dict(orient="records") if not ranked_rows.empty else []
    return AutoTraderCycleResult(
        cycle_time=utc_now().isoformat(),
        mode_label=runtime.mode_label,
        status=f"active_positions={active_managed} available_balance={available_balance:.2f}",
        messages=messages,
        candidate_rows=candidate_rows,
        managed_rows=managed_rows,
    )
