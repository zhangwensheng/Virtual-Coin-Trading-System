from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any
from urllib.parse import urlencode

import requests


BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
BINANCE_FUTURES_TESTNET_BASE_URL = "https://demo-fapi.binance.com"


class BinanceApiError(RuntimeError):
    pass


@dataclass
class BinanceCredentials:
    api_key: str
    api_secret: str


def signed_query_string(params: dict[str, Any], secret: str) -> str:
    query = urlencode([(key, value) for key, value in params.items() if value is not None], doseq=True)
    signature = hmac.new(secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{query}&signature={signature}"


def normalize_symbol_list(raw: str | list[str]) -> list[str]:
    if isinstance(raw, str):
        values = [item.strip().upper() for item in raw.split(",") if item.strip()]
        return values
    return [str(item).strip().upper() for item in raw if str(item).strip()]


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


def close_side_for_position_amt(position_amt: float) -> str:
    if position_amt > 0:
        return "SELL"
    if position_amt < 0:
        return "BUY"
    raise ValueError("position_amt must be non-zero")


def summarize_income(rows: list[dict[str, Any]]) -> dict[str, float]:
    now_ms = int(time.time() * 1000)
    one_day_ms = 24 * 60 * 60 * 1000
    seven_day_ms = 7 * one_day_ms

    realized_24h = 0.0
    realized_7d = 0.0
    funding_7d = 0.0
    commission_7d = 0.0
    net_7d = 0.0

    for row in rows:
        income = _to_float(row.get("income"))
        income_type = str(row.get("incomeType", ""))
        event_time = _to_int(row.get("time"))
        if now_ms - event_time <= seven_day_ms:
            net_7d += income
            if income_type == "REALIZED_PNL":
                realized_7d += income
            if income_type == "FUNDING_FEE":
                funding_7d += income
            if income_type == "COMMISSION":
                commission_7d += income
        if now_ms - event_time <= one_day_ms and income_type == "REALIZED_PNL":
            realized_24h += income

    return {
        "realized_pnl_24h": round(realized_24h, 4),
        "realized_pnl_7d": round(realized_7d, 4),
        "funding_fee_7d": round(funding_7d, 4),
        "commission_7d": round(commission_7d, 4),
        "net_income_7d": round(net_7d, 4),
    }


class BinanceFuturesAccountClient:
    def __init__(
        self,
        credentials: BinanceCredentials,
        *,
        base_url: str = BINANCE_FUTURES_BASE_URL,
        recv_window: int = 5000,
        timeout: int = 20,
        session: requests.Session | None = None,
    ) -> None:
        self.credentials = credentials
        self.base_url = base_url.rstrip("/")
        self.recv_window = recv_window
        self.timeout = timeout
        self.session = session or requests.Session()
        self._exchange_info_cache: dict[str, Any] | None = None

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> Any:
        payload = dict(params or {})
        headers = {"X-MBX-APIKEY": self.credentials.api_key}
        url = f"{self.base_url}{path}"

        if signed:
            payload["timestamp"] = int(time.time() * 1000)
            payload["recvWindow"] = self.recv_window
            query = signed_query_string(payload, self.credentials.api_secret)
            response = self.session.request(
                method.upper(),
                f"{url}?{query}",
                headers=headers,
                timeout=self.timeout,
            )
        else:
            response = self.session.request(
                method.upper(),
                url,
                params=payload,
                headers=headers,
                timeout=self.timeout,
            )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            raise BinanceApiError(f"Binance API request failed: {detail or exc}") from exc

        try:
            return response.json()
        except ValueError as exc:
            raise BinanceApiError("Binance API returned a non-JSON response") from exc

    def ping_account(self) -> dict[str, Any]:
        return self.account_info()

    def account_info(self) -> dict[str, Any]:
        return self._request("GET", "/fapi/v3/account", signed=True)

    def change_position_mode(self, dual_side_position: bool) -> dict[str, Any]:
        try:
            return self._request(
                "POST",
                "/fapi/v1/positionSide/dual",
                params={"dualSidePosition": "true" if dual_side_position else "false"},
                signed=True,
            )
        except BinanceApiError as exc:
            if "-4059" in str(exc):
                return {"code": -4059, "msg": "No need to change position mode."}
            raise

    def exchange_info(self, *, refresh: bool = False) -> dict[str, Any]:
        if self._exchange_info_cache is None or refresh:
            self._exchange_info_cache = self._request("GET", "/fapi/v1/exchangeInfo", signed=False)
        return self._exchange_info_cache

    def symbol_info(self, symbol: str) -> dict[str, Any]:
        info = self.exchange_info()
        for row in info.get("symbols", []):
            if row.get("symbol") == symbol:
                return row
        raise BinanceApiError(f"Symbol not found in exchangeInfo: {symbol}")

    def change_initial_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        return self._request(
            "POST",
            "/fapi/v1/leverage",
            params={"symbol": symbol, "leverage": int(leverage)},
            signed=True,
        )

    def change_margin_type(self, symbol: str, margin_type: str) -> dict[str, Any]:
        try:
            return self._request(
                "POST",
                "/fapi/v1/marginType",
                params={"symbol": symbol, "marginType": margin_type.upper()},
                signed=True,
            )
        except BinanceApiError as exc:
            if "-4046" in str(exc):
                return {"code": -4046, "msg": "No need to change margin type."}
            raise

    def place_order(self, **params: Any) -> dict[str, Any]:
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)

    def place_algo_order(self, **params: Any) -> dict[str, Any]:
        return self._request("POST", "/fapi/v1/algoOrder", params=params, signed=True)

    def place_close_position_algo_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        trigger_price: float,
        working_type: str = "MARK_PRICE",
        client_algo_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "triggerPrice": trigger_price,
            "closePosition": "true",
            "workingType": working_type,
            "newOrderRespType": "RESULT",
        }
        if client_algo_id:
            params["clientAlgoId"] = client_algo_id
        return self.place_algo_order(**params)

    def close_position_market(
        self,
        symbol: str,
        quantity: float,
        *,
        position_amt: float | None = None,
        position_side: str = "BOTH",
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_amt = quantity if position_amt is None else position_amt
        close_side = close_side_for_position_amt(resolved_amt)
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": close_side,
            "type": "MARKET",
            "quantity": quantity,
            "newOrderRespType": "RESULT",
        }
        normalized_position_side = str(position_side or "BOTH").upper()
        if normalized_position_side != "BOTH":
            params["positionSide"] = normalized_position_side
        else:
            params["reduceOnly"] = "true"
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        return self.place_order(**params)

    def cancel_order(self, symbol: str, order_id: int | None = None, client_order_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = int(order_id)
        if client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        return self._request("DELETE", "/fapi/v1/order", params=params, signed=True)

    def cancel_algo_order(self, algo_id: int | None = None, client_algo_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if algo_id is not None:
            params["algoId"] = int(algo_id)
        if client_algo_id is not None:
            params["clientAlgoId"] = client_algo_id
        return self._request("DELETE", "/fapi/v1/algoOrder", params=params, signed=True)

    def cancel_all_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        cancelled: list[dict[str, Any]] = []
        for order in self.open_orders(symbol):
            try:
                cancelled.append(self.cancel_order(symbol, order_id=_to_int(order.get("orderId"))))
            except BinanceApiError:
                continue
        for order in self.open_algo_orders(symbol):
            try:
                cancelled.append(self.cancel_algo_order(algo_id=_to_int(order.get("algoId"))))
            except BinanceApiError:
                continue
        return cancelled

    def position_risk(self) -> list[dict[str, Any]]:
        return self._request("GET", "/fapi/v3/positionRisk", signed=True)

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {"symbol": symbol} if symbol else {}
        return self._request("GET", "/fapi/v1/openOrders", params=params, signed=True)

    def open_algo_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {"symbol": symbol} if symbol else {}
        return self._request("GET", "/fapi/v1/openAlgoOrders", params=params, signed=True)

    def account_trades(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        params = {"symbol": symbol, "limit": limit}
        return self._request("GET", "/fapi/v1/userTrades", params=params, signed=True)

    def income_history(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._request("GET", "/fapi/v1/income", params={"limit": limit}, signed=True)

    def fetch_dashboard_snapshot(
        self,
        tracked_symbols: list[str],
        *,
        include_history: bool = True,
        trade_limit_per_symbol: int = 50,
        income_limit: int = 100,
    ) -> dict[str, Any]:
        tracked = normalize_symbol_list(tracked_symbols)
        account = self.account_info()
        positions = self.position_risk()
        open_orders: list[dict[str, Any]] = []
        for symbol in tracked:
            try:
                open_orders.extend(self.open_orders(symbol))
            except Exception:
                continue
            try:
                open_orders.extend(self.open_algo_orders(symbol))
            except Exception:
                continue

        trades: list[dict[str, Any]] = []
        incomes: list[dict[str, Any]] = []
        if include_history:
            for symbol in tracked:
                try:
                    trades.extend(self.account_trades(symbol, limit=trade_limit_per_symbol))
                except Exception:
                    continue
            try:
                incomes = self.income_history(limit=income_limit)
            except Exception:
                incomes = []

        return build_dashboard_payload(
            account=account,
            positions=positions,
            open_orders=open_orders,
            trades=trades,
            incomes=incomes,
            tracked_symbols=tracked,
        )

    def quantize_price(self, symbol: str, price: float, *, rounding: str = "down") -> float:
        info = self.symbol_info(symbol)
        price_filter = next((row for row in info.get("filters", []) if row.get("filterType") == "PRICE_FILTER"), None)
        tick_size = _to_float(price_filter.get("tickSize")) if price_filter else 0.0
        if tick_size <= 0:
            return price
        return quantize_to_step(price, tick_size, rounding=rounding)

    def quantize_quantity(self, symbol: str, quantity: float, *, market: bool = True) -> float:
        info = self.symbol_info(symbol)
        filter_type = "MARKET_LOT_SIZE" if market else "LOT_SIZE"
        lot_filter = next((row for row in info.get("filters", []) if row.get("filterType") == filter_type), None)
        if lot_filter is None:
            lot_filter = next((row for row in info.get("filters", []) if row.get("filterType") == "LOT_SIZE"), None)
        if lot_filter is None:
            return quantity

        step_size = _to_float(lot_filter.get("stepSize"))
        min_qty = _to_float(lot_filter.get("minQty"))
        max_qty = _to_float(lot_filter.get("maxQty"))
        value = quantize_to_step(quantity, step_size) if step_size > 0 else quantity
        if min_qty > 0:
            value = max(value, min_qty)
        if max_qty > 0:
            value = min(value, max_qty)
        return value

    def min_notional(self, symbol: str) -> float:
        info = self.symbol_info(symbol)
        min_notional_filter = next(
            (row for row in info.get("filters", []) if row.get("filterType") == "MIN_NOTIONAL"),
            None,
        )
        if min_notional_filter is None:
            return 0.0
        return _to_float(min_notional_filter.get("notional"))


def build_dashboard_payload(
    *,
    account: dict[str, Any],
    positions: list[dict[str, Any]],
    open_orders: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    incomes: list[dict[str, Any]],
    tracked_symbols: list[str],
) -> dict[str, Any]:
    asset_rows: list[dict[str, Any]] = []
    for asset in account.get("assets", []):
        wallet_balance = _to_float(asset.get("walletBalance"))
        unrealized = _to_float(asset.get("unrealizedProfit"))
        margin_balance = _to_float(asset.get("marginBalance"))
        if abs(wallet_balance) <= 1e-8 and abs(unrealized) <= 1e-8 and abs(margin_balance) <= 1e-8:
            continue
        asset_rows.append(
            {
                "asset": asset.get("asset", ""),
                "wallet_balance": round(wallet_balance, 4),
                "available_balance": round(_to_float(asset.get("availableBalance")), 4),
                "unrealized_profit": round(unrealized, 4),
                "margin_balance": round(margin_balance, 4),
            }
        )

    position_rows: list[dict[str, Any]] = []
    for row in positions:
        qty = _to_float(row.get("positionAmt"))
        notional = _to_float(row.get("notional"))
        initial_margin = abs(_to_float(row.get("initialMargin")))
        open_order_initial_margin = abs(_to_float(row.get("openOrderInitialMargin")))
        if abs(qty) <= 1e-8 and open_order_initial_margin <= 1e-8:
            continue
        unrealized = _to_float(row.get("unRealizedProfit"))
        roe_pct = (unrealized / initial_margin * 100.0) if initial_margin > 0 else 0.0
        position_rows.append(
            {
                "symbol": row.get("symbol", ""),
                "side": "LONG" if qty > 0 else "SHORT" if qty < 0 else "FLAT",
                "qty": round(qty, 6),
                "entry_price": round(_to_float(row.get("entryPrice")), 4),
                "mark_price": round(_to_float(row.get("markPrice")), 4),
                "notional": round(notional, 4),
                "liquidation_price": round(_to_float(row.get("liquidationPrice")), 4),
                "unrealized_pnl": round(unrealized, 4),
                "roe_pct": round(roe_pct, 2),
                "leverage": _to_int(row.get("leverage")),
                "margin_type": row.get("marginType", ""),
                "position_side": row.get("positionSide", ""),
            }
        )

    def order_trigger_price(row: dict[str, Any]) -> float:
        stop_price = _to_float(row.get("stopPrice"))
        if stop_price > 0:
            return stop_price
        return _to_float(row.get("triggerPrice"))

    def infer_trade_position_side(row: dict[str, Any]) -> str:
        position_side = str(row.get("positionSide", "") or "").upper()
        if position_side and position_side != "BOTH":
            return position_side
        side = str(row.get("side", "") or "").upper()
        realized = _to_float(row.get("realizedPnl"))
        if abs(realized) > 1e-12:
            return "SHORT" if side == "BUY" else "LONG" if side == "SELL" else position_side
        return "LONG" if side == "BUY" else "SHORT" if side == "SELL" else position_side

    order_rows = [
        {
            "symbol": row.get("symbol", ""),
            "side": row.get("side", ""),
            "type": row.get("type", ""),
            "status": row.get("status", ""),
            "orig_qty": round(_to_float(row.get("origQty")), 6),
            "price": round(_to_float(row.get("price")), 6),
            "stop_price": round(order_trigger_price(row), 6),
            "avg_price": round(_to_float(row.get("avgPrice")), 6),
            "update_time": _to_int(row.get("updateTime")),
        }
        for row in open_orders
    ]

    trade_rows = sorted(
        [
            {
                "symbol": row.get("symbol", ""),
                "side": row.get("side", ""),
                "position_side": infer_trade_position_side(row),
                "qty": round(_to_float(row.get("qty")), 6),
                "price": round(_to_float(row.get("price")), 6),
                "realized_pnl": round(_to_float(row.get("realizedPnl")), 6),
                "commission": round(_to_float(row.get("commission")), 6),
                "commission_asset": row.get("commissionAsset", ""),
                "time": _to_int(row.get("time")),
            }
            for row in trades
        ],
        key=lambda item: item["time"],
        reverse=True,
    )

    income_rows = sorted(
        [
            {
                "symbol": row.get("symbol", ""),
                "income_type": row.get("incomeType", ""),
                "income": round(_to_float(row.get("income")), 6),
                "asset": row.get("asset", ""),
                "info": row.get("info", ""),
                "time": _to_int(row.get("time")),
            }
            for row in incomes
        ],
        key=lambda item: item["time"],
        reverse=True,
    )

    income_summary = summarize_income(incomes)
    summary = {
        "total_wallet_balance": round(_to_float(account.get("totalWalletBalance")), 4),
        "available_balance": round(_to_float(account.get("availableBalance")), 4),
        "total_margin_balance": round(_to_float(account.get("totalMarginBalance")), 4),
        "total_unrealized_profit": round(_to_float(account.get("totalUnrealizedProfit")), 4),
        "total_initial_margin": round(_to_float(account.get("totalInitialMargin")), 4),
        "total_maint_margin": round(_to_float(account.get("totalMaintMargin")), 4),
        "max_withdraw_amount": round(_to_float(account.get("maxWithdrawAmount")), 4),
        "position_count": len(position_rows),
        "open_order_count": len(order_rows),
        "tracked_symbols": tracked_symbols,
        **income_summary,
    }

    return {
        "summary": summary,
        "assets": sorted(asset_rows, key=lambda item: item["wallet_balance"], reverse=True),
        "positions": sorted(position_rows, key=lambda item: abs(item["notional"]), reverse=True),
        "open_orders": order_rows,
        "trades": trade_rows,
        "income": income_rows,
    }


def quantize_to_step(value: float, step: float, *, rounding: str = "down") -> float:
    if step <= 0:
        return value
    step_dec = Decimal(str(step))
    value_dec = Decimal(str(value))
    rounding_mode = ROUND_UP if str(rounding).lower() == "up" else ROUND_DOWN
    quantized = (value_dec / step_dec).quantize(Decimal("1"), rounding=rounding_mode) * step_dec
    return float(quantized)
