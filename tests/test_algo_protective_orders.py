from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from futures_strategy.auto_trader import place_stop_order as place_auto_stop_order
from run_shortline_downtrend_live_trader import (
    ShortlineLiveRuntime,
    build_runtime_from_args,
    build_order_plan,
    score_long_momentum,
    open_orders_by_symbol,
    parse_args,
    place_stop_order as place_shortline_stop_order,
    symbols_requiring_order_checks,
)


class DummyAlgoClient:
    def __init__(self) -> None:
        self.algo_calls: list[dict[str, object]] = []

    def place_close_position_algo_order(self, **kwargs: object) -> dict[str, object]:
        self.algo_calls.append(kwargs)
        return {"algoId": 12345}


class DummyOrderClient(DummyAlgoClient):
    def open_orders(self, symbol: str) -> list[dict[str, object]]:
        return [{"symbol": symbol, "orderId": 11, "type": "LIMIT"}]

    def open_algo_orders(self, symbol: str) -> list[dict[str, object]]:
        return [{"symbol": symbol, "algoId": 22, "type": "STOP_MARKET"}]


class DummyPlanClient:
    def quantize_price(self, symbol: str, price: float, *, rounding: str = "down") -> float:
        if rounding == "up":
            return round(price + 1e-12, 2)
        return round(price - 1e-12, 2)

    def quantize_quantity(self, symbol: str, quantity: float, *, market: bool = True) -> float:
        return round(quantity, 3)

    def min_notional(self, symbol: str) -> float:
        return 5.0


class AlgoProtectiveOrdersTest(unittest.TestCase):
    def test_shortline_stop_order_uses_algo_close_position_order(self) -> None:
        client = DummyAlgoClient()
        runtime = ShortlineLiveRuntime(trade_side="SHORT", working_type="MARK_PRICE")

        response = place_shortline_stop_order(
            client,  # type: ignore[arg-type]
            symbol="BTCUSDT",
            quantity=0.01,
            stop_price=105.5,
            runtime=runtime,
        )

        self.assertEqual(response["algoId"], 12345)
        self.assertEqual(client.algo_calls[0]["symbol"], "BTCUSDT")
        self.assertEqual(client.algo_calls[0]["side"], "BUY")
        self.assertEqual(client.algo_calls[0]["order_type"], "STOP_MARKET")
        self.assertEqual(client.algo_calls[0]["trigger_price"], 105.5)
        self.assertEqual(client.algo_calls[0]["working_type"], "MARK_PRICE")

    def test_shortline_long_stop_order_uses_sell_algo_close_position_order(self) -> None:
        client = DummyAlgoClient()
        runtime = ShortlineLiveRuntime(trade_side="LONG", working_type="MARK_PRICE")

        response = place_shortline_stop_order(
            client,  # type: ignore[arg-type]
            symbol="BTCUSDT",
            quantity=0.01,
            stop_price=98.0,
            runtime=runtime,
        )

        self.assertEqual(response["algoId"], 12345)
        self.assertEqual(client.algo_calls[0]["symbol"], "BTCUSDT")
        self.assertEqual(client.algo_calls[0]["side"], "SELL")
        self.assertEqual(client.algo_calls[0]["order_type"], "STOP_MARKET")
        self.assertEqual(client.algo_calls[0]["trigger_price"], 98.0)

    def test_auto_stop_order_uses_algo_close_position_order(self) -> None:
        client = DummyAlgoClient()

        response = place_auto_stop_order(
            client,  # type: ignore[arg-type]
            symbol="ETHUSDT",
            quantity=0.2,
            stop_price=3500.0,
            working_type="CONTRACT_PRICE",
            client_order_id="AUTOST_ETHUSDT_1",
        )

        self.assertEqual(response["algoId"], 12345)
        self.assertEqual(client.algo_calls[0]["symbol"], "ETHUSDT")
        self.assertEqual(client.algo_calls[0]["side"], "BUY")
        self.assertEqual(client.algo_calls[0]["order_type"], "STOP_MARKET")
        self.assertEqual(client.algo_calls[0]["trigger_price"], 3500.0)
        self.assertEqual(client.algo_calls[0]["working_type"], "CONTRACT_PRICE")
        self.assertEqual(client.algo_calls[0]["client_algo_id"], "AUTOST_ETHUSDT_1")

    def test_shortline_open_orders_include_algo_orders(self) -> None:
        client = DummyOrderClient()

        result = open_orders_by_symbol(client, ["BTCUSDT"])  # type: ignore[arg-type]

        ids = {row.get("orderId") or row.get("algoId") for row in result["BTCUSDT"]}
        self.assertEqual(ids, {11, 22})

    def test_shortline_order_plan_uses_roe_targets(self) -> None:
        runtime = ShortlineLiveRuntime(
            trade_side="SHORT",
            leverage=10,
            margin_usdt=1.0,
            take_profit_roe_pct=10.0,
            stop_loss_roe_pct=20.0,
        )
        row = pd.Series({"close": 100.0, "atr": 1.0, "swing_high": 101.0})

        plan = build_order_plan(DummyPlanClient(), "BTCUSDT", row, runtime)  # type: ignore[arg-type]

        assert plan is not None
        self.assertAlmostEqual(plan["tp_price"], 99.0)
        self.assertAlmostEqual(plan["stop_price"], 102.0)
        self.assertAlmostEqual(plan["tp_qty"], plan["qty"])

    def test_shortline_long_order_plan_uses_roe_targets(self) -> None:
        runtime = ShortlineLiveRuntime(
            trade_side="LONG",
            leverage=10,
            margin_usdt=1.0,
            take_profit_roe_pct=20.0,
            stop_loss_roe_pct=10.0,
        )
        row = pd.Series({"close": 100.0, "atr": 1.0, "swing_low": 99.0})

        plan = build_order_plan(DummyPlanClient(), "BTCUSDT", row, runtime)  # type: ignore[arg-type]

        assert plan is not None
        self.assertAlmostEqual(plan["tp_price"], 102.0)
        self.assertAlmostEqual(plan["stop_price"], 99.0)
        self.assertAlmostEqual(plan["tp_qty"], plan["qty"])

    def test_score_long_momentum_detects_buy_setup(self) -> None:
        row = pd.Series(
            {
                "close": 102.0,
                "ema_fast": 101.0,
                "ema_slow": 100.0,
                "ema_trend": 99.0,
                "session_vwap": 101.0,
                "volume_ratio": 2.0,
                "trade_ratio": 1.5,
                "delta_ratio": 0.45,
                "atr_pct": 0.01,
                "rsi": 58.0,
                "day_close_location": 0.82,
                "vwap_gap": 0.0099,
            }
        )

        score = score_long_momentum(row)

        self.assertTrue(score["long_signal"])
        self.assertGreaterEqual(score["rank_score"], 74.0)

    def test_shortline_runtime_has_no_signal_exit_switch(self) -> None:
        runtime = ShortlineLiveRuntime()

        self.assertFalse(hasattr(runtime, "exit_on_signal"))
        self.assertEqual(runtime.max_bars_in_trade, 0)

    def test_live_parser_rejects_signal_exit_switch(self) -> None:
        with patch("sys.argv", ["live", "--exit-on-signal"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_runtime_allows_five_positions(self) -> None:
        with patch("sys.argv", ["live", "--max-positions", "5"]):
            args = parse_args()

        runtime = build_runtime_from_args(args)

        self.assertEqual(runtime.max_positions, 5)

    def test_runtime_accepts_long_side(self) -> None:
        with patch(
            "sys.argv",
            ["live", "--trade-side", "LONG", "--take-profit-roe-pct", "20", "--stop-loss-roe-pct", "10"],
        ):
            args = parse_args()

        runtime = build_runtime_from_args(args)

        self.assertEqual(runtime.trade_side, "LONG")
        self.assertEqual(runtime.take_profit_roe_pct, 20.0)
        self.assertEqual(runtime.stop_loss_roe_pct, 10.0)

    def test_order_checks_only_include_managed_and_live_position_symbols(self) -> None:
        symbols = symbols_requiring_order_checks(
            managed_symbols=["BTCUSDT", "ETHUSDT"],
            exchange_positions={"ETHUSDT": {}, "SOLUSDT": {}},
        )

        self.assertEqual(symbols, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])


if __name__ == "__main__":
    unittest.main()
