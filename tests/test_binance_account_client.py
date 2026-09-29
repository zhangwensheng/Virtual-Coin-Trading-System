from __future__ import annotations

import unittest

from futures_strategy.binance_account_client import (
    build_dashboard_payload,
    close_side_for_position_amt,
    signed_query_string,
    summarize_income,
)


class BinanceAccountClientHelpersTest(unittest.TestCase):
    def test_close_side_for_position_amt(self) -> None:
        self.assertEqual(close_side_for_position_amt(1.5), "SELL")
        self.assertEqual(close_side_for_position_amt(-2.0), "BUY")

    def test_signed_query_string_appends_signature(self) -> None:
        query = signed_query_string(
            {
                "symbol": "BTCUSDT",
                "timestamp": 1700000000000,
                "recvWindow": 5000,
            },
            "test-secret",
        )

        self.assertIn("symbol=BTCUSDT", query)
        self.assertIn("timestamp=1700000000000", query)
        self.assertIn("signature=", query)

    def test_summarize_income_groups_recent_values(self) -> None:
        rows = [
            {"incomeType": "REALIZED_PNL", "income": "12.5", "time": 4102444800000},
            {"incomeType": "FUNDING_FEE", "income": "-1.2", "time": 4102444800000},
            {"incomeType": "COMMISSION", "income": "-0.8", "time": 4102444800000},
        ]

        summary = summarize_income(rows)

        self.assertAlmostEqual(summary["realized_pnl_7d"], 12.5)
        self.assertAlmostEqual(summary["funding_fee_7d"], -1.2)
        self.assertAlmostEqual(summary["commission_7d"], -0.8)

    def test_build_dashboard_payload_filters_zero_assets_and_positions(self) -> None:
        payload = build_dashboard_payload(
            account={
                "totalWalletBalance": "1000",
                "availableBalance": "800",
                "totalMarginBalance": "1015",
                "totalUnrealizedProfit": "15",
                "totalInitialMargin": "120",
                "totalMaintMargin": "30",
                "maxWithdrawAmount": "780",
                "assets": [
                    {
                        "asset": "USDT",
                        "walletBalance": "1000",
                        "availableBalance": "800",
                        "unrealizedProfit": "15",
                        "marginBalance": "1015",
                    },
                    {
                        "asset": "BNB",
                        "walletBalance": "0",
                        "availableBalance": "0",
                        "unrealizedProfit": "0",
                        "marginBalance": "0",
                    },
                ],
            },
            positions=[
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "-0.01",
                    "notional": "-820",
                    "initialMargin": "82",
                    "openOrderInitialMargin": "0",
                    "unRealizedProfit": "12",
                    "entryPrice": "84000",
                    "markPrice": "82800",
                    "liquidationPrice": "95000",
                    "leverage": "20",
                    "marginType": "cross",
                    "positionSide": "BOTH",
                },
                {
                    "symbol": "ETHUSDT",
                    "positionAmt": "0",
                    "notional": "0",
                    "initialMargin": "0",
                    "openOrderInitialMargin": "0",
                    "unRealizedProfit": "0",
                    "entryPrice": "0",
                    "markPrice": "0",
                    "liquidationPrice": "0",
                    "leverage": "20",
                    "marginType": "cross",
                    "positionSide": "BOTH",
                },
            ],
            open_orders=[
                {
                    "symbol": "BTCUSDT",
                    "side": "SELL",
                    "type": "LIMIT",
                    "status": "NEW",
                    "origQty": "0.01",
                    "price": "83000",
                    "stopPrice": "0",
                    "avgPrice": "0",
                    "updateTime": 1700000000000,
                }
            ],
            trades=[
                {
                    "symbol": "BTCUSDT",
                    "side": "SELL",
                    "positionSide": "BOTH",
                    "qty": "0.01",
                    "price": "83500",
                    "realizedPnl": "5.5",
                    "commission": "0.3",
                    "commissionAsset": "USDT",
                    "time": 1700000000000,
                }
            ],
            incomes=[
                {
                    "symbol": "BTCUSDT",
                    "incomeType": "REALIZED_PNL",
                    "income": "5.5",
                    "asset": "USDT",
                    "info": "close",
                    "time": 1700000000000,
                }
            ],
            tracked_symbols=["BTCUSDT", "ETHUSDT"],
        )

        self.assertEqual(payload["summary"]["position_count"], 1)
        self.assertEqual(payload["summary"]["open_order_count"], 1)
        self.assertEqual(len(payload["assets"]), 1)
        self.assertEqual(payload["positions"][0]["symbol"], "BTCUSDT")

    def test_build_dashboard_payload_reads_algo_order_trigger_price(self) -> None:
        payload = build_dashboard_payload(
            account={
                "totalWalletBalance": "100",
                "availableBalance": "90",
                "totalMarginBalance": "100",
                "totalUnrealizedProfit": "0",
                "totalInitialMargin": "0",
                "totalMaintMargin": "0",
                "assets": [],
            },
            positions=[],
            open_orders=[
                {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "type": "STOP_MARKET",
                    "status": "NEW",
                    "origQty": "0",
                    "price": "0",
                    "triggerPrice": "105.5",
                    "avgPrice": "0",
                    "updateTime": 1700000000000,
                    "algoId": 123,
                }
            ],
            trades=[],
            incomes=[],
            tracked_symbols=["BTCUSDT"],
        )

        self.assertEqual(payload["summary"]["open_order_count"], 1)
        self.assertEqual(payload["open_orders"][0]["stop_price"], 105.5)

    def test_build_dashboard_payload_infers_one_way_trade_position_side(self) -> None:
        base_account = {
            "totalWalletBalance": "100",
            "availableBalance": "90",
            "totalMarginBalance": "100",
            "totalUnrealizedProfit": "0",
            "totalInitialMargin": "0",
            "totalMaintMargin": "0",
            "assets": [],
        }

        payload = build_dashboard_payload(
            account=base_account,
            positions=[],
            open_orders=[],
            trades=[
                {
                    "symbol": "EVAAUSDT",
                    "side": "SELL",
                    "positionSide": "BOTH",
                    "qty": "26.9",
                    "price": "0.3712",
                    "realizedPnl": "0",
                    "commission": "0.005",
                    "commissionAsset": "USDT",
                    "time": 1700000000000,
                },
                {
                    "symbol": "EVAAUSDT",
                    "side": "BUY",
                    "positionSide": "BOTH",
                    "qty": "26.9",
                    "price": "0.3700",
                    "realizedPnl": "0.032",
                    "commission": "0.005",
                    "commissionAsset": "USDT",
                    "time": 1700000001000,
                },
            ],
            incomes=[],
            tracked_symbols=["EVAAUSDT"],
        )

        self.assertEqual(payload["trades"][0]["position_side"], "SHORT")
        self.assertEqual(payload["trades"][1]["position_side"], "SHORT")


if __name__ == "__main__":
    unittest.main()
