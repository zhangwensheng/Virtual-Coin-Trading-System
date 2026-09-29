from __future__ import annotations

import unittest

import pandas as pd

from futures_strategy.factor_daily_scan import rank_scan_results


class FactorDailyScanTest(unittest.TestCase):
    def test_rank_scan_results_prefers_setup_then_score(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "symbol": "AAA/USDT:USDT",
                    "action": "FUNDING_OI_BEAR_SHORT_BIAS",
                    "action_priority": 2,
                    "rank_score": 250.0,
                    "factor_score": 3,
                    "risk_multiplier": 2.0,
                    "oi_value_change_pct": 1.5,
                    "funding_rate": 0.0001,
                },
                {
                    "symbol": "BBB/USDT:USDT",
                    "action": "FUNDING_OI_BEAR_SHORT_SETUP",
                    "action_priority": 3,
                    "rank_score": 180.0,
                    "factor_score": 2,
                    "risk_multiplier": 1.8,
                    "oi_value_change_pct": 0.8,
                    "funding_rate": 0.00005,
                },
                {
                    "symbol": "CCC/USDT:USDT",
                    "action": "NO_TRADE_ZONE",
                    "action_priority": 1,
                    "rank_score": 999.0,
                    "factor_score": 4,
                    "risk_multiplier": 2.5,
                    "oi_value_change_pct": 2.0,
                    "funding_rate": 0.0002,
                },
            ]
        )

        ranked = rank_scan_results(rows)

        self.assertEqual(ranked.iloc[0]["symbol"], "BBB/USDT:USDT")
        self.assertEqual(ranked.iloc[1]["symbol"], "AAA/USDT:USDT")
        self.assertEqual(ranked.iloc[2]["symbol"], "CCC/USDT:USDT")
        self.assertEqual(ranked.iloc[0]["portfolio_rank"], 1)


if __name__ == "__main__":
    unittest.main()
