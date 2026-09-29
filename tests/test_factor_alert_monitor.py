from __future__ import annotations

import unittest

import pandas as pd

from futures_strategy.factor_alert_monitor import detect_alerts
from futures_strategy.factor_daily_scan import DailyScanResult


class FactorAlertMonitorTest(unittest.TestCase):
    def test_detect_alerts_fires_when_symbol_upgrades_to_setup(self) -> None:
        previous_state = {
            "ADA/USDT:USDT": {
                "action": "FUNDING_OI_BEAR_SHORT_BIAS",
                "portfolio_rank": 2,
            }
        }
        rows = pd.DataFrame(
            [
                {
                    "symbol": "ADA/USDT:USDT",
                    "timestamp": "2026-03-19T08:00:00+00:00",
                    "action": "FUNDING_OI_BEAR_SHORT_SETUP",
                    "portfolio_rank": 1,
                    "rank_score": 355.2,
                    "factor_score": 4,
                    "risk_multiplier": 2.0,
                    "funding_rate": -0.00012,
                    "oi_value_change_pct": 1.4,
                    "stop_hint": 0.3012,
                }
            ]
        )
        scan = DailyScanResult(
            scan_time=pd.Timestamp("2026-03-19 08:00:00+00:00"),
            rows=rows,
        )

        alerts = detect_alerts(previous_state, scan, top_n=2)

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].symbol, "ADA/USDT:USDT")
        self.assertEqual(alerts[0].reason, "action_upgraded_to_setup")

    def test_detect_alerts_ignores_duplicate_setup(self) -> None:
        previous_state = {
            "ADA/USDT:USDT": {
                "action": "FUNDING_OI_BEAR_SHORT_SETUP",
                "portfolio_rank": 1,
            }
        }
        rows = pd.DataFrame(
            [
                {
                    "symbol": "ADA/USDT:USDT",
                    "timestamp": "2026-03-19T08:00:00+00:00",
                    "action": "FUNDING_OI_BEAR_SHORT_SETUP",
                    "portfolio_rank": 1,
                    "rank_score": 355.2,
                    "factor_score": 4,
                    "risk_multiplier": 2.0,
                    "funding_rate": -0.00012,
                    "oi_value_change_pct": 1.4,
                    "stop_hint": 0.3012,
                }
            ]
        )
        scan = DailyScanResult(
            scan_time=pd.Timestamp("2026-03-19 08:00:00+00:00"),
            rows=rows,
        )

        alerts = detect_alerts(previous_state, scan, top_n=2)

        self.assertEqual(alerts, [])


if __name__ == "__main__":
    unittest.main()
