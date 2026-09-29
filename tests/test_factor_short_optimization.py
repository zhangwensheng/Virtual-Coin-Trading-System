from __future__ import annotations

import unittest

import pandas as pd

from futures_strategy.config import AppConfig
from futures_strategy.factor_short_optimization import apply_tunable_params, build_rolling_windows


class FactorShortOptimizationHelpersTest(unittest.TestCase):
    def test_build_rolling_windows_uses_common_overlap(self) -> None:
        index_a = pd.date_range("2026-02-16 00:00:00", periods=24 * 15, freq="h", tz="UTC")
        index_b = pd.date_range("2026-02-17 00:00:00", periods=(24 * 16) + 1, freq="h", tz="UTC")
        datasets = {
            "BTCUSDT": pd.DataFrame({"close": range(len(index_a))}, index=index_a),
            "ETHUSDT": pd.DataFrame({"close": range(len(index_b))}, index=index_b),
        }

        windows = build_rolling_windows(datasets, window_days=7, step_days=7)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["start_time"], pd.Timestamp("2026-02-17 00:00:00+0000", tz="UTC"))
        self.assertEqual(windows[0]["end_time"], pd.Timestamp("2026-02-24 00:00:00+0000", tz="UTC"))
        self.assertEqual(windows[0]["month"], "2026-02")

    def test_apply_tunable_params_updates_strategy_and_risk(self) -> None:
        config = AppConfig()

        apply_tunable_params(
            config,
            {
                "enhanced_entry_min_score": 3,
                "enhanced_risk_step": 0.4,
                "risk_per_trade": 0.03,
            },
        )

        self.assertEqual(config.strategy.enhanced_entry_min_score, 3)
        self.assertEqual(config.strategy.enhanced_risk_step, 0.4)
        self.assertEqual(config.risk.risk_per_trade, 0.03)


if __name__ == "__main__":
    unittest.main()
