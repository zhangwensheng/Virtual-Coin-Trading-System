from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from futures_strategy.universal_short_screen import (
    archive_output_path,
    compute_survivor_score,
    is_survivor_candidate,
)


class UniversalShortScreenTest(unittest.TestCase):
    def test_archive_output_path_contains_symbol_and_range(self) -> None:
        path = archive_output_path("data/cache", "BTCUSDT", "1h", "2024-03", "2026-02")
        self.assertEqual(path, Path("data/cache/btcusdt_1h_2024-03_2026-02.csv"))

    def test_compute_survivor_score_rewards_better_profile(self) -> None:
        strong = compute_survivor_score(
            full_return_pct=24.0,
            profitable_month_ratio=0.62,
            active_month_ratio=0.7,
            mean_month_return_pct=1.2,
            avg_month_drawdown_pct=4.0,
            avg_month_profit_factor=1.4,
            worst_month_return_pct=-5.0,
            full_profit_factor=1.6,
        )
        weak = compute_survivor_score(
            full_return_pct=-3.0,
            profitable_month_ratio=0.35,
            active_month_ratio=0.3,
            mean_month_return_pct=-0.2,
            avg_month_drawdown_pct=9.0,
            avg_month_profit_factor=0.9,
            worst_month_return_pct=-14.0,
            full_profit_factor=0.8,
        )
        self.assertGreater(strong, weak)

    def test_is_survivor_candidate_requires_cross_month_stability(self) -> None:
        good = pd.Series(
            {
                "full_return_pct": 18.0,
                "profitable_month_ratio": 0.58,
                "active_month_ratio": 0.45,
                "mean_month_return_pct": 0.8,
                "avg_month_profit_factor": 1.18,
                "avg_month_drawdown_pct": 7.5,
                "worst_month_return_pct": -8.0,
                "full_profit_factor": 1.22,
                "full_total_trades": 18,
                "months_tested": 24,
            }
        )
        bad = pd.Series(
            {
                "full_return_pct": 12.0,
                "profitable_month_ratio": 0.42,
                "active_month_ratio": 0.2,
                "mean_month_return_pct": -0.1,
                "avg_month_profit_factor": 0.98,
                "avg_month_drawdown_pct": 14.0,
                "worst_month_return_pct": -16.0,
                "full_profit_factor": 1.01,
                "full_total_trades": 5,
                "months_tested": 24,
            }
        )
        self.assertTrue(is_survivor_candidate(good))
        self.assertFalse(is_survivor_candidate(bad))


if __name__ == "__main__":
    unittest.main()
