from __future__ import annotations

import unittest

import pandas as pd

from futures_strategy.optimization import (
    aggregate_combo_metrics,
    build_param_grid,
    build_selection_table,
    expand_months,
)


class OptimizationHelpersTest(unittest.TestCase):
    def test_expand_months_supports_lists_and_ranges(self) -> None:
        months = expand_months("2026-01,2026-02:2026-03,2026-03")
        self.assertEqual(months, ["2026-01", "2026-02", "2026-03"])

    def test_build_param_grid_cartesian_product(self) -> None:
        grid = build_param_grid(
            {
                "a": [1, 2],
                "b": [10],
                "c": [100, 200],
            }
        )
        self.assertEqual(len(grid), 4)
        self.assertEqual(grid[0], {"a": 1, "b": 10, "c": 100})
        self.assertEqual(grid[-1], {"a": 2, "b": 10, "c": 200})

    def test_selection_prefers_more_robust_combo(self) -> None:
        symbol_results = pd.DataFrame(
            [
                {
                    "combo_id": 1,
                    "split": "train",
                    "symbol": "AAA",
                    "return_pct": 0.8,
                    "max_drawdown_pct": -0.4,
                    "total_trades": 10,
                    "win_rate_pct": 60.0,
                    "profit_factor": 1.8,
                    "net_profit": 80.0,
                    "param_intraday_day_return_threshold": 0.03,
                },
                {
                    "combo_id": 1,
                    "split": "train",
                    "symbol": "BBB",
                    "return_pct": 0.4,
                    "max_drawdown_pct": -0.5,
                    "total_trades": 9,
                    "win_rate_pct": 55.0,
                    "profit_factor": 1.4,
                    "net_profit": 40.0,
                    "param_intraday_day_return_threshold": 0.03,
                },
                {
                    "combo_id": 1,
                    "split": "valid",
                    "symbol": "AAA",
                    "return_pct": 0.6,
                    "max_drawdown_pct": -0.4,
                    "total_trades": 8,
                    "win_rate_pct": 62.0,
                    "profit_factor": 1.7,
                    "net_profit": 60.0,
                    "param_intraday_day_return_threshold": 0.03,
                },
                {
                    "combo_id": 1,
                    "split": "valid",
                    "symbol": "BBB",
                    "return_pct": 0.2,
                    "max_drawdown_pct": -0.5,
                    "total_trades": 7,
                    "win_rate_pct": 50.0,
                    "profit_factor": 1.3,
                    "net_profit": 20.0,
                    "param_intraday_day_return_threshold": 0.03,
                },
                {
                    "combo_id": 2,
                    "split": "train",
                    "symbol": "AAA",
                    "return_pct": -0.1,
                    "max_drawdown_pct": -0.9,
                    "total_trades": 10,
                    "win_rate_pct": 40.0,
                    "profit_factor": 0.8,
                    "net_profit": -10.0,
                    "param_intraday_day_return_threshold": 0.05,
                },
                {
                    "combo_id": 2,
                    "split": "train",
                    "symbol": "BBB",
                    "return_pct": 0.0,
                    "max_drawdown_pct": -1.2,
                    "total_trades": 8,
                    "win_rate_pct": 42.0,
                    "profit_factor": 0.9,
                    "net_profit": 0.0,
                    "param_intraday_day_return_threshold": 0.05,
                },
                {
                    "combo_id": 2,
                    "split": "valid",
                    "symbol": "AAA",
                    "return_pct": -0.3,
                    "max_drawdown_pct": -1.0,
                    "total_trades": 9,
                    "win_rate_pct": 38.0,
                    "profit_factor": 0.7,
                    "net_profit": -30.0,
                    "param_intraday_day_return_threshold": 0.05,
                },
                {
                    "combo_id": 2,
                    "split": "valid",
                    "symbol": "BBB",
                    "return_pct": -0.2,
                    "max_drawdown_pct": -1.1,
                    "total_trades": 9,
                    "win_rate_pct": 39.0,
                    "profit_factor": 0.75,
                    "net_profit": -20.0,
                    "param_intraday_day_return_threshold": 0.05,
                },
            ]
        )

        aggregate = aggregate_combo_metrics(symbol_results)
        selection = build_selection_table(aggregate)

        self.assertEqual(int(selection.iloc[0]["combo_id"]), 1)
        self.assertGreater(selection.iloc[0]["selection_score"], selection.iloc[1]["selection_score"])


if __name__ == "__main__":
    unittest.main()
