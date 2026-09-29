from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from futures_strategy.pairs import PairConfig, prepare_pair_data, run_pair_backtest


class PairBacktestSmokeTest(unittest.TestCase):
    def test_pair_strategy_generates_trade(self) -> None:
        periods = 420
        index = pd.date_range("2025-01-01", periods=periods, freq="5min", tz="UTC")
        base = 100 + np.linspace(0, 3, periods)
        wave = 0.25 * np.sin(np.arange(periods) / 11)
        close_a = base + wave
        close_b = (base * 0.8) + (wave * 0.78)

        close_a[220:245] += np.linspace(0.2, 2.8, 25)
        close_a[245:280] -= np.linspace(2.6, 0.1, 35)
        close_b[220:280] += np.linspace(0.1, 0.25, 60)

        open_a = np.roll(close_a, 1)
        open_b = np.roll(close_b, 1)
        open_a[0] = close_a[0] - 0.05
        open_b[0] = close_b[0] - 0.04

        frame_a = pd.DataFrame(
            {
                "open": open_a,
                "high": np.maximum(open_a, close_a) + 0.08,
                "low": np.minimum(open_a, close_a) - 0.08,
                "close": close_a,
                "volume": np.full(periods, 1000.0),
            },
            index=index,
        )
        frame_b = pd.DataFrame(
            {
                "open": open_b,
                "high": np.maximum(open_b, close_b) + 0.06,
                "low": np.minimum(open_b, close_b) - 0.06,
                "close": close_b,
                "volume": np.full(periods, 1200.0),
            },
            index=index,
        )

        config = PairConfig()
        config.strategy.beta_window = 96
        config.strategy.zscore_window = 96
        config.strategy.entry_z = 1.6
        config.strategy.exit_z = 0.35
        config.strategy.stop_z = 3.8
        config.strategy.min_correlation = 0.4
        config.strategy.max_holding_bars = 72
        config.strategy.cooldown_bars = 3
        config.risk.fee_rate = 0.0
        config.risk.slippage_bps = 0.0

        prepared = prepare_pair_data(frame_a, frame_b, config.strategy)
        result = run_pair_backtest(prepared, config)

        self.assertFalse(prepared.empty)
        self.assertFalse(result.trades.empty)
        self.assertIn("return_pct", result.summary)
        self.assertGreaterEqual(result.summary["total_trades"], 1)


if __name__ == "__main__":
    unittest.main()
