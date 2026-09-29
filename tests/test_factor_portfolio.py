from __future__ import annotations

import unittest

import pandas as pd

from futures_strategy.config import AppConfig
from futures_strategy.factor_portfolio import override_portfolio_risk, run_factor_portfolio_backtest


class FactorPortfolioBacktestTest(unittest.TestCase):
    def test_override_portfolio_risk_clones_configs(self) -> None:
        config = AppConfig()
        config.exchange.symbol = "AAA/USDT:USDT"
        updated = override_portfolio_risk({config.exchange.symbol: config}, risk_per_trade=0.04)

        self.assertEqual(updated[config.exchange.symbol].risk.risk_per_trade, 0.04)
        self.assertEqual(config.risk.risk_per_trade, 0.005)

    def test_portfolio_prefers_higher_ranked_symbol_when_slots_are_limited(self) -> None:
        index = pd.date_range("2026-03-01 00:00:00", periods=4, freq="h", tz="UTC")

        def build_frame(start_price: float, factor_score: int) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "open": [start_price, start_price, start_price - 3, start_price - 3],
                    "high": [start_price + 1, start_price + 1, start_price - 2, start_price - 2],
                    "low": [start_price - 1, start_price - 1, start_price - 4, start_price - 4],
                    "close": [start_price, start_price - 2, start_price - 3, start_price - 3],
                    "atr": [1.0, 1.0, 1.0, 1.0],
                    "swing_high": [start_price + 1, start_price + 1, start_price + 1, start_price + 1],
                    "swing_low": [start_price - 4, start_price - 4, start_price - 4, start_price - 4],
                    "long_signal": [False, False, False, False],
                    "short_signal": [True, False, False, False],
                    "exit_long_signal": [False, False, False, False],
                    "exit_short_signal": [False, False, False, False],
                    "short_regime": [True, True, True, True],
                    "long_regime": [False, False, False, False],
                    "ema_entry": [start_price, start_price, start_price - 2, start_price - 2],
                    "rsi": [45.0, 40.0, 35.0, 35.0],
                    "factor_score": [factor_score, factor_score, factor_score, factor_score],
                    "risk_multiplier": [2.0, 2.0, 2.0, 2.0],
                    "oi_value_change_pct": [0.02, 0.02, 0.02, 0.02],
                    "funding_rate": [0.0001, 0.0001, 0.0001, 0.0001],
                    "bounce_pct": [0.01, 0.01, 0.01, 0.01],
                    "volume_ratio": [1.5, 1.5, 1.5, 1.5],
                },
                index=index,
            )

        config_a = AppConfig()
        config_a.exchange.symbol = "AAA/USDT:USDT"
        config_a.strategy.allow_long = False
        config_a.strategy.allow_short = True
        config_a.strategy.max_bars_in_trade = 1
        config_a.risk.fee_rate = 0.0
        config_a.risk.slippage_bps = 0.0

        config_b = AppConfig()
        config_b.exchange.symbol = "BBB/USDT:USDT"
        config_b.strategy.allow_long = False
        config_b.strategy.allow_short = True
        config_b.strategy.max_bars_in_trade = 1
        config_b.risk.fee_rate = 0.0
        config_b.risk.slippage_bps = 0.0

        result = run_factor_portfolio_backtest(
            configs={
                config_a.exchange.symbol: config_a,
                config_b.exchange.symbol: config_b,
            },
            prepared_frames={
                config_a.exchange.symbol: build_frame(100.0, factor_score=4),
                config_b.exchange.symbol: build_frame(90.0, factor_score=2),
            },
            max_positions=1,
            portfolio_notional_fraction=0.6,
        )

        self.assertEqual(result.summary["total_trades"], 1)
        self.assertEqual(result.trades.iloc[0]["symbol"], "AAA/USDT:USDT")
        self.assertGreater(result.summary["return_pct"], 0.0)

    def test_portfolio_can_open_long_signals(self) -> None:
        index = pd.date_range("2026-03-01 00:00:00", periods=4, freq="h", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [100.0, 100.0, 103.0, 103.0],
                "high": [101.0, 104.0, 104.0, 104.0],
                "low": [99.0, 99.0, 102.0, 102.0],
                "close": [100.0, 103.0, 103.0, 103.0],
                "atr": [1.0, 1.0, 1.0, 1.0],
                "swing_high": [104.0, 104.0, 104.0, 104.0],
                "swing_low": [99.0, 99.0, 99.0, 99.0],
                "long_signal": [True, False, False, False],
                "short_signal": [False, False, False, False],
                "exit_long_signal": [False, False, False, False],
                "exit_short_signal": [False, False, False, False],
                "short_regime": [False, False, False, False],
                "long_regime": [True, True, True, True],
                "ema_entry": [100.0, 101.0, 102.0, 102.0],
                "rsi": [60.0, 62.0, 64.0, 64.0],
                "factor_score": [3, 3, 3, 3],
                "risk_multiplier": [1.0, 1.0, 1.0, 1.0],
                "oi_value_change_pct": [0.0, 0.0, 0.0, 0.0],
                "funding_rate": [0.0, 0.0, 0.0, 0.0],
                "bounce_pct": [0.02, 0.02, 0.02, 0.02],
                "volume_ratio": [1.5, 1.5, 1.5, 1.5],
            },
            index=index,
        )

        config = AppConfig()
        config.exchange.symbol = "AAA/USDT:USDT"
        config.strategy.allow_long = True
        config.strategy.allow_short = False
        config.strategy.max_bars_in_trade = 1
        config.risk.fee_rate = 0.0
        config.risk.slippage_bps = 0.0

        result = run_factor_portfolio_backtest(
            configs={config.exchange.symbol: config},
            prepared_frames={config.exchange.symbol: frame},
            max_positions=1,
            portfolio_notional_fraction=0.6,
        )

        self.assertEqual(result.summary["total_trades"], 1)
        self.assertEqual(result.trades.iloc[0]["side"], "long")
        self.assertGreater(result.summary["return_pct"], 0.0)

    def test_symbol_daily_loss_gate_blocks_reentry(self) -> None:
        index = pd.date_range("2026-03-01 00:00:00", periods=6, freq="h", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                "high": [101.0, 103.0, 101.0, 103.0, 101.0, 101.0],
                "low": [99.0, 99.0, 98.0, 99.0, 99.0, 99.0],
                "close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                "atr": [1.0] * 6,
                "swing_high": [101.0] * 6,
                "swing_low": [99.0] * 6,
                "long_signal": [True, False, True, False, False, False],
                "short_signal": [False] * 6,
                "exit_long_signal": [False] * 6,
                "exit_short_signal": [False] * 6,
                "short_regime": [False] * 6,
                "long_regime": [True] * 6,
                "ema_entry": [100.0] * 6,
                "rsi": [60.0] * 6,
                "factor_score": [3] * 6,
                "risk_multiplier": [1.0] * 6,
                "oi_value_change_pct": [0.0] * 6,
                "funding_rate": [0.0] * 6,
                "bounce_pct": [0.02] * 6,
                "volume_ratio": [1.5] * 6,
            },
            index=index,
        )

        config = AppConfig()
        config.exchange.symbol = "AAA/USDT:USDT"
        config.strategy.allow_long = True
        config.strategy.allow_short = False
        config.strategy.max_bars_in_trade = 3
        config.strategy.atr_stop_mult = 1.0
        config.risk.fee_rate = 0.0
        config.risk.slippage_bps = 0.0
        config.risk.max_consecutive_losses = 99

        result = run_factor_portfolio_backtest(
            configs={config.exchange.symbol: config},
            prepared_frames={config.exchange.symbol: frame},
            max_positions=1,
            portfolio_notional_fraction=0.6,
            symbol_daily_max_losses=1,
        )

        self.assertEqual(result.summary["total_trades"], 1)
        self.assertLess(result.trades.iloc[0]["pnl"], 0.0)


if __name__ == "__main__":
    unittest.main()
