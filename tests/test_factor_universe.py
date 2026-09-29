from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from futures_strategy.config import AppConfig, ExchangeSettings, OutputSettings, RiskSettings, StrategySettings
from futures_strategy.factor_universe import build_shared_symbol_configs, select_strict_short_universe


class FactorUniverseTest(unittest.TestCase):
    def test_select_strict_short_universe_prefers_preferred_symbols(self) -> None:
        rows = [
            {"symbol": "AAAUSDT", "quote_volume": 100.0, "listing_days": 500},
            {"symbol": "BTCUSDT", "quote_volume": 90.0, "listing_days": 500},
            {"symbol": "ETHUSDT", "quote_volume": 80.0, "listing_days": 500},
            {"symbol": "ZZZUSDT", "quote_volume": 70.0, "listing_days": 500},
        ]
        with patch("futures_strategy.factor_universe.fetch_liquid_usdt_perpetual_rows", return_value=rows):
            selected = select_strict_short_universe(
                size=3,
                min_listing_days=30,
                min_quote_volume=10.0,
                preferred_symbols=["ETHUSDT", "BTCUSDT"],
            )

        self.assertEqual([row["symbol"] for row in selected], ["ETHUSDT", "BTCUSDT", "AAAUSDT"])

    def test_build_shared_symbol_configs_writes_yaml_files(self) -> None:
        base_config = AppConfig(
            exchange=ExchangeSettings(symbol="BTC/USDT:USDT", timeframe="1h"),
            strategy=StrategySettings(strategy_kind="funding_oi_bear_short", enhanced_entry_min_score=1),
            risk=RiskSettings(initial_capital=10000, risk_per_trade=0.03, leverage=20, max_notional_fraction=0.9),
            output=OutputSettings(output_dir="outputs/base"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            created = build_shared_symbol_configs(
                base_config,
                symbols=["BTCUSDT", "ETHUSDT"],
                output_dir=output_dir,
                shared_params={"enhanced_entry_min_score": 2},
            )

            self.assertEqual(len(created), 2)
            contents = created[0].read_text(encoding="utf-8")
            self.assertIn("BTC/USDT:USDT", contents)
            self.assertIn("enhanced_entry_min_score: 2", contents)


if __name__ == "__main__":
    unittest.main()
