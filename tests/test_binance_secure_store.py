from __future__ import annotations

import unittest

from futures_strategy.binance_secure_store import DashboardSettings, default_auto_trader_config_dir


class BinanceSecureStoreTest(unittest.TestCase):
    def test_from_dict_migrates_legacy_auto_trader_dir(self) -> None:
        settings = DashboardSettings.from_dict(
            {
                "auto_trader_config_dir": r"E:\项目\虚拟币合约策略\outputs\funding_oi_short_optimizer_profit_2026-03-18_universe10\symbol_configs"
            }
        )

        self.assertEqual(settings.auto_trader_config_dir, default_auto_trader_config_dir())

    def test_from_dict_migrates_top30_auto_trader_dir(self) -> None:
        settings = DashboardSettings.from_dict(
            {
                "auto_trader_config_dir": r"E:\项目\虚拟币合约策略\outputs\funding_oi_short_strict_liquid_top30\symbol_configs"
            }
        )

        self.assertEqual(settings.auto_trader_config_dir, default_auto_trader_config_dir())


if __name__ == "__main__":
    unittest.main()
