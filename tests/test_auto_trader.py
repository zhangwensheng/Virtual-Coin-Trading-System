from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from futures_strategy.auto_trader import (
    AutoTraderRuntimeConfig,
    ManagedTradeState,
    ShortOrderPlan,
    SymbolContext,
    apply_manual_flat_cooldown,
    build_short_order_plan,
    load_symbol_contexts,
    load_auto_trader_state,
    save_auto_trader_state,
)
from futures_strategy.config import AppConfig, ExchangeSettings, OutputSettings, RiskSettings, StrategySettings


class DummyClient:
    def quantize_price(self, symbol: str, price: float, *, rounding: str = "down") -> float:
        return round(price, 2)

    def quantize_quantity(self, symbol: str, quantity: float, *, market: bool = True) -> float:
        return round(quantity, 3)

    def min_notional(self, symbol: str) -> float:
        return 5.0


class AutoTraderHelpersTest(unittest.TestCase):
    def test_state_roundtrip_preserves_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            state = {
                "managed_trades": {
                    "ADAUSDT": {
                        "native_symbol": "ADAUSDT",
                        "display_symbol": "ADA/USDT:USDT",
                        "config_path": "configs/ada.yaml",
                        "signal_timestamp": "2026-03-23T08:00:00+00:00",
                        "entry_time": "2026-03-23T08:01:00+00:00",
                        "entry_price": 0.72,
                        "entry_qty": 1200,
                        "stop_price": 0.75,
                        "tp_price": 0.69,
                        "stop_order_id": 1,
                        "tp_order_id": 2,
                        "partial_taken": False,
                        "last_status": "OPEN",
                    }
                },
                "last_signal_timestamps": {"ADAUSDT": "2026-03-23T08:00:00+00:00"},
                "one_way_mode_confirmed": True,
            }
            save_auto_trader_state(path, state)
            loaded = load_auto_trader_state(path)

        trade = ManagedTradeState.from_dict(loaded["managed_trades"]["ADAUSDT"])
        self.assertEqual(trade.native_symbol, "ADAUSDT")
        self.assertAlmostEqual(trade.entry_price, 0.72)
        self.assertEqual(loaded["last_signal_timestamps"]["ADAUSDT"], "2026-03-23T08:00:00+00:00")
        self.assertTrue(loaded["one_way_mode_confirmed"])

    def test_build_short_order_plan_computes_qty_and_take_profit(self) -> None:
        config = AppConfig(
            exchange=ExchangeSettings(symbol="ADA/USDT:USDT", timeframe="1h"),
            strategy=StrategySettings(
                strategy_kind="funding_oi_bear_short",
                partial_rr=1.0,
                partial_close_ratio=0.5,
                atr_stop_mult=1.0,
            ),
            risk=RiskSettings(
                initial_capital=10000,
                risk_per_trade=0.03,
                leverage=20,
                max_notional_fraction=0.9,
            ),
            output=OutputSettings(output_dir="outputs/test"),
        )
        context = SymbolContext(
            config_path=Path("configs/test.yaml"),
            config=config,
            prepared=pd.DataFrame({"close": [10.0]}),
            native_symbol="ADAUSDT",
            display_symbol="ADA/USDT:USDT",
            scan_row={
                "symbol": "ADA/USDT:USDT",
                "native_symbol": "ADAUSDT",
                "timestamp": "2026-03-23T08:00:00+00:00",
                "action": "FUNDING_OI_BEAR_SHORT_SETUP",
                "rank_score": 320.0,
                "factor_score": 3,
                "risk_multiplier": 2.0,
                "close": 10.0,
                "atr": 1.0,
                "stop_hint": 11.0,
            },
        )
        runtime = AutoTraderRuntimeConfig(
            config_dir="configs",
            cache_dir="data/binance_factor_bundle",
            state_path="outputs/state.json",
            risk_per_trade=0.03,
            leverage=20,
            max_positions=1,
        )

        plan = build_short_order_plan(DummyClient(), context, 1000.0, runtime)

        self.assertIsInstance(plan, ShortOrderPlan)
        assert plan is not None
        self.assertEqual(plan.native_symbol, "ADAUSDT")
        self.assertAlmostEqual(plan.qty, 60.0)
        self.assertAlmostEqual(plan.stop_price, 11.0)
        self.assertAlmostEqual(plan.tp_price or 0.0, 9.0)
        self.assertAlmostEqual(plan.tp_qty, 30.0)

    def test_apply_manual_flat_cooldown_sets_cooldown_and_clears_managed_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            save_auto_trader_state(
                path,
                {
                    "managed_trades": {
                        "ADAUSDT": {
                            "native_symbol": "ADAUSDT",
                            "display_symbol": "ADA/USDT:USDT",
                            "config_path": "configs/ada.yaml",
                            "signal_timestamp": "2026-03-23T08:00:00+00:00",
                            "entry_time": "2026-03-23T08:01:00+00:00",
                            "entry_price": 0.72,
                            "entry_qty": 1200,
                            "stop_price": 0.75,
                        }
                    }
                },
            )
            apply_manual_flat_cooldown(path, "ADAUSDT", cooldown_hours=2)
            loaded = load_auto_trader_state(path)

        self.assertNotIn("ADAUSDT", loaded["managed_trades"])
        self.assertIn("ADAUSDT", loaded["manual_flat_cooldowns"])

    def test_load_symbol_contexts_skips_bad_config(self) -> None:
        runtime = AutoTraderRuntimeConfig(
            config_dir="configs",
            cache_dir="data/binance_factor_bundle",
            state_path="outputs/state.json",
        )
        ok_context = SymbolContext(
            config_path=Path("configs/ok.yaml"),
            config=AppConfig(
                exchange=ExchangeSettings(symbol="ADA/USDT:USDT", timeframe="1h"),
                strategy=StrategySettings(strategy_kind="funding_oi_bear_short"),
                risk=RiskSettings(),
                output=OutputSettings(output_dir="outputs/test"),
            ),
            prepared=pd.DataFrame({"close": [1.0]}),
            native_symbol="ADAUSDT",
            display_symbol="ADA/USDT:USDT",
            scan_row={
                "symbol": "ADA/USDT:USDT",
                "native_symbol": "ADAUSDT",
                "action": "NO_TRADE_ZONE",
                "action_priority": 1,
                "rank_score": 1.0,
                "factor_score": 1,
                "risk_multiplier": 1.0,
                "oi_value_change_pct": 0.0,
                "funding_rate": 0.0,
            },
        )

        with patch.object(
            AutoTraderRuntimeConfig,
            "resolved_config_paths",
            return_value=[Path("configs/bad.yaml"), Path("configs/ok.yaml")],
        ), patch(
            "futures_strategy.auto_trader.build_symbol_context",
            side_effect=[RuntimeError("boom"), ok_context],
        ):
            ranked, contexts = load_symbol_contexts(runtime)

        self.assertEqual(list(contexts), ["ADAUSDT"])
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked.iloc[0]["native_symbol"], "ADAUSDT")


if __name__ == "__main__":
    unittest.main()
