from __future__ import annotations

import unittest

import pandas as pd

from futures_strategy.daily_short_gate import (
    DailyShortGateSettings,
    build_daily_short_gate,
    daily_short_signal_mask,
    enrich_intraday_with_daily_short_gate,
    prepare_daily_short_features,
)


def make_hourly(days: int, *, bearish_signal_day: int | None = None, base: float = 100.0) -> pd.DataFrame:
    rows = []
    index = pd.date_range("2026-01-01", periods=days * 24, freq="h", tz="UTC")
    price = base
    for day in range(days):
        day_open = price
        for hour in range(24):
            bearish = bearish_signal_day is not None and day == bearish_signal_day
            if bearish:
                open_price = day_open * (1.0 - 0.001 * hour)
                close_price = open_price * 0.994
                high = open_price * 1.002
                low = close_price * 0.996
                volume = 1000.0 if hour < 16 else 1600.0
            else:
                open_price = price
                close_price = open_price * (1.001 if hour % 3 == 0 else 0.9995)
                high = max(open_price, close_price) * 1.002
                low = min(open_price, close_price) * 0.998
                volume = 500.0
            rows.append({"open": open_price, "high": high, "low": low, "close": close_price, "volume": volume})
            price = close_price
    return pd.DataFrame(rows, index=index)


class DailyShortGateTest(unittest.TestCase):
    def test_daily_features_detect_sell_pressure_day(self) -> None:
        hourly = make_hourly(70, bearish_signal_day=60)
        daily = prepare_daily_short_features(hourly)
        settings = DailyShortGateSettings(min_symbol_trades=1, min_atr_pct=0.005)
        mask = daily_short_signal_mask(daily, settings)
        self.assertTrue(bool(mask.iloc[60]))

    def test_build_gate_requires_signal_and_rolling_whitelist(self) -> None:
        hourly_a = make_hourly(130, bearish_signal_day=128)
        hourly_b = make_hourly(130, bearish_signal_day=None, base=80.0)
        # Add two historical bearish days for A so it can pass the rolling trade-count whitelist.
        hourly_a = hourly_a.copy()
        for day in [80, 100]:
            day_slice = hourly_a.index.floor("D") == hourly_a.index[day * 24].floor("D")
            hourly_a.loc[day_slice, "close"] = hourly_a.loc[day_slice, "open"] * 0.994
            hourly_a.loc[day_slice, "low"] = hourly_a.loc[day_slice, "close"] * 0.996
            hourly_a.loc[day_slice, "volume"] = 1500.0
            next_day_slice = hourly_a.index.floor("D") == hourly_a.index[(day + 1) * 24].floor("D")
            hourly_a.loc[next_day_slice, "high"] = hourly_a.loc[next_day_slice, "open"] * 1.001
            hourly_a.loc[next_day_slice, "low"] = hourly_a.loc[next_day_slice, "open"] * 0.95
            hourly_a.loc[next_day_slice, "close"] = hourly_a.loc[next_day_slice, "open"] * 0.96

        settings = DailyShortGateSettings(min_symbol_trades=1, min_symbol_profit_factor=0.1, top_symbols=1, min_atr_pct=0.005)
        gate, _ = build_daily_short_gate(
            {"AAAUSDT": hourly_a, "BBBUSDT": hourly_b},
            settings,
            target_trade_date=pd.Timestamp("2026-05-10", tz="UTC"),
        )

        allowed = gate[gate["daily_short_allowed"]]
        self.assertEqual(allowed["symbol"].tolist(), ["AAAUSDT"])

    def test_enrich_intraday_with_gate_adds_allowed_flag(self) -> None:
        intraday = pd.DataFrame(
            {"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.0], "volume": [100]},
            index=pd.to_datetime(["2026-05-09T01:00:00Z"], utc=True),
        )
        gate = pd.DataFrame(
            {
                "symbol": ["AAAUSDT"],
                "trade_date": [pd.Timestamp("2026-05-09", tz="UTC")],
                "daily_short_allowed": [True],
                "gate_rank": [1],
                "score_profit_factor": [2.0],
            }
        )

        enriched = enrich_intraday_with_daily_short_gate(intraday, "AAAUSDT", gate)
        self.assertTrue(bool(enriched.iloc[0]["daily_short_allowed"]))
        self.assertEqual(int(enriched.iloc[0]["daily_short_gate_rank"]), 1)


if __name__ == "__main__":
    unittest.main()
