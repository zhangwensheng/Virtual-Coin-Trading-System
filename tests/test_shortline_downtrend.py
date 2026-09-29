from __future__ import annotations

import unittest

import pandas as pd

from futures_strategy.shortline_downtrend import (
    ShortlineDowntrendSettings,
    prepare_shortline_downtrend_frame,
    score_shortline_downtrend,
)


def make_intraday_frame(*, bearish: bool) -> pd.DataFrame:
    rows = []
    index = pd.date_range("2026-06-11T00:00:00Z", periods=240, freq="min")
    price = 100.0
    for i, ts in enumerate(index):
        if bearish:
            if i < 150:
                change = 0.00025
            elif i < 220:
                change = -0.00015
            elif i < 238:
                change = -0.0002 if i % 3 else 0.00005
            else:
                change = -0.0009
        else:
            if i < 150:
                change = -0.00015
            else:
                change = 0.00055
        open_price = price
        close_price = open_price * (1.0 + change)
        high = max(open_price, close_price) * 1.001
        low = min(open_price, close_price) * 0.999
        volume = 120.0
        if bearish and i >= 230:
            volume = 450.0
        if bearish and i >= 238:
            volume = 650.0
        if (not bearish) and i >= 205:
            volume = 260.0
        quote_volume = volume * close_price
        buy_ratio = 0.35 if bearish and i >= 230 else 0.62
        if not bearish and i >= 205:
            buy_ratio = 0.66
        rows.append(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": volume,
                "quote_volume": quote_volume,
                "trade_count": volume / 3.0,
                "taker_buy_quote": quote_volume * buy_ratio,
            }
        )
        price = close_price
    return pd.DataFrame(rows, index=index)


class ShortlineDowntrendTest(unittest.TestCase):
    def test_bearish_flow_scores_as_short_signal(self) -> None:
        raw = make_intraday_frame(bearish=True)
        prepared = prepare_shortline_downtrend_frame(raw)
        settings = ShortlineDowntrendSettings(min_entry_score=65.0, min_rsi=4.0)
        score = score_shortline_downtrend(
            prepared,
            symbol="TESTUSDT",
            market_context={"bearish_count": 2},
            settings=settings,
        )

        self.assertGreaterEqual(score["rank_score"], 65.0)
        self.assertTrue(score["short_signal"], score["failed_checks"])

    def test_bullish_flow_does_not_short(self) -> None:
        raw = make_intraday_frame(bearish=False)
        prepared = prepare_shortline_downtrend_frame(raw)
        score = score_shortline_downtrend(
            prepared,
            symbol="TESTUSDT",
            market_context={"bearish_count": 0},
        )

        self.assertFalse(score["short_signal"])
        self.assertIn("close_below_vwap", score["failed_checks"])


if __name__ == "__main__":
    unittest.main()
