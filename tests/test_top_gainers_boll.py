from __future__ import annotations

import unittest

import pandas as pd

from futures_strategy.top_gainers_boll import (
    compute_daily_selection,
    compute_realtime_intraday_selection,
    enrich_intraday_with_daily_selection,
)


class TopGainersBollTest(unittest.TestCase):
    def test_compute_daily_selection_marks_top_rank_above_upper(self) -> None:
        index = pd.to_datetime(
            [
                "2026-01-31T00:00:00Z",
                "2026-02-01T00:00:00Z",
                "2026-02-02T00:00:00Z",
            ],
            utc=True,
        )
        frames = {
            "AAAUSDT": pd.DataFrame(
                {
                    "open": [10.0, 10.0, 10.0],
                    "high": [10.2, 11.8, 11.5],
                    "low": [9.9, 9.8, 10.2],
                    "close": [10.0, 11.5, 11.0],
                    "volume": [1000, 1500, 1200],
                },
                index=index,
            ),
            "BBBUSDT": pd.DataFrame(
                {
                    "open": [10.0, 10.0, 10.0],
                    "high": [10.1, 10.8, 10.5],
                    "low": [9.9, 9.9, 9.7],
                    "close": [10.0, 10.3, 9.9],
                    "volume": [1000, 1100, 1200],
                },
                index=index,
            ),
        }

        selection = compute_daily_selection(
            frames,
            start_month="2026-02",
            end_month="2026-02",
            daily_boll_window=2,
            daily_boll_std=0.5,
            top_n=1,
            min_daily_return_pct=0.05,
        )

        chosen = selection[selection["selected_day"]]
        self.assertEqual(chosen["symbol"].tolist(), ["AAAUSDT"])
        self.assertEqual(int(chosen.iloc[0]["daily_rank"]), 1)

    def test_enrich_intraday_with_daily_selection_adds_daily_columns(self) -> None:
        intraday_index = pd.date_range("2026-02-01 00:00:00", periods=3, freq="h", tz="UTC")
        intraday = pd.DataFrame(
            {
                "open": [1.0, 1.1, 1.2],
                "high": [1.1, 1.2, 1.3],
                "low": [0.9, 1.0, 1.1],
                "close": [1.05, 1.15, 1.25],
                "volume": [100, 120, 130],
            },
            index=intraday_index,
        )
        selection = pd.DataFrame(
            {
                "symbol": ["AAAUSDT"],
                "trade_date": [pd.Timestamp("2026-02-01", tz="UTC")],
                "daily_open": [1.0],
                "daily_high": [1.3],
                "daily_low": [0.9],
                "daily_close": [1.25],
                "daily_volume": [1000],
                "daily_return_pct": [0.08],
                "daily_bb_mid": [1.1],
                "daily_bb_upper": [1.2],
                "daily_bb_lower": [1.0],
                "daily_rank": [1],
                "selected_day": [True],
            }
        )

        enriched = enrich_intraday_with_daily_selection(
            intraday,
            symbol="AAAUSDT",
            selection_frame=selection,
            start_month="2026-02",
            end_month="2026-02",
        )

        self.assertIn("daily_return_pct", enriched.columns)
        self.assertIn("selected_day", enriched.columns)
        self.assertTrue(bool(enriched.iloc[0]["selected_day"]))
        self.assertAlmostEqual(float(enriched.iloc[0]["daily_bb_upper"]), 1.2)

    def test_compute_realtime_intraday_selection_uses_current_rank(self) -> None:
        daily_index = pd.date_range("2026-01-28", periods=5, freq="D", tz="UTC")
        daily_frames = {
            "AAAUSDT": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0, 1.0, 1.0],
                    "high": [1.02, 1.03, 1.04, 1.05, 1.06],
                    "low": [0.98, 0.98, 0.98, 0.98, 0.98],
                    "close": [1.0, 1.01, 1.02, 1.03, 1.04],
                    "volume": [1000, 1000, 1000, 1000, 1000],
                },
                index=daily_index,
            ),
            "BBBUSDT": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0, 1.0, 1.0],
                    "high": [1.01, 1.02, 1.03, 1.04, 1.05],
                    "low": [0.98, 0.98, 0.98, 0.98, 0.98],
                    "close": [1.0, 1.0, 1.01, 1.01, 1.02],
                    "volume": [1000, 1000, 1000, 1000, 1000],
                },
                index=daily_index,
            ),
        }
        intraday_index = pd.date_range("2026-02-01 00:00:00", periods=3, freq="min", tz="UTC")
        intraday_frames = {
            "AAAUSDT": pd.DataFrame(
                {
                    "open": [1.0, 1.04, 1.08],
                    "high": [1.04, 1.08, 1.12],
                    "low": [1.0, 1.03, 1.07],
                    "close": [1.04, 1.08, 1.12],
                    "volume": [100, 120, 140],
                },
                index=intraday_index,
            ),
            "BBBUSDT": pd.DataFrame(
                {
                    "open": [1.0, 1.02, 1.03],
                    "high": [1.02, 1.03, 1.04],
                    "low": [1.0, 1.01, 1.02],
                    "close": [1.02, 1.03, 1.04],
                    "volume": [100, 120, 140],
                },
                index=intraday_index,
            ),
        }

        enriched = compute_realtime_intraday_selection(
            intraday_frames,
            daily_frames,
            start_month="2026-02",
            end_month="2026-02",
            daily_boll_window=2,
            daily_boll_std=0.5,
            top_n=1,
            min_daily_return_pct=0.05,
        )

        self.assertTrue(bool(enriched["AAAUSDT"].iloc[-1]["selected_day"]))
        self.assertFalse(bool(enriched["BBBUSDT"].iloc[-1]["selected_day"]))
        self.assertEqual(int(enriched["AAAUSDT"].iloc[-1]["daily_rank"]), 1)


if __name__ == "__main__":
    unittest.main()
