from __future__ import annotations

import unittest

import pandas as pd

from futures_strategy.pair_optimization import (
    build_rolling_windows,
    parse_pair_list,
    slice_frame_by_months,
)


class PairOptimizationHelpersTest(unittest.TestCase):
    def test_parse_pair_list_supports_colon_and_slash(self) -> None:
        pairs = parse_pair_list("BTCUSDT:ETHUSDT,SOLUSDT/ETHUSDT")
        self.assertEqual(pairs, [("BTCUSDT", "ETHUSDT"), ("SOLUSDT", "ETHUSDT")])

    def test_build_rolling_windows(self) -> None:
        windows = build_rolling_windows(
            ["2025-09", "2025-10", "2025-11", "2025-12", "2026-01"],
            train_size=2,
            valid_size=1,
        )
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0]["train_months"], ["2025-09", "2025-10"])
        self.assertEqual(windows[0]["valid_months"], ["2025-11"])
        self.assertEqual(windows[-1]["valid_months"], ["2026-01"])

    def test_slice_frame_by_months(self) -> None:
        index = pd.date_range("2025-09-01", periods=120, freq="D", tz="UTC")
        frame = pd.DataFrame({"value": range(len(index))}, index=index)
        sliced = slice_frame_by_months(frame, ["2025-10", "2025-11"])
        self.assertEqual(sliced.index.min().month, 10)
        self.assertEqual(sliced.index.max().month, 11)


if __name__ == "__main__":
    unittest.main()
