from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from futures_strategy.data import load_csv


class DataCsvLoadTest(unittest.TestCase):
    def test_load_csv_handles_stringdtype_timestamp_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            frame = pd.DataFrame(
                {
                    "timestamp": pd.Series(
                        ["2026-03-24T00:00:00Z", "2026-03-24T01:00:00Z"],
                        dtype="string",
                    ),
                    "open": ["1", "2"],
                    "high": ["1.1", "2.1"],
                    "low": ["0.9", "1.9"],
                    "close": ["1.05", "2.05"],
                    "volume": ["100", "120"],
                }
            )
            frame.to_csv(path, index=False)

            loaded = load_csv(path)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(str(loaded.index.tz), "UTC")
        self.assertAlmostEqual(float(loaded.iloc[0]["close"]), 1.05)

    def test_load_csv_preserves_orderflow_proxy_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "orderflow.csv"
            frame = pd.DataFrame(
                {
                    "timestamp": ["2026-03-24T00:00:00Z", "2026-03-24T00:01:00Z"],
                    "open": ["100", "100.5"],
                    "high": ["100.8", "101.0"],
                    "low": ["99.9", "100.2"],
                    "close": ["100.4", "100.9"],
                    "volume": ["200", "220"],
                    "quote_volume": ["20080", "22198"],
                    "trade_count": ["55", "78"],
                    "taker_buy_base": ["110", "150"],
                    "taker_buy_quote": ["11044", "15135"],
                }
            )
            frame.to_csv(path, index=False)

            loaded = load_csv(path)

        self.assertIn("quote_volume", loaded.columns)
        self.assertIn("trade_count", loaded.columns)
        self.assertIn("taker_buy_base", loaded.columns)
        self.assertIn("taker_buy_quote", loaded.columns)
        self.assertAlmostEqual(float(loaded.iloc[1]["taker_buy_quote"]), 15135.0)


if __name__ == "__main__":
    unittest.main()
