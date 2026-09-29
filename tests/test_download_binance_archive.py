from __future__ import annotations

import unittest

import pandas as pd

from download_binance_archive import ARCHIVE_COLUMNS, normalize


class DownloadBinanceArchiveTest(unittest.TestCase):
    def test_normalize_preserves_orderflow_proxy_columns(self) -> None:
        raw = pd.DataFrame(
            [
                [
                    1_706_745_600_000,
                    "100",
                    "101",
                    "99",
                    "100.5",
                    "200",
                    1_706_745_659_999,
                    "20080",
                    "55",
                    "110",
                    "11044",
                    "0",
                ],
                [
                    1_706_745_660_000,
                    "100.5",
                    "101.2",
                    "100.1",
                    "101.0",
                    "230",
                    1_706_745_719_999,
                    "23230",
                    "72",
                    "150",
                    "15150",
                    "0",
                ],
            ],
            columns=ARCHIVE_COLUMNS,
        )

        normalized = normalize(raw)

        self.assertEqual(
            normalized.columns.tolist(),
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "trade_count",
                "taker_buy_base",
                "taker_buy_quote",
            ],
        )
        self.assertAlmostEqual(float(normalized.iloc[1]["trade_count"]), 72.0)
        self.assertAlmostEqual(float(normalized.iloc[1]["taker_buy_quote"]), 15150.0)


if __name__ == "__main__":
    unittest.main()
