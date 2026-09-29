from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from futures_strategy.binance_funding_arbitrage_data import (
    BinanceFundingArbitrageDataClient,
    build_historical_liquidity_universe,
    merge_cached_frame,
    validate_symbol_history,
)


@dataclass
class FakeResponse:
    payload: object

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class FakeSession:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, *, params: dict[str, object], timeout: int) -> FakeResponse:
        self.calls.append((url, params))
        return FakeResponse(self.responses[url])


def exchange_info(market: str) -> dict[str, object]:
    spot = market == "spot"
    symbols = []
    for symbol, base in (("AAAUSDT", "AAA"), ("USDCUSDT", "USDC"), ("ONLYSPOTUSDT", "ONLYSPOT")):
        if not spot and symbol == "ONLYSPOTUSDT":
            continue
        symbols.append(
            {
                "symbol": symbol,
                "baseAsset": base,
                "quoteAsset": "USDT",
                "status": "TRADING",
                "contractType": "PERPETUAL" if not spot else None,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                ],
            }
        )
    return {"symbols": symbols}


def test_common_symbols_require_tradable_spot_and_perpetual_and_exclude_stablecoins() -> None:
    responses = {
        "https://api.binance.com/api/v3/exchangeInfo": exchange_info("spot"),
        "https://fapi.binance.com/fapi/v1/exchangeInfo": exchange_info("perp"),
    }
    session = FakeSession(responses)
    client = BinanceFundingArbitrageDataClient(session=session)

    common = client.fetch_common_tradable_symbols()

    assert [item.symbol for item in common] == ["AAAUSDT"]
    assert common[0].spot_step_size == 0.001
    assert common[0].perp_min_notional == 5.0


def test_spot_and_perp_klines_are_normalized_with_quote_volume_and_trades() -> None:
    row = [1_767_225_600_000, "100", "102", "99", "101", "10", 1_767_225_899_999, "1005", 42, "5", "500", "0"]
    responses = {
        "https://api.binance.com/api/v3/klines": [row],
        "https://fapi.binance.com/fapi/v1/klines": [row],
    }
    client = BinanceFundingArbitrageDataClient(session=FakeSession(responses))

    spot = client.fetch_spot_klines("AAAUSDT", "5m", 1_767_225_600_000, 1_767_225_900_000)
    perp = client.fetch_perp_klines("AAAUSDT", "5m", 1_767_225_600_000, 1_767_225_900_000)

    assert list(spot.columns) == ["open", "high", "low", "close", "volume", "quote_volume", "trades"]
    assert spot.iloc[0]["quote_volume"] == 1005.0
    assert perp.iloc[0]["trades"] == 42
    assert spot.index.tz is not None


def test_funding_history_is_normalized_and_unique() -> None:
    payload = [
        {"symbol": "AAAUSDT", "fundingTime": 1_767_225_600_000, "fundingRate": "0.001", "markPrice": "100.1"},
        {"symbol": "AAAUSDT", "fundingTime": 1_767_225_600_000, "fundingRate": "0.001", "markPrice": "100.1"},
    ]
    session = FakeSession({"https://fapi.binance.com/fapi/v1/fundingRate": payload})
    frame = BinanceFundingArbitrageDataClient(session=session).fetch_funding_history(
        "AAAUSDT", 1_767_225_600_000, 1_767_254_400_000
    )
    assert len(frame) == 1
    assert frame.iloc[0]["funding_rate"] == 0.001
    assert frame.iloc[0]["mark_price"] == 100.1


def test_24h_tickers_are_keyed_by_symbol_with_price_and_quote_volume() -> None:
    payload = [
        {"symbol": "AAAUSDT", "lastPrice": "101.5", "quoteVolume": "123456.7"},
        {"symbol": "BBBUSDT", "lastPrice": "20.0", "quoteVolume": "7654.3"},
    ]
    session = FakeSession({"https://api.binance.com/api/v3/ticker/24hr": payload})
    tickers = BinanceFundingArbitrageDataClient(session=session).fetch_24h_tickers("spot")
    assert tickers["AAAUSDT"]["last_price"] == 101.5
    assert tickers["AAAUSDT"]["quote_volume"] == 123456.7


def test_historical_universe_does_not_use_future_volume() -> None:
    index = pd.date_range("2026-01-01", periods=30, freq="1h", tz="UTC")
    base = pd.DataFrame({"quote_volume": [100.0] * 30}, index=index)
    future_spike = pd.DataFrame({"quote_volume": [1.0] * 25 + [10_000.0] * 5}, index=index)
    spot = {"AAAUSDT": base, "BBBUSDT": future_spike}
    perp = {"AAAUSDT": base, "BBBUSDT": future_spike}

    universe = build_historical_liquidity_universe(spot, perp, universe_size=1)

    assert universe[index[24]] == ["AAAUSDT"]
    assert universe[index[25]] == ["AAAUSDT"]
    assert universe[index[29]] == ["BBBUSDT"]


def test_history_validation_marks_large_gap_untradable() -> None:
    index = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:20:00Z")]
    )
    frame = pd.DataFrame(
        {"open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0], "volume": [1.0, 1.0], "quote_volume": [1.0, 1.0], "trades": [1, 1]},
        index=index,
    )
    result = validate_symbol_history(frame, expected_interval="5min")
    assert result.valid is False
    assert result.max_gap_minutes == 20.0
    assert result.reason == "UNRECOVERABLE_KLINE_GAP"


def test_cache_merge_replaces_duplicate_timestamp_and_keeps_sorted_rows(tmp_path) -> None:
    path = tmp_path / "AAAUSDT_spot_1h.csv"
    first = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC", name="open_time"),
    )
    merged = merge_cached_frame(path, first)
    update = pd.DataFrame(
        {"close": [111.0, 102.0]},
        index=pd.date_range("2026-01-01T01:00:00Z", periods=2, freq="1h", name="open_time"),
    )
    merged = merge_cached_frame(path, update)
    assert len(merged) == 3
    assert merged.iloc[1]["close"] == 111.0
    assert merged.index.is_monotonic_increasing
