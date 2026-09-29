from __future__ import annotations

import unittest

from futures_strategy.exchanges import (
    CandleEvent,
    get_exchange_spec,
    parse_binance_kline,
    parse_bitget_kline,
    parse_bybit_kline,
    parse_deribit_kline,
    parse_gate_kline,
    parse_hyperliquid_kline,
    parse_okx_kline,
    timeframe_to_bitget_channel,
    timeframe_to_bybit,
    timeframe_to_deribit_resolution,
    timeframe_to_okx_channel,
    to_deribit_symbol,
    to_gate_symbol,
    to_hyperliquid_symbol,
    to_linear_symbol,
    to_okx_symbol,
)
from futures_strategy.realtime import CandleStateBuffer


class ExchangeAdapterTest(unittest.TestCase):
    def test_supported_exchange_specs_are_exposed(self) -> None:
        supported = {"binance", "okx", "bybit", "bitget", "gate", "kucoinfutures", "deribit", "hyperliquid"}
        resolved = {name for name in supported if get_exchange_spec(name).name == name}
        self.assertEqual(resolved, supported)

    def test_symbol_mapping_helpers(self) -> None:
        self.assertEqual(to_linear_symbol("BTC/USDT:USDT"), "BTCUSDT")
        self.assertEqual(to_okx_symbol("BTC/USDT:USDT"), "BTC-USDT-SWAP")
        self.assertEqual(to_gate_symbol("BTC/USDT:USDT"), "BTC_USDT")
        self.assertEqual(to_deribit_symbol("BTC/USDT:USDT"), "BTC-PERPETUAL")
        self.assertEqual(to_hyperliquid_symbol("BTC/USDC:USDC"), "BTC")

    def test_timeframe_mapping_helpers(self) -> None:
        self.assertEqual(timeframe_to_bybit("1h"), "60")
        self.assertEqual(timeframe_to_okx_channel("1h"), "candle1H")
        self.assertEqual(timeframe_to_bitget_channel("1h"), "candle1H")
        self.assertEqual(timeframe_to_deribit_resolution("1h"), "60")

    def test_parse_binance_message(self) -> None:
        message = {
            "e": "kline",
            "k": {
                "t": 1711111111000,
                "T": 1711114710999,
                "o": "65000",
                "h": "65100",
                "l": "64900",
                "c": "65050",
                "v": "123.4",
                "i": "1h",
                "x": True,
            },
        }
        events = parse_binance_kline(message, "BTC/USDT:USDT", "1h")
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].closed)
        self.assertAlmostEqual(events[0].close, 65050.0)

    def test_parse_okx_message(self) -> None:
        message = {
            "arg": {"channel": "candle1H", "instId": "BTC-USDT-SWAP"},
            "data": [["1711111111000", "65000", "65100", "64900", "65050", "12", "0", "0", "1"]],
        }
        events = parse_okx_kline(message, "BTC/USDT:USDT", "1h")
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].closed)

    def test_parse_bybit_message(self) -> None:
        message = {
            "topic": "kline.60.BTCUSDT",
            "data": [
                {
                    "start": 1711111111000,
                    "end": 1711114710999,
                    "open": "65000",
                    "high": "65100",
                    "low": "64900",
                    "close": "65050",
                    "volume": "20",
                    "confirm": False,
                }
            ],
        }
        events = parse_bybit_kline(message, "BTC/USDT:USDT", "1h")
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0].closed)

    def test_parse_bitget_and_finalize_previous_candle(self) -> None:
        first_message = {
            "action": "snapshot",
            "arg": {"channel": "candle1H", "instId": "BTCUSDT"},
            "data": [["1711111111000", "65000", "65100", "64900", "65050", "20"]],
        }
        second_message = {
            "action": "update",
            "arg": {"channel": "candle1H", "instId": "BTCUSDT"},
            "data": [["1711114711000", "65050", "65200", "65020", "65120", "22"]],
        }
        first_event = parse_bitget_kline(first_message, "BTC/USDT:USDT", "1h")[0]
        second_event = parse_bitget_kline(second_message, "BTC/USDT:USDT", "1h")[0]

        buffer = CandleStateBuffer()
        self.assertEqual(buffer.process(first_event, emit_updates=False), [])
        emitted = buffer.process(second_event, emit_updates=False)
        self.assertEqual(len(emitted), 1)
        self.assertTrue(emitted[0].closed)
        self.assertEqual(emitted[0].open_time, first_event.open_time)

    def test_parse_gate_message(self) -> None:
        message = {
            "channel": "futures.candlesticks",
            "event": "update",
            "result": [{"t": 1711111111, "o": "65000", "h": "65100", "l": "64900", "c": "65050", "v": "10", "w": True}],
        }
        events = parse_gate_kline(message, "BTC/USDT:USDT", "1h")
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].closed)

    def test_parse_deribit_message(self) -> None:
        message = {
            "method": "subscription",
            "params": {
                "channel": "chart.trades.BTC-PERPETUAL.60",
                "data": {"tick": 1711111111000, "open": 65000, "high": 65100, "low": 64900, "close": 65050, "volume": 88},
            },
        }
        events = parse_deribit_kline(message, "BTC-PERPETUAL", "1h")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].volume, 88.0)

    def test_parse_hyperliquid_message(self) -> None:
        message = {
            "channel": "candle",
            "data": {"t": 1711111111000, "T": 1711114710999, "o": "65000", "h": "65100", "l": "64900", "c": "65050", "v": "9.9"},
        }
        events = parse_hyperliquid_kline(message, "BTC/USDC:USDC", "1h")
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0].closed)


class CandleEventShapeTest(unittest.TestCase):
    def test_as_dict_contains_core_fields(self) -> None:
        event = CandleEvent(
            exchange="binance",
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            open_time=1,
            close_time=2,
            open=1.0,
            high=2.0,
            low=0.5,
            close=1.5,
            volume=3.0,
            closed=True,
        )
        payload = event.as_dict()
        self.assertEqual(payload["exchange"], "binance")
        self.assertEqual(payload["close"], 1.5)


if __name__ == "__main__":
    unittest.main()
