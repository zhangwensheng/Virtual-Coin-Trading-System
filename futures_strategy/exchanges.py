from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import ccxt


@dataclass(frozen=True)
class CandleEvent:
    exchange: str
    symbol: str
    timeframe: str
    open_time: int
    close_time: int | None
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool | None
    raw: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "open_time": self.open_time,
            "close_time": self.close_time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "closed": self.closed,
        }


WsPayloadFactory = Callable[[str, str], Any]
WsParser = Callable[[Any, str, str], list[CandleEvent]]
TextFactory = Callable[[str, str], str]


@dataclass(frozen=True)
class ExchangeSpec:
    name: str
    ccxt_id: str | None = None
    ccxt_options: dict[str, Any] = field(default_factory=dict)
    rest_limit_cap: int = 1000
    ws_url_builder: TextFactory | None = None
    ws_subscribe_builder: WsPayloadFactory | None = None
    ws_message_parser: WsParser | None = None
    ws_symbol_builder: TextFactory | None = None
    ws_timeframe_builder: Callable[[str], str] | None = None
    heartbeat_interval: float | None = None
    heartbeat_payload_builder: Callable[[], Any] | None = None

    def build_ccxt_exchange(self) -> ccxt.Exchange:
        if not self.ccxt_id:
            raise ValueError(f"{self.name} does not expose a ccxt adapter")

        options: dict[str, Any] = {
            "enableRateLimit": True,
            "timeout": 20_000,
        }
        for key, value in self.ccxt_options.items():
            if key == "options":
                options.setdefault("options", {}).update(value)
            else:
                options[key] = value
        exchange_class = getattr(ccxt, self.ccxt_id)
        return exchange_class(options)

    def build_ws_url(self, symbol: str, timeframe: str) -> str:
        if self.ws_url_builder is None:
            raise ValueError(f"{self.name} does not expose a websocket candle stream")
        return self.ws_url_builder(symbol, timeframe)

    def build_ws_symbol(self, symbol: str) -> str:
        if self.ws_symbol_builder is None:
            return symbol
        return self.ws_symbol_builder(symbol, "")

    def build_ws_timeframe(self, timeframe: str) -> str:
        if self.ws_timeframe_builder is None:
            return timeframe
        return self.ws_timeframe_builder(timeframe)


def passthrough(symbol: str, _: str = "") -> str:
    return symbol


def parse_json_message(message: Any) -> Any:
    if isinstance(message, (bytes, bytearray)):
        message = message.decode("utf-8")
    if isinstance(message, str):
        stripped = message.strip()
        if stripped in {"ping", "pong"}:
            return stripped
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    return message


def split_unified_symbol(symbol: str) -> tuple[str | None, str | None, str | None]:
    normalized = symbol.strip()
    if "/" not in normalized:
        return None, None, None

    base, quote_part = normalized.split("/", 1)
    if ":" in quote_part:
        quote, settle = quote_part.split(":", 1)
    else:
        quote = quote_part
        settle = quote_part
    return base.upper(), quote.upper(), settle.upper()


def base_asset(symbol: str) -> str:
    unified_base, _, _ = split_unified_symbol(symbol)
    if unified_base:
        return unified_base

    normalized = symbol.upper().strip()
    if normalized.endswith("-PERPETUAL"):
        return normalized.split("-", 1)[0]
    if normalized.endswith("-SWAP"):
        return normalized.split("-", 1)[0]
    if "_" in normalized:
        return normalized.split("_", 1)[0]
    if "-" in normalized:
        return normalized.split("-", 1)[0]
    for suffix in ["USDT", "USDC", "USD"]:
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            return normalized[: -len(suffix)]
    return normalized


def settle_asset(symbol: str) -> str | None:
    _, quote, settle = split_unified_symbol(symbol)
    if settle:
        return settle

    normalized = symbol.upper().strip()
    if normalized.endswith("-SWAP"):
        parts = normalized.split("-")
        if len(parts) >= 2:
            return parts[1]
    if normalized.endswith("-PERPETUAL"):
        return "USD"
    if "_" in normalized:
        return normalized.split("_", 1)[1]
    if "-" in normalized:
        parts = normalized.split("-")
        if len(parts) >= 2:
            return parts[1]
    for suffix in ["USDT", "USDC", "USD"]:
        if normalized.endswith(suffix):
            return suffix
    return None


def to_linear_symbol(symbol: str, _: str = "") -> str:
    base, quote, settle = split_unified_symbol(symbol)
    if base:
        return f"{base}{settle or quote}".upper()

    normalized = symbol.upper().strip()
    if normalized.endswith("-SWAP"):
        parts = normalized.split("-")
        if len(parts) >= 2:
            return f"{parts[0]}{parts[1]}"
    if normalized.endswith("-PERPETUAL"):
        return f"{base_asset(normalized)}USD"
    return normalized.replace("/", "").replace(":", "").replace("-", "").replace("_", "")


def to_okx_symbol(symbol: str, _: str = "") -> str:
    base, quote, settle = split_unified_symbol(symbol)
    if base:
        return f"{base}-{settle or quote}-SWAP".upper()

    normalized = symbol.upper().strip()
    if normalized.endswith("-SWAP"):
        return normalized
    if normalized.endswith("USDT"):
        return f"{normalized[:-4]}-USDT-SWAP"
    if normalized.endswith("USDC"):
        return f"{normalized[:-4]}-USDC-SWAP"
    if normalized.endswith("USD"):
        return f"{normalized[:-3]}-USD-SWAP"
    return normalized.replace("_", "-")


def to_gate_symbol(symbol: str, _: str = "") -> str:
    base, quote, settle = split_unified_symbol(symbol)
    if base:
        return f"{base}_{settle or quote}".upper()

    normalized = symbol.upper().strip()
    if "_" in normalized:
        return normalized
    if normalized.endswith("-SWAP"):
        parts = normalized.split("-")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
    if normalized.endswith("-PERPETUAL"):
        return f"{base_asset(normalized)}_USD"
    for suffix in ["USDT", "USDC", "USD"]:
        if normalized.endswith(suffix):
            return f"{normalized[: -len(suffix)]}_{suffix}"
    return normalized.replace("-", "_")


def to_deribit_symbol(symbol: str, _: str = "") -> str:
    normalized = symbol.upper().strip()
    if "-" in normalized:
        return normalized
    return f"{base_asset(normalized)}-PERPETUAL"


def to_hyperliquid_symbol(symbol: str, _: str = "") -> str:
    return base_asset(symbol)


def timeframe_to_bybit(timeframe: str) -> str:
    mapping = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D",
        "1w": "W",
        "1M": "M",
    }
    return require_mapping("bybit timeframe", timeframe, mapping)


def timeframe_to_okx_channel(timeframe: str) -> str:
    mapping = {
        "1m": "candle1m",
        "3m": "candle3m",
        "5m": "candle5m",
        "15m": "candle15m",
        "30m": "candle30m",
        "1h": "candle1H",
        "2h": "candle2H",
        "4h": "candle4H",
        "6h": "candle6H",
        "12h": "candle12H",
        "1d": "candle1D",
        "1w": "candle1W",
        "1M": "candle1M",
    }
    return require_mapping("okx timeframe", timeframe, mapping)


def timeframe_to_bitget_channel(timeframe: str) -> str:
    mapping = {
        "1m": "candle1m",
        "3m": "candle3m",
        "5m": "candle5m",
        "15m": "candle15m",
        "30m": "candle30m",
        "1h": "candle1H",
        "4h": "candle4H",
        "6h": "candle6H",
        "12h": "candle12H",
        "1d": "candle1D",
        "3d": "candle3D",
        "1w": "candle1W",
        "1M": "candle1M",
    }
    return require_mapping("bitget timeframe", timeframe, mapping)


def timeframe_to_deribit_resolution(timeframe: str) -> str:
    mapping = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "10m": "10",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "3h": "180",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "1D",
    }
    return require_mapping("deribit timeframe", timeframe, mapping)


def require_mapping(label: str, key: str, mapping: dict[str, str]) -> str:
    if key not in mapping:
        supported = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported {label}: {key}. Supported: {supported}")
    return mapping[key]


def build_binance_url(symbol: str, timeframe: str) -> str:
    native_symbol = to_linear_symbol(symbol).lower()
    return f"wss://fstream.binance.com/ws/{native_symbol}@kline_{timeframe}"


def build_okx_url(_: str, __: str) -> str:
    return "wss://ws.okx.com:8443/ws/v5/business"


def build_bybit_url(symbol: str, _: str) -> str:
    settle = settle_asset(symbol)
    if settle == "USD":
        return "wss://stream.bybit.com/v5/public/inverse"
    return "wss://stream.bybit.com/v5/public/linear"


def build_bitget_url(_: str, __: str) -> str:
    return "wss://ws.bitget.com/v2/ws/public"


def build_gate_url(symbol: str, _: str) -> str:
    settle = (settle_asset(symbol) or "USDT").lower()
    return f"wss://fx-ws.gateio.ws/v4/ws/{settle}"


def build_deribit_url(_: str, __: str) -> str:
    return "wss://www.deribit.com/ws/api/v2"


def build_hyperliquid_url(_: str, __: str) -> str:
    return "wss://api.hyperliquid.xyz/ws"


def build_okx_subscribe(symbol: str, timeframe: str) -> dict[str, Any]:
    return {
        "op": "subscribe",
        "args": [
            {
                "channel": timeframe_to_okx_channel(timeframe),
                "instId": to_okx_symbol(symbol),
            }
        ],
    }


def build_bybit_subscribe(symbol: str, timeframe: str) -> dict[str, Any]:
    native_symbol = to_linear_symbol(symbol)
    native_interval = timeframe_to_bybit(timeframe)
    return {
        "op": "subscribe",
        "args": [f"kline.{native_interval}.{native_symbol}"],
    }


def build_bitget_subscribe(symbol: str, timeframe: str) -> dict[str, Any]:
    settle = settle_asset(symbol) or "USDT"
    inst_type = {
        "USDT": "USDT-FUTURES",
        "USDC": "USDC-FUTURES",
        "USD": "COIN-FUTURES",
    }.get(settle.upper(), "USDT-FUTURES")
    return {
        "op": "subscribe",
        "args": [
            {
                "instType": inst_type,
                "channel": timeframe_to_bitget_channel(timeframe),
                "instId": to_linear_symbol(symbol),
            }
        ],
    }


def build_gate_subscribe(symbol: str, timeframe: str) -> dict[str, Any]:
    return {
        "time": int(time.time()),
        "channel": "futures.candlesticks",
        "event": "subscribe",
        "payload": [timeframe, to_gate_symbol(symbol)],
    }


def build_deribit_subscribe(symbol: str, timeframe: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "public/subscribe",
        "params": {
            "channels": [f"chart.trades.{to_deribit_symbol(symbol)}.{timeframe_to_deribit_resolution(timeframe)}"]
        },
    }


def build_hyperliquid_subscribe(symbol: str, timeframe: str) -> dict[str, Any]:
    return {
        "method": "subscribe",
        "subscription": {
            "type": "candle",
            "coin": to_hyperliquid_symbol(symbol),
            "interval": timeframe,
        },
    }


def bybit_ping_payload() -> dict[str, Any]:
    return {"op": "ping"}


def parse_binance_kline(message: Any, symbol: str, timeframe: str) -> list[CandleEvent]:
    payload = parse_json_message(message)
    if not isinstance(payload, dict) or payload.get("e") != "kline":
        return []
    kline = payload["k"]
    return [
        CandleEvent(
            exchange="binance",
            symbol=symbol,
            timeframe=timeframe,
            open_time=int(kline["t"]),
            close_time=int(kline["T"]),
            open=float(kline["o"]),
            high=float(kline["h"]),
            low=float(kline["l"]),
            close=float(kline["c"]),
            volume=float(kline["v"]),
            closed=bool(kline["x"]),
            raw=payload,
        )
    ]


def parse_okx_kline(message: Any, symbol: str, timeframe: str) -> list[CandleEvent]:
    payload = parse_json_message(message)
    if not isinstance(payload, dict):
        return []
    if payload.get("event") or payload.get("op") == "pong":
        return []
    if payload.get("arg", {}).get("channel", "").startswith("candle") is False:
        return []

    events: list[CandleEvent] = []
    for item in payload.get("data", []):
        if len(item) < 6:
            continue
        close_time = candle_close_time(int(item[0]), timeframe)
        closed = None
        if len(item) >= 9:
            closed = item[8] == "1"
        events.append(
            CandleEvent(
                exchange="okx",
                symbol=symbol,
                timeframe=timeframe,
                open_time=int(item[0]),
                close_time=close_time,
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]),
                closed=closed,
                raw=payload,
            )
        )
    return events


def parse_bybit_kline(message: Any, symbol: str, timeframe: str) -> list[CandleEvent]:
    payload = parse_json_message(message)
    if not isinstance(payload, dict):
        return []
    if payload.get("op") == "ping" or payload.get("ret_msg") == "pong":
        return []
    if not str(payload.get("topic", "")).startswith("kline."):
        return []

    events: list[CandleEvent] = []
    for item in payload.get("data", []):
        events.append(
            CandleEvent(
                exchange="bybit",
                symbol=symbol,
                timeframe=timeframe,
                open_time=int(item["start"]),
                close_time=int(item["end"]),
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=float(item["volume"]),
                closed=bool(item.get("confirm")),
                raw=payload,
            )
        )
    return events


def parse_bitget_kline(message: Any, symbol: str, timeframe: str) -> list[CandleEvent]:
    payload = parse_json_message(message)
    if not isinstance(payload, dict):
        return []
    if payload.get("event") or payload.get("action") not in {None, "snapshot", "update"}:
        return []
    if not str(payload.get("arg", {}).get("channel", "")).startswith("candle"):
        return []

    events: list[CandleEvent] = []
    for item in payload.get("data", []):
        if len(item) < 6:
            continue
        events.append(
            CandleEvent(
                exchange="bitget",
                symbol=symbol,
                timeframe=timeframe,
                open_time=int(item[0]),
                close_time=candle_close_time(int(item[0]), timeframe),
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]),
                closed=None,
                raw=payload,
            )
        )
    return events


def parse_gate_kline(message: Any, symbol: str, timeframe: str) -> list[CandleEvent]:
    payload = parse_json_message(message)
    if not isinstance(payload, dict):
        return []
    if payload.get("channel") != "futures.candlesticks" or payload.get("event") != "update":
        return []

    events: list[CandleEvent] = []
    for item in payload.get("result", []):
        open_time = int(item["t"]) * 1000
        events.append(
            CandleEvent(
                exchange="gate",
                symbol=symbol,
                timeframe=timeframe,
                open_time=open_time,
                close_time=candle_close_time(open_time, timeframe),
                open=float(item["o"]),
                high=float(item["h"]),
                low=float(item["l"]),
                close=float(item["c"]),
                volume=float(item["v"]),
                closed=bool(item["w"]) if "w" in item else None,
                raw=payload,
            )
        )
    return events


def parse_deribit_kline(message: Any, symbol: str, timeframe: str) -> list[CandleEvent]:
    payload = parse_json_message(message)
    if not isinstance(payload, dict):
        return []
    if payload.get("method") != "subscription":
        return []
    params = payload.get("params", {})
    if not str(params.get("channel", "")).startswith("chart.trades."):
        return []
    data = params.get("data", {})
    if not isinstance(data, dict) or "tick" not in data:
        return []

    open_time = int(data["tick"])
    return [
        CandleEvent(
            exchange="deribit",
            symbol=symbol,
            timeframe=timeframe,
            open_time=open_time,
            close_time=candle_close_time(open_time, timeframe),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data.get("volume", 0.0)),
            closed=None,
            raw=payload,
        )
    ]


def parse_hyperliquid_kline(message: Any, symbol: str, timeframe: str) -> list[CandleEvent]:
    payload = parse_json_message(message)
    if not isinstance(payload, dict):
        return []
    if payload.get("channel") in {"subscriptionResponse", "pong"}:
        return []
    if payload.get("channel") != "candle":
        return []

    data = payload.get("data", [])
    if isinstance(data, dict):
        data = [data]

    events: list[CandleEvent] = []
    for item in data:
        if "t" not in item:
            continue
        events.append(
            CandleEvent(
                exchange="hyperliquid",
                symbol=symbol,
                timeframe=timeframe,
                open_time=int(item["t"]),
                close_time=int(item.get("T")) if item.get("T") is not None else candle_close_time(int(item["t"]), timeframe),
                open=float(item["o"]),
                high=float(item["h"]),
                low=float(item["l"]),
                close=float(item["c"]),
                volume=float(item["v"]),
                closed=None,
                raw=payload,
            )
        )
    return events


def timeframe_ms(timeframe: str) -> int:
    suffix = timeframe[-1]
    value = int(timeframe[:-1])
    unit_ms = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
        "M": 2_592_000_000,
    }
    if suffix not in unit_ms:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return value * unit_ms[suffix]


def candle_close_time(open_time: int, timeframe: str) -> int:
    return open_time + timeframe_ms(timeframe) - 1


EXCHANGE_SPECS: dict[str, ExchangeSpec] = {
    "binance": ExchangeSpec(
        name="binance",
        ccxt_id="binance",
        ccxt_options={"options": {"defaultType": "future"}},
        rest_limit_cap=1500,
        ws_url_builder=build_binance_url,
        ws_message_parser=parse_binance_kline,
    ),
    "okx": ExchangeSpec(
        name="okx",
        ccxt_id="okx",
        ccxt_options={"options": {"defaultType": "swap"}},
        rest_limit_cap=100,
        ws_url_builder=build_okx_url,
        ws_subscribe_builder=build_okx_subscribe,
        ws_message_parser=parse_okx_kline,
        heartbeat_interval=20.0,
        heartbeat_payload_builder=lambda: "ping",
    ),
    "bybit": ExchangeSpec(
        name="bybit",
        ccxt_id="bybit",
        ccxt_options={"options": {"defaultType": "linear"}},
        rest_limit_cap=1000,
        ws_url_builder=build_bybit_url,
        ws_subscribe_builder=build_bybit_subscribe,
        ws_message_parser=parse_bybit_kline,
        heartbeat_interval=20.0,
        heartbeat_payload_builder=bybit_ping_payload,
    ),
    "bitget": ExchangeSpec(
        name="bitget",
        ccxt_id="bitget",
        ccxt_options={"options": {"defaultType": "swap"}},
        rest_limit_cap=1000,
        ws_url_builder=build_bitget_url,
        ws_subscribe_builder=build_bitget_subscribe,
        ws_message_parser=parse_bitget_kline,
        heartbeat_interval=30.0,
        heartbeat_payload_builder=lambda: "ping",
    ),
    "gate": ExchangeSpec(
        name="gate",
        ccxt_id="gate",
        ccxt_options={"options": {"defaultType": "swap"}},
        rest_limit_cap=1000,
        ws_url_builder=build_gate_url,
        ws_subscribe_builder=build_gate_subscribe,
        ws_message_parser=parse_gate_kline,
        heartbeat_interval=25.0,
        heartbeat_payload_builder=lambda: {"time": int(time.time()), "channel": "futures.ping"},
    ),
    "kucoinfutures": ExchangeSpec(
        name="kucoinfutures",
        ccxt_id="kucoinfutures",
        rest_limit_cap=500,
    ),
    "deribit": ExchangeSpec(
        name="deribit",
        ccxt_id="deribit",
        rest_limit_cap=500,
        ws_url_builder=build_deribit_url,
        ws_subscribe_builder=build_deribit_subscribe,
        ws_message_parser=parse_deribit_kline,
    ),
    "hyperliquid": ExchangeSpec(
        name="hyperliquid",
        ccxt_id="hyperliquid",
        rest_limit_cap=5000,
        ws_url_builder=build_hyperliquid_url,
        ws_subscribe_builder=build_hyperliquid_subscribe,
        ws_message_parser=parse_hyperliquid_kline,
        heartbeat_interval=30.0,
        heartbeat_payload_builder=lambda: {"method": "ping"},
    ),
}


def get_exchange_spec(name: str) -> ExchangeSpec:
    key = name.lower()
    if key not in EXCHANGE_SPECS:
        supported = ", ".join(sorted(EXCHANGE_SPECS))
        raise ValueError(f"Unsupported exchange: {name}. Supported: {supported}")
    return EXCHANGE_SPECS[key]
