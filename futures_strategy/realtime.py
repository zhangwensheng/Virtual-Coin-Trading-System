from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import replace
from pathlib import Path
from typing import AsyncIterator

import pandas as pd
import websockets

from futures_strategy.exchanges import CandleEvent, ExchangeSpec, get_exchange_spec


class CandleStateBuffer:
    def __init__(self) -> None:
        self._latest: dict[tuple[str, str, str], CandleEvent] = {}

    def process(self, event: CandleEvent, emit_updates: bool) -> list[CandleEvent]:
        key = (event.exchange, event.symbol, event.timeframe)
        previous = self._latest.get(key)
        emitted: list[CandleEvent] = []

        if previous and event.open_time > previous.open_time and previous.closed is not True:
            emitted.append(replace(previous, closed=True))

        self._latest[key] = event

        if event.closed is True:
            emitted.append(event)
        elif emit_updates:
            emitted.append(event)

        return dedupe_events(emitted)


def dedupe_events(events: list[CandleEvent]) -> list[CandleEvent]:
    seen: set[tuple[int, bool | None]] = set()
    unique: list[CandleEvent] = []
    for event in events:
        marker = (event.open_time, event.closed)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(event)
    return unique


async def stream_klines(
    exchange_name: str,
    symbol: str,
    timeframe: str,
    *,
    closed_only: bool = True,
    reconnect: bool = True,
    reconnect_delay: float = 5.0,
) -> AsyncIterator[CandleEvent]:
    spec = get_exchange_spec(exchange_name)
    buffer = CandleStateBuffer()

    while True:
        try:
            async for event in _stream_once(spec, symbol, timeframe, buffer, closed_only):
                yield event
        except Exception:
            if not reconnect:
                raise
            await asyncio.sleep(reconnect_delay)


async def _stream_once(
    spec: ExchangeSpec,
    symbol: str,
    timeframe: str,
    buffer: CandleStateBuffer,
    closed_only: bool,
) -> AsyncIterator[CandleEvent]:
    url = spec.build_ws_url(symbol, timeframe)
    async with websockets.connect(url, ping_interval=None, max_size=None) as websocket:
        subscribe_payload = None
        if spec.ws_subscribe_builder is not None:
            subscribe_payload = spec.ws_subscribe_builder(symbol, timeframe)
            await websocket.send(serialize_ws_payload(subscribe_payload))

        heartbeat_task = None
        if spec.heartbeat_interval and spec.heartbeat_payload_builder:
            heartbeat_task = asyncio.create_task(_heartbeat_loop(websocket, spec))

        try:
            async for message in websocket:
                parser = spec.ws_message_parser
                if parser is None:
                    continue
                events = parser(message, symbol, timeframe)
                for event in events:
                    normalized = buffer.process(event, emit_updates=not closed_only)
                    for item in normalized:
                        if closed_only and item.closed is not True:
                            continue
                        yield item
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task


async def _heartbeat_loop(websocket, spec: ExchangeSpec) -> None:
    while True:
        await asyncio.sleep(spec.heartbeat_interval or 20.0)
        payload = spec.heartbeat_payload_builder() if spec.heartbeat_payload_builder else None
        if payload is not None:
            await websocket.send(serialize_ws_payload(payload))


def serialize_ws_payload(payload) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def candle_to_row(event: CandleEvent) -> dict:
    return {
        "timestamp": pd.to_datetime(event.open_time, unit="ms", utc=True),
        "open": event.open,
        "high": event.high,
        "low": event.low,
        "close": event.close,
        "volume": event.volume,
        "closed": event.closed,
        "exchange": event.exchange,
        "symbol": event.symbol,
        "timeframe": event.timeframe,
    }


def upsert_candle(frame: pd.DataFrame, event: CandleEvent, max_rows: int | None = None) -> pd.DataFrame:
    row = candle_to_row(event)
    timestamp = row.pop("timestamp")
    updated = frame.copy()
    updated.loc[timestamp, ["open", "high", "low", "close", "volume"]] = [
        row["open"],
        row["high"],
        row["low"],
        row["close"],
        row["volume"],
    ]
    updated = updated.sort_index()
    if max_rows is not None and len(updated) > max_rows:
        updated = updated.iloc[-max_rows:]
    return updated


def append_event_jsonl(path: str | Path, event: CandleEvent) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.as_dict(), ensure_ascii=False) + "\n")
    return output_path
