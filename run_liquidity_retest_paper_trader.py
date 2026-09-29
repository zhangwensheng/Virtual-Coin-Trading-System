from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from futures_strategy.factor_universe import fetch_liquid_usdt_perpetual_rows
from futures_strategy.liquidity_retest import (
    EntryCounters,
    EventLogger,
    LiquidityRetestPortfolioSimulator,
    LiquidityRetestSettings,
    LiquidityRetestSignalEngine,
    SignalCandidate,
    SimulatedPosition,
    Zone,
    normalize_timestamp,
    process_raw_candidate,
    select_top_quote_volume_symbols,
)
from run_daily_short_gate_paper_trader import klines_to_frame
from run_daily_short_gate_scan import request_json
from run_liquidity_retest_backtest import load_settings_config, parse_symbols, prepare_5m_frame


BINANCE_FAPI = "https://fapi.binance.com"


@dataclass
class PaperRuntime:
    output_dir: str = "outputs/liquidity_retest_paper"
    poll_sec: int = 60
    lookback_hours: int = 72
    initial_equity: float = 10_000.0
    once: bool = False
    symbols: list[str] | None = None


@dataclass
class PaperState:
    processed_bar_keys: set[str] = field(default_factory=set)
    counters: EntryCounters = field(default_factory=EntryCounters)
    cooldowns: dict[str, str] = field(default_factory=dict)
    pending: list[SignalCandidate] = field(default_factory=list)
    engines: dict[str, LiquidityRetestSignalEngine] = field(default_factory=dict)
    cash: float | None = None
    peak_equity: float | None = None
    last_marked_equity: float | None = None
    transform_mode: str = ""
    positions: dict[str, SimulatedPosition] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "processed_bar_keys": sorted(self.processed_bar_keys),
            "counters": self.counters.to_dict(),
            "cooldowns": self.cooldowns,
            "pending": [_candidate_to_dict(candidate) for candidate in self.pending],
            "cash": self.cash,
            "peak_equity": self.peak_equity,
            "last_marked_equity": self.last_marked_equity,
            "transform_mode": self.transform_mode,
            "positions": {symbol: _position_to_dict(position) for symbol, position in self.positions.items()},
        }

    @classmethod
    def from_jsonable(cls, payload: dict[str, Any] | None) -> "PaperState":
        payload = payload or {}
        return cls(
            processed_bar_keys=set(payload.get("processed_bar_keys", [])),
            counters=EntryCounters.from_dict(payload.get("counters")),
            cooldowns=dict(payload.get("cooldowns", {})),
            pending=[_candidate_from_dict(item) for item in payload.get("pending", [])],
            cash=_optional_float(payload.get("cash")),
            peak_equity=_optional_float(payload.get("peak_equity")),
            last_marked_equity=_optional_float(payload.get("last_marked_equity")),
            transform_mode=str(payload.get("transform_mode", "")),
            positions={symbol: _position_from_dict(item) for symbol, item in dict(payload.get("positions", {})).items()},
        )

    @classmethod
    def from_file(cls, path: Path) -> "PaperState":
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_jsonable(payload)


def _optional_float(value: Any) -> float | None:
    return None if value is None or value == "" else float(value)


def _zone_to_dict(zone: Zone) -> dict[str, Any]:
    payload = asdict(zone)
    payload["last_touch_time"] = zone.last_touch_time.isoformat()
    return payload


def _zone_from_dict(payload: dict[str, Any]) -> Zone:
    return Zone(
        side=payload["side"],
        timeframe=str(payload["timeframe"]),
        low=float(payload["low"]),
        high=float(payload["high"]),
        touches=int(payload["touches"]),
        score=float(payload["score"]),
        last_touch_time=normalize_timestamp(payload["last_touch_time"]),
    )


def _candidate_to_dict(candidate: SignalCandidate) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["signal_time"] = candidate.signal_time.isoformat()
    payload["zone"] = _zone_to_dict(candidate.zone)
    return payload


def _candidate_from_dict(payload: dict[str, Any]) -> SignalCandidate:
    values = dict(payload)
    values["signal_time"] = normalize_timestamp(values["signal_time"])
    values["zone"] = _zone_from_dict(values["zone"])
    return SignalCandidate(**values)


def _position_to_dict(position: SimulatedPosition) -> dict[str, Any]:
    payload = asdict(position)
    payload["entry_time"] = position.entry_time.isoformat()
    return payload


def _position_from_dict(payload: dict[str, Any]) -> SimulatedPosition:
    values = dict(payload)
    values["entry_time"] = normalize_timestamp(values["entry_time"])
    return SimulatedPosition(**values)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def active_symbols_from_binance(session: requests.Session, limit: int = 20) -> list[str]:
    rows = fetch_liquid_usdt_perpetual_rows(session=session)
    return [row["symbol"] for row in rows[:limit]]


def fetch_5m_history(session: requests.Session, symbol: str, *, lookback_hours: int, server_time_ms: int) -> pd.DataFrame:
    start_ms = int(server_time_ms - (lookback_hours * 60 * 60 * 1000))
    rows = request_json(
        session,
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": "5m", "startTime": start_ms, "endTime": server_time_ms, "limit": 1000},
    )
    return klines_to_frame(rows, server_time_ms=server_time_ms)


def fetch_frames_for_cycle(session: requests.Session, runtime: PaperRuntime) -> dict[str, pd.DataFrame]:
    server_time_ms = int(request_json(session, "/fapi/v1/time")["serverTime"])
    symbols = runtime.symbols or active_symbols_from_binance(session)
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            frame = fetch_5m_history(session, symbol, lookback_hours=runtime.lookback_hours, server_time_ms=server_time_ms)
        except Exception:
            continue
        if not frame.empty:
            frames[symbol.upper()] = frame
        time.sleep(0.03)
    return frames


def run_paper_cycle_from_frames(
    frames_by_symbol: dict[str, pd.DataFrame],
    *,
    settings: LiquidityRetestSettings,
    runtime: PaperRuntime,
    state: PaperState,
    run_id: str = "liquidity-retest-paper",
) -> dict[str, Any]:
    output_dir = Path(runtime.output_dir)
    logger = EventLogger(output_dir, run_id=run_id)
    if state.transform_mode and state.transform_mode != settings.trade_transform_mode:
        raise ValueError(
            f"Paper state transform mode {state.transform_mode!r} does not match configured mode {settings.trade_transform_mode!r}"
        )
    state.transform_mode = settings.trade_transform_mode
    prepared = {symbol.upper(): prepare_5m_frame(frame) for symbol, frame in frames_by_symbol.items() if not frame.empty}
    prepared = {symbol: frame for symbol, frame in prepared.items() if not frame.empty}
    if not prepared:
        status = {"mode": "PAPER", "strategy": "liquidity_retest", "signals_processed": 0, "positions": 0, "error": "NO_USABLE_FRAMES"}
        write_json(output_dir / "latest_status.json", status)
        write_json(output_dir / "state.json", state.to_jsonable())
        logger.ensure_output_files()
        logger.write_summary(status)
        return status

    for symbol in prepared:
        state.engines.setdefault(symbol, LiquidityRetestSignalEngine(settings))

    simulator = LiquidityRetestPortfolioSimulator(
        settings,
        logger=logger,
        initial_equity=runtime.initial_equity,
        counters=state.counters,
        cooldowns=state.cooldowns,
    )
    simulator.cash = float(state.cash if state.cash is not None else runtime.initial_equity)
    simulator.peak_equity = float(state.peak_equity if state.peak_equity is not None else runtime.initial_equity)
    simulator.last_marked_equity = float(
        state.last_marked_equity if state.last_marked_equity is not None else simulator.cash
    )
    simulator.positions = dict(state.positions)
    all_times = sorted(set().union(*(frame.index.to_list() for frame in prepared.values())))
    signals_processed = 0
    latest_top: list[str] = []
    latest_ranking = pd.DataFrame()

    for timestamp in all_times:
        timestamp = normalize_timestamp(timestamp)
        bar_keys = {f"{symbol}:{timestamp.isoformat()}" for symbol, frame in prepared.items() if timestamp in frame.index}
        if bar_keys and bar_keys.issubset(state.processed_bar_keys):
            continue
        as_of = timestamp + pd.Timedelta(minutes=5)
        rows_by_symbol = {symbol: frame.loc[timestamp] for symbol, frame in prepared.items() if timestamp in frame.index}
        simulator.manage_positions(timestamp, rows_by_symbol)

        still_pending: list[SignalCandidate] = []
        for candidate in state.pending:
            row = rows_by_symbol.get(candidate.symbol)
            if row is None:
                still_pending.append(candidate)
                continue
            simulator.open_from_candidate(candidate, fill_price=float(row["open"]), fill_time=timestamp)
        state.pending = still_pending

        if not latest_top or as_of.minute == 0:
            latest_top, latest_ranking = select_top_quote_volume_symbols({symbol: frame.loc[:timestamp] for symbol, frame in prepared.items()}, as_of, settings)
            logger.log_event({"event": "UNIVERSE_UPDATE", "timestamp": as_of.isoformat(), "symbol": "", "side": "", "top_symbols": latest_top})

        ranking_by_symbol = latest_ranking.set_index("symbol").to_dict("index") if not latest_ranking.empty else {}
        for symbol, frame in prepared.items():
            if timestamp not in frame.index:
                continue
            candidates = state.engines[symbol].on_bar(symbol, frame.loc[:timestamp], latest_top, as_of, ranking_by_symbol.get(symbol, {}))
            for raw_candidate in candidates:
                transformed = process_raw_candidate(raw_candidate, settings, logger)
                if transformed is not None:
                    state.pending.append(transformed)
                    signals_processed += 1
        simulator.mark_equity(timestamp, {symbol: float(row["close"]) for symbol, row in rows_by_symbol.items()})
        state.processed_bar_keys.update(bar_keys)

    state.cooldowns = {symbol: str(value) for symbol, value in simulator.cooldowns.items()}
    state.counters = simulator.counters
    state.cash = simulator.cash
    state.peak_equity = simulator.peak_equity
    state.last_marked_equity = simulator.last_marked_equity
    state.positions = dict(simulator.positions)
    trades = pd.DataFrame(simulator.trades)
    logger.write_daily_summary(trades)
    summary = simulator.summary()
    summary.update(
        {
            "mode": "PAPER",
            "strategy": "liquidity_retest",
            "signals_processed": signals_processed,
            "settings": asdict(settings),
            "transform_mode": settings.trade_transform_mode,
            "drawdown_pct": simulator.current_drawdown_pct,
            "entries_enabled": simulator.entries_enabled,
        }
    )
    logger.ensure_output_files()
    logger.write_summary(summary)
    status = {
        "mode": "PAPER",
        "strategy": "liquidity_retest",
        "signals_processed": signals_processed,
        "top_symbols": latest_top,
        "positions": len(simulator.positions),
        "pending_signals": len(state.pending),
        "paper_equity": simulator.equity,
        "peak_equity": simulator.peak_equity,
        "drawdown_pct": simulator.current_drawdown_pct,
        "entries_enabled": simulator.entries_enabled,
        "transform_mode": settings.trade_transform_mode,
        "output_dir": str(output_dir.resolve()),
    }
    write_json(output_dir / "latest_status.json", status)
    write_json(output_dir / "state.json", state.to_jsonable())
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run liquidity retest paper signal loop using Binance public data.")
    parser.add_argument("--output-dir", default="outputs/liquidity_retest_paper")
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--lookback-hours", type=int, default=72)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--config", default="configs/binance_liquidity_retest_5m.yaml")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    runtime = PaperRuntime(
        output_dir=args.output_dir,
        poll_sec=max(10, args.poll_sec),
        lookback_hours=max(24, args.lookback_hours),
        initial_equity=max(1.0, args.initial_equity),
        once=bool(args.once),
        symbols=parse_symbols(args.symbols) if args.symbols else None,
    )
    settings = load_settings_config(args.config)
    state_path = Path(runtime.output_dir) / "state.json"
    state = PaperState.from_file(state_path)
    session = requests.Session()
    while True:
        try:
            frames = fetch_frames_for_cycle(session, runtime)
            status = run_paper_cycle_from_frames(frames, settings=settings, runtime=runtime, state=state)
            print(json.dumps(status, ensure_ascii=False, default=str))
        except Exception as exc:
            error = {"mode": "PAPER", "strategy": "liquidity_retest", "timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "error": str(exc)}
            write_json(Path(runtime.output_dir) / "latest_error.json", error)
            print(json.dumps(error, ensure_ascii=False))
        if runtime.once:
            break
        time.sleep(runtime.poll_sec)


if __name__ == "__main__":
    main()
