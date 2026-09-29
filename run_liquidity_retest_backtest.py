from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from futures_strategy.liquidity_retest import (
    EventLogger,
    LiquidityRetestBacktestResult,
    LiquidityRetestPortfolioSimulator,
    LiquidityRetestSettings,
    LiquidityRetestSignalEngine,
    SignalCandidate,
    normalize_timestamp,
    process_raw_candidate,
    select_top_quote_volume_symbols,
)
from futures_strategy.indicators import atr
from futures_strategy.top_gainers_boll import download_archive_range, load_archive_csv


def parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def load_settings_config(path: str | Path | None) -> LiquidityRetestSettings:
    if path is None:
        return LiquidityRetestSettings()
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    values = payload.get("liquidity_retest", payload)
    allowed = set(LiquidityRetestSettings.__dataclass_fields__)
    filtered = {key: value for key, value in values.items() if key in allowed}
    if "stablecoin_bases" in filtered and isinstance(filtered["stablecoin_bases"], list):
        filtered["stablecoin_bases"] = tuple(str(item).upper() for item in filtered["stablecoin_bases"])
    return LiquidityRetestSettings(**filtered)


def prepare_5m_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy().sort_index()
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")
    if "quote_volume" not in data.columns:
        data["quote_volume"] = data["close"] * data["volume"]
    data["atr"] = atr(data, 14)
    return data.dropna(subset=["open", "high", "low", "close", "volume", "quote_volume"])


def run_backtest_from_frames(
    frames_by_symbol: dict[str, pd.DataFrame],
    *,
    settings: LiquidityRetestSettings,
    output_dir: str | Path,
    initial_equity: float = 10_000.0,
    run_id: str = "liquidity-retest-backtest",
    trade_start: pd.Timestamp | str | None = None,
    trade_end: pd.Timestamp | str | None = None,
) -> LiquidityRetestBacktestResult:
    output_path = Path(output_dir)
    logger = EventLogger(output_path, run_id=run_id)
    prepared = {symbol.upper(): prepare_5m_frame(frame) for symbol, frame in frames_by_symbol.items() if not frame.empty}
    prepared = {symbol: frame for symbol, frame in prepared.items() if not frame.empty}
    if not prepared:
        raise ValueError("No usable 5m frames for liquidity retest backtest.")

    all_times = sorted(set().union(*(frame.index.to_list() for frame in prepared.values())))
    trade_start_ts = normalize_timestamp(trade_start) if trade_start is not None else None
    trade_end_ts = normalize_timestamp(trade_end) if trade_end is not None else None
    if trade_start_ts is not None and trade_end_ts is not None and trade_start_ts >= trade_end_ts:
        raise ValueError("trade_start must be earlier than trade_end")
    engines = {symbol: LiquidityRetestSignalEngine(settings) for symbol in prepared}
    simulator = LiquidityRetestPortfolioSimulator(settings, logger=logger, initial_equity=initial_equity)
    pending: list[SignalCandidate] = []
    latest_top: list[str] = []
    latest_ranking = pd.DataFrame()
    last_trade_time: pd.Timestamp | None = None
    last_trade_marks: dict[str, float] = {}

    for timestamp in all_times:
        timestamp = normalize_timestamp(timestamp)
        if trade_end_ts is not None and timestamp >= trade_end_ts:
            break
        within_trade_window = (trade_start_ts is None or timestamp >= trade_start_ts) and (trade_end_ts is None or timestamp < trade_end_ts)
        as_of = timestamp + pd.Timedelta(minutes=5)
        rows_by_symbol = {symbol: frame.loc[timestamp] for symbol, frame in prepared.items() if timestamp in frame.index}
        simulator.manage_positions(timestamp, rows_by_symbol)

        still_pending: list[SignalCandidate] = []
        for candidate in pending if within_trade_window else []:
            row = rows_by_symbol.get(candidate.symbol)
            if row is None:
                still_pending.append(candidate)
                continue
            simulator.open_from_candidate(candidate, fill_price=float(row["open"]), fill_time=timestamp)
        pending = still_pending

        if not latest_top or as_of.minute == 0:
            history_for_rank = {symbol: frame.loc[:timestamp] for symbol, frame in prepared.items()}
            latest_top, latest_ranking = select_top_quote_volume_symbols(history_for_rank, as_of, settings)
            logger.log_event(
                {
                    "event": "UNIVERSE_UPDATE",
                    "timestamp": as_of.isoformat(),
                    "symbol": "",
                    "side": "",
                    "top_symbols": latest_top,
                    "universe_size": len(latest_top),
                }
            )
            if len(latest_top) < settings.universe_size:
                logger.log_event({"event": "UNIVERSE_UNDERSIZED", "timestamp": as_of.isoformat(), "symbol": "", "side": "", "usable_symbols": len(latest_top)})

        ranking_by_symbol: dict[str, dict[str, Any]] = {}
        if not latest_ranking.empty:
            ranking_by_symbol = latest_ranking.set_index("symbol").to_dict("index")

        signal_symbols = set(latest_top)
        signal_symbols.update(candidate.symbol for candidate in pending)
        signal_symbols.update(simulator.positions.keys())
        for symbol in sorted(signal_symbols):
            frame = prepared.get(symbol)
            if frame is None:
                continue
            if timestamp not in frame.index:
                continue
            candidates = engines[symbol].on_bar(
                symbol,
                frame.loc[:timestamp],
                latest_top,
                as_of,
                ranking_by_symbol.get(symbol, {}),
            )
            if within_trade_window:
                for raw_candidate in candidates:
                    transformed = process_raw_candidate(raw_candidate, settings, logger)
                    if transformed is not None:
                        pending.append(transformed)

        marks = {symbol: float(row["close"]) for symbol, row in rows_by_symbol.items()}
        if within_trade_window:
            simulator.mark_equity(timestamp, marks)
            last_trade_time = timestamp
            last_trade_marks = marks

    if last_trade_time is not None:
        simulator.close_all(last_trade_time, last_trade_marks, reason="end_of_test")
    trades = pd.DataFrame(simulator.trades)
    equity_curve = pd.DataFrame(simulator.equity_rows)
    logger.write_daily_summary(trades)
    summary = simulator.summary()
    summary.update(
        {
            "strategy": "liquidity_retest",
            "mode": "BACKTEST",
            "symbols_loaded": sorted(prepared.keys()),
            "settings": asdict(settings),
            "trade_start": trade_start_ts.isoformat() if trade_start_ts is not None else "",
            "trade_end": trade_end_ts.isoformat() if trade_end_ts is not None else "",
            "output_files": {
                "events": str((output_path / "events.jsonl").resolve()),
                "signals": str((output_path / "signals.csv").resolve()),
                "trades": str((output_path / "trades.csv").resolve()),
                "daily_summary": str((output_path / "daily_summary.csv").resolve()),
                "equity_curve": str((output_path / "equity_curve.csv").resolve()),
            },
        }
    )
    logger.ensure_output_files()
    logger.write_summary(summary)
    signals = pd.read_csv(output_path / "signals.csv") if (output_path / "signals.csv").exists() else pd.DataFrame()
    return LiquidityRetestBacktestResult(summary=summary, trades=trades, equity_curve=equity_curve, signals=signals)


def load_archive_frames(symbols: list[str], start_month: str, end_month: str, cache_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    with requests.Session() as session:
        for symbol in symbols:
            path = download_archive_range(symbol, "5m", start_month, end_month, cache_dir=cache_dir, session=session)
            frame = load_archive_csv(path)
            if not frame.empty:
                frames[symbol] = frame
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Run liquidity top-volume breakout-retest 5m backtest.")
    parser.add_argument("--start-month", required=True)
    parser.add_argument("--end-month", required=True)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--cache-dir", default="data/liquidity_retest_5m")
    parser.add_argument("--output-dir", default="outputs/liquidity_retest_backtest")
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--config", default="configs/binance_liquidity_retest_5m.yaml")
    args = parser.parse_args()

    settings = load_settings_config(args.config)
    frames = load_archive_frames(parse_symbols(args.symbols), args.start_month, args.end_month, Path(args.cache_dir))
    result = run_backtest_from_frames(frames, settings=settings, output_dir=args.output_dir, initial_equity=args.initial_equity)
    print("=== Liquidity Retest Backtest ===")
    print(json.dumps(result.summary, ensure_ascii=False, indent=2, default=str))
    print(f"output_dir: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
