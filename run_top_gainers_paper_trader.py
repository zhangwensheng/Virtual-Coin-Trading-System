from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from futures_strategy.backtest import apply_slippage
from futures_strategy.config import AppConfig, load_config
from futures_strategy.factor_portfolio import candidate_rank_score
from futures_strategy.indicators import bollinger_bands
from futures_strategy.strategy import latest_signal_snapshot, prepare_market_data, suggested_stop


BINANCE_FAPI = "https://fapi.binance.com"
INTERVAL_MS = 60_000


@dataclass
class PaperRuntime:
    config_path: str
    output_dir: str
    poll_sec: int = 60
    scan_limit: int = 80
    leverage: int = 10
    margin_usdt: float = 1.0
    max_positions: int = 1
    initial_balance: float = 100.0
    once: bool = False


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def request_json(session: requests.Session, path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{BINANCE_FAPI}{path}"
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Binance request failed: {path} {params or {}}") from last_error


def native_to_unified(symbol: str) -> str:
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USDT:USDT"
    return symbol


def fetch_active_usdt_perps(session: requests.Session) -> set[str]:
    payload = request_json(session, "/fapi/v1/exchangeInfo")
    symbols: set[str] = set()
    for row in payload.get("symbols", []):
        if (
            row.get("contractType") == "PERPETUAL"
            and row.get("quoteAsset") == "USDT"
            and row.get("status") == "TRADING"
        ):
            symbols.add(str(row["symbol"]))
    return symbols


def fetch_prefilter_symbols(session: requests.Session, active: set[str], limit: int) -> list[str]:
    payload = request_json(session, "/fapi/v1/ticker/24hr")
    rows: list[tuple[float, float, str]] = []
    for row in payload:
        symbol = str(row.get("symbol", ""))
        if symbol not in active:
            continue
        try:
            change_pct = float(row.get("priceChangePercent", 0.0))
            quote_volume = float(row.get("quoteVolume", 0.0))
        except (TypeError, ValueError):
            continue
        if quote_volume <= 0:
            continue
        rows.append((change_pct, quote_volume, symbol))
    rows.sort(reverse=True)
    return [symbol for _, _, symbol in rows[: max(5, limit)]]


def klines_to_frame(rows: list[list[Any]], *, interval_ms: int, server_time_ms: int | None = None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_quote",
        "close_time",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if server_time_ms is not None:
        frame = frame[frame["close_time"] <= server_time_ms]
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame = frame.set_index("timestamp").sort_index()
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    return frame[["open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_quote"]]


def fetch_1m_frame(session: requests.Session, symbol: str, *, server_time_ms: int) -> pd.DataFrame:
    end_time = pd.to_datetime(server_time_ms, unit="ms", utc=True)
    start = (end_time.floor("D") - pd.Timedelta(minutes=120)).value // 1_000_000
    rows = request_json(
        session,
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": "1m", "startTime": int(start), "limit": 1500},
    )
    return klines_to_frame(rows, interval_ms=INTERVAL_MS, server_time_ms=server_time_ms)


def fetch_daily_lookup(session: requests.Session, symbol: str, config: AppConfig, *, server_time_ms: int) -> dict[str, float] | None:
    rows = request_json(
        session,
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": "1d", "limit": max(40, config.strategy.top_gainers_daily_boll_window + 5)},
    )
    daily = klines_to_frame(rows, interval_ms=86_400_000, server_time_ms=server_time_ms)
    if daily.empty:
        return None
    middle, upper, lower = bollinger_bands(
        daily["close"],
        config.strategy.top_gainers_daily_boll_window,
        config.strategy.top_gainers_daily_boll_std,
    )
    latest = daily.iloc[-1]
    return {
        "daily_bb_mid": float(middle.shift(1).iloc[-1]),
        "daily_bb_upper": float(upper.shift(1).iloc[-1]),
        "daily_bb_lower": float(lower.shift(1).iloc[-1]),
        "prev_daily_close": float(daily["close"].shift(1).iloc[-1]),
        "daily_open": float(latest["open"]),
    }


def day_open_for_frame(frame: pd.DataFrame, *, now: pd.Timestamp) -> float | None:
    day = now.floor("D")
    today = frame[frame.index >= day]
    if today.empty:
        return None
    value = float(today["open"].iloc[0])
    return value if value > 0 else None


def build_live_frames(
    session: requests.Session,
    config: AppConfig,
    symbols: list[str],
    *,
    open_symbols: set[str],
    daily_cache: dict[str, dict[str, float]],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]], int]:
    server_time_ms = int(request_json(session, "/fapi/v1/time")["serverTime"])
    now = pd.to_datetime(server_time_ms, unit="ms", utc=True)
    scan_symbols = list(dict.fromkeys([*symbols, *sorted(open_symbols)]))
    frames: dict[str, pd.DataFrame] = {}
    ranking_rows: list[dict[str, Any]] = []

    for symbol in scan_symbols:
        try:
            frame = fetch_1m_frame(session, symbol, server_time_ms=server_time_ms)
            if len(frame) < 80:
                continue
            day_open = day_open_for_frame(frame, now=now)
            if day_open is None:
                continue
            if symbol not in daily_cache:
                lookup = fetch_daily_lookup(session, symbol, config, server_time_ms=server_time_ms)
                if lookup is None or not math.isfinite(lookup["daily_bb_upper"]):
                    continue
                daily_cache[symbol] = lookup
            close = float(frame["close"].iloc[-1])
            daily_return_pct = (close / day_open) - 1.0
            lookup = daily_cache[symbol]
            upper_break = close > float(lookup["daily_bb_upper"])
            frames[symbol] = frame
            ranking_rows.append(
                {
                    "symbol": symbol,
                    "daily_return_pct": daily_return_pct,
                    "daily_close": close,
                    "daily_upper_break": upper_break,
                }
            )
        except Exception:
            continue

    if not ranking_rows:
        return {}, [], server_time_ms

    ranking_rows.sort(key=lambda row: float(row["daily_return_pct"]), reverse=True)
    rank_by_symbol = {row["symbol"]: rank for rank, row in enumerate(ranking_rows, start=1)}
    selected_rows: list[dict[str, Any]] = []
    enriched: dict[str, pd.DataFrame] = {}
    for row in ranking_rows:
        symbol = row["symbol"]
        rank = rank_by_symbol[symbol]
        lookup = daily_cache[symbol]
        selected = (
            rank <= int(config.strategy.top_gainers_daily_top_n)
            and float(row["daily_return_pct"]) >= float(config.strategy.top_gainers_daily_min_return_pct)
            and float(row["daily_return_pct"]) <= float(config.strategy.top_gainers_daily_max_return_pct)
            and bool(row["daily_upper_break"])
        )
        frame = frames[symbol].copy()
        frame["daily_rank"] = 999.0
        frame["selected_day"] = False
        frame["daily_return_pct"] = (frame["close"] / float(lookup["daily_open"])) - 1.0
        frame["daily_close"] = frame["close"]
        frame["daily_bb_mid"] = float(lookup["daily_bb_mid"])
        frame["daily_bb_upper"] = float(lookup["daily_bb_upper"])
        frame["daily_bb_lower"] = float(lookup["daily_bb_lower"])
        frame.loc[frame.index[-1], "daily_rank"] = float(rank)
        frame.loc[frame.index[-1], "selected_day"] = bool(selected)
        enriched[symbol] = frame
        selected_rows.append({**row, "daily_rank": rank, "selected_day": selected})
    return enriched, selected_rows, server_time_ms


def load_state(path: Path, initial_balance: float) -> dict[str, Any]:
    if not path.exists():
        return {"positions": {}, "realized_pnl": 0.0, "initial_balance": initial_balance}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("positions", {})
    payload.setdefault("realized_pnl", 0.0)
    payload.setdefault("initial_balance", initial_balance)
    return payload


def position_unrealized(position: dict[str, Any], mark: float) -> float:
    return (mark - float(position["entry_price"])) * float(position["remaining_qty"])


def close_position(
    state: dict[str, Any],
    symbol: str,
    position: dict[str, Any],
    *,
    timestamp: pd.Timestamp,
    exit_price: float,
    reason: str,
    fee_rate: float,
    trades_path: Path,
) -> dict[str, Any]:
    remaining_qty = float(position["remaining_qty"])
    gross = (exit_price - float(position["entry_price"])) * remaining_qty
    exit_fee = exit_price * remaining_qty * fee_rate
    pnl = float(position["realized_pnl"]) + gross - exit_fee
    event = {
        "event": "CLOSE",
        "timestamp": timestamp.isoformat(),
        "symbol": symbol,
        "side": "long",
        "entry_time": position["entry_time"],
        "entry_price": position["entry_price"],
        "exit_price": exit_price,
        "qty": position["qty"],
        "remaining_qty": remaining_qty,
        "notional_usdt": position["notional_usdt"],
        "margin_usdt": position["margin_usdt"],
        "leverage": position["leverage"],
        "pnl": pnl,
        "reason": reason,
    }
    append_jsonl(trades_path, event)
    state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + pnl
    state["positions"].pop(symbol, None)
    return event


def manage_positions(
    state: dict[str, Any],
    prepared_frames: dict[str, pd.DataFrame],
    config: AppConfig,
    *,
    trades_path: Path,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for symbol, position in list(state["positions"].items()):
        frame = prepared_frames.get(symbol)
        if frame is None or frame.empty:
            continue
        row = frame.iloc[-1]
        bar_time = frame.index[-1]
        last_bar = position.get("last_bar_time")
        if last_bar and pd.Timestamp(last_bar) >= bar_time:
            continue
        position["last_bar_time"] = bar_time.isoformat()
        position["bars_held"] = int(position.get("bars_held", 0)) + 1
        position["highest_price"] = max(float(position.get("highest_price", position["entry_price"])), float(row["high"]), float(row["close"]))

        exit_price = None
        reason = ""
        if float(row["low"]) <= float(position["stop_price"]):
            exit_price = apply_slippage("long", float(position["stop_price"]), is_entry=False, slippage_bps=config.risk.slippage_bps)
            reason = "stop_loss" if not position.get("partial_taken") else "trail_stop"
        elif not position.get("partial_taken") and float(row["high"]) >= float(position["tp_price"]):
            partial_qty = float(position["qty"]) * float(config.strategy.partial_close_ratio)
            partial_qty = min(partial_qty, float(position["remaining_qty"]))
            partial_price = apply_slippage("long", float(position["tp_price"]), is_entry=False, slippage_bps=config.risk.slippage_bps)
            gross = (partial_price - float(position["entry_price"])) * partial_qty
            fee = partial_price * partial_qty * config.risk.fee_rate
            position["remaining_qty"] = float(position["remaining_qty"]) - partial_qty
            position["realized_pnl"] = float(position["realized_pnl"]) + gross - fee
            position["partial_taken"] = True
            position["stop_price"] = max(float(position["stop_price"]), float(position["entry_price"]))
            event = {
                "event": "PARTIAL_TAKE_PROFIT",
                "timestamp": bar_time.isoformat(),
                "symbol": symbol,
                "qty": partial_qty,
                "price": partial_price,
                "realized_pnl": position["realized_pnl"],
            }
            append_jsonl(trades_path, event)
            events.append(event)

        if position.get("partial_taken") and exit_price is None:
            trail_stop = float(position["highest_price"]) - (float(row["atr"]) * config.strategy.trail_atr_mult)
            position["stop_price"] = max(float(position["stop_price"]), trail_stop, float(position["entry_price"]))

        if exit_price is None and bool(row.get("exit_long_signal", False)):
            exit_price = apply_slippage("long", float(row["close"]), is_entry=False, slippage_bps=config.risk.slippage_bps)
            reason = "signal_exit"
        if exit_price is None and int(position.get("bars_held", 0)) >= int(config.strategy.max_bars_in_trade):
            exit_price = apply_slippage("long", float(row["close"]), is_entry=False, slippage_bps=config.risk.slippage_bps)
            reason = "time_exit"
        if exit_price is None and bool(config.strategy.flat_on_new_day):
            if pd.Timestamp(position["entry_time"]).floor("D") < bar_time.floor("D"):
                exit_price = apply_slippage("long", float(row["close"]), is_entry=False, slippage_bps=config.risk.slippage_bps)
                reason = "new_day_exit"

        if exit_price is not None:
            events.append(
                close_position(
                    state,
                    symbol,
                    position,
                    timestamp=bar_time,
                    exit_price=float(exit_price),
                    reason=reason,
                    fee_rate=config.risk.fee_rate,
                    trades_path=trades_path,
                )
            )
    return events


def open_paper_position(
    state: dict[str, Any],
    symbol: str,
    row: pd.Series,
    config: AppConfig,
    runtime: PaperRuntime,
    *,
    trades_path: Path,
) -> dict[str, Any] | None:
    entry_raw = float(row["close"])
    if entry_raw <= 0:
        return None
    entry_price = apply_slippage("long", entry_raw, is_entry=True, slippage_bps=config.risk.slippage_bps)
    stop = suggested_stop(row, "long", config.strategy)
    if stop is None or not math.isfinite(float(stop)) or float(stop) >= entry_price:
        atr_value = float(row.get("atr", 0.0))
        stop = entry_price - max(atr_value * config.strategy.atr_stop_mult, entry_price * 0.003)
    risk_distance = entry_price - float(stop)
    if risk_distance <= 0:
        return None
    notional = float(runtime.margin_usdt) * float(runtime.leverage)
    qty = notional / entry_price
    entry_fee = notional * float(config.risk.fee_rate)
    tp_price = entry_price + (risk_distance * float(config.strategy.partial_rr))
    event = {
        "event": "OPEN",
        "timestamp": row.name.isoformat(),
        "symbol": symbol,
        "side": "long",
        "entry_price": entry_price,
        "qty": qty,
        "notional_usdt": notional,
        "margin_usdt": runtime.margin_usdt,
        "leverage": runtime.leverage,
        "stop_price": float(stop),
        "tp_price": float(tp_price),
        "daily_rank": int(row.get("daily_rank", 999)),
        "daily_return_pct": float(row.get("daily_return_pct", 0.0)),
        "vwap_gap": float(row.get("vwap_gap", 0.0)),
        "volume_ratio": float(row.get("volume_ratio", 0.0)),
        "rank_score": float(candidate_rank_score(row)),
    }
    state["positions"][symbol] = {
        "symbol": symbol,
        "side": "long",
        "entry_time": row.name.isoformat(),
        "entry_price": entry_price,
        "qty": qty,
        "remaining_qty": qty,
        "notional_usdt": notional,
        "margin_usdt": runtime.margin_usdt,
        "leverage": runtime.leverage,
        "stop_price": float(stop),
        "tp_price": float(tp_price),
        "highest_price": entry_price,
        "bars_held": 0,
        "partial_taken": False,
        "realized_pnl": -entry_fee,
        "last_bar_time": row.name.isoformat(),
    }
    append_jsonl(trades_path, event)
    return event


def append_equity_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_signal_diagnostic(symbol: str, row: pd.Series, config: AppConfig) -> dict[str, Any]:
    checks = {
        "selected": bool(row.get("selected_day", False)),
        "long_regime": bool(row.get("long_regime", False)),
        "long_signal": bool(row.get("long_signal", False)),
        "close_gt_ema": bool(row["close"] > row["ema_entry"]),
        "ema_stack": bool(row["ema_entry"] > row["ema_slow"]),
        "close_gt_vwap": bool(row["close"] > row["session_vwap"]),
        "vwap_ok": bool(row.get("vwap_gap", 999.0) <= config.strategy.top_gainers_momentum_max_vwap_extension_pct),
        "htf_mid_ok": bool(row["htf_close"] >= row["htf_bb_mid"]),
        "bullish_bar": bool(row["close"] > row["open"]),
        "break_high": bool(row["close"] > row["momentum_break_high"]),
        "volume_ok": bool(
            config.strategy.top_gainers_momentum_min_volume_ratio
            <= row.get("volume_ratio", 0.0)
            <= config.strategy.top_gainers_momentum_max_volume_ratio
        ),
        "trade_ok": bool(row.get("trade_ratio", 0.0) >= config.strategy.top_gainers_momentum_min_trade_ratio),
        "delta_ok": bool(
            config.strategy.top_gainers_momentum_min_delta_ratio
            <= row.get("delta_ratio", 0.0)
            <= config.strategy.top_gainers_momentum_max_delta_ratio
        ),
        "rsi_ok": bool(
            config.strategy.top_gainers_momentum_min_rsi
            <= row.get("rsi", 0.0)
            <= config.strategy.top_gainers_momentum_max_rsi
        ),
        "htf_volume_ok": bool(
            row.get("htf_volume_ratio", 0.0) >= config.strategy.top_gainers_momentum_min_htf_volume_ratio
        ),
        "atr_ok": bool(config.strategy.min_atr_pct <= row.get("atr_pct", 0.0) <= config.strategy.max_atr_pct),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "symbol": symbol,
        "daily_rank": int(row.get("daily_rank", 999)),
        "daily_return_pct": round(float(row.get("daily_return_pct", 0.0)) * 100, 2),
        "close": float(row["close"]),
        "rsi": round(float(row.get("rsi", 0.0)), 2),
        "volume_ratio": round(float(row.get("volume_ratio", 0.0)), 2),
        "trade_ratio": round(float(row.get("trade_ratio", 0.0)), 2),
        "delta_ratio": round(float(row.get("delta_ratio", 0.0)), 3),
        "htf_volume_ratio": round(float(row.get("htf_volume_ratio", 0.0)), 2),
        "vwap_gap_pct": round(float(row.get("vwap_gap", 0.0)) * 100, 2),
        "atr_pct": round(float(row.get("atr_pct", 0.0)) * 100, 3),
        "failed_checks": failed,
        "checks": checks,
    }


def run_cycle(
    session: requests.Session,
    config: AppConfig,
    runtime: PaperRuntime,
    state: dict[str, Any],
    *,
    active_symbols: set[str],
    daily_cache: dict[str, dict[str, float]],
    output_dir: Path,
) -> dict[str, Any]:
    trades_path = output_dir / "trades.jsonl"
    signals_path = output_dir / "signals.jsonl"
    symbols = fetch_prefilter_symbols(session, active_symbols, runtime.scan_limit)
    frames, ranking_rows, server_time_ms = build_live_frames(
        session,
        config,
        symbols,
        open_symbols=set(state["positions"].keys()),
        daily_cache=daily_cache,
    )
    prepared_frames: dict[str, pd.DataFrame] = {}
    candidates: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for symbol, frame in frames.items():
        try:
            prepared = prepare_market_data(frame, config.strategy)
        except Exception:
            continue
        if prepared.empty:
            continue
        prepared_frames[symbol] = prepared
        row = prepared.iloc[-1]
        if int(row.get("daily_rank", 999)) <= max(10, config.strategy.top_gainers_daily_top_n):
            diagnostics.append(build_signal_diagnostic(symbol, row, config))
        snapshot = latest_signal_snapshot(prepared, config.strategy)
        if bool(row.get("long_signal", False)):
            candidates.append(
                {
                    "symbol": symbol,
                    "display_symbol": native_to_unified(symbol),
                    "timestamp": prepared.index[-1].isoformat(),
                    "rank_score": float(candidate_rank_score(row)),
                    "close": float(row["close"]),
                    "daily_rank": int(row["daily_rank"]),
                    "daily_return_pct": float(row["daily_return_pct"]),
                    "vwap_gap": float(row.get("vwap_gap", 0.0)),
                    "volume_ratio": float(row.get("volume_ratio", 0.0)),
                    "snapshot": snapshot,
                }
            )
    candidates.sort(key=lambda item: item["rank_score"], reverse=True)
    events = manage_positions(state, prepared_frames, config, trades_path=trades_path)

    slots = max(0, int(runtime.max_positions) - len(state["positions"]))
    opened: list[dict[str, Any]] = []
    if slots > 0:
        for candidate in candidates:
            symbol = candidate["symbol"]
            if symbol in state["positions"]:
                continue
            row = prepared_frames[symbol].iloc[-1]
            event = open_paper_position(state, symbol, row, config, runtime, trades_path=trades_path)
            if event is not None:
                opened.append(event)
                slots -= 1
            if slots <= 0:
                break

    unrealized = 0.0
    marks: dict[str, float] = {}
    for symbol, position in state["positions"].items():
        frame = prepared_frames.get(symbol)
        if frame is None or frame.empty:
            continue
        mark = float(frame.iloc[-1]["close"])
        marks[symbol] = mark
        unrealized += position_unrealized(position, mark)
    equity = float(state.get("initial_balance", runtime.initial_balance)) + float(state.get("realized_pnl", 0.0)) + unrealized
    cycle_time = pd.to_datetime(server_time_ms, unit="ms", utc=True).isoformat()
    status = {
        "mode": "PAPER",
        "cycle_time": cycle_time,
        "strategy": f"top_gainers_pump_dump_1m_top{config.strategy.top_gainers_daily_top_n}_opportunity",
        "leverage": runtime.leverage,
        "margin_usdt_per_trade": runtime.margin_usdt,
        "notional_usdt_per_trade": runtime.margin_usdt * runtime.leverage,
        "scanned_symbols": len(symbols),
        "prepared_symbols": len(prepared_frames),
        "candidate_count": len(candidates),
        "top_ranked": ranking_rows[:10],
        "candidates": candidates[:10],
        "diagnostics": sorted(diagnostics, key=lambda item: item["daily_rank"])[:20],
        "opened": opened,
        "position_events": events,
        "positions": state["positions"],
        "marks": marks,
        "realized_pnl": float(state.get("realized_pnl", 0.0)),
        "unrealized_pnl": unrealized,
        "paper_equity": equity,
    }
    append_jsonl(signals_path, status)
    write_json(output_dir / "latest_status.json", status)
    write_json(output_dir / "positions.json", {"positions": state["positions"], "marks": marks})
    append_equity_csv(
        output_dir / "equity.csv",
        {
            "timestamp": cycle_time,
            "paper_equity": round(equity, 8),
            "realized_pnl": round(float(state.get("realized_pnl", 0.0)), 8),
            "unrealized_pnl": round(unrealized, 8),
            "open_positions": len(state["positions"]),
            "candidate_count": len(candidates),
            "prepared_symbols": len(prepared_frames),
        },
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Top2 pump sniper paper trader on Binance Futures public data")
    parser.add_argument("--config", default="configs/binance_top_gainers_pump_dump_1m_turbo.yaml")
    parser.add_argument("--output-dir", default="outputs/top2_sniper_paper_10x_1u")
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--scan-limit", type=int, default=80)
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--margin-usdt", type=float, default=1.0)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--initial-balance", type=float, default=100.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    runtime = PaperRuntime(
        config_path=args.config,
        output_dir=args.output_dir,
        poll_sec=max(15, args.poll_sec),
        scan_limit=max(5, args.scan_limit),
        leverage=max(1, args.leverage),
        margin_usdt=max(0.1, args.margin_usdt),
        max_positions=max(1, args.max_positions),
        initial_balance=max(1.0, args.initial_balance),
        once=bool(args.once),
    )
    output_dir = Path(runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(runtime.config_path)
    config.risk.leverage = runtime.leverage
    state_path = output_dir / "state.json"
    state = load_state(state_path, runtime.initial_balance)
    daily_cache: dict[str, dict[str, float]] = {}

    with requests.Session() as session:
        active_symbols = fetch_active_usdt_perps(session)
        write_json(
            output_dir / "runtime.json",
            {"runtime": asdict(runtime), "active_symbol_count": len(active_symbols), "config": runtime.config_path},
        )
        while True:
            try:
                status = run_cycle(
                    session,
                    config,
                    runtime,
                    state,
                    active_symbols=active_symbols,
                    daily_cache=daily_cache,
                    output_dir=output_dir,
                )
                save_payload = {**state, "last_cycle_time": status["cycle_time"]}
                write_json(state_path, save_payload)
                print(
                    json.dumps(
                        {
                            "cycle_time": status["cycle_time"],
                            "candidates": status["candidate_count"],
                            "positions": len(status["positions"]),
                            "paper_equity": status["paper_equity"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                error = {"timestamp": utc_now().isoformat(), "error": str(exc)}
                append_jsonl(output_dir / "errors.jsonl", error)
                write_json(output_dir / "latest_error.json", error)
                print(json.dumps(error, ensure_ascii=False), flush=True)
            if runtime.once:
                break
            time.sleep(runtime.poll_sec)


if __name__ == "__main__":
    main()
