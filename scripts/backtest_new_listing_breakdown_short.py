"""Research-only backtest for shorting new listings after prior-day-low breaks.

This script is intentionally standalone: it reads cached Binance futures candles
and public exchangeInfo metadata, then writes inspectable CSV/JSON/MD outputs.
It does not read or write account state.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from download_binance_archive import download_month, month_range

FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002


@dataclass(frozen=True)
class StrategyParams:
    confirm: str
    max_listing_age_days: int
    min_drawdown_pct: float
    require_prev_green: bool
    sl_pct: float
    tp_pct: float
    max_hold_minutes: int

    def key(self) -> str:
        prev = "prev_green" if self.require_prev_green else "any_prev"
        return (
            f"{self.confirm}|age{self.max_listing_age_days}|dd{self.min_drawdown_pct:.0%}|"
            f"{prev}|sl{self.sl_pct:.0%}|tp{self.tp_pct:.0%}|hold{self.max_hold_minutes // 60}h"
        )


def month_start(month: str) -> pd.Timestamp:
    return pd.Timestamp(f"{month}-01", tz="UTC")


def month_after(month: str) -> pd.Timestamp:
    return pd.Timestamp((pd.Period(month, freq="M") + 1).start_time, tz="UTC")


def symbol_from_file(path: Path, timeframe: str) -> str | None:
    match = re.match(rf"(.+?)_{re.escape(timeframe)}_\d{{4}}-\d{{2}}_\d{{4}}-\d{{2}}\.csv$", path.name)
    return match.group(1).upper() if match else None


def load_candles(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "timestamp" not in frame.columns:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp"])
    merged = merged.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    for column in ["open", "high", "low", "close", "volume", "quote_volume", "trade_count"]:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged.dropna(subset=["open", "high", "low", "close"])


def fetch_onboard_dates(timeout: float = 20.0) -> tuple[dict[str, pd.Timestamp], str | None]:
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        payload = requests.get(url, timeout=timeout).json()
    except Exception as exc:  # pragma: no cover - network dependent
        return {}, f"{type(exc).__name__}: {exc}"
    result: dict[str, pd.Timestamp] = {}
    for row in payload.get("symbols", []):
        symbol = str(row.get("symbol", "")).upper()
        if not symbol or row.get("contractType") != "PERPETUAL" or row.get("quoteAsset") != "USDT":
            continue
        onboard_ms = pd.to_numeric(row.get("onboardDate"), errors="coerce")
        if pd.notna(onboard_ms):
            result[symbol] = pd.Timestamp(float(onboard_ms), unit="ms", tz="UTC").normalize()
    return result, None


def archive_cache_path(cache_dir: Path, symbol: str, timeframe: str, start_month: str, end_month: str) -> Path:
    return cache_dir / f"{symbol.lower()}_{timeframe}_{start_month}_{end_month}.csv"


def download_archive_range(
    symbol: str,
    timeframe: str,
    start_month: str,
    end_month: str,
    *,
    cache_dir: Path,
    session: requests.Session,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = archive_cache_path(cache_dir, symbol, timeframe, start_month, end_month)
    if output_path.exists():
        return output_path
    frames = [download_month(symbol, timeframe, month, "um", session) for month in month_range(start_month, end_month)]
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    merged.to_csv(output_path, index=False)
    return output_path


def download_new_listing_archives(
    onboard_dates: dict[str, pd.Timestamp],
    *,
    cache_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_listing_age_days: int,
) -> list[dict[str, Any]]:
    intraday_dir = cache_dir / "intraday"
    daily_dir = cache_dir / "daily"
    rows: list[dict[str, Any]] = []
    start_cutoff = start - pd.Timedelta(days=max_listing_age_days + 2)
    session = requests.Session()
    symbols = [
        (symbol, onboard)
        for symbol, onboard in sorted(onboard_dates.items(), key=lambda item: (item[1], item[0]))
        if start_cutoff <= onboard < end
    ]
    for symbol, onboard in symbols:
        first_month = str(pd.Period((onboard - pd.Timedelta(days=1)).to_pydatetime(), freq="M"))
        last_needed = min(end - pd.Timedelta(minutes=1), onboard + pd.Timedelta(days=max_listing_age_days + 2))
        last_month = str(pd.Period(last_needed.to_pydatetime(), freq="M"))
        record = {
            "symbol": symbol,
            "onboard": str(onboard),
            "first_month": first_month,
            "last_month": last_month,
            "status": "downloaded_or_cached",
            "errors": [],
        }
        for timeframe, target_dir in [("1d", daily_dir), ("1m", intraday_dir)]:
            try:
                download_archive_range(
                    symbol,
                    timeframe,
                    first_month,
                    last_month,
                    cache_dir=target_dir,
                    session=session,
                )
            except Exception as exc:  # pragma: no cover - network/archive dependent
                record["status"] = "partial_or_failed"
                record["errors"].append(f"{timeframe}: {type(exc).__name__}: {exc}")
        rows.append(record)
        print(f"archive {symbol} {first_month}->{last_month} {record['status']}", flush=True)
    return rows


def discover_symbols(intraday_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(intraday_dir.glob("*_1m_*.csv")):
        symbol = symbol_from_file(path, "1m")
        if symbol:
            grouped.setdefault(symbol, []).append(path)
    return grouped


def daily_paths_for(daily_dir: Path, symbol: str) -> list[Path]:
    return sorted(daily_dir.glob(f"{symbol.lower()}_1d_*.csv"))


def listing_date_for(
    symbol: str,
    daily: pd.DataFrame,
    onboard_dates: dict[str, pd.Timestamp],
    start: pd.Timestamp,
) -> tuple[pd.Timestamp | None, str]:
    official = onboard_dates.get(symbol)
    if official is not None:
        return official, "official_onboardDate"
    if daily.empty:
        return None, "missing_daily"
    first_daily = daily.index.min().normalize()
    if first_daily > start + pd.Timedelta(days=1):
        return first_daily, "local_first_daily_fallback"
    return None, "not_new_listing_without_official_onboardDate"


def generate_candidates_for_symbol(
    symbol: str,
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    listing_date: pd.Timestamp,
    listing_source: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_age_to_collect: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    daily = daily[(daily.index >= listing_date - pd.Timedelta(days=2)) & (daily.index < end)].copy()
    if len(daily) < 2:
        return rows
    daily["trade_date"] = daily.index.normalize()
    daily_by_date = {ts.normalize(): row for ts, row in daily.iterrows()}
    intraday = intraday[(intraday.index >= start) & (intraday.index < end)].copy()
    if intraday.empty:
        return rows

    for trade_date, day_row in daily_by_date.items():
        age_days = int((trade_date - listing_date).days)
        if age_days < 1 or age_days > max_age_to_collect:
            continue
        if trade_date < start or trade_date >= end:
            continue
        prev_date = trade_date - pd.Timedelta(days=1)
        prev_row = daily_by_date.get(prev_date)
        if prev_row is None:
            continue
        day_minutes = intraday[(intraday.index >= trade_date) & (intraday.index < trade_date + pd.Timedelta(days=1))].copy()
        if len(day_minutes) < 60:
            continue
        day_open = float(day_minutes["open"].iloc[0])
        prev_low = float(prev_row["low"])
        prev_high = float(prev_row["high"])
        prev_open = float(prev_row["open"])
        prev_close = float(prev_row["close"])
        day_minutes["minute_in_day"] = ((day_minutes.index - trade_date).total_seconds() / 60).astype(int)
        day_minutes["session_high_so_far"] = day_minutes["high"].cummax()
        day_minutes["drawdown_pct"] = day_minutes["close"] / day_minutes["session_high_so_far"] - 1.0
        day_minutes["break_depth_pct"] = prev_low / day_minutes["close"] - 1.0

        for confirm in ["wick_break", "close_break"]:
            if confirm == "wick_break":
                mask = day_minutes["low"] < prev_low
            else:
                mask = day_minutes["close"] < prev_low
            confirmed = day_minutes[mask & (day_minutes["minute_in_day"] >= 15)]
            if confirmed.empty:
                continue
            signal_time = confirmed.index[0]
            pos = intraday.index.get_indexer([signal_time])[0]
            if pos < 0 or pos + 1 >= len(intraday):
                continue
            entry_time = intraday.index[pos + 1]
            if entry_time != signal_time + pd.Timedelta(minutes=1):
                continue
            signal_bar = day_minutes.loc[signal_time]
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "listing_date": listing_date,
                    "listing_source": listing_source,
                    "listing_age_days": age_days,
                    "confirm": confirm,
                    "signal_time": signal_time,
                    "entry_time": entry_time,
                    "entry_open": float(intraday.iloc[pos + 1]["open"]),
                    "prev_low": prev_low,
                    "prev_high": prev_high,
                    "prev_return_pct": prev_close / prev_open - 1.0 if prev_open else np.nan,
                    "prev_green": bool(prev_close > prev_open),
                    "day_open": day_open,
                    "signal_close": float(signal_bar["close"]),
                    "signal_day_return_pct": float(signal_bar["close"] / day_open - 1.0) if day_open else np.nan,
                    "session_high_so_far": float(signal_bar["session_high_so_far"]),
                    "session_drawdown_pct": float(signal_bar["drawdown_pct"]),
                    "break_depth_pct": float(signal_bar["break_depth_pct"]),
                    "signal_volume": float(signal_bar.get("volume", np.nan)),
                }
            )
    return rows


def simulate_trade(
    frame: pd.DataFrame,
    candidate: dict[str, Any],
    params: StrategyParams,
) -> dict[str, Any] | None:
    entry_time = candidate["entry_time"]
    if entry_time not in frame.index:
        return None
    start_pos = frame.index.get_loc(entry_time)
    if isinstance(start_pos, slice) or isinstance(start_pos, np.ndarray):
        return None
    max_end = entry_time + pd.Timedelta(minutes=params.max_hold_minutes)
    window = frame.iloc[start_pos:]
    window = window[window.index <= max_end]
    if window.empty:
        return None

    entry = float(candidate["entry_open"]) * (1.0 - SLIPPAGE_RATE)
    stop = entry * (1.0 + params.sl_pct)
    target = entry * (1.0 - params.tp_pct)
    qty = 1000.0 / entry
    exit_price = float(window.iloc[-1]["close"]) * (1.0 + SLIPPAGE_RATE)
    exit_time = window.index[-1]
    exit_reason = "TIME_EXIT" if exit_time >= max_end else "END_DATA"

    for ts, bar in window.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])
        open_price = float(bar["open"])
        if high >= stop:
            exit_price = max(open_price, stop) * (1.0 + SLIPPAGE_RATE)
            exit_time = ts
            exit_reason = "STOP"
            break
        if low <= target:
            exit_price = min(open_price, target) * (1.0 + SLIPPAGE_RATE)
            exit_time = ts
            exit_reason = "TARGET"
            break

    gross_pnl = qty * (entry - exit_price)
    fee = qty * (entry + exit_price) * FEE_RATE
    net_pnl = gross_pnl - fee
    return {
        **candidate,
        "param_key": params.key(),
        **asdict(params),
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_pnl": gross_pnl,
        "fee": fee,
        "net_pnl": net_pnl,
        "return_pct_on_notional": net_pnl / 1000.0,
        "hold_minutes": (exit_time - entry_time).total_seconds() / 60,
    }


def metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "trades": 0,
            "symbols": 0,
            "net_pnl_u": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "max_closed_dd_pct": 0.0,
            "avg_pnl_u": 0.0,
            "median_pnl_u": 0.0,
            "avg_hold_minutes": 0.0,
            "target_rate_pct": 0.0,
            "stop_rate_pct": 0.0,
        }
    ordered = trades.sort_values("exit_time")
    pnl = ordered["net_pnl"].astype(float)
    equity = 1000.0 + pnl.cumsum()
    running_high = np.maximum.accumulate(np.r_[1000.0, equity.to_numpy()])
    equity_with_initial = np.r_[1000.0, equity.to_numpy()]
    drawdown = 1.0 - equity_with_initial / running_high
    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl <= 0].sum())
    return {
        "trades": int(len(trades)),
        "symbols": int(trades["symbol"].nunique()),
        "net_pnl_u": float(pnl.sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100.0),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else None,
        "max_closed_dd_pct": float(drawdown.max() * 100.0),
        "avg_pnl_u": float(pnl.mean()),
        "median_pnl_u": float(pnl.median()),
        "avg_hold_minutes": float(trades["hold_minutes"].mean()),
        "target_rate_pct": float((trades["exit_reason"] == "TARGET").mean() * 100.0),
        "stop_rate_pct": float((trades["exit_reason"] == "STOP").mean() * 100.0),
    }


def passes_filters(candidate: dict[str, Any], params: StrategyParams) -> bool:
    if candidate["confirm"] != params.confirm:
        return False
    if int(candidate["listing_age_days"]) > params.max_listing_age_days:
        return False
    if abs(float(candidate["session_drawdown_pct"])) < params.min_drawdown_pct:
        return False
    if params.require_prev_green and not bool(candidate["prev_green"]):
        return False
    if float(candidate["signal_close"]) >= float(candidate["day_open"]):
        return False
    return True


def build_param_grid() -> list[StrategyParams]:
    grid: list[StrategyParams] = []
    for confirm in ["close_break", "wick_break"]:
        for max_age in [7, 14, 30]:
            for min_dd in [0.03, 0.05, 0.08]:
                for prev_green in [False, True]:
                    for sl_pct, tp_pct in [(0.03, 0.06), (0.05, 0.10), (0.08, 0.12), (0.05, 0.15)]:
                        for hold in [1440, 2880]:
                            grid.append(
                                StrategyParams(
                                    confirm=confirm,
                                    max_listing_age_days=max_age,
                                    min_drawdown_pct=min_dd,
                                    require_prev_green=prev_green,
                                    sl_pct=sl_pct,
                                    tp_pct=tp_pct,
                                    max_hold_minutes=hold,
                                )
                            )
    return grid


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = month_start(args.start_month)
    end = month_after(args.end_month)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intraday_dir = Path(args.cache_dir) / "intraday"
    daily_dir = Path(args.cache_dir) / "daily"
    onboard_dates, onboard_error = fetch_onboard_dates()
    download_audit: list[dict[str, Any]] = []
    if args.download_new_listings:
        download_audit = download_new_listing_archives(
            onboard_dates,
            cache_dir=Path(args.cache_dir),
            start=start,
            end=end,
            max_listing_age_days=30,
        )
        pd.DataFrame(download_audit).to_csv(output_dir / "download_audit.csv", index=False)
    symbol_paths = discover_symbols(intraday_dir)

    all_candidates: list[dict[str, Any]] = []
    symbol_audit: list[dict[str, Any]] = []
    max_collect_age = 30
    for index, (symbol, intraday_paths) in enumerate(symbol_paths.items(), start=1):
        daily_paths = daily_paths_for(daily_dir, symbol)
        daily = load_candles(daily_paths)
        listing_date, listing_source = listing_date_for(symbol, daily, onboard_dates, start)
        if listing_date is None:
            symbol_audit.append(
                {
                    "symbol": symbol,
                    "status": "skipped",
                    "reason": listing_source,
                    "intraday_files": len(intraday_paths),
                    "daily_files": len(daily_paths),
                }
            )
            continue
        if listing_date >= end or listing_date < start - pd.Timedelta(days=max_collect_age + 2):
            symbol_audit.append(
                {
                    "symbol": symbol,
                    "status": "skipped",
                    "reason": "listing_outside_research_window",
                    "listing_date": str(listing_date),
                    "listing_source": listing_source,
                    "intraday_files": len(intraday_paths),
                    "daily_files": len(daily_paths),
                }
            )
            continue
        intraday = load_candles(intraday_paths)
        candidates = generate_candidates_for_symbol(
            symbol,
            intraday,
            daily,
            listing_date,
            listing_source,
            start,
            end,
            max_age_to_collect=max_collect_age,
        )
        all_candidates.extend(candidates)
        symbol_audit.append(
            {
                "symbol": symbol,
                "status": "used" if candidates else "no_signal",
                "listing_date": str(listing_date),
                "listing_source": listing_source,
                "intraday_files": len(intraday_paths),
                "daily_files": len(daily_paths),
                "first_intraday": str(intraday.index.min()) if not intraday.empty else None,
                "last_intraday": str(intraday.index.max()) if not intraday.empty else None,
                "candidates": len(candidates),
            }
        )
        if index % 25 == 0:
            print(f"scanned {index}/{len(symbol_paths)} symbols; candidates={len(all_candidates)}", flush=True)

    candidates_frame = pd.DataFrame(all_candidates)
    for candidate_id, candidate in enumerate(all_candidates):
        candidate["candidate_id"] = candidate_id
    candidates_frame = pd.DataFrame(all_candidates)
    if not candidates_frame.empty:
        candidates_frame.to_csv(output_dir / "signal_candidates.csv", index=False)
    pd.DataFrame(symbol_audit).to_csv(output_dir / "symbol_audit.csv", index=False)

    params_grid = build_param_grid()
    frames_cache: dict[str, pd.DataFrame] = {}
    exit_settings = sorted({(params.sl_pct, params.tp_pct, params.max_hold_minutes) for params in params_grid})
    priced_by_exit: dict[tuple[float, float, int], dict[int, dict[str, Any]]] = {setting: {} for setting in exit_settings}
    for index, candidate in enumerate(all_candidates, start=1):
        symbol = str(candidate["symbol"])
        if symbol not in frames_cache:
            frames_cache[symbol] = load_candles(symbol_paths[symbol])
        for sl_pct, tp_pct, hold_minutes in exit_settings:
            placeholder = StrategyParams(
                confirm=str(candidate["confirm"]),
                max_listing_age_days=30,
                min_drawdown_pct=0.0,
                require_prev_green=False,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                max_hold_minutes=hold_minutes,
            )
            trade = simulate_trade(frames_cache[symbol], candidate, placeholder)
            if trade is not None:
                priced_by_exit[(sl_pct, tp_pct, hold_minutes)][int(candidate["candidate_id"])] = trade
        if index % 250 == 0:
            print(f"priced {index}/{len(all_candidates)} candidates", flush=True)

    all_trades: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    for params in params_grid:
        selected = [row for row in all_candidates if passes_filters(row, params)]
        selected = sorted(selected, key=lambda row: row["entry_time"])
        variant_trades: list[dict[str, Any]] = []
        last_exit_by_symbol: dict[str, pd.Timestamp] = {}
        priced_exit = priced_by_exit[(params.sl_pct, params.tp_pct, params.max_hold_minutes)]
        for candidate in selected:
            symbol = str(candidate["symbol"])
            entry_time = candidate["entry_time"]
            if entry_time < last_exit_by_symbol.get(symbol, pd.Timestamp.min.tz_localize("UTC")):
                continue
            base_trade = priced_exit.get(int(candidate["candidate_id"]))
            if base_trade is None:
                continue
            trade = {
                **base_trade,
                "param_key": params.key(),
                **asdict(params),
            }
            variant_trades.append(trade)
            last_exit_by_symbol[symbol] = trade["exit_time"]
        trades_frame = pd.DataFrame(variant_trades)
        row = {"param_key": params.key(), **asdict(params), **metrics(trades_frame)}
        sweep_rows.append(row)
        all_trades.extend(variant_trades)

    trades = pd.DataFrame(all_trades)
    sweep = pd.DataFrame(sweep_rows)
    if not trades.empty:
        trades.to_csv(output_dir / "all_variant_trades.csv", index=False)
    if not sweep.empty:
        sweep = sweep.sort_values(["net_pnl_u", "profit_factor", "trades"], ascending=[False, False, False])
        sweep.to_csv(output_dir / "parameter_sweep.csv", index=False)
        viable = sweep[sweep["trades"] >= max(3, int(args.min_trades_for_best))]
        best_row = viable.iloc[0].to_dict() if not viable.empty else sweep.iloc[0].to_dict()
        best_trades = trades[trades["param_key"] == best_row["param_key"]].sort_values("entry_time")
        best_trades.to_csv(output_dir / "best_variant_trades.csv", index=False)
    else:
        best_row = {}
        best_trades = pd.DataFrame()

    feature_summary = pd.DataFrame()
    if not best_trades.empty:
        feature_summary = (
            best_trades.assign(winner=best_trades["net_pnl"] > 0)
            .groupby("winner")
            .agg(
                trades=("symbol", "count"),
                median_age_days=("listing_age_days", "median"),
                median_prev_return_pct=("prev_return_pct", "median"),
                median_signal_day_return_pct=("signal_day_return_pct", "median"),
                median_session_drawdown_pct=("session_drawdown_pct", "median"),
                median_break_depth_pct=("break_depth_pct", "median"),
                net_pnl_u=("net_pnl", "sum"),
            )
            .reset_index()
        )
        feature_summary.to_csv(output_dir / "best_variant_feature_summary.csv", index=False)

    audit = {
        "strategy": "new_listing_prior_day_low_break_short",
        "start_month": args.start_month,
        "end_month": args.end_month,
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "intraday_symbols": len(symbol_paths),
        "official_onboard_dates": len(onboard_dates),
        "onboard_error": onboard_error,
        "download_new_listings": bool(args.download_new_listings),
        "download_attempts": len(download_audit),
        "candidate_rows": int(len(candidates_frame)),
        "sweep_variants": int(len(params_grid)),
        "fee_rate_per_side": FEE_RATE,
        "slippage_rate_per_side": SLIPPAGE_RATE,
        "notional_per_trade_u": 1000.0,
        "best": best_row,
        "notes": [
            "Entry is next 1-minute open after the first qualifying break, not the intrabar low.",
            "If stop and target occur in the same candle, the adverse stop is applied first.",
            "PnL is per 1000U signal notional and does not enforce cross-symbol margin or exchange funding fees.",
            "Official Binance exchangeInfo onboardDate is used when available; otherwise local first daily candle is used only when it starts after the research start.",
        ],
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(output_dir, audit, sweep, best_trades, feature_summary)
    print(json.dumps({"output_dir": str(output_dir.resolve()), "best": best_row}, ensure_ascii=False, indent=2, default=str))
    return audit


def fmt_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def write_report(
    output_dir: Path,
    audit: dict[str, Any],
    sweep: pd.DataFrame,
    best_trades: pd.DataFrame,
    feature_summary: pd.DataFrame,
) -> None:
    best = audit.get("best", {}) or {}
    lines = [
        "# New Listing Prior-Day-Low Break Short Backtest",
        "",
        "## TL;DR",
        "",
    ]
    if best:
        pf = best.get("profit_factor")
        pf_text = "NA" if pf is None or pd.isna(pf) else f"{float(pf):.2f}"
        lines.extend(
            [
                f"- Best tested variant: `{best.get('param_key')}`",
                f"- Trades: {int(best.get('trades', 0))}, symbols: {int(best.get('symbols', 0))}",
                f"- Net PnL on 1000U per signal: {float(best.get('net_pnl_u', 0.0)):.2f}U",
                f"- Win rate: {float(best.get('win_rate_pct', 0.0)):.2f}%, PF: {pf_text}, closed-trade DD: {float(best.get('max_closed_dd_pct', 0.0)):.2f}%",
            ]
        )
    else:
        lines.append("- No qualified trades were found.")
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Universe: cached Binance USD-M 1m symbols with matching local daily candles.",
            "- New listing date: Binance `exchangeInfo.onboardDate` when available; otherwise local first daily candle fallback only if it starts after the research window begins.",
            "- Signal: after listing day 1, first minute where today's low/close breaks the previous daily low, with the signal-day close below day open and a minimum pullback from the session high.",
            "- Fill model: short at next minute open, 0.05% fee per side, 0.02% slippage per side, adverse stop first inside a candle.",
            "",
            "## Data Quality",
            "",
            f"- Intraday symbols scanned: {audit['intraday_symbols']}",
            f"- Candidate rows before parameter filters: {audit['candidate_rows']}",
            f"- Parameter variants tested: {audit['sweep_variants']}",
            f"- Official onboard metadata rows: {audit['official_onboard_dates']}",
        ]
    )
    if audit.get("onboard_error"):
        lines.append(f"- Onboard metadata warning: {audit['onboard_error']}")
    if not sweep.empty:
        lines.extend(["", "## Top 10 Variants", ""])
        top = sweep.head(10).copy()
        columns = ["param_key", "trades", "symbols", "net_pnl_u", "win_rate_pct", "profit_factor", "max_closed_dd_pct"]
        lines.append(top[columns].to_markdown(index=False))
    if not feature_summary.empty:
        lines.extend(["", "## Winner vs Loser Feature Summary", ""])
        lines.append(feature_summary.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is an exploratory, in-sample research backtest, not a live-trading guarantee.",
            "- The cached Jan-Feb 2026 universe may miss delisted listings, later listings, or symbols without local 1m history.",
            "- Funding, partial fills, live order-book slippage, liquidation risk, and cross-position margin pressure are not modeled.",
            "- Because variants are searched on the same sample, the best row is likely optimistic and needs out-of-sample validation.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest new-listing prior-day-low-break short strategy")
    parser.add_argument("--start-month", default="2026-01")
    parser.add_argument("--end-month", default="2026-02")
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "top_gainers_boll_short_1m"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "new_listing_breakdown_short_20260917"))
    parser.add_argument("--min-trades-for-best", type=int, default=5)
    parser.add_argument("--download-new-listings", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
