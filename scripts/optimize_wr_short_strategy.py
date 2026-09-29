"""Sweep Williams %R variants for the top-gainers short strategy.

This is an offline research script.  It reads cached Binance kline CSVs and
does not touch the running paper account, SQLite ledger, or live order path.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/top_gainers_boll_short_1m/realtime_enriched"
DAILY_DIR = ROOT / "data/top_gainers_boll_short_1m/daily"
OUT = ROOT / "outputs/wr_short_optimization_20260911"

FEE = 0.0005
SLIP = 0.0002
INITIAL_EQUITY = 1000.0
RESERVE = 200.0
MAX_POSITIONS = 2
POSITION_NOTIONAL = 320.0
MAX_GROSS = 640.0
DAILY_LOSS_LIMIT = 10.0
DAILY_COIN_LIMIT = 3
COOLDOWN_MINUTES = 30
MAX_HOLD_MINUTES = 30


@dataclass(frozen=True)
class Params:
    daily_period: int
    daily_wr_min: float
    minute_period: int
    pump_wr_min: float
    reversal_wr_max: float
    min_volume_ratio: float
    min_pump_body_pct: float
    max_breakdown_body_pct: float
    block_late_utc: bool

    def key(self) -> str:
        return (
            f"d{self.daily_period}_dw{self.daily_wr_min:g}_m{self.minute_period}"
            f"_p{self.pump_wr_min:g}_r{self.reversal_wr_max:g}_v{self.min_volume_ratio:g}"
            f"_pb{self.min_pump_body_pct:g}_bb{self.max_breakdown_body_pct:g}"
            f"_{'noLate' if self.block_late_utc else 'allHours'}"
        ).replace("-", "n").replace(".", "p")


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    width = highest - lowest
    return -100.0 * (highest - close) / width.replace(0.0, np.nan)


def previous_daily_range(daily: pd.DataFrame, period: int) -> pd.DataFrame:
    day = daily.copy().sort_index()
    highest = day["high"].rolling(period, min_periods=period).max().shift(1)
    lowest = day["low"].rolling(period, min_periods=period).min().shift(1)
    return pd.DataFrame(
        {
            "trade_date": day.index.normalize(),
            f"daily_highest_{period}": highest,
            f"daily_lowest_{period}": lowest,
        }
    ).dropna()


def stats(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "net_pnl": 0.0,
            "avg_trade": 0.0,
            "profit_factor": None,
            "max_drawdown_pct": 0.0,
            "months_profitable": 0,
        }
    pnl = frame["net_pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_loss = float(-losses.sum())
    by_month = frame.groupby(frame["entry_time"].dt.strftime("%Y-%m"))["net_pnl"].sum()
    return {
        "trades": int(len(frame)),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl <= 0).sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100.0),
        "net_pnl": float(pnl.sum()),
        "avg_trade": float(pnl.mean()),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else None,
        "max_drawdown_pct": float(frame.attrs.get("max_drawdown_pct", 0.0)),
        "months_profitable": int((by_month > 0).sum()),
    }


def load_symbol_frame(path: Path, daily_periods: set[int]) -> pd.DataFrame | None:
    symbol = path.name.split("_1m_")[0].upper()
    raw_files = sorted(DATA_DIR.glob(f"{symbol.lower()}_1m_*.csv"))
    frame = pd.concat([pd.read_csv(file) for file in raw_files], ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    if frame.empty:
        return None
    for column in frame.columns:
        if column not in {"symbol", "selected_day"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["symbol"] = symbol
    frame["trade_date"] = frame.index.normalize()

    daily_files = sorted(DAILY_DIR.glob(f"{symbol.lower()}_1d_*.csv"))
    if not daily_files:
        return None
    daily = pd.concat([pd.read_csv(file) for file in daily_files], ignore_index=True)
    daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)
    daily = daily.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    for column in ["open", "high", "low", "close", "volume"]:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    for period in daily_periods:
        lookup = previous_daily_range(daily, period)
        frame = frame.reset_index().merge(lookup, on="trade_date", how="left").set_index("timestamp")
        high = frame[f"daily_highest_{period}"]
        low = frame[f"daily_lowest_{period}"]
        frame[f"daily_wr_{period}"] = -100.0 * (high - frame["close"]) / (high - low).replace(0.0, np.nan)
    return frame.sort_index()


def make_unit_trade(
    symbol: str,
    frame: pd.DataFrame,
    breakdown_pos: int,
    pump_pos: int,
    reversal_pos: int,
    volume_ratio: float,
    params: Params,
) -> dict[str, Any] | None:
    if breakdown_pos + 1 >= len(frame):
        return None
    previous = frame.iloc[breakdown_pos - 14 : breakdown_pos]
    if len(previous) != 14:
        return None
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)
    atr = float(true_range.iloc[breakdown_pos - 14 : breakdown_pos].mean())
    if not np.isfinite(atr) or atr <= 0:
        return None

    breakdown = frame.iloc[breakdown_pos]
    entry_time = frame.index[breakdown_pos] + pd.Timedelta(minutes=1)
    entry = float(breakdown["close"]) * (1.0 - SLIP)
    peak = float(frame["high"].iloc[pump_pos : breakdown_pos + 1].max())
    stop = peak + max(0.2 * atr, entry * 0.0002)
    risk_distance = stop - entry
    target = entry - 2.0 * risk_distance
    if risk_distance <= 0 or target <= 0:
        return None

    exit_time = frame.index[min(breakdown_pos + MAX_HOLD_MINUTES, len(frame) - 1)]
    exit_price = float(frame.loc[exit_time, "close"]) * (1.0 + SLIP)
    exit_reason = "TIME_EXIT"
    trailing = False
    lowest_buy = entry
    live_stop = stop
    for pos in range(breakdown_pos + 1, min(len(frame), breakdown_pos + 1 + MAX_HOLD_MINUTES)):
        row = frame.iloc[pos]
        high = float(row["high"])
        low = float(row["low"])
        current_time = frame.index[pos]
        if high >= live_stop:
            exit_time = current_time
            exit_price = live_stop * (1.0 + SLIP)
            exit_reason = "STOP" if not trailing else "TRAIL_STOP"
            break
        if low <= target:
            exit_time = current_time
            exit_price = target * (1.0 + SLIP)
            exit_reason = "TARGET"
            break
        if low <= entry - risk_distance:
            trailing = True
        if trailing:
            lowest_buy = min(lowest_buy, low * (1.0 + SLIP))
            live_stop = min(live_stop, lowest_buy + risk_distance)

    unit_price_pnl = entry - exit_price
    return {
        "symbol": symbol,
        "param_key": params.key(),
        "entry_time": entry_time,
        "breakdown_time": frame.index[breakdown_pos],
        "entry_price": entry,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "stop_price": stop,
        "target_price": target,
        "risk_distance": risk_distance,
        "unit_risk": stop * (1.0 + SLIP) - entry + FEE * (entry + stop * (1.0 + SLIP)),
        "unit_price_pnl": unit_price_pnl,
        "peak": peak,
        "atr": atr,
        "volume_ratio": volume_ratio,
        "daily_rank": float(breakdown["daily_rank"]),
        "daily_return_pct": float(breakdown["daily_return_pct"]),
        "daily_wr": float(breakdown[f"daily_wr_{params.daily_period}"]),
        "minute_wr": float(breakdown[f"wr_{params.minute_period}"]),
        "pump_body_pct": float((frame["close"].iloc[pump_pos] / frame["open"].iloc[pump_pos] - 1.0) * 100.0),
        "breakdown_body_pct": float((breakdown["open"] - breakdown["close"]) / breakdown["open"] * 100.0),
        "pump_to_entry_minutes": int(breakdown_pos - pump_pos + 1),
        "reversal_to_breakdown_minutes": int(breakdown_pos - reversal_pos),
    }


def detect_symbol_trades(frame: pd.DataFrame, params: Params) -> list[dict[str, Any]]:
    symbol = str(frame["symbol"].iloc[0])
    data = frame.copy()
    wr_col = f"wr_{params.minute_period}"
    data[wr_col] = williams_r(data["high"], data["low"], data["close"], params.minute_period)
    prev_vol = data["volume"].shift(1).rolling(20, min_periods=20).mean()
    data["volume_ratio_wr"] = data["volume"] / prev_vol.replace(0.0, np.nan)
    daily_wr_col = f"daily_wr_{params.daily_period}"

    trades: list[dict[str, Any]] = []
    state = "WAIT_PUMP"
    pump_pos: int | None = None
    pump_high: float | None = None
    reversal_pos: int | None = None
    reversal_low: float | None = None
    peak_through_reversal: float | None = None
    pump_ratio: float | None = None

    def clear() -> None:
        nonlocal state, pump_pos, pump_high, reversal_pos, reversal_low, peak_through_reversal, pump_ratio
        state = "WAIT_PUMP"
        pump_pos = None
        pump_high = None
        reversal_pos = None
        reversal_low = None
        peak_through_reversal = None
        pump_ratio = None

    def begin(position: int, ratio: float) -> None:
        nonlocal state, pump_pos, pump_high, reversal_pos, reversal_low, peak_through_reversal, pump_ratio
        state = "WAIT_REVERSAL"
        pump_pos = position
        pump_high = float(data["high"].iloc[position])
        reversal_pos = None
        reversal_low = None
        peak_through_reversal = None
        pump_ratio = ratio

    for position, (_, row) in enumerate(data.iterrows()):
        if position < max(params.minute_period, 35):
            continue
        if params.block_late_utc and data.index[position].hour >= 14:
            clear()
            continue
        minute_wr = row[wr_col]
        daily_wr = row[daily_wr_col]
        ratio = row["volume_ratio_wr"]
        body = (row["close"] / row["open"] - 1.0) * 100.0
        eligible = (
            np.isfinite(minute_wr)
            and np.isfinite(daily_wr)
            and np.isfinite(ratio)
            and row["daily_rank"] <= 5
            and row["daily_return_pct"] >= 0.15
            and daily_wr >= params.daily_wr_min
        )
        pump_now = bool(
            eligible
            and row["close"] > row["open"]
            and minute_wr >= params.pump_wr_min
            and ratio >= params.min_volume_ratio
            and body >= params.min_pump_body_pct
        )

        if state == "WAIT_PUMP":
            if pump_now:
                begin(position, float(ratio))
            continue

        if pump_pos is None or data.index[position].normalize() != data.index[pump_pos].normalize():
            clear()
            if pump_now:
                begin(position, float(ratio))
            continue

        if state == "WAIT_REVERSAL":
            distance = position - pump_pos
            if distance > 5 or (pump_high is not None and row["high"] > pump_high):
                clear()
                if pump_now:
                    begin(position, float(ratio))
                continue
            if distance >= 1 and row["close"] < row["open"] and minute_wr <= params.reversal_wr_max:
                state = "WAIT_BREAKDOWN"
                reversal_pos = position
                reversal_low = float(row["low"])
                peak_through_reversal = float(data["high"].iloc[pump_pos : position + 1].max())
            continue

        if reversal_pos is None or reversal_low is None or peak_through_reversal is None or pump_ratio is None:
            clear()
            continue
        distance = position - reversal_pos
        if distance > 3 or row["high"] > peak_through_reversal:
            clear()
            if pump_now:
                begin(position, float(ratio))
            continue
        breakdown_body = (row["open"] - row["close"]) / row["open"] * 100.0
        if (
            distance >= 1
            and row["close"] < reversal_low
            and row["close"] < row["open"]
            and breakdown_body <= params.max_breakdown_body_pct
        ):
            trade = make_unit_trade(symbol, data, position, pump_pos, reversal_pos, pump_ratio, params)
            if trade is not None:
                trades.append(trade)
            clear()
    return trades


def simulate_portfolio(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals
    rows = signals.sort_values(["entry_time", "daily_rank", "symbol"]).to_dict("records")
    cash = INITIAL_EQUITY
    peak = INITIAL_EQUITY
    day = None
    day_equity = INITIAL_EQUITY
    risk_paused = False
    open_positions: list[dict[str, Any]] = []
    closed_rows: list[dict[str, Any]] = []
    counts: dict[tuple[str, str], int] = {}
    cooldowns: dict[str, pd.Timestamp] = {}

    def close_due(now: pd.Timestamp) -> None:
        nonlocal cash, peak, day, day_equity, risk_paused
        remaining = []
        for pos in open_positions:
            if pos["exit_time"] <= now:
                cash += pos["price_pnl"] - pos["exit_fee"]
                closed_rows.append(pos)
            else:
                remaining.append(pos)
        open_positions[:] = remaining
        equity = cash
        for pos in open_positions:
            equity += 0.0
        peak = max(peak, equity)
        if day is None or now.normalize() > day:
            day = now.normalize()
            day_equity = equity
            risk_paused = False
        if equity - day_equity <= -DAILY_LOSS_LIMIT:
            risk_paused = True

    for signal in rows:
        now = signal["entry_time"]
        close_due(now)
        if risk_paused:
            continue
        if len(open_positions) >= MAX_POSITIONS:
            continue
        if any(pos["symbol"] == signal["symbol"] for pos in open_positions):
            continue
        if now < cooldowns.get(signal["symbol"], pd.Timestamp(0, unit="s", tz="UTC")):
            continue
        count_key = (str(now.date()), signal["symbol"])
        if counts.get(count_key, 0) >= DAILY_COIN_LIMIT:
            continue
        gross = sum(pos["gross_notional"] for pos in open_positions)
        available = min(cash, cash) - gross
        notional_cap = min(POSITION_NOTIONAL, MAX_GROSS - gross, available - RESERVE)
        if notional_cap <= 1.0 or signal["unit_risk"] <= 0:
            continue
        risk = min(3.0, cash * 0.003)
        qty = min(risk / signal["unit_risk"], notional_cap / signal["entry_price"])
        if qty <= 0:
            continue
        entry_fee = qty * signal["entry_price"] * FEE
        exit_fee = qty * signal["exit_price"] * FEE
        price_pnl = qty * signal["unit_price_pnl"]
        net_pnl = price_pnl - entry_fee - exit_fee
        cash -= entry_fee
        pos = dict(signal)
        pos.update(
            qty=qty,
            gross_notional=qty * signal["entry_price"],
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            price_pnl=price_pnl,
            planned_risk=qty * signal["unit_risk"],
            net_pnl=net_pnl,
        )
        open_positions.append(pos)
        counts[count_key] = counts.get(count_key, 0) + 1
        cooldowns[signal["symbol"]] = signal["exit_time"] + pd.Timedelta(minutes=COOLDOWN_MINUTES)

    close_due(pd.Timestamp.max.tz_localize("UTC") - pd.Timedelta(days=1))
    result = pd.DataFrame(closed_rows)
    if not result.empty:
        equity = INITIAL_EQUITY + result.sort_values("exit_time")["net_pnl"].cumsum()
        drawdown = 1.0 - equity / equity.cummax()
        result.attrs["max_drawdown_pct"] = float(drawdown.max() * 100.0)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params_grid = [
        Params(*values)
        for values in itertools.product(
            [14, 20],
            [-10.0, -15.0],
            [14, 20],
            [-5.0, -10.0],
            [-20.0, -30.0],
            [3.0],
            [0.65],
            [0.7],
            [True],
        )
    ]
    daily_periods = {p.daily_period for p in params_grid}
    paths = sorted(DATA_DIR.glob("*_1m_*.csv"))
    # Each symbol can have split month files; use the first path per symbol.
    unique_paths = []
    seen: set[str] = set()
    for path in paths:
        symbol = path.name.split("_1m_")[0]
        if symbol not in seen:
            unique_paths.append(path)
            seen.add(symbol)

    all_signals: dict[str, list[dict[str, Any]]] = {p.key(): [] for p in params_grid}
    for idx, path in enumerate(unique_paths, start=1):
        frame = load_symbol_frame(path, daily_periods)
        if frame is None or frame.empty:
            continue
        for params in params_grid:
            all_signals[params.key()].extend(detect_symbol_trades(frame, params))
        if idx % 10 == 0:
            print(f"processed {idx}/{len(unique_paths)} symbols", flush=True)

    summary_rows: list[dict[str, Any]] = []
    best_trades: pd.DataFrame | None = None
    best_key = ""
    for params in params_grid:
        key = params.key()
        signal_frame = pd.DataFrame(all_signals[key])
        trades = simulate_portfolio(signal_frame)
        overall = stats(trades)
        jan = stats(trades[trades["entry_time"].dt.strftime("%Y-%m") == "2026-01"] if not trades.empty else trades)
        feb_mar = stats(trades[trades["entry_time"].dt.strftime("%Y-%m").isin(["2026-02", "2026-03"])] if not trades.empty else trades)
        row = {
            **params.__dict__,
            "param_key": key,
            **{f"all_{k}": v for k, v in overall.items()},
            **{f"jan_{k}": v for k, v in jan.items()},
            **{f"feb_mar_{k}": v for k, v in feb_mar.items()},
            "raw_signals": int(len(signal_frame)),
        }
        summary_rows.append(row)
        if (
            overall["trades"] >= 40
            and jan["trades"] >= 12
            and feb_mar["trades"] >= 20
            and (overall["profit_factor"] or 0) > 1.0
            and row["all_net_pnl"] > 0
        ):
            score = row["feb_mar_net_pnl"] + row["jan_net_pnl"] * 0.35 - row["all_max_drawdown_pct"] * 0.5
            if best_trades is None or score > best_trades.attrs.get("score", -1e9):
                trades.attrs["score"] = score
                best_trades = trades
                best_key = key

    summary = pd.DataFrame(summary_rows).sort_values(
        ["feb_mar_net_pnl", "all_profit_factor", "all_net_pnl"], ascending=[False, False, False]
    )
    summary.to_csv(OUT / "wr_param_summary.csv", index=False)
    if best_trades is not None:
        best_trades.to_csv(OUT / "best_trades.csv", index=False)
    top = summary.head(20)
    (OUT / "top20.json").write_text(
        json.dumps(top.to_dict("records"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    audit = {
        "params_tested": len(params_grid),
        "symbols": len(unique_paths),
        "best_key": best_key,
        "output_dir": str(OUT),
        "notes": [
            "Uses cached realtime_enriched symbols, so symbols that never passed the old BOLL cache may be absent.",
            "Historical ranking uses cached minute daily_rank from the prior full-universe builder.",
            "Williams %R daily gate uses current minute close against the previous N daily high/low range.",
            "Portfolio simulation approximates the paper account: 1000U, two positions, 320U per position, 3U risk cap, 30m max hold.",
        ],
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_key": best_key, "top": top.head(5).to_dict("records")}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
