from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from futures_strategy.daily_short_gate import DailyShortGateSettings, build_daily_short_gate


BINANCE_FAPI = "https://fapi.binance.com"
HOUR_MS = 60 * 60 * 1000
DEFAULT_UNIVERSE = [
    "1000PEPEUSDT",
    "AAVEUSDT",
    "ADAUSDT",
    "APTUSDT",
    "AVAXUSDT",
    "BCHUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "ETHUSDT",
    "FETUSDT",
    "FILUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "NEARUSDT",
    "SOLUSDT",
    "SUIUSDT",
    "TRXUSDT",
    "UNIUSDT",
    "WLDUSDT",
    "XRPUSDT",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan daily short gate from Binance Futures 1h public klines.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE), help="Comma-separated Binance native symbols.")
    parser.add_argument("--lookback-days", type=int, default=230)
    parser.add_argument("--output-dir", default="outputs/daily_short_gate_live")
    parser.add_argument("--top-symbols", type=int, default=8)
    parser.add_argument("--min-profit-factor", type=float, default=1.05)
    parser.add_argument("--min-trades", type=int, default=2)
    parser.add_argument("--sleep-sec", type=float, default=0.08)
    return parser.parse_args()


def request_json(session: requests.Session, path: str, params: dict[str, Any] | None = None) -> Any:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = session.get(f"{BINANCE_FAPI}{path}", params=params, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Binance request failed: {path} {params or {}}") from last_error


def fetch_1h_history(session: requests.Session, symbol: str, *, lookback_days: int, sleep_sec: float) -> pd.DataFrame:
    server_time = int(request_json(session, "/fapi/v1/time")["serverTime"])
    start_ms = server_time - (lookback_days * 24 * HOUR_MS)
    rows: list[list[Any]] = []
    next_start = start_ms
    while next_start < server_time:
        batch = request_json(
            session,
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": "1h",
                "startTime": int(next_start),
                "endTime": int(server_time),
                "limit": 1500,
            },
        )
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_start = last_open + HOUR_MS
        if len(batch) < 1500:
            break
        time.sleep(sleep_sec)
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
    frame = frame.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame = frame.set_index("timestamp")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["open", "high", "low", "close", "volume"]].dropna()


def main() -> None:
    args = parse_args()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = DailyShortGateSettings(
        top_symbols=max(1, args.top_symbols),
        min_symbol_profit_factor=max(0.1, args.min_profit_factor),
        min_symbol_trades=max(1, args.min_trades),
    )

    session = requests.Session()
    hourly_by_symbol: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for symbol in symbols:
        try:
            frame = fetch_1h_history(session, symbol, lookback_days=args.lookback_days, sleep_sec=args.sleep_sec)
            if not frame.empty:
                hourly_by_symbol[symbol] = frame
        except Exception as exc:  # noqa: BLE001
            errors[symbol] = str(exc)
        time.sleep(args.sleep_sec)

    gate, _ = build_daily_short_gate(hourly_by_symbol, settings)
    allowed = gate.loc[gate["daily_short_allowed"], "symbol"].tolist() if not gate.empty else []
    whitelisted = gate.loc[gate["whitelisted"], "symbol"].tolist() if not gate.empty else []
    generated_at = pd.Timestamp.now(tz="UTC").isoformat()
    payload = {
        "generated_at": generated_at,
        "mode": "DAILY_SHORT_GATE",
        "symbols_requested": symbols,
        "symbols_loaded": sorted(hourly_by_symbol.keys()),
        "allowed_symbols": allowed,
        "whitelisted_symbols": whitelisted,
        "errors": errors,
        "settings": settings.__dict__,
        "rows": gate.to_dict("records") if not gate.empty else [],
    }
    (output_dir / "latest_gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output_dir / "allowed_symbols.txt").write_text("\n".join(allowed), encoding="utf-8")
    if not gate.empty:
        gate.to_csv(output_dir / "gate_rows.csv", index=False)

    print(f"Generated: {generated_at}")
    print(f"Loaded symbols: {len(hourly_by_symbol)}/{len(symbols)}")
    print(f"Whitelisted: {', '.join(whitelisted) if whitelisted else '-'}")
    print(f"Allowed today: {', '.join(allowed) if allowed else '-'}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
