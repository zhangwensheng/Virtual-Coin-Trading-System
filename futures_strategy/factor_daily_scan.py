from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from futures_strategy.binance_factors import build_factor_dataset, ensure_utc_timestamp, save_factor_dataset
from futures_strategy.config import AppConfig, load_config
from futures_strategy.data import load_market_data
from futures_strategy.exchanges import to_linear_symbol
from futures_strategy.factor_portfolio import candidate_rank_score
from futures_strategy.strategy import latest_signal_snapshot, prepare_market_data


@dataclass
class DailyScanResult:
    scan_time: pd.Timestamp
    rows: pd.DataFrame


def parse_config_paths(config_dir: str | None = None, configs: str | None = None) -> list[Path]:
    if configs:
        return [Path(item.strip()) for item in configs.split(",") if item.strip()]
    if config_dir:
        return sorted(Path(config_dir).glob("*.yaml"))
    raise ValueError("Please provide --config-dir or --configs.")


def action_priority(action: str) -> int:
    priorities = {
        "FUNDING_OI_BEAR_SHORT_SETUP": 3,
        "FUNDING_OI_BEAR_SHORT_BIAS": 2,
        "NO_TRADE_ZONE": 1,
    }
    return priorities.get(action, 0)


def rank_scan_results(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    ranked = rows.copy()
    ranked = ranked.sort_values(
        [
            "action_priority",
            "rank_score",
            "factor_score",
            "risk_multiplier",
            "oi_value_change_pct",
            "funding_rate",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    ranked["portfolio_rank"] = range(1, len(ranked) + 1)
    return ranked


def refresh_factor_csv(
    config: AppConfig,
    *,
    output_dir: Path,
    days_back: int,
    end_time: str | None,
    session: requests.Session,
) -> Path:
    native_symbol = to_linear_symbol(config.exchange.symbol)
    dataset = build_factor_dataset(
        symbol=native_symbol,
        interval=config.exchange.timeframe,
        days_back=days_back,
        end_time=end_time,
        session=session,
    )
    end_label = dataset.index[-1].strftime("%Y-%m-%d")
    output_path = output_dir / f"{native_symbol.lower()}_{config.exchange.timeframe}_{end_label}_factors.csv"
    return save_factor_dataset(dataset, output_path)


def scan_one_config(
    config_path: Path,
    *,
    refresh_data: bool,
    cache_dir: Path,
    days_back: int,
    end_time: str | None,
    session: requests.Session,
) -> dict[str, Any]:
    config = load_config(config_path)
    if refresh_data:
        csv_path = refresh_factor_csv(
            config,
            output_dir=cache_dir,
            days_back=days_back,
            end_time=end_time,
            session=session,
        )
        config.exchange.csv_path = str(csv_path)

    raw = load_market_data(config.exchange)
    prepared = prepare_market_data(raw, config.strategy)
    if prepared.empty:
        raise ValueError(f"No prepared rows for {config.exchange.symbol}")

    signal = latest_signal_snapshot(prepared, config.strategy)
    latest_row = prepared.iloc[-1]
    timestamp = prepared.index[-1]
    action = str(signal.get("action", "UNKNOWN"))
    row = {
        "symbol": config.exchange.symbol,
        "native_symbol": to_linear_symbol(config.exchange.symbol),
        "config_path": str(config_path),
        "csv_path": str(config.exchange.csv_path or ""),
        "timestamp": timestamp.isoformat(),
        "action": action,
        "action_priority": action_priority(action),
        "ready_to_short": action == "FUNDING_OI_BEAR_SHORT_SETUP",
        "has_short_bias": action in {"FUNDING_OI_BEAR_SHORT_SETUP", "FUNDING_OI_BEAR_SHORT_BIAS"},
        "rank_score": round(float(candidate_rank_score(latest_row)), 4),
        "factor_score": int(signal.get("factor_score", 0)),
        "risk_multiplier": float(signal.get("risk_multiplier", 1.0)),
        "close": float(signal.get("close", latest_row["close"])),
        "ema_entry": float(signal.get("ema_entry", latest_row.get("ema_entry", latest_row["close"]))),
        "atr": float(signal.get("atr", latest_row.get("atr", 0.0))),
        "rsi": float(signal.get("rsi", latest_row.get("rsi", 0.0))),
        "htf_adx": float(signal.get("htf_adx", latest_row.get("htf_adx", 0.0))),
        "funding_rate": float(signal.get("funding_rate", latest_row.get("funding_rate", 0.0))),
        "funding_change": float(signal.get("funding_change", latest_row.get("funding_change", 0.0))),
        "oi_value_change_pct": float(signal.get("oi_value_change_pct", 0.0)),
        "price_change_pct": float(signal.get("price_change_pct", 0.0)),
        "bounce_pct": float(signal.get("bounce_pct", 0.0)),
        "stop_hint": signal.get("stop_hint"),
    }
    return row


def run_daily_factor_scan(
    config_paths: list[str | Path],
    *,
    refresh_data: bool = True,
    cache_dir: str | Path = "data/binance_factor_bundle",
    days_back: int = 27,
    end_time: str | None = None,
) -> DailyScanResult:
    session = requests.Session()
    scan_time = ensure_utc_timestamp(end_time).floor("h") if end_time else pd.Timestamp.now(tz="UTC").floor("h")
    rows: list[dict[str, Any]] = []
    resolved_cache_dir = Path(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in config_paths:
        row = scan_one_config(
            Path(raw_path),
            refresh_data=refresh_data,
            cache_dir=resolved_cache_dir,
            days_back=days_back,
            end_time=end_time,
            session=session,
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    return DailyScanResult(scan_time=scan_time, rows=rank_scan_results(frame))


def render_daily_scan_report(result: DailyScanResult) -> str:
    rows = result.rows
    lines = [
        "# 每日空头筛币报告",
        "",
        f"- 扫描时间(UTC): `{result.scan_time.isoformat()}`",
        f"- 候选数量: `{len(rows)}`",
        "",
        "## 当下可开空",
        "",
    ]
    ready = rows[rows["ready_to_short"]] if not rows.empty else rows
    if ready.empty:
        lines.append("- 当前没有满足直接开空条件的币。")
    else:
        for _, row in ready.iterrows():
            lines.append(
                f"- `{row['symbol']}` rank=`{row['portfolio_rank']}` score=`{row['rank_score']}` "
                f"factor=`{row['factor_score']}` risk=`{row['risk_multiplier']}` "
                f"funding=`{row['funding_rate']}` oi_change=`{row['oi_value_change_pct']}%` "
                f"stop=`{row['stop_hint']}`"
            )

    lines.extend(["", "## 观察名单", ""])
    if rows.empty:
        lines.append("- 没有可用结果。")
    else:
        for _, row in rows.iterrows():
            lines.append(
                f"- `{row['symbol']}` action=`{row['action']}` rank=`{row['portfolio_rank']}` "
                f"score=`{row['rank_score']}` factor=`{row['factor_score']}` "
                f"funding=`{row['funding_rate']}` oi_change=`{row['oi_value_change_pct']}%` "
                f"bounce=`{row['bounce_pct']}%`"
            )

    return "\n".join(lines) + "\n"
