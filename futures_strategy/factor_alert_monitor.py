from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from futures_strategy.factor_daily_scan import DailyScanResult, run_daily_factor_scan


@dataclass
class AlertEvent:
    symbol: str
    timestamp: str
    action: str
    previous_action: str | None
    portfolio_rank: int
    rank_score: float
    factor_score: int
    risk_multiplier: float
    funding_rate: float
    oi_value_change_pct: float
    stop_hint: float | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "action": self.action,
            "previous_action": self.previous_action,
            "portfolio_rank": self.portfolio_rank,
            "rank_score": self.rank_score,
            "factor_score": self.factor_score,
            "risk_multiplier": self.risk_multiplier,
            "funding_rate": self.funding_rate,
            "oi_value_change_pct": self.oi_value_change_pct,
            "stop_hint": self.stop_hint,
            "reason": self.reason,
        }


def state_map_from_scan(scan_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if scan_rows.empty:
        return {}
    indexed = scan_rows.set_index("symbol", drop=False)
    return indexed.to_dict(orient="index")


def detect_alerts(
    previous_state: dict[str, dict[str, Any]],
    current_scan: DailyScanResult,
    *,
    top_n: int = 2,
) -> list[AlertEvent]:
    alerts: list[AlertEvent] = []
    if current_scan.rows.empty:
        return alerts

    top_symbols = set(current_scan.rows.head(max(1, top_n))["symbol"].tolist())
    for _, row in current_scan.rows.iterrows():
        symbol = str(row["symbol"])
        action = str(row["action"])
        previous_action = None
        if symbol in previous_state:
            previous_action = str(previous_state[symbol].get("action"))

        if action != "FUNDING_OI_BEAR_SHORT_SETUP":
            continue

        reason: str | None = None
        if previous_action != "FUNDING_OI_BEAR_SHORT_SETUP":
            reason = "action_upgraded_to_setup"
        elif symbol in top_symbols and symbol not in {
            key for key, payload in previous_state.items() if int(payload.get("portfolio_rank", 999)) <= top_n
        }:
            reason = "entered_top_rank_setup"

        if reason is None:
            continue

        alerts.append(
            AlertEvent(
                symbol=symbol,
                timestamp=str(row["timestamp"]),
                action=action,
                previous_action=previous_action,
                portfolio_rank=int(row["portfolio_rank"]),
                rank_score=float(row["rank_score"]),
                factor_score=int(row["factor_score"]),
                risk_multiplier=float(row["risk_multiplier"]),
                funding_rate=float(row["funding_rate"]),
                oi_value_change_pct=float(row["oi_value_change_pct"]),
                stop_hint=None if pd.isna(row["stop_hint"]) else float(row["stop_hint"]),
                reason=reason,
            )
        )
    return alerts


def load_monitor_state(path: str | Path) -> dict[str, dict[str, Any]]:
    state_path = Path(path)
    if not state_path.exists():
        return {}
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    symbols = payload.get("symbols")
    if not isinstance(symbols, dict):
        return {}
    return symbols


def save_monitor_state(path: str | Path, scan: DailyScanResult) -> Path:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scan_time": scan.scan_time.isoformat(),
        "symbols": state_map_from_scan(scan.rows),
    }
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return state_path


def append_alerts_jsonl(path: str | Path, alerts: list[AlertEvent]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not alerts:
        return output_path
    with output_path.open("a", encoding="utf-8") as handle:
        for alert in alerts:
            handle.write(json.dumps(alert.as_dict(), ensure_ascii=False) + "\n")
    return output_path


def save_latest_alerts(path: str | Path, alerts: list[AlertEvent]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [alert.as_dict() for alert in alerts]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def run_alert_scan(
    *,
    config_paths: list[str | Path],
    state_path: str | Path,
    top_n: int = 2,
    refresh_data: bool = True,
    cache_dir: str | Path = "data/binance_factor_bundle",
    days_back: int = 27,
    end_time: str | None = None,
) -> tuple[DailyScanResult, list[AlertEvent]]:
    previous_state = load_monitor_state(state_path)
    scan = run_daily_factor_scan(
        config_paths=config_paths,
        refresh_data=refresh_data,
        cache_dir=cache_dir,
        days_back=days_back,
        end_time=end_time,
    )
    alerts = detect_alerts(previous_state, scan, top_n=top_n)
    save_monitor_state(state_path, scan)
    return scan, alerts
