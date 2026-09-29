"""Append-only audit logs for funding arbitrage runs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd

from .funding_arbitrage import CandidateEvaluation


CANDIDATE_FIELDS = (
    "run_id",
    "timestamp",
    "symbol",
    "accepted",
    "reason",
    "funding_lookback",
    "funding_interval_hours",
    "predicted_funding_rate",
    "round_trip_cost_rate",
    "expected_rebalance_cost_rate",
    "basis_risk_buffer_rate",
    "predicted_net_rate",
    "entry_basis_pct",
    "liquidity_score",
)

ORDER_FIELDS = (
    "run_id",
    "timestamp",
    "group_id",
    "symbol",
    "market",
    "side",
    "reference_price",
    "fill_price",
    "quantity",
    "notional_usdt",
    "fee_usdt",
    "slippage_usdt",
    "status",
    "reason",
)

FUNDING_FIELDS = (
    "run_id",
    "funding_time",
    "group_id",
    "symbol",
    "funding_rate",
    "mark_price",
    "perp_notional_usdt",
    "cashflow_usdt",
    "settlement_key",
    "status",
)

REBALANCE_FIELDS = (
    "run_id",
    "timestamp",
    "group_id",
    "symbol",
    "drift_before_pct",
    "drift_after_pct",
    "quantity_delta",
    "notional_usdt",
    "fee_usdt",
    "slippage_usdt",
)

TRADE_FIELDS = (
    "run_id",
    "group_id",
    "symbol",
    "opened_at",
    "closed_at",
    "holding_hours",
    "initial_notional_usdt",
    "funding_pnl_usdt",
    "gross_price_pnl_usdt",
    "net_pnl_usdt",
    "exit_reason",
)

EQUITY_FIELDS = (
    "run_id",
    "timestamp",
    "equity_usdt",
    "cash_usdt",
    "spot_market_value_usdt",
    "perp_margin_usdt",
    "perp_unrealized_pnl_usdt",
    "gross_price_pnl_usdt",
    "drawdown_pct",
    "open_positions",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        timestamp = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return timestamp.isoformat()
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    return value


class FundingArbitrageLogger:
    def __init__(self, output_dir: Path | str, *, run_id: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def _append_csv(self, filename: str, fields: tuple[str, ...], row: Mapping[str, Any]) -> None:
        path = self.output_dir / filename
        write_header = not path.exists() or path.stat().st_size == 0
        normalized = {field: _json_value(row.get(field, "")) for field in fields}
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(normalized)

    def log_event(
        self,
        *,
        event_type: str,
        timestamp: pd.Timestamp,
        symbol: str | None,
        payload: Mapping[str, Any],
    ) -> str:
        event_id = uuid4().hex
        event = {
            "run_id": self.run_id,
            "event_id": event_id,
            "timestamp": _json_value(timestamp),
            "event_type": event_type,
            "symbol": symbol,
            "payload": _json_value(payload),
        }
        with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return event_id

    def log_candidate(self, evaluation: CandidateEvaluation) -> None:
        candidate = evaluation.candidate
        row = {
            "run_id": self.run_id,
            "timestamp": evaluation.timestamp,
            "symbol": evaluation.symbol,
            "accepted": evaluation.accepted,
            "reason": evaluation.reason or "",
            "funding_lookback": "|".join(f"{rate:.8f}" for rate in evaluation.funding_lookback),
            "funding_interval_hours": candidate.funding_interval_hours if candidate else "",
            "predicted_funding_rate": evaluation.predicted_funding_rate,
            "round_trip_cost_rate": evaluation.round_trip_cost_rate,
            "expected_rebalance_cost_rate": candidate.expected_rebalance_cost_rate if candidate else "",
            "basis_risk_buffer_rate": candidate.basis_risk_buffer_rate if candidate else "",
            "predicted_net_rate": evaluation.predicted_net_rate,
            "entry_basis_pct": candidate.entry_basis_pct if candidate else "",
            "liquidity_score": candidate.liquidity_score if candidate else "",
        }
        self._append_csv("candidates.csv", CANDIDATE_FIELDS, row)
        self.log_event(
            event_type="CANDIDATE_ACCEPTED" if evaluation.accepted else "CANDIDATE_REJECTED",
            timestamp=evaluation.timestamp,
            symbol=evaluation.symbol,
            payload=row,
        )

    def log_order(self, row: Mapping[str, Any]) -> None:
        self._append_csv("orders.csv", ORDER_FIELDS, {"run_id": self.run_id, **row})

    def log_funding(self, row: Mapping[str, Any]) -> None:
        self._append_csv("funding_settlements.csv", FUNDING_FIELDS, {"run_id": self.run_id, **row})

    def log_rebalance(self, row: Mapping[str, Any]) -> None:
        self._append_csv("rebalances.csv", REBALANCE_FIELDS, {"run_id": self.run_id, **row})

    def log_trade(self, row: Mapping[str, Any]) -> None:
        self._append_csv("trades.csv", TRADE_FIELDS, {"run_id": self.run_id, **row})

    def log_equity(self, row: Mapping[str, Any]) -> None:
        self._append_csv("equity.csv", EQUITY_FIELDS, {"run_id": self.run_id, **row})


__all__ = ["FundingArbitrageLogger"]
