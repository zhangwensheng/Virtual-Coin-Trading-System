"""Pure rules for positive-funding spot/perpetual arbitrage."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Iterable, Sequence

import pandas as pd


def _utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class FundingArbitrageSettings:
    initial_capital_usdt: float = 1000.0
    max_deployable_usdt: float = 800.0
    reserve_usdt: float = 200.0
    universe_size: int = 20
    max_positions: int = 5
    min_position_notional_usdt: float = 50.0
    max_position_notional_usdt: float = 320.0
    absolute_position_cap_usdt: float = 1000.0
    max_drawdown_pct: float = 0.05
    spot_fee_rate: float = 0.001
    perp_fee_rate: float = 0.0005
    slippage_bps: float = 2.0
    emergency_slippage_bps: float = 10.0
    perp_margin_leverage: float = 4.0
    hedge_drift_limit_pct: float = 0.005
    forecast_lookback: int = 3
    forecast_horizon_days: int = 7
    cost_coverage_multiple: float = 2.0
    max_hold_days: int = 7
    max_abs_entry_basis_pct: float = 0.01
    max_basis_widening_pct: float = 0.005
    expected_rebalance_cost_rate: float = 0.0002
    basis_risk_buffer_rate: float = 0.001
    min_observed_funding_interval_hours: float = 1.0
    max_observed_funding_interval_hours: float = 24.0

    def __post_init__(self) -> None:
        positive_values = {
            "initial_capital_usdt": self.initial_capital_usdt,
            "max_deployable_usdt": self.max_deployable_usdt,
            "reserve_usdt": self.reserve_usdt,
            "universe_size": self.universe_size,
            "max_positions": self.max_positions,
            "min_position_notional_usdt": self.min_position_notional_usdt,
            "max_position_notional_usdt": self.max_position_notional_usdt,
            "absolute_position_cap_usdt": self.absolute_position_cap_usdt,
            "perp_margin_leverage": self.perp_margin_leverage,
            "forecast_lookback": self.forecast_lookback,
            "forecast_horizon_days": self.forecast_horizon_days,
            "cost_coverage_multiple": self.cost_coverage_multiple,
            "max_hold_days": self.max_hold_days,
        }
        invalid = [name for name, value in positive_values.items() if float(value) <= 0.0]
        if invalid:
            raise ValueError(f"SETTINGS_MUST_BE_POSITIVE: {','.join(invalid)}")
        if self.max_deployable_usdt + self.reserve_usdt > self.initial_capital_usdt:
            raise ValueError("DEPLOYABLE_PLUS_RESERVE_EXCEEDS_CAPITAL")
        rate_values = (
            self.max_drawdown_pct,
            self.spot_fee_rate,
            self.perp_fee_rate,
            self.slippage_bps,
            self.emergency_slippage_bps,
            self.hedge_drift_limit_pct,
            self.max_abs_entry_basis_pct,
            self.max_basis_widening_pct,
            self.expected_rebalance_cost_rate,
            self.basis_risk_buffer_rate,
        )
        if any(not isfinite(float(value)) or float(value) < 0.0 for value in rate_values):
            raise ValueError("SETTINGS_RATE_INVALID")
        if self.min_position_notional_usdt > self.max_position_notional_usdt:
            raise ValueError("MIN_POSITION_EXCEEDS_MAX_POSITION")
        if self.min_observed_funding_interval_hours > self.max_observed_funding_interval_hours:
            raise ValueError("FUNDING_INTERVAL_BOUNDS_INVALID")


@dataclass(frozen=True)
class FundingObservation:
    symbol: str
    funding_time: pd.Timestamp
    funding_rate: float
    mark_price: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "funding_time", _utc_timestamp(self.funding_time))


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    timestamp: pd.Timestamp
    spot_price: float
    perp_price: float
    spot_quote_volume_24h: float
    perp_quote_volume_24h: float
    spot_tradable: bool
    perp_tradable: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "timestamp", _utc_timestamp(self.timestamp))

    @property
    def basis_pct(self) -> float:
        return self.perp_price / self.spot_price - 1.0

    @property
    def liquidity_score(self) -> float:
        return min(self.spot_quote_volume_24h, self.perp_quote_volume_24h)


@dataclass(frozen=True)
class ArbitrageCandidate:
    symbol: str
    timestamp: pd.Timestamp
    spot_price: float
    perp_price: float
    entry_basis_pct: float
    liquidity_score: float
    funding_lookback: tuple[float, ...]
    funding_interval_hours: float
    predicted_funding_rate: float
    round_trip_cost_rate: float
    expected_rebalance_cost_rate: float
    basis_risk_buffer_rate: float
    predicted_net_rate: float
    score: float


@dataclass(frozen=True)
class CandidateEvaluation:
    symbol: str
    timestamp: pd.Timestamp
    accepted: bool
    reason: str | None
    candidate: ArbitrageCandidate | None
    funding_lookback: tuple[float, ...] = ()
    round_trip_cost_rate: float = 0.0
    predicted_funding_rate: float = 0.0
    predicted_net_rate: float = 0.0


@dataclass(frozen=True)
class AllocationDecision:
    symbol: str
    notional_usdt: float
    score: float
    candidate: ArbitrageCandidate


def round_trip_cost_rate(settings: FundingArbitrageSettings) -> float:
    slippage_rate = settings.slippage_bps / 10_000.0
    return 2.0 * settings.spot_fee_rate + 2.0 * settings.perp_fee_rate + 4.0 * slippage_rate


def infer_funding_interval_hours(observations: Sequence[FundingObservation]) -> float:
    ordered = sorted(observations, key=lambda item: item.funding_time)
    if len(ordered) < 2:
        raise ValueError("INSUFFICIENT_FUNDING_INTERVALS")
    intervals = [
        (current.funding_time - previous.funding_time).total_seconds() / 3600.0
        for previous, current in zip(ordered, ordered[1:])
    ]
    if any(value <= 0.0 for value in intervals):
        raise ValueError("INVALID_FUNDING_INTERVAL")
    return float(median(intervals[-2:]))


def forecast_seven_day_funding_rate(
    observations: Sequence[FundingObservation],
    *,
    lookback: int = 3,
    horizon_days: int = 7,
) -> float:
    recent = sorted(observations, key=lambda item: item.funding_time)[-lookback:]
    if len(recent) != lookback or any(item.funding_rate <= 0.0 for item in recent):
        return 0.0
    conservative_rate = min(
        median(item.funding_rate for item in recent),
        min(item.funding_rate for item in recent),
    )
    interval_hours = infer_funding_interval_hours(recent)
    settlements = int((horizon_days * 24) // interval_hours)
    return float(conservative_rate * settlements)


def _rejected(
    snapshot: MarketSnapshot,
    reason: str,
    *,
    rates: Iterable[float] = (),
    costs: float = 0.0,
    predicted: float = 0.0,
    net: float = 0.0,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        symbol=snapshot.symbol,
        timestamp=snapshot.timestamp,
        accepted=False,
        reason=reason,
        candidate=None,
        funding_lookback=tuple(float(rate) for rate in rates),
        round_trip_cost_rate=costs,
        predicted_funding_rate=predicted,
        predicted_net_rate=net,
    )


def evaluate_candidate(
    snapshot: MarketSnapshot,
    observations: Sequence[FundingObservation],
    settings: FundingArbitrageSettings,
) -> CandidateEvaluation:
    if any(item.funding_time >= snapshot.timestamp for item in observations):
        raise ValueError("FUTURE_FUNDING_OBSERVATION")
    if not snapshot.spot_tradable or not snapshot.perp_tradable:
        return _rejected(snapshot, "MARKET_NOT_TRADABLE")
    numeric = (
        snapshot.spot_price,
        snapshot.perp_price,
        snapshot.spot_quote_volume_24h,
        snapshot.perp_quote_volume_24h,
    )
    if any(not isfinite(float(value)) or float(value) <= 0.0 for value in numeric):
        return _rejected(snapshot, "INVALID_MARKET_SNAPSHOT")
    recent = sorted(observations, key=lambda item: item.funding_time)[-settings.forecast_lookback :]
    rates = tuple(float(item.funding_rate) for item in recent)
    if len(recent) != settings.forecast_lookback:
        return _rejected(snapshot, "INSUFFICIENT_FUNDING_HISTORY", rates=rates)
    if any(item.symbol != snapshot.symbol for item in recent):
        return _rejected(snapshot, "FUNDING_SYMBOL_MISMATCH", rates=rates)
    if any(rate <= 0.0 for rate in rates):
        return _rejected(snapshot, "NON_POSITIVE_FUNDING_LOOKBACK", rates=rates)
    try:
        interval_hours = infer_funding_interval_hours(recent)
    except ValueError:
        return _rejected(snapshot, "INVALID_FUNDING_INTERVAL", rates=rates)
    if not (
        settings.min_observed_funding_interval_hours
        <= interval_hours
        <= settings.max_observed_funding_interval_hours
    ):
        return _rejected(snapshot, "FUNDING_INTERVAL_OUT_OF_RANGE", rates=rates)
    if abs(snapshot.basis_pct) > settings.max_abs_entry_basis_pct:
        return _rejected(snapshot, "ENTRY_BASIS_TOO_WIDE", rates=rates)
    costs = round_trip_cost_rate(settings)
    predicted = forecast_seven_day_funding_rate(
        recent,
        lookback=settings.forecast_lookback,
        horizon_days=settings.forecast_horizon_days,
    )
    if predicted < costs * settings.cost_coverage_multiple:
        return _rejected(
            snapshot,
            "INSUFFICIENT_COST_COVERAGE",
            rates=rates,
            costs=costs,
            predicted=predicted,
        )
    net = predicted - costs - settings.expected_rebalance_cost_rate - settings.basis_risk_buffer_rate
    if net <= 0.0:
        return _rejected(
            snapshot,
            "NON_POSITIVE_PREDICTED_NET",
            rates=rates,
            costs=costs,
            predicted=predicted,
            net=net,
        )
    candidate = ArbitrageCandidate(
        symbol=snapshot.symbol,
        timestamp=snapshot.timestamp,
        spot_price=float(snapshot.spot_price),
        perp_price=float(snapshot.perp_price),
        entry_basis_pct=float(snapshot.basis_pct),
        liquidity_score=float(snapshot.liquidity_score),
        funding_lookback=rates,
        funding_interval_hours=interval_hours,
        predicted_funding_rate=predicted,
        round_trip_cost_rate=costs,
        expected_rebalance_cost_rate=settings.expected_rebalance_cost_rate,
        basis_risk_buffer_rate=settings.basis_risk_buffer_rate,
        predicted_net_rate=net,
        score=net,
    )
    return CandidateEvaluation(
        symbol=snapshot.symbol,
        timestamp=snapshot.timestamp,
        accepted=True,
        reason=None,
        candidate=candidate,
        funding_lookback=rates,
        round_trip_cost_rate=costs,
        predicted_funding_rate=predicted,
        predicted_net_rate=net,
    )


def allocate_candidates(
    candidates: Sequence[ArbitrageCandidate],
    free_deployable_usdt: float,
    open_symbols: frozenset[str],
    settings: FundingArbitrageSettings,
) -> list[AllocationDecision]:
    slots = max(0, settings.max_positions - len(open_symbols))
    available = min(float(free_deployable_usdt), settings.max_deployable_usdt)
    eligible = sorted(
        (candidate for candidate in candidates if candidate.symbol not in open_symbols),
        key=lambda candidate: (-candidate.score, candidate.symbol),
    )
    count = min(slots, len(eligible), int(available // settings.min_position_notional_usdt))
    if count <= 0:
        return []
    selected = eligible[:count]
    equal_notional = min(
        available / count,
        settings.max_position_notional_usdt,
        settings.absolute_position_cap_usdt,
    )
    if equal_notional < settings.min_position_notional_usdt:
        return []
    return [
        AllocationDecision(
            symbol=candidate.symbol,
            notional_usdt=equal_notional,
            score=candidate.score,
            candidate=candidate,
        )
        for candidate in selected
    ]


__all__ = [
    "AllocationDecision",
    "ArbitrageCandidate",
    "CandidateEvaluation",
    "FundingArbitrageSettings",
    "FundingObservation",
    "MarketSnapshot",
    "allocate_candidates",
    "evaluate_candidate",
    "forecast_seven_day_funding_rate",
    "infer_funding_interval_hours",
    "round_trip_cost_rate",
]
