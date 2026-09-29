"""Two-leg portfolio accounting for funding-rate arbitrage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import pandas as pd

from .funding_arbitrage import ArbitrageCandidate, FundingArbitrageSettings
from .funding_arbitrage_logging import FundingArbitrageLogger


def _utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def apply_buy_slippage(price: float, bps: float) -> float:
    return float(price) * (1.0 + float(bps) / 10_000.0)


def apply_sell_slippage(price: float, bps: float) -> float:
    return float(price) * (1.0 - float(bps) / 10_000.0)


@dataclass(frozen=True)
class FundingSettlement:
    symbol: str
    funding_time: pd.Timestamp
    funding_rate: float
    mark_price: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "funding_time", _utc_timestamp(self.funding_time))

    @property
    def key(self) -> str:
        return f"{self.symbol}|{self.funding_time.isoformat()}"


@dataclass(frozen=True)
class ArbitragePosition:
    group_id: str
    symbol: str
    opened_at: pd.Timestamp
    initial_notional_usdt: float
    spot_quantity: float
    perp_quantity: float
    spot_entry_price: float
    perp_entry_price: float
    spot_reference_entry: float
    perp_reference_entry: float
    entry_basis_pct: float
    perp_margin_usdt: float
    last_spot_price: float
    last_perp_price: float
    predicted_net_rate: float
    entry_fee_usdt: float = 0.0
    rebalance_fee_usdt: float = 0.0
    realized_rebalance_price_pnl_usdt: float = 0.0
    cumulative_funding_usdt: float = 0.0
    rebalance_count: int = 0

    def hedge_drift_pct(self, spot_price: float, perp_price: float) -> float:
        spot_notional = self.spot_quantity * float(spot_price)
        perp_notional = self.perp_quantity * float(perp_price)
        denominator = max(spot_notional, perp_notional)
        return 0.0 if denominator <= 0.0 else abs(spot_notional - perp_notional) / denominator


@dataclass(frozen=True)
class PortfolioSnapshot:
    timestamp: pd.Timestamp
    equity_usdt: float
    cash_usdt: float
    spot_market_value_usdt: float
    perp_margin_usdt: float
    perp_unrealized_pnl_usdt: float
    gross_price_pnl_usdt: float
    drawdown_pct: float
    open_positions: int


class FundingArbitragePortfolioSimulator:
    def __init__(
        self,
        settings: FundingArbitrageSettings,
        *,
        output_dir: Path | str,
        run_id: str = "funding-arbitrage",
        persist_state: bool = True,
    ) -> None:
        self.settings = settings
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.persist_state = persist_state
        self.logger = FundingArbitrageLogger(self.output_dir, run_id=run_id)
        self.cash_usdt = float(settings.initial_capital_usdt)
        self.peak_equity_usdt = float(settings.initial_capital_usdt)
        self.positions: dict[str, ArbitragePosition] = {}
        self.processed_funding_keys: set[str] = set()
        self.realized_fees_usdt = 0.0
        self.realized_slippage_usdt = 0.0
        self.funding_pnl_usdt = 0.0
        self.realized_price_pnl_usdt = 0.0
        self.rebalance_cost_usdt = 0.0
        self.rebalance_count = 0
        self.hedge_drift_observations = 0
        self.hedge_drift_within_limit = 0
        self.duplicate_funding_ignored = 0
        self.closed_trades: list[dict[str, object]] = []
        self.circuit_breaker = False
        self.last_timestamp: pd.Timestamp | None = None

    @property
    def deployed_notional_usdt(self) -> float:
        return sum(position.initial_notional_usdt for position in self.positions.values())

    @property
    def unrealized_price_pnl_usdt(self) -> float:
        return sum(
            position.spot_quantity * (position.last_spot_price - position.spot_entry_price)
            + position.perp_quantity * (position.perp_entry_price - position.last_perp_price)
            for position in self.positions.values()
        )

    @property
    def equity_usdt(self) -> float:
        spot_value = sum(position.spot_quantity * position.last_spot_price for position in self.positions.values())
        perp_margin = sum(position.perp_margin_usdt for position in self.positions.values())
        perp_pnl = sum(
            position.perp_quantity * (position.perp_entry_price - position.last_perp_price)
            for position in self.positions.values()
        )
        return self.cash_usdt + spot_value + perp_margin + perp_pnl

    @property
    def current_drawdown_pct(self) -> float:
        if self.peak_equity_usdt <= 0.0:
            return 0.0
        return max(0.0, (self.peak_equity_usdt - self.equity_usdt) / self.peak_equity_usdt)

    def open_pair(
        self,
        candidate: ArbitrageCandidate,
        notional_usdt: float,
        timestamp: pd.Timestamp,
        *,
        second_leg_succeeds: bool = True,
    ) -> bool:
        timestamp = _utc_timestamp(timestamp)
        notional = float(notional_usdt)
        if self.circuit_breaker or candidate.symbol in self.positions:
            return False
        if len(self.positions) >= self.settings.max_positions:
            return False
        max_single = min(
            self.settings.max_position_notional_usdt,
            self.settings.absolute_position_cap_usdt,
        )
        if notional < self.settings.min_position_notional_usdt or notional > max_single:
            return False
        if self.deployed_notional_usdt + notional > self.settings.max_deployable_usdt + 1e-9:
            return False

        quantity = notional / candidate.perp_price
        spot_fill = apply_buy_slippage(candidate.spot_price, self.settings.slippage_bps)
        spot_cost = quantity * spot_fill
        spot_fee = spot_cost * self.settings.spot_fee_rate
        spot_slippage = quantity * (spot_fill - candidate.spot_price)

        if not second_leg_succeeds:
            emergency_fill = apply_sell_slippage(candidate.spot_price, self.settings.emergency_slippage_bps)
            emergency_proceeds = quantity * emergency_fill
            emergency_fee = emergency_proceeds * self.settings.spot_fee_rate
            emergency_slippage = quantity * (candidate.spot_price - emergency_fill)
            self.cash_usdt -= spot_cost + spot_fee
            self.cash_usdt += emergency_proceeds - emergency_fee
            self.realized_fees_usdt += spot_fee + emergency_fee
            self.realized_slippage_usdt += spot_slippage + emergency_slippage
            self.realized_price_pnl_usdt += emergency_proceeds - spot_cost
            self.closed_trades.append(
                {
                    "group_id": uuid4().hex,
                    "symbol": candidate.symbol,
                    "opened_at": timestamp,
                    "closed_at": timestamp,
                    "exit_reason": "ONE_LEG_FAILURE",
                    "net_pnl_usdt": emergency_proceeds - spot_cost - spot_fee - emergency_fee,
                }
            )
            self.logger.log_order(
                {
                    "timestamp": timestamp,
                    "group_id": self.closed_trades[-1]["group_id"],
                    "symbol": candidate.symbol,
                    "market": "SPOT",
                    "side": "SELL",
                    "reference_price": candidate.spot_price,
                    "fill_price": emergency_fill,
                    "quantity": quantity,
                    "notional_usdt": emergency_proceeds,
                    "fee_usdt": emergency_fee,
                    "slippage_usdt": emergency_slippage,
                    "status": "EMERGENCY_FILLED",
                    "reason": "ONE_LEG_FAILURE",
                }
            )
            self.logger.log_event(
                event_type="ONE_LEG_FAILURE",
                timestamp=timestamp,
                symbol=candidate.symbol,
                payload=self.closed_trades[-1],
            )
            self.save_state()
            return False

        perp_fill = apply_sell_slippage(candidate.perp_price, self.settings.slippage_bps)
        perp_notional = quantity * perp_fill
        perp_fee = perp_notional * self.settings.perp_fee_rate
        perp_slippage = quantity * (candidate.perp_price - perp_fill)
        margin = perp_notional / self.settings.perp_margin_leverage
        required_cash = spot_cost + margin + spot_fee + perp_fee
        emergency_reserve = notional * (
            2.0 * self.settings.emergency_slippage_bps / 10_000.0
            + self.settings.spot_fee_rate
            + self.settings.perp_fee_rate
        )
        if required_cash + emergency_reserve > self.cash_usdt + 1e-9:
            return False

        self.cash_usdt -= required_cash
        self.realized_fees_usdt += spot_fee + perp_fee
        self.realized_slippage_usdt += spot_slippage + perp_slippage
        self.positions[candidate.symbol] = ArbitragePosition(
            group_id=uuid4().hex,
            symbol=candidate.symbol,
            opened_at=timestamp,
            initial_notional_usdt=notional,
            spot_quantity=quantity,
            perp_quantity=quantity,
            spot_entry_price=spot_fill,
            perp_entry_price=perp_fill,
            spot_reference_entry=candidate.spot_price,
            perp_reference_entry=candidate.perp_price,
            entry_basis_pct=candidate.entry_basis_pct,
            perp_margin_usdt=margin,
            last_spot_price=candidate.spot_price,
            last_perp_price=candidate.perp_price,
            predicted_net_rate=candidate.predicted_net_rate,
            entry_fee_usdt=spot_fee + perp_fee,
        )
        self.last_timestamp = timestamp
        position = self.positions[candidate.symbol]
        for market, side, reference, fill, fee, slippage in (
            ("SPOT", "BUY", candidate.spot_price, spot_fill, spot_fee, spot_slippage),
            ("PERP", "SELL", candidate.perp_price, perp_fill, perp_fee, perp_slippage),
        ):
            self.logger.log_order(
                {
                    "timestamp": timestamp,
                    "group_id": position.group_id,
                    "symbol": candidate.symbol,
                    "market": market,
                    "side": side,
                    "reference_price": reference,
                    "fill_price": fill,
                    "quantity": quantity,
                    "notional_usdt": quantity * fill,
                    "fee_usdt": fee,
                    "slippage_usdt": slippage,
                    "status": "SIMULATED_FILLED",
                    "reason": "OPEN_PAIR",
                }
            )
        self.logger.log_event(
            event_type="PAIR_OPENED",
            timestamp=timestamp,
            symbol=candidate.symbol,
            payload=asdict(position),
        )
        self._update_peak()
        self.save_state()
        return True

    def close_pair(
        self,
        symbol: str,
        *,
        spot_price: float,
        perp_price: float,
        timestamp: pd.Timestamp,
        reason: str,
    ) -> dict[str, object] | None:
        position = self.positions.pop(symbol, None)
        if position is None:
            return None
        timestamp = _utc_timestamp(timestamp)
        spot_fill = apply_sell_slippage(spot_price, self.settings.slippage_bps)
        perp_fill = apply_buy_slippage(perp_price, self.settings.slippage_bps)
        spot_proceeds = position.spot_quantity * spot_fill
        perp_pnl = position.perp_quantity * (position.perp_entry_price - perp_fill)
        spot_fee = spot_proceeds * self.settings.spot_fee_rate
        perp_close_notional = position.perp_quantity * perp_fill
        perp_fee = perp_close_notional * self.settings.perp_fee_rate
        close_slippage = (
            position.spot_quantity * (spot_price - spot_fill)
            + position.perp_quantity * (perp_fill - perp_price)
        )
        self.cash_usdt += spot_proceeds - spot_fee + position.perp_margin_usdt + perp_pnl - perp_fee
        self.realized_fees_usdt += spot_fee + perp_fee
        self.realized_slippage_usdt += close_slippage
        closing_price_pnl = (
            position.spot_quantity * (spot_fill - position.spot_entry_price)
            + position.perp_quantity * (position.perp_entry_price - perp_fill)
        )
        gross_price_pnl = position.realized_rebalance_price_pnl_usdt + closing_price_pnl
        self.realized_price_pnl_usdt += closing_price_pnl
        exit_fee = spot_fee + perp_fee
        trade = {
            "group_id": position.group_id,
            "symbol": position.symbol,
            "opened_at": position.opened_at,
            "closed_at": timestamp,
            "holding_hours": (timestamp - position.opened_at).total_seconds() / 3600.0,
            "initial_notional_usdt": position.initial_notional_usdt,
            "funding_pnl_usdt": position.cumulative_funding_usdt,
            "gross_price_pnl_usdt": gross_price_pnl,
            "entry_fee_usdt": position.entry_fee_usdt,
            "exit_fee_usdt": exit_fee,
            "rebalance_fee_usdt": position.rebalance_fee_usdt,
            "total_fee_usdt": position.entry_fee_usdt + position.rebalance_fee_usdt + exit_fee,
            "exit_reason": reason,
        }
        trade["net_pnl_usdt"] = (
            gross_price_pnl
            + position.cumulative_funding_usdt
            - position.entry_fee_usdt
            - position.rebalance_fee_usdt
            - exit_fee
        )
        self.closed_trades.append(trade)
        self.logger.log_trade(trade)
        self.logger.log_event(
            event_type="PAIR_CLOSED",
            timestamp=timestamp,
            symbol=symbol,
            payload=trade,
        )
        self.last_timestamp = timestamp
        self.save_state()
        return trade

    def apply_funding(self, settlement: FundingSettlement) -> float:
        if settlement.key in self.processed_funding_keys:
            self.duplicate_funding_ignored += 1
            return 0.0
        self.processed_funding_keys.add(settlement.key)
        position = self.positions.get(settlement.symbol)
        if position is None or position.opened_at >= settlement.funding_time:
            self.logger.log_funding(
                {
                    "funding_time": settlement.funding_time,
                    "group_id": position.group_id if position else "",
                    "symbol": settlement.symbol,
                    "funding_rate": settlement.funding_rate,
                    "mark_price": settlement.mark_price,
                    "perp_notional_usdt": 0.0,
                    "cashflow_usdt": 0.0,
                    "settlement_key": settlement.key,
                    "status": "NOT_ELIGIBLE",
                }
            )
            self.save_state()
            return 0.0
        cashflow = position.perp_quantity * settlement.mark_price * settlement.funding_rate
        self.cash_usdt += cashflow
        self.funding_pnl_usdt += cashflow
        self.positions[settlement.symbol] = replace(
            position,
            cumulative_funding_usdt=position.cumulative_funding_usdt + cashflow,
            last_perp_price=settlement.mark_price,
        )
        self.logger.log_funding(
            {
                "funding_time": settlement.funding_time,
                "group_id": position.group_id,
                "symbol": settlement.symbol,
                "funding_rate": settlement.funding_rate,
                "mark_price": settlement.mark_price,
                "perp_notional_usdt": position.perp_quantity * settlement.mark_price,
                "cashflow_usdt": cashflow,
                "settlement_key": settlement.key,
                "status": "APPLIED",
            }
        )
        if settlement.funding_rate < 0.0:
            current = self.positions[settlement.symbol]
            self.close_pair(
                settlement.symbol,
                spot_price=current.last_spot_price,
                perp_price=settlement.mark_price,
                timestamp=settlement.funding_time,
                reason="NEGATIVE_FUNDING",
            )
        self.last_timestamp = settlement.funding_time
        self.save_state()
        return cashflow

    def mark_to_market(
        self,
        *,
        timestamp: pd.Timestamp,
        prices: Mapping[str, tuple[float, float]],
        rebalance: bool = False,
    ) -> PortfolioSnapshot:
        timestamp = _utc_timestamp(timestamp)
        for symbol, position in list(self.positions.items()):
            current = prices.get(symbol)
            if current is None:
                continue
            spot_price, perp_price = map(float, current)
            updated = replace(position, last_spot_price=spot_price, last_perp_price=perp_price)
            self.positions[symbol] = updated
            drift = updated.hedge_drift_pct(spot_price, perp_price)
            self.hedge_drift_observations += 1
            if drift <= self.settings.hedge_drift_limit_pct:
                self.hedge_drift_within_limit += 1
            elif rebalance:
                self._rebalance_perp(symbol, spot_price, perp_price, timestamp)

        for symbol, position in list(self.positions.items()):
            if timestamp - position.opened_at >= pd.Timedelta(days=self.settings.max_hold_days):
                self.close_pair(
                    symbol,
                    spot_price=position.last_spot_price,
                    perp_price=position.last_perp_price,
                    timestamp=timestamp,
                    reason="MAX_HOLD_TIME",
                )
                continue
            current_basis = position.last_perp_price / position.last_spot_price - 1.0
            if (
                abs(current_basis) > self.settings.max_abs_entry_basis_pct
                or abs(current_basis - position.entry_basis_pct) > self.settings.max_basis_widening_pct
            ):
                self.close_pair(
                    symbol,
                    spot_price=position.last_spot_price,
                    perp_price=position.last_perp_price,
                    timestamp=timestamp,
                    reason="BASIS_WIDENING",
                )

        equity_before_circuit = self.equity_usdt
        drawdown = (
            max(0.0, (self.peak_equity_usdt - equity_before_circuit) / self.peak_equity_usdt)
            if self.peak_equity_usdt > 0.0
            else 0.0
        )
        if drawdown >= self.settings.max_drawdown_pct:
            self.circuit_breaker = True
            for symbol, position in list(self.positions.items()):
                self.close_pair(
                    symbol,
                    spot_price=position.last_spot_price,
                    perp_price=position.last_perp_price,
                    timestamp=timestamp,
                    reason="DRAWDOWN_CIRCUIT_BREAKER",
                )
        self.last_timestamp = timestamp
        self._update_peak()
        snapshot = self.portfolio_snapshot(timestamp)
        self.logger.log_equity(asdict(snapshot))
        self.save_state()
        return snapshot

    def _rebalance_perp(self, symbol: str, spot_price: float, perp_price: float, timestamp: pd.Timestamp) -> None:
        position = self.positions[symbol]
        target_quantity = position.spot_quantity * spot_price / perp_price
        delta_quantity = target_quantity - position.perp_quantity
        adjustment_notional = abs(delta_quantity) * perp_price
        if adjustment_notional <= 0.0:
            return
        if delta_quantity > 0.0:
            fill_price = apply_sell_slippage(perp_price, self.settings.slippage_bps)
        else:
            fill_price = apply_buy_slippage(perp_price, self.settings.slippage_bps)
        fee = abs(delta_quantity) * fill_price * self.settings.perp_fee_rate
        slippage = abs(delta_quantity) * abs(perp_price - fill_price)
        old_margin = position.perp_margin_usdt
        new_margin = target_quantity * perp_price / self.settings.perp_margin_leverage
        realized_rebalance_pnl = 0.0
        if delta_quantity < 0.0:
            closed_quantity = -delta_quantity
            realized_rebalance_pnl = closed_quantity * (position.perp_entry_price - fill_price)
        self.cash_usdt += old_margin - new_margin + realized_rebalance_pnl - fee
        self.realized_fees_usdt += fee
        self.realized_slippage_usdt += slippage
        self.rebalance_cost_usdt += fee + slippage
        self.realized_price_pnl_usdt += realized_rebalance_pnl
        self.rebalance_count += 1
        if delta_quantity > 0.0:
            entry_price = (
                position.perp_quantity * position.perp_entry_price + delta_quantity * fill_price
            ) / target_quantity
        else:
            entry_price = position.perp_entry_price
        self.positions[symbol] = replace(
            position,
            perp_quantity=target_quantity,
            perp_entry_price=entry_price,
            perp_margin_usdt=new_margin,
            rebalance_fee_usdt=position.rebalance_fee_usdt + fee,
            realized_rebalance_price_pnl_usdt=(
                position.realized_rebalance_price_pnl_usdt + realized_rebalance_pnl
            ),
            rebalance_count=position.rebalance_count + 1,
        )
        drift_after = self.positions[symbol].hedge_drift_pct(spot_price, perp_price)
        self.logger.log_rebalance(
            {
                "timestamp": timestamp,
                "group_id": position.group_id,
                "symbol": symbol,
                "drift_before_pct": position.hedge_drift_pct(spot_price, perp_price),
                "drift_after_pct": drift_after,
                "quantity_delta": delta_quantity,
                "notional_usdt": adjustment_notional,
                "fee_usdt": fee,
                "slippage_usdt": slippage,
            }
        )

    def _update_peak(self) -> None:
        self.peak_equity_usdt = max(self.peak_equity_usdt, self.equity_usdt)

    def portfolio_snapshot(self, timestamp: pd.Timestamp) -> PortfolioSnapshot:
        spot_value = sum(position.spot_quantity * position.last_spot_price for position in self.positions.values())
        perp_margin = sum(position.perp_margin_usdt for position in self.positions.values())
        perp_pnl = sum(
            position.perp_quantity * (position.perp_entry_price - position.last_perp_price)
            for position in self.positions.values()
        )
        return PortfolioSnapshot(
            timestamp=_utc_timestamp(timestamp),
            equity_usdt=self.equity_usdt,
            cash_usdt=self.cash_usdt,
            spot_market_value_usdt=spot_value,
            perp_margin_usdt=perp_margin,
            perp_unrealized_pnl_usdt=perp_pnl,
            gross_price_pnl_usdt=self.unrealized_price_pnl_usdt,
            drawdown_pct=self.current_drawdown_pct,
            open_positions=len(self.positions),
        )

    def save_state(self) -> Path:
        destination = self.output_dir / "state.json"
        if not self.persist_state:
            return destination
        state = {
            "version": 1,
            "run_id": self.run_id,
            "settings": asdict(self.settings),
            "cash_usdt": self.cash_usdt,
            "peak_equity_usdt": self.peak_equity_usdt,
            "positions": {symbol: asdict(position) for symbol, position in self.positions.items()},
            "processed_funding_keys": sorted(self.processed_funding_keys),
            "realized_fees_usdt": self.realized_fees_usdt,
            "realized_slippage_usdt": self.realized_slippage_usdt,
            "funding_pnl_usdt": self.funding_pnl_usdt,
            "realized_price_pnl_usdt": self.realized_price_pnl_usdt,
            "rebalance_cost_usdt": self.rebalance_cost_usdt,
            "rebalance_count": self.rebalance_count,
            "hedge_drift_observations": self.hedge_drift_observations,
            "hedge_drift_within_limit": self.hedge_drift_within_limit,
            "duplicate_funding_ignored": self.duplicate_funding_ignored,
            "closed_trades": self.closed_trades,
            "circuit_breaker": self.circuit_breaker,
            "last_timestamp": self.last_timestamp,
        }

        def default(value: object) -> object:
            if isinstance(value, pd.Timestamp):
                return value.isoformat()
            raise TypeError(f"Unsupported state value: {type(value).__name__}")

        temporary = self.output_dir / "state.json.tmp"
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2, default=default),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    @classmethod
    def load_state(
        cls,
        settings: FundingArbitrageSettings,
        *,
        output_dir: Path | str,
    ) -> "FundingArbitragePortfolioSimulator":
        output_path = Path(output_dir)
        state = json.loads((output_path / "state.json").read_text(encoding="utf-8"))
        simulator = cls(settings, output_dir=output_path, run_id=state.get("run_id", "funding-arbitrage"))
        simulator.cash_usdt = float(state["cash_usdt"])
        simulator.peak_equity_usdt = float(state["peak_equity_usdt"])
        simulator.positions = {
            symbol: ArbitragePosition(
                **{
                    **payload,
                    "opened_at": _utc_timestamp(payload["opened_at"]),
                }
            )
            for symbol, payload in state.get("positions", {}).items()
        }
        simulator.processed_funding_keys = set(state.get("processed_funding_keys", []))
        for name in (
            "realized_fees_usdt",
            "realized_slippage_usdt",
            "funding_pnl_usdt",
            "realized_price_pnl_usdt",
            "rebalance_cost_usdt",
        ):
            setattr(simulator, name, float(state.get(name, 0.0)))
        for name in (
            "rebalance_count",
            "hedge_drift_observations",
            "hedge_drift_within_limit",
            "duplicate_funding_ignored",
        ):
            setattr(simulator, name, int(state.get(name, 0)))
        simulator.closed_trades = list(state.get("closed_trades", []))
        simulator.circuit_breaker = bool(state.get("circuit_breaker", False))
        last_timestamp = state.get("last_timestamp")
        simulator.last_timestamp = _utc_timestamp(last_timestamp) if last_timestamp else None
        return simulator


__all__ = [
    "ArbitragePosition",
    "FundingArbitragePortfolioSimulator",
    "FundingSettlement",
    "PortfolioSnapshot",
    "apply_buy_slippage",
    "apply_sell_slippage",
]
