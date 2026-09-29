"""Historical backtest runner for Binance positive-funding arbitrage."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import yaml

from futures_strategy.funding_arbitrage import (
    FundingArbitrageSettings,
    FundingObservation,
    MarketSnapshot,
    allocate_candidates,
    evaluate_candidate,
)
from futures_strategy.funding_arbitrage_simulator import (
    FundingArbitragePortfolioSimulator,
    FundingSettlement,
)


def _utc(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _parse_utc_datetime(values):
    return pd.to_datetime(values, utc=True, format="mixed")


def run_backtest_cycle(
    *,
    timestamp: pd.Timestamp,
    snapshots: Mapping[str, MarketSnapshot],
    funding_history: Mapping[str, Sequence[FundingObservation]],
    settlements: Sequence[FundingSettlement],
    simulator: FundingArbitragePortfolioSimulator,
    settings: FundingArbitrageSettings,
    allow_new_entries: bool,
) -> list[str]:
    timestamp = _utc(timestamp)
    for settlement in sorted(settlements, key=lambda item: (item.funding_time, item.symbol)):
        if settlement.funding_time <= timestamp:
            simulator.apply_funding(settlement)

    visible_prices = {
        symbol: (snapshot.spot_price, snapshot.perp_price)
        for symbol, snapshot in snapshots.items()
        if snapshot.timestamp <= timestamp
    }
    simulator.mark_to_market(timestamp=timestamp, prices=visible_prices, rebalance=True)

    for symbol, position in list(simulator.positions.items()):
        snapshot = snapshots.get(symbol)
        if snapshot is None:
            continue
        visible_history = [item for item in funding_history.get(symbol, ()) if item.funding_time < timestamp]
        evaluation = evaluate_candidate(snapshot, visible_history, settings)
        if not evaluation.accepted and evaluation.reason in {
            "NON_POSITIVE_FUNDING_LOOKBACK",
            "INSUFFICIENT_COST_COVERAGE",
            "NON_POSITIVE_PREDICTED_NET",
        }:
            simulator.close_pair(
                symbol,
                spot_price=snapshot.spot_price,
                perp_price=snapshot.perp_price,
                timestamp=timestamp,
                reason="FORECAST_EDGE_GONE",
            )

    if not allow_new_entries or simulator.circuit_breaker:
        return []
    candidates = []
    for symbol in sorted(snapshots):
        if symbol in simulator.positions:
            continue
        snapshot = snapshots[symbol]
        if snapshot.timestamp > timestamp:
            continue
        visible_history = [item for item in funding_history.get(symbol, ()) if item.funding_time < timestamp]
        evaluation = evaluate_candidate(snapshot, visible_history, settings)
        simulator.logger.log_candidate(evaluation)
        if evaluation.candidate is not None:
            candidates.append(evaluation.candidate)
    free = max(0.0, settings.max_deployable_usdt - simulator.deployed_notional_usdt)
    decisions = allocate_candidates(candidates, free, frozenset(simulator.positions), settings)
    opened: list[str] = []
    for decision in decisions:
        if simulator.open_pair(decision.candidate, decision.notional_usdt, timestamp):
            opened.append(decision.symbol)
    return opened


def _frame_time_filter(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError("EQUITY_INDEX_MUST_BE_DATETIME")
    result.index = _parse_utc_datetime(result.index)
    return result.loc[(result.index >= start) & (result.index <= end)]


def build_period_summary(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    settlements: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    start = _utc(start)
    end = _utc(end)
    normalized = equity.copy()
    normalized.index = _parse_utc_datetime(normalized.index)
    before_start = normalized.loc[normalized.index <= start]
    period = normalized.loc[(normalized.index >= start) & (normalized.index <= end)]
    if period.empty and before_start.empty:
        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "starting_equity_usdt": None,
            "ending_equity_usdt": None,
            "net_profit_usdt": None,
            "net_return_pct": None,
            "max_drawdown_pct": None,
            "closed_trade_groups": 0,
            "verified_funding_settlements": 0,
        }
    starting_equity = float(
        before_start.iloc[-1]["equity_usdt"] if not before_start.empty else period.iloc[0]["equity_usdt"]
    )
    ending_equity = float(period.iloc[-1]["equity_usdt"] if not period.empty else starting_equity)
    path = pd.concat(
        [
            pd.Series([starting_equity], index=pd.DatetimeIndex([start])),
            period["equity_usdt"].astype(float),
        ]
    ).sort_index()
    path = path[~path.index.duplicated(keep="last")]
    running_peak = path.cummax()
    max_drawdown = float(((running_peak - path) / running_peak).max()) if not path.empty else 0.0

    trade_count = 0
    wins = 0
    gross_wins = 0.0
    gross_losses = 0.0
    if not trades.empty:
        trade_frame = trades.copy()
        closed_column = "closed_at" if "closed_at" in trade_frame else None
        if closed_column:
            closed = _parse_utc_datetime(trade_frame[closed_column])
            trade_frame = trade_frame.loc[(closed >= start) & (closed <= end)]
        trade_count = len(trade_frame)
        if "net_pnl_usdt" in trade_frame:
            pnl = pd.to_numeric(trade_frame["net_pnl_usdt"], errors="coerce").dropna()
            wins = int((pnl > 0.0).sum())
            gross_wins = float(pnl[pnl > 0.0].sum())
            gross_losses = float(-pnl[pnl < 0.0].sum())

    verified = 0
    funding_income = 0.0
    funding_expense = 0.0
    if not settlements.empty and "funding_time" in settlements:
        funding_frame = settlements.copy()
        funding_times = _parse_utc_datetime(funding_frame["funding_time"])
        funding_frame = funding_frame.loc[(funding_times >= start) & (funding_times <= end)]
        applied = funding_frame.loc[funding_frame.get("status", "") == "APPLIED"]
        verified = len(applied)
        cashflows = pd.to_numeric(applied.get("cashflow_usdt", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        funding_income = float(cashflows[cashflows > 0.0].sum())
        funding_expense = float(-cashflows[cashflows < 0.0].sum())

    profit_factor = None if gross_losses == 0.0 else gross_wins / gross_losses
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "starting_equity_usdt": starting_equity,
        "ending_equity_usdt": ending_equity,
        "net_profit_usdt": ending_equity - starting_equity,
        "net_return_pct": ending_equity / starting_equity - 1.0,
        "max_drawdown_pct": max_drawdown,
        "closed_trade_groups": trade_count,
        "winning_trade_groups": wins,
        "win_rate": wins / trade_count if trade_count else None,
        "profit_factor": profit_factor,
        "profit_factor_reason": "NO_LOSING_TRADES" if gross_losses == 0.0 else None,
        "verified_funding_settlements": verified,
        "gross_funding_income_usdt": funding_income,
        "negative_funding_expense_usdt": funding_expense,
    }


def _read_indexed_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = _parse_utc_datetime(frame.index.astype(str))
    return frame.sort_index()


def _interval_delta(interval: str) -> pd.Timedelta:
    if interval.endswith("m"):
        return pd.Timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return pd.Timedelta(hours=int(interval[:-1]))
    raise ValueError(f"UNSUPPORTED_INTERVAL: {interval}")


def _load_data(data_dir: Path, interval: str) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, list[FundingObservation]], dict[str, object]]:
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    symbols = metadata.get("detailed_complete_symbols", [])
    spot: dict[str, pd.DataFrame] = {}
    perp: dict[str, pd.DataFrame] = {}
    funding: dict[str, list[FundingObservation]] = {}
    delta = _interval_delta(interval)
    for symbol in symbols:
        spot_frame = _read_indexed_csv(data_dir / f"{symbol}_spot_{interval}.csv")
        perp_frame = _read_indexed_csv(data_dir / f"{symbol}_perp_{interval}.csv")
        spot_frame.index = spot_frame.index + delta
        perp_frame.index = perp_frame.index + delta
        spot[symbol] = spot_frame
        perp[symbol] = perp_frame
        funding_frame = _read_indexed_csv(data_dir / f"{symbol}_funding.csv")
        funding[symbol] = [
            FundingObservation(symbol, timestamp, float(row["funding_rate"]), float(row["mark_price"]))
            for timestamp, row in funding_frame.iterrows()
        ]
    return spot, perp, funding, metadata


def _filter_symbols_for_window(
    spot: Mapping[str, pd.DataFrame],
    perp: Mapping[str, pd.DataFrame],
    funding_history: Mapping[str, Sequence[FundingObservation]],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, list[FundingObservation]], list[dict[str, object]]]:
    start = _utc(start)
    end = _utc(end)
    filtered_spot: dict[str, pd.DataFrame] = {}
    filtered_perp: dict[str, pd.DataFrame] = {}
    filtered_funding: dict[str, list[FundingObservation]] = {}
    exclusions: list[dict[str, object]] = []
    for symbol in sorted(set(spot) | set(perp)):
        spot_frame = spot.get(symbol)
        perp_frame = perp.get(symbol)
        if spot_frame is None or perp_frame is None or spot_frame.empty or perp_frame.empty:
            exclusions.append({"symbol": symbol, "reason": "MISSING_MARKET_DATA"})
            continue
        spot_start, spot_end = _utc(spot_frame.index.min()), _utc(spot_frame.index.max())
        perp_start, perp_end = _utc(perp_frame.index.min()), _utc(perp_frame.index.max())
        if spot_start > start or perp_start > start or spot_end < end or perp_end < end:
            exclusions.append(
                {
                    "symbol": symbol,
                    "reason": "INSUFFICIENT_WINDOW_COVERAGE",
                    "spot_start": spot_start.isoformat(),
                    "spot_end": spot_end.isoformat(),
                    "perp_start": perp_start.isoformat(),
                    "perp_end": perp_end.isoformat(),
                }
            )
            continue
        filtered_spot[symbol] = spot_frame
        filtered_perp[symbol] = perp_frame
        filtered_funding[symbol] = list(funding_history.get(symbol, ()))
    return filtered_spot, filtered_perp, filtered_funding, exclusions


def run_backtest(
    *,
    data_dir: Path,
    output_dir: Path,
    settings: FundingArbitrageSettings,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str = "5m",
) -> dict[str, object]:
    start = _utc(start)
    end = _utc(end)
    spot, perp, funding_history, metadata = _load_data(data_dir, interval)
    spot, perp, funding_history, excluded_symbols = _filter_symbols_for_window(
        spot,
        perp,
        funding_history,
        start=start,
        end=end,
    )
    if not spot:
        raise ValueError(f"NO_COMPLETE_SYMBOL_DATA_FOR_WINDOW: excluded={len(excluded_symbols)}")
    earliest = max(frame.index.min() for frame in spot.values())
    latest = min(frame.index.max() for frame in spot.values())
    if earliest > start or latest < end:
        raise ValueError(f"INSUFFICIENT_BACKTEST_WINDOW: available={earliest.isoformat()}..{latest.isoformat()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    simulator = FundingArbitragePortfolioSimulator(
        settings,
        output_dir=output_dir,
        run_id="backtest",
        persist_state=False,
    )
    rolling_spot = {
        symbol: frame["quote_volume"].rolling(288, min_periods=288).sum()
        for symbol, frame in spot.items()
    }
    rolling_perp = {
        symbol: frame["quote_volume"].rolling(288, min_periods=288).sum()
        for symbol, frame in perp.items()
    }
    funding_events: dict[pd.Timestamp, list[FundingSettlement]] = {}
    for symbol, items in funding_history.items():
        for item in items:
            if start <= item.funding_time <= end:
                funding_events.setdefault(item.funding_time, []).append(
                    FundingSettlement(symbol, item.funding_time, item.funding_rate, item.mark_price)
                )
    timestamps = sorted(
        set().union(*(set(frame.loc[(frame.index >= start) & (frame.index <= end)].index) for frame in spot.values()))
        | set(funding_events)
    )
    equity_rows: list[dict[str, object]] = []
    for timestamp in timestamps:
        snapshots: dict[str, MarketSnapshot] = {}
        liquidity_rows: list[tuple[str, float]] = []
        for symbol in spot.keys() & perp.keys():
            if timestamp not in spot[symbol].index or timestamp not in perp[symbol].index:
                continue
            spot_volume = rolling_spot[symbol].get(timestamp, float("nan"))
            perp_volume = rolling_perp[symbol].get(timestamp, float("nan"))
            if pd.isna(spot_volume) or pd.isna(perp_volume):
                continue
            liquidity_rows.append((symbol, min(float(spot_volume), float(perp_volume))))
        top_symbols = {
            symbol for symbol, _ in sorted(liquidity_rows, key=lambda item: (-item[1], item[0]))[: settings.universe_size]
        }
        for symbol in top_symbols | set(simulator.positions):
            if symbol not in spot or symbol not in perp:
                continue
            if timestamp not in spot[symbol].index or timestamp not in perp[symbol].index:
                continue
            spot_row = spot[symbol].loc[timestamp]
            perp_row = perp[symbol].loc[timestamp]
            snapshots[symbol] = MarketSnapshot(
                symbol=symbol,
                timestamp=timestamp,
                spot_price=float(spot_row["close"]),
                perp_price=float(perp_row["close"]),
                spot_quote_volume_24h=float(rolling_spot[symbol].get(timestamp, 0.0)),
                perp_quote_volume_24h=float(rolling_perp[symbol].get(timestamp, 0.0)),
                spot_tradable=symbol in top_symbols or symbol in simulator.positions,
                perp_tradable=symbol in top_symbols or symbol in simulator.positions,
            )
        allow_entries = timestamp.minute == 0 and timestamp.second == 0 and bool(top_symbols)
        run_backtest_cycle(
            timestamp=timestamp,
            snapshots=snapshots,
            funding_history=funding_history,
            settlements=funding_events.get(timestamp, []),
            simulator=simulator,
            settings=settings,
            allow_new_entries=allow_entries,
        )
        equity_rows.append(asdict(simulator.portfolio_snapshot(timestamp)))
    if timestamps:
        final_time = timestamps[-1]
        for symbol, position in list(simulator.positions.items()):
            simulator.close_pair(
                symbol,
                spot_price=position.last_spot_price,
                perp_price=position.last_perp_price,
                timestamp=final_time,
                reason="BACKTEST_END",
            )
        equity_rows.append(asdict(simulator.portfolio_snapshot(final_time)))
    equity = pd.DataFrame(equity_rows).set_index("timestamp")
    trades = pd.DataFrame(simulator.closed_trades)
    funding_path = output_dir / "funding_settlements.csv"
    settlements = pd.read_csv(funding_path, low_memory=False) if funding_path.exists() else pd.DataFrame()
    full = build_period_summary(equity, trades, settlements, start=start, end=end)
    month_start = max(start, end - pd.Timedelta(days=30))
    last_month = build_period_summary(equity, trades, settlements, start=month_start, end=end)
    drift_ratio = (
        simulator.hedge_drift_within_limit / simulator.hedge_drift_observations
        if simulator.hedge_drift_observations
        else None
    )
    full.update(
        {
            "fees_usdt": simulator.realized_fees_usdt,
            "slippage_usdt": simulator.realized_slippage_usdt,
            "rebalance_cost_usdt": simulator.rebalance_cost_usdt,
            "gross_price_pnl_usdt": simulator.realized_price_pnl_usdt,
            "hedge_drift_observations": simulator.hedge_drift_observations,
            "hedge_drift_within_limit_ratio": drift_ratio,
            "rebalance_count": simulator.rebalance_count,
            "duplicate_funding_ignored": simulator.duplicate_funding_ignored,
            "symbols_used": len(spot),
            "excluded_symbol_count": len(excluded_symbols),
            "excluded_symbols": excluded_symbols,
        }
    )
    checks = {
        "full_3m_positive": bool(full["net_return_pct"] is not None and full["net_return_pct"] > 0.0),
        "last_1m_positive": bool(last_month["net_return_pct"] is not None and last_month["net_return_pct"] > 0.0),
        "drawdown_below_5pct": bool(full["max_drawdown_pct"] is not None and full["max_drawdown_pct"] < 0.05),
        "at_least_20_settlements": full["verified_funding_settlements"] >= 20,
        "drift_compliance_at_least_95pct": bool(drift_ratio is not None and drift_ratio >= 0.95),
    }
    summary = {
        "mode": "backtest",
        "full_3m": full,
        "last_1m": last_month,
        "acceptance": {**checks, "qualified": all(checks.values())},
        "current_listing_universe_bias": bool(metadata.get("current_listing_universe_bias", True)),
        "survivorship_bias": bool(metadata.get("survivorship_bias", True)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    equity.to_csv(output_dir / "equity_curve.csv")
    return summary


def settings_from_config(path: Path) -> tuple[FundingArbitrageSettings, dict[str, object]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    strategy = payload.get("strategy", {})
    if payload.get("paper", {}).get("live_ordering_enabled") is True:
        raise ValueError("LIVE_ORDERING_NOT_SUPPORTED")
    return FundingArbitrageSettings(**strategy), payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binance 资金费率套利历史回测")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings, config = settings_from_config(args.config)
    interval = str(config.get("data", {}).get("interval", "5m"))
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    end = _utc(metadata["end"])
    start = end - pd.Timedelta(days=args.days)
    summary = run_backtest(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        settings=settings,
        start=start,
        end=end,
        interval=interval,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
