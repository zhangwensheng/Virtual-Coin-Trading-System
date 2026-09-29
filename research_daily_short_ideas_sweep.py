from __future__ import annotations

import argparse
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import research_daily_short_pressure as base


warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")


@dataclass(frozen=True)
class IdeaSpec:
    name: str
    base_variant: str
    event_filter: str
    selector: str
    train_months: int
    top_symbols: int
    risk_per_trade: float
    max_positions: int
    cooldown_after_loss_pct: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep daily short ideas with rolling validation.")
    parser.add_argument("--train-dir", default="data/binance_universal_short_2y_1h")
    parser.add_argument("--oos-dir", default="data/binance_universal_short_oos_1h")
    parser.add_argument("--seed-summary", default="outputs/daily_short_pressure_research_marketfilter/variant_summary.csv")
    parser.add_argument("--output-dir", default="outputs/daily_short_ideas_sweep")
    parser.add_argument("--start-month", default="2024-10-01")
    parser.add_argument("--end-month", default="2026-04-01")
    parser.add_argument("--max-base-variants", type=int, default=14)
    parser.add_argument("--base-variants", default="", help="Comma-separated base variant names. Overrides seed selection.")
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--max-notional-fraction", type=float, default=0.12)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    return parser.parse_args()


def load_daily_data(train_dir: Path, oos_dir: Path) -> dict[str, pd.DataFrame]:
    hourly = base.load_hourly_data(train_dir, oos_dir)
    daily = {symbol: base.prepare_daily_features(frame) for symbol, frame in hourly.items()}
    return {symbol: frame for symbol, frame in daily.items() if len(frame) >= 80}


def build_market_context(daily_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol, frame in daily_by_symbol.items():
        chunk = pd.DataFrame(
            {
                "symbol": symbol,
                "day_return": frame["day_return"],
                "below_ema20": frame["close"] < frame["ema20"],
                "below_ema50": frame["close"] < frame["ema50"],
            }
        )
        rows.append(chunk)
    stacked = pd.concat(rows).reset_index(names="date")
    context = stacked.groupby("date").agg(
        market_median_return=("day_return", "median"),
        market_breadth_below_ema20=("below_ema20", "mean"),
        market_breadth_below_ema50=("below_ema50", "mean"),
    )
    btc = daily_by_symbol.get("BTCUSDT")
    if btc is not None:
        context["btc_day_return"] = btc["day_return"].reindex(context.index)
        context["btc_5d_return"] = (btc["close"] / btc["close"].shift(5) - 1.0).reindex(context.index)
        context["btc_below_ema20"] = (btc["close"] < btc["ema20"]).reindex(context.index)
        context["btc_below_ema50"] = (btc["close"] < btc["ema50"]).reindex(context.index)
        context["btc_ema_bear"] = (btc["ema20"] < btc["ema50"]).reindex(context.index)
    else:
        context["btc_day_return"] = np.nan
        context["btc_5d_return"] = np.nan
        context["btc_below_ema20"] = False
        context["btc_below_ema50"] = False
        context["btc_ema_bear"] = False
    return context.ffill()


def enrich_events(events_by_variant: dict[str, pd.DataFrame], daily_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    context = build_market_context(daily_by_symbol)
    enriched: dict[str, pd.DataFrame] = {}
    for name, events in events_by_variant.items():
        if events.empty:
            enriched[name] = events
            continue
        frame = events.copy()
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], utc=True)
        frame["entry_date"] = pd.to_datetime(frame["entry_date"], utc=True)
        ctx = context.reindex(frame["signal_date"]).reset_index(drop=True)
        for column in ctx.columns:
            frame[column] = ctx[column].to_numpy()

        symbol_5d: list[float] = []
        symbol_10d: list[float] = []
        symbol_atr_pct: list[float] = []
        for _, row in frame.iterrows():
            daily = daily_by_symbol.get(str(row["symbol"]))
            signal_date = row["signal_date"]
            if daily is None or signal_date not in daily.index:
                symbol_5d.append(np.nan)
                symbol_10d.append(np.nan)
                symbol_atr_pct.append(np.nan)
                continue
            loc = daily.index.get_loc(signal_date)
            if not isinstance(loc, int):
                symbol_5d.append(np.nan)
                symbol_10d.append(np.nan)
            else:
                close = float(daily.iloc[loc]["close"])
                ret5 = close / float(daily.iloc[max(0, loc - 5)]["close"]) - 1.0 if loc >= 5 else np.nan
                ret10 = close / float(daily.iloc[max(0, loc - 10)]["close"]) - 1.0 if loc >= 10 else np.nan
                symbol_5d.append(ret5)
                symbol_10d.append(ret10)
            symbol_atr_pct.append(float(daily.loc[signal_date].get("atr_pct", np.nan)))
        frame["symbol_5d_return"] = symbol_5d
        frame["symbol_10d_return"] = symbol_10d
        frame["atr_pct"] = symbol_atr_pct
        frame["rel_to_btc_1d"] = frame["day_return"] - frame["btc_day_return"]
        frame["rel_to_btc_5d"] = frame["symbol_5d_return"] - frame["btc_5d_return"]
        frame["rel_to_market_1d"] = frame["day_return"] - frame["market_median_return"]
        enriched[name] = frame.replace([np.inf, -np.inf], np.nan)
    return enriched


def choose_base_variants(seed_summary_path: Path, all_variants: list[base.Variant], limit: int) -> list[str]:
    fallback = [
        "bear_cont_s0.58_cl0.35_mnone",
        "bear_cont_s0.54_cl0.35_mnone",
        "bear_cont_s0.62_cl0.45_mnone",
        "bear_cont_s0.62_cl0.35_mbtc_below_ema50",
        "pump_dist_r0.015_v1.05_s0.50_cl0.60_mnone",
        "failed_upper_s0.52_w0.28_mnone",
    ]
    available = {variant.name for variant in all_variants}
    selected: list[str] = []
    if seed_summary_path.exists():
        summary = pd.read_csv(seed_summary_path)
        broad = summary[(summary["all_total_trades"] >= 20) & (summary["train_return_pct"] > 0)]
        broad = broad.sort_values(["robust_score", "all_profit_factor"], ascending=False)
        selected.extend([name for name in broad["variant"].tolist() if name in available])
        sparse = summary[
            (summary["family"].isin(["pump_distribution", "failed_upper_breakout"]))
            & (summary["all_total_trades"] >= 8)
            & (summary["all_profit_factor"] >= 1.0)
        ].sort_values(["future_3d_win_pct", "all_profit_factor"], ascending=False)
        selected.extend([name for name in sparse["variant"].tolist() if name in available])
    selected.extend([name for name in fallback if name in available])
    deduped = list(dict.fromkeys(selected))
    return deduped[:limit]


def filter_events(events: pd.DataFrame, filter_name: str) -> pd.DataFrame:
    if events.empty or filter_name == "none":
        return events.copy()
    rules: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "btc_weak": lambda df: df["btc_below_ema20"].fillna(False),
        "market_weak": lambda df: df["market_breadth_below_ema20"].fillna(0.0) >= 0.55,
        "deep_market_weak": lambda df: df["market_breadth_below_ema50"].fillna(0.0) >= 0.55,
        "relative_weak_1d": lambda df: df["rel_to_btc_1d"].fillna(0.0) <= -0.003,
        "relative_weak_5d": lambda df: df["rel_to_btc_5d"].fillna(0.0) <= -0.01,
        "not_panic": lambda df: df["atr_pct"].fillna(99.0).between(0.01, 0.09),
        "quality_sell": lambda df: (df["sell_volume_ratio"].fillna(0.0) >= 0.58)
        & (df["down_hour_ratio"].fillna(0.0) >= 0.50),
        "late_dump": lambda df: df["late_sell_ratio"].fillna(0.0) >= 0.16,
        "weak_and_quality": lambda df: (df["market_breadth_below_ema20"].fillna(0.0) >= 0.50)
        & (df["sell_volume_ratio"].fillna(0.0) >= 0.56)
        & (df["atr_pct"].fillna(99.0).between(0.01, 0.10)),
        "relative_market_weak": lambda df: df["rel_to_market_1d"].fillna(0.0) <= -0.003,
    }
    if filter_name not in rules:
        raise ValueError(f"Unsupported event filter: {filter_name}")
    return events[rules[filter_name](events)].copy()


def select_symbols(train_trades: pd.DataFrame, selector: str, top_symbols: int) -> list[str]:
    if selector == "all":
        return []
    if train_trades.empty:
        return []
    trades = train_trades.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], utc=True)

    grouped = trades.groupby("symbol").agg(
        trades=("pnl", "size"),
        pnl=("pnl", "sum"),
        win_rate=("pnl", lambda values: float((values > 0).mean())),
        avg_pnl=("pnl", "mean"),
        gross_win=("pnl", lambda values: float(values[values > 0].sum())),
        gross_loss=("pnl", lambda values: float(values[values <= 0].sum())),
    )
    grouped["pf"] = grouped["gross_win"] / grouped["gross_loss"].abs().replace(0.0, np.nan)
    grouped["pf"] = grouped["pf"].replace([np.inf, -np.inf], 99.0).fillna(0.0)
    month_symbol = trades.assign(month=trades["entry_date"].dt.to_period("M").astype(str)).groupby(["symbol", "month"])["pnl"].sum()
    positive_month_ratio = month_symbol.groupby("symbol").apply(lambda values: float((values > 0).mean()))
    grouped["positive_month_ratio"] = positive_month_ratio.reindex(grouped.index).fillna(0.0)

    if selector == "profitable_min2":
        chosen = grouped[(grouped["pnl"] > 0) & (grouped["trades"] >= 2)].sort_values("pnl", ascending=False)
    elif selector == "profitable_min4":
        chosen = grouped[(grouped["pnl"] > 0) & (grouped["trades"] >= 4)].sort_values("pnl", ascending=False)
    elif selector == "pf_top":
        chosen = grouped[(grouped["trades"] >= 2) & (grouped["pf"] >= 1.05)].sort_values(
            ["pf", "pnl"], ascending=False
        )
    elif selector == "stable_top":
        chosen = grouped[
            (grouped["pnl"] > 0)
            & (grouped["trades"] >= 3)
            & (grouped["positive_month_ratio"] >= 0.45)
        ].sort_values(["positive_month_ratio", "pnl"], ascending=False)
    elif selector == "anti_loser":
        chosen = grouped[(grouped["pnl"] > 0) | ((grouped["win_rate"] >= 0.50) & (grouped["trades"] >= 3))].sort_values(
            ["pnl", "win_rate"], ascending=False
        )
    else:
        raise ValueError(f"Unsupported selector: {selector}")
    return chosen.head(top_symbols).index.tolist()


def run_month(
    events: pd.DataFrame,
    variant: base.Variant,
    daily_by_symbol: dict[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float,
    risk_per_trade: float,
    leverage: float,
    max_notional_fraction: float,
    max_positions: int,
    fee_rate: float,
    slippage_bps: float,
) -> tuple[dict, pd.DataFrame]:
    summary, trades, _ = base.backtest_portfolio(
        events,
        daily_by_symbol,
        variant,
        start=start,
        end=end,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        leverage=leverage,
        max_notional_fraction=max_notional_fraction,
        max_positions=max_positions,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
    )
    return summary, trades


def rolling_validate(
    spec: IdeaSpec,
    events_by_variant: dict[str, pd.DataFrame],
    variant_lookup: dict[str, base.Variant],
    daily_by_symbol: dict[str, pd.DataFrame],
    *,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
    initial_capital: float,
    leverage: float,
    max_notional_fraction: float,
    fee_rate: float,
    slippage_bps: float,
) -> tuple[dict, pd.DataFrame]:
    variant = variant_lookup[spec.base_variant]
    raw_events = events_by_variant[spec.base_variant]
    filtered_events = filter_events(raw_events, spec.event_filter)
    months = pd.date_range(start_month, end_month, freq="MS", tz="UTC")
    equity = initial_capital
    rows: list[dict] = []
    previous_return = 0.0

    for month in months[:-1]:
        train_start = month - pd.DateOffset(months=spec.train_months)
        train_end = month
        test_start = month
        test_end = month + pd.DateOffset(months=1)

        skipped = False
        if spec.cooldown_after_loss_pct is not None and previous_return <= spec.cooldown_after_loss_pct:
            skipped = True
            rows.append(
                {
                    "idea": spec.name,
                    "month": month.strftime("%Y-%m"),
                    "return_pct": 0.0,
                    "trades": 0,
                    "pf": 0.0,
                    "win_rate_pct": 0.0,
                    "whitelist_count": 0,
                    "whitelist": "",
                    "skipped": True,
                    "compounded_equity": round(equity, 2),
                }
            )
            previous_return = 0.0
            continue

        train_summary, train_trades = run_month(
            filtered_events,
            variant,
            daily_by_symbol,
            start=train_start,
            end=train_end,
            initial_capital=initial_capital,
            risk_per_trade=spec.risk_per_trade,
            leverage=leverage,
            max_notional_fraction=max_notional_fraction,
            max_positions=spec.max_positions,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
        )
        whitelist = select_symbols(train_trades, spec.selector, spec.top_symbols)
        test_events = filtered_events if spec.selector == "all" else filtered_events[filtered_events["symbol"].isin(whitelist)]
        test_summary, _ = run_month(
            test_events,
            variant,
            daily_by_symbol,
            start=test_start,
            end=test_end,
            initial_capital=initial_capital,
            risk_per_trade=spec.risk_per_trade,
            leverage=leverage,
            max_notional_fraction=max_notional_fraction,
            max_positions=spec.max_positions,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
        )
        month_return = float(test_summary["return_pct"])
        equity *= 1.0 + (month_return / 100.0)
        previous_return = month_return
        rows.append(
            {
                "idea": spec.name,
                "month": month.strftime("%Y-%m"),
                "return_pct": month_return,
                "trades": int(test_summary["total_trades"]),
                "pf": float(test_summary["profit_factor"]),
                "win_rate_pct": float(test_summary["win_rate_pct"]),
                "whitelist_count": len(whitelist) if spec.selector != "all" else int(filtered_events["symbol"].nunique()),
                "whitelist": ",".join(whitelist) if spec.selector != "all" else "ALL",
                "skipped": skipped,
                "compounded_equity": round(equity, 2),
                "train_trades": int(train_summary["total_trades"]),
                "train_return_pct": float(train_summary["return_pct"]),
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {
            "idea": spec.name,
            **asdict(spec),
            "months": 0,
            "total_return_pct": 0.0,
            "avg_month_ret": 0.0,
            "positive_months": 0,
            "positive_month_ratio": 0.0,
            "worst_month_pct": 0.0,
            "total_trades": 0,
            "active_months": 0,
            "score": -999.0,
        }, detail

    active = detail[~detail["skipped"]]
    total_return = (equity / initial_capital - 1.0) * 100.0
    positive_months = int((detail["return_pct"] > 0).sum())
    total_trades = int(detail["trades"].sum())
    worst_month = float(detail["return_pct"].min())
    positive_ratio = positive_months / len(detail) * 100.0 if len(detail) else 0.0
    avg_month = float(detail["return_pct"].mean())
    avg_pf = float(active[active["trades"] > 0]["pf"].replace(99.0, np.nan).mean()) if not active.empty else 0.0
    if not np.isfinite(avg_pf):
        avg_pf = 0.0
    avg_whitelist = float(active["whitelist_count"].mean()) if not active.empty else 0.0
    score = total_return + positive_ratio * 0.20 + avg_pf * 2.0 - abs(worst_month) * 1.4
    if total_trades < 12:
        score -= 20.0
    if positive_ratio < 45.0:
        score -= 8.0
    summary = {
        "idea": spec.name,
        **asdict(spec),
        "months": int(len(detail)),
        "active_months": int(len(active)),
        "total_return_pct": round(total_return, 2),
        "avg_month_ret": round(avg_month, 3),
        "positive_months": positive_months,
        "positive_month_ratio": round(positive_ratio, 2),
        "worst_month_pct": round(worst_month, 2),
        "total_trades": total_trades,
        "avg_pf": round(avg_pf, 2) if np.isfinite(avg_pf) else 0.0,
        "avg_whitelist": round(avg_whitelist, 2),
        "score": round(score, 4),
    }
    return summary, detail


def generate_specs(base_variants: list[str]) -> list[IdeaSpec]:
    event_filters = [
        "none",
        "market_weak",
        "relative_weak_5d",
        "quality_sell",
        "weak_and_quality",
        "not_panic",
    ]
    selectors = ["profitable_min2", "pf_top", "stable_top"]
    train_windows = [6]
    risk_profiles = [(0.004, 3), (0.006, 5)]
    specs: list[IdeaSpec] = []
    for base_variant in base_variants:
        family_hint = "pump" if base_variant.startswith("pump") else ("fail" if base_variant.startswith("failed") else "bear")
        selected_filters = event_filters
        if family_hint != "bear":
            selected_filters = ["none", "market_weak", "btc_weak", "quality_sell", "not_panic"]
        for event_filter in selected_filters:
            for selector in selectors:
                for train_months in train_windows:
                    for risk_per_trade, max_positions in risk_profiles:
                        top_symbols = 8 if selector in {"pf_top", "stable_top"} else 12
                        specs.append(
                            IdeaSpec(
                                name=(
                                    f"{base_variant}|f={event_filter}|sel={selector}|tw={train_months}"
                                    f"|r={risk_per_trade:.3f}|mp={max_positions}"
                                ),
                                base_variant=base_variant,
                                event_filter=event_filter,
                                selector=selector,
                                train_months=train_months,
                                top_symbols=top_symbols,
                                risk_per_trade=risk_per_trade,
                                max_positions=max_positions,
                            )
                        )
    return specs


def render_report(output_dir: Path, summary: pd.DataFrame, detail: pd.DataFrame) -> None:
    lines = [
        "# Daily Short Ideas Sweep",
        "",
        "This sweep tests extra filters on top of the daily short-pressure framework: relative weakness, BTC/market regime, sell-pressure quality, volatility avoidance, rolling symbol selection and risk sizing.",
        "",
        "## Best Ideas",
        summary.head(20).to_markdown(index=False),
        "",
    ]
    if not detail.empty and not summary.empty:
        best = str(summary.iloc[0]["idea"])
        best_detail = detail[detail["idea"] == best]
        lines.extend(["## Best Monthly Detail", best_detail.to_markdown(index=False), ""])
    lines.extend(
        [
            "## Interpretation",
            "- Prefer ideas with positive rolling return, more than 45% positive months, at least 12 trades and tolerable worst month.",
            "- Sparse pump/distribution ideas can be directionally interesting, but should not be promoted unless they keep producing signals in forward data.",
            "- A negative OOS month after strong training is a warning that the idea is a filter, not a standalone money machine.",
        ]
    )
    (output_dir / "ideas_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily = load_daily_data(Path(args.train_dir), Path(args.oos_dir))
    variants = base.build_variants()
    variant_lookup = {variant.name: variant for variant in variants}
    if args.base_variants.strip():
        base_variants = [item.strip() for item in args.base_variants.split(",") if item.strip()]
        missing = [name for name in base_variants if name not in variant_lookup]
        if missing:
            raise SystemExit(f"Unknown base variants: {missing}")
    else:
        base_variants = choose_base_variants(Path(args.seed_summary), variants, args.max_base_variants)
    selected_variants = [variant_lookup[name] for name in base_variants]
    raw_events = base.build_signal_events(daily, selected_variants)
    events = enrich_events(raw_events, daily)
    specs = generate_specs(base_variants)

    start_month = pd.Timestamp(args.start_month, tz="UTC")
    end_month = pd.Timestamp(args.end_month, tz="UTC")
    summary_rows: list[dict] = []
    detail_rows: list[pd.DataFrame] = []
    for index, spec in enumerate(specs, start=1):
        summary, detail = rolling_validate(
            spec,
            events,
            variant_lookup,
            daily,
            start_month=start_month,
            end_month=end_month,
            initial_capital=args.initial_capital,
            leverage=args.leverage,
            max_notional_fraction=args.max_notional_fraction,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
        )
        summary_rows.append(summary)
        if not detail.empty:
            detail_rows.append(detail)
        if index % 100 == 0:
            print(f"tested {index}/{len(specs)} ideas")

    summary_frame = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
    detail_frame = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    summary_frame.to_csv(output_dir / "ideas_summary.csv", index=False)
    detail_frame.to_csv(output_dir / "ideas_monthly_detail.csv", index=False)
    render_report(output_dir, summary_frame, detail_frame)

    print(f"Loaded symbols: {len(daily)}")
    print(f"Base variants: {len(base_variants)}")
    print(f"Ideas tested: {len(specs)}")
    print(f"Output: {output_dir}")
    print(
        summary_frame.head(20)[
            [
                "base_variant",
                "event_filter",
                "selector",
                "train_months",
                "risk_per_trade",
                "max_positions",
                "total_return_pct",
                "positive_month_ratio",
                "worst_month_pct",
                "total_trades",
                "avg_pf",
                "avg_whitelist",
                "score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
