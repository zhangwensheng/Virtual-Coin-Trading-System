from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from futures_strategy.config import load_config
from futures_strategy.factor_portfolio import (
    load_prepared_symbol_frames,
    override_portfolio_risk,
    run_factor_portfolio_backtest,
)
from futures_strategy.top_gainers_boll import (
    build_realtime_top_gainers_boll_datasets,
    build_top_gainers_boll_datasets,
    write_symbol_configs,
)


def parse_symbols(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    values = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return values or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run top gainers Bollinger short universe backtest")
    parser.add_argument(
        "--base-config",
        default="configs/binance_top_gainers_boll_short_1m_base.yaml",
        help="Base YAML config for the top gainers Bollinger short strategy",
    )
    parser.add_argument("--start-month", required=True, help="Backtest start month, such as 2026-02")
    parser.add_argument("--end-month", required=True, help="Backtest end month, such as 2026-02")
    parser.add_argument("--symbols", default=None, help="Optional comma separated Binance native symbols")
    parser.add_argument(
        "--selection-mode",
        choices=["realtime", "daily-close"],
        default="realtime",
        help="Use minute-by-minute current-day ranking or completed daily-close ranking",
    )
    parser.add_argument("--max-positions", default=2, type=int, help="Maximum concurrent positions")
    parser.add_argument("--override-risk-per-trade", default=None, type=float, help="Optional risk override")
    parser.add_argument(
        "--portfolio-notional-fraction",
        default=0.9,
        type=float,
        help="Total notional cap across concurrent positions",
    )
    parser.add_argument(
        "--symbol-daily-max-losses",
        default=None,
        type=int,
        help="Disable a symbol for the rest of a UTC day after this many losing trades",
    )
    parser.add_argument(
        "--symbol-daily-loss-limit-pct",
        default=None,
        type=float,
        help="Disable a symbol for the rest of a UTC day after losing this fraction of initial equity",
    )
    parser.add_argument(
        "--portfolio-daily-loss-limit-pct",
        default=None,
        type=float,
        help="Stop opening new positions for the UTC day after this portfolio loss fraction",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/top_gainers_boll_short_1m",
        help="Directory for downloaded archive caches and enriched CSVs",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/top_gainers_boll_short_1m",
        help="Directory for configs, manifests and portfolio outputs",
    )
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "symbol_configs"

    with requests.Session() as session:
        if args.selection_mode == "realtime":
            bundle = build_realtime_top_gainers_boll_datasets(
                base_config=base_config,
                start_month=args.start_month,
                end_month=args.end_month,
                symbols=parse_symbols(args.symbols),
                cache_dir=Path(args.cache_dir),
                session=session,
            )
        else:
            bundle = build_top_gainers_boll_datasets(
                base_config=base_config,
                start_month=args.start_month,
                end_month=args.end_month,
                symbols=parse_symbols(args.symbols),
                cache_dir=Path(args.cache_dir),
                session=session,
            )

    enriched_paths = bundle["enriched_paths"]
    if not enriched_paths:
        raise RuntimeError("No enriched top-gainer datasets available for backtest.")

    config_paths = write_symbol_configs(
        base_config,
        enriched_paths,
        output_dir=config_dir,
        output_name_suffix="top_gainers_boll_short_1m",
    )

    configs, prepared_frames = load_prepared_symbol_frames(config_paths)
    configs = override_portfolio_risk(configs, risk_per_trade=args.override_risk_per_trade)
    result = run_factor_portfolio_backtest(
        configs=configs,
        prepared_frames=prepared_frames,
        max_positions=max(1, args.max_positions),
        portfolio_notional_fraction=max(0.1, float(args.portfolio_notional_fraction)),
        symbol_daily_max_losses=args.symbol_daily_max_losses,
        symbol_daily_loss_limit_pct=args.symbol_daily_loss_limit_pct,
        portfolio_daily_loss_limit_pct=args.portfolio_daily_loss_limit_pct,
    )

    result.trades.to_csv(output_dir / "trades.csv", index=False)
    result.equity_curve.to_csv(output_dir / "equity_curve.csv")
    result.symbol_summary.to_csv(output_dir / "symbol_summary.csv", index=False)
    selection_frame = bundle["selection_frame"]
    if isinstance(selection_frame, pd.DataFrame) and not selection_frame.empty:
        selection_frame.to_csv(output_dir / "daily_selection.csv", index=False)

    manifest = {
        "base_config": str(Path(args.base_config).resolve()),
        "selection_mode": args.selection_mode,
        "start_month": args.start_month,
        "end_month": args.end_month,
        "max_positions": args.max_positions,
        "portfolio_notional_fraction": args.portfolio_notional_fraction,
        "override_risk_per_trade": args.override_risk_per_trade,
        "symbol_daily_max_losses": args.symbol_daily_max_losses,
        "symbol_daily_loss_limit_pct": args.symbol_daily_loss_limit_pct,
        "portfolio_daily_loss_limit_pct": args.portfolio_daily_loss_limit_pct,
        "symbols_requested": parse_symbols(args.symbols),
        "universe_size": len(bundle["universe"]),
        "selected_symbol_count": len(bundle["selected_symbols"]),
        "selected_symbols": bundle["selected_symbols"],
        "usable_symbol_count": len(prepared_frames),
        "usable_symbols": sorted(prepared_frames.keys()),
        "skipped_daily": bundle["skipped_daily"],
        "skipped_intraday": bundle["skipped_intraday"],
        "summary": result.summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("=== Top Gainers Boll Short 回测结果 ===")
    for key, value in result.summary.items():
        print(f"{key}: {value}")
    print(f"selected_symbols: {','.join(bundle['selected_symbols'])}")
    print(f"usable_symbols: {','.join(sorted(prepared_frames.keys()))}")
    print(f"output_dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
