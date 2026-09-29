from __future__ import annotations

import argparse
import json
from pathlib import Path

from futures_strategy.config import load_config
from futures_strategy.factor_short_optimization import (
    DEFAULT_FACTOR_PARAM_GRID,
    DEFAULT_FACTOR_SYMBOLS,
    build_param_grid,
    dump_best_config,
    dump_symbol_configs,
    optimize_factor_strategy,
    parse_csv_list,
    parse_float_list,
    parse_int_list,
    render_factor_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the funding + OI + price short strategy with rolling backtests")
    parser.add_argument(
        "--base-config",
        default="configs/binance_majors_funding_oi_short_base.yaml",
        help="Base YAML config for the enhanced factor short strategy",
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_FACTOR_SYMBOLS),
        help="Comma separated Binance native symbols, such as BTCUSDT,ETHUSDT",
    )
    parser.add_argument("--interval", default="1h", help="Binance kline/OI interval")
    parser.add_argument("--days-back", default=30, type=int, help="Recent lookback days, capped by Binance OI limits")
    parser.add_argument("--end-time", default=None, help="Optional UTC end time, such as 2026-03-18T00:00:00Z")
    parser.add_argument("--window-days", default=7, type=int, help="Rolling window size in days")
    parser.add_argument("--step-days", default=7, type=int, help="Rolling step in days")
    parser.add_argument("--cache-dir", default="data/binance_factor_bundle", help="Factor dataset cache folder")
    parser.add_argument("--output-dir", default=None, help="Folder for optimization outputs")
    parser.add_argument("--min-funding-rates", default=None, help="Comma separated values for enhanced_min_funding_rate")
    parser.add_argument("--min-funding-changes", default=None, help="Comma separated values for enhanced_min_funding_change")
    parser.add_argument("--min-oi-change-pcts", default=None, help="Comma separated values for enhanced_min_oi_change_pct")
    parser.add_argument("--min-price-bounce-pcts", default=None, help="Comma separated values for enhanced_min_price_bounce_pct")
    parser.add_argument("--entry-min-scores", default=None, help="Comma separated values for enhanced_entry_min_score")
    parser.add_argument("--risk-steps", default=None, help="Comma separated values for enhanced_risk_step")
    parser.add_argument("--max-risk-multipliers", default=None, help="Comma separated values for enhanced_max_risk_multiplier")
    parser.add_argument("--risk-per-trades", default=None, help="Comma separated values for risk.risk_per_trade")
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    symbols = [symbol.upper() for symbol in parse_csv_list(args.symbols, DEFAULT_FACTOR_SYMBOLS)]
    grid_values = {
        "enhanced_min_funding_rate": parse_float_list(
            args.min_funding_rates,
            DEFAULT_FACTOR_PARAM_GRID["enhanced_min_funding_rate"],
        ),
        "enhanced_min_funding_change": parse_float_list(
            args.min_funding_changes,
            DEFAULT_FACTOR_PARAM_GRID["enhanced_min_funding_change"],
        ),
        "enhanced_min_oi_change_pct": parse_float_list(
            args.min_oi_change_pcts,
            DEFAULT_FACTOR_PARAM_GRID["enhanced_min_oi_change_pct"],
        ),
        "enhanced_min_price_bounce_pct": parse_float_list(
            args.min_price_bounce_pcts,
            DEFAULT_FACTOR_PARAM_GRID["enhanced_min_price_bounce_pct"],
        ),
        "enhanced_entry_min_score": parse_int_list(
            args.entry_min_scores,
            [int(base_config.strategy.enhanced_entry_min_score)],
        ),
        "enhanced_risk_step": parse_float_list(
            args.risk_steps,
            [float(base_config.strategy.enhanced_risk_step)],
        ),
        "enhanced_max_risk_multiplier": parse_float_list(
            args.max_risk_multipliers,
            [float(base_config.strategy.enhanced_max_risk_multiplier)],
        ),
        "risk_per_trade": parse_float_list(
            args.risk_per_trades,
            [float(base_config.risk.risk_per_trade)],
        ),
    }
    param_grid = build_param_grid(grid_values)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        end_label = (args.end_time or "latest").replace(":", "-")
        output_dir = Path("outputs") / f"funding_oi_short_optimizer_{args.interval}_{end_label}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 增强版空头优化设置 ===")
    print(f"base_config: {Path(args.base_config).resolve()}")
    print(f"symbols: {symbols}")
    print(f"interval: {args.interval}")
    print(f"days_back: {args.days_back}")
    print(f"window_days: {args.window_days}")
    print(f"step_days: {args.step_days}")
    print(f"grid_size: {len(param_grid)}")
    print(f"cache_dir: {Path(args.cache_dir).resolve()}")
    print(f"output_dir: {output_dir.resolve()}")
    print("")

    results = optimize_factor_strategy(
        base_config=base_config,
        symbols=symbols,
        interval=args.interval,
        days_back=args.days_back,
        end_time=args.end_time,
        cache_dir=Path(args.cache_dir),
        param_grid=param_grid,
        window_days=args.window_days,
        step_days=args.step_days,
        logger=print,
    )

    results["window_results"].to_csv(output_dir / "window_results.csv", index=False)
    results["rolling_summary"].to_csv(output_dir / "rolling_summary.csv", index=False)
    results["symbol_rolling_summary"].to_csv(output_dir / "symbol_rolling_summary.csv", index=False)
    results["symbol_best_summary"].to_csv(output_dir / "symbol_best_summary.csv", index=False)
    results["symbol_best_backtests"].to_csv(output_dir / "symbol_best_backtests.csv", index=False)
    results["shortlist"].to_csv(output_dir / "shortlist.csv", index=False)
    results["month_symbol_scores"].to_csv(output_dir / "month_symbol_scores.csv", index=False)
    (output_dir / "best_params.json").write_text(
        json.dumps(results["best_params"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_factor_report(results), encoding="utf-8")
    dump_best_config(base_config, results["best_params"], output_dir / "best_config.yaml")
    dump_symbol_configs(base_config, results["shortlist"], output_dir / "symbol_configs")

    print("")
    print("=== 最优参数 ===")
    print(f"best_combo_id: {results['best_combo_id']}")
    for key, value in results["best_params"].items():
        print(f"{key}: {value}")

    print("")
    print("=== 月份/币种筛选 ===")
    preview = results["month_symbol_scores"][["month", "symbol", "suitability_score", "return_pct", "max_drawdown_pct"]]
    for _, row in preview.iterrows():
        print(
            f"{row['month']} {row['symbol']} score={row['suitability_score']} "
            f"return={row['return_pct']}% dd={row['max_drawdown_pct']}%"
        )

    print("")
    print("=== 激进利润候选 ===")
    shortlist = results["shortlist"][
        ["symbol", "return_pct", "max_drawdown_pct", "profit_factor", "total_trades", "full_profit_focus_score"]
    ]
    if shortlist.empty:
        print("no_recommended_symbols")
    else:
        for _, row in shortlist.iterrows():
            print(
                f"{row['symbol']} return={row['return_pct']}% dd={row['max_drawdown_pct']}% "
                f"pf={row['profit_factor']} trades={row['total_trades']} score={row['full_profit_focus_score']}"
            )

    print("")
    print(f"results_dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
