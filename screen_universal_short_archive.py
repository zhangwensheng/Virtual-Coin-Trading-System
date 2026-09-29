from __future__ import annotations

import argparse
import json
from pathlib import Path

from futures_strategy.factor_short_optimization import parse_csv_list
from futures_strategy.universal_short_screen import (
    DEFAULT_STRATEGY_CONFIGS,
    DEFAULT_UNIVERSAL_SHORT_SYMBOLS,
    dump_survivor_configs,
    render_universal_short_report,
    screen_universal_short_strategies,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen universal short strategies on 2y Binance futures archives")
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_UNIVERSAL_SHORT_SYMBOLS),
        help="Comma separated Binance native symbols",
    )
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGY_CONFIGS),
        help="Comma separated base YAML strategy configs",
    )
    parser.add_argument("--timeframe", default="1h", help="Archive timeframe, such as 1h")
    parser.add_argument("--start-month", default="2024-03", help="Inclusive start month YYYY-MM")
    parser.add_argument("--end-month", default="2026-02", help="Inclusive end month YYYY-MM")
    parser.add_argument("--market", default="um", choices=["um", "cm"], help="Binance archive market")
    parser.add_argument("--data-dir", default="data/binance_universal_short_2y_1h", help="Archive cache folder")
    parser.add_argument(
        "--output-dir",
        default="outputs/universal_short_screen_2024-03__2026-02",
        help="Directory for reports and survivor configs",
    )
    parser.add_argument("--force-download", action="store_true", help="Redownload CSV archives even if cached")
    args = parser.parse_args()

    symbols = [symbol.upper() for symbol in parse_csv_list(args.symbols, DEFAULT_UNIVERSAL_SHORT_SYMBOLS)]
    strategies = parse_csv_list(args.strategies, DEFAULT_STRATEGY_CONFIGS)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 两年空头筛选设置 ===")
    print(f"symbols: {symbols}")
    print(f"strategies: {strategies}")
    print(f"window: {args.start_month} -> {args.end_month}")
    print(f"data_dir: {Path(args.data_dir).resolve()}")
    print(f"output_dir: {output_dir.resolve()}")
    print("")

    results = screen_universal_short_strategies(
        symbols=symbols,
        strategy_config_paths=strategies,
        data_dir=args.data_dir,
        start_month=args.start_month,
        end_month=args.end_month,
        timeframe=args.timeframe,
        market=args.market,
        force_download=args.force_download,
        logger=print,
    )

    results["summary"].to_csv(output_dir / "strategy_symbol_summary.csv", index=False)
    results["monthly_results"].to_csv(output_dir / "monthly_results.csv", index=False)
    results["survivor_summary"].to_csv(output_dir / "survivor_summary.csv", index=False)
    results["best_per_symbol"].to_csv(output_dir / "best_per_symbol.csv", index=False)
    results["best_survivor_per_symbol"].to_csv(output_dir / "best_survivor_per_symbol.csv", index=False)
    results["strategy_summary"].to_csv(output_dir / "strategy_summary.csv", index=False)
    (output_dir / "report.md").write_text(render_universal_short_report(results), encoding="utf-8")
    (output_dir / "meta.json").write_text(
        json.dumps(
            {
                "start_month": results["start_month"],
                "end_month": results["end_month"],
                "symbols": results["symbols"],
                "strategy_candidates": results["strategy_candidates"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    dump_survivor_configs(
        results["survivor_summary"],
        results["candidate_map"],
        results["dataset_paths"],
        output_dir / "survivor_configs",
    )

    print("")
    print("=== 策略总览 ===")
    if results["strategy_summary"].empty:
        print("no_strategy_summary")
    else:
        for _, row in results["strategy_summary"].iterrows():
            print(
                f"{row['strategy_name']} survivors={int(row['survivor_count'])} "
                f"avg_full_return={row['avg_full_return_pct']}% "
                f"avg_month_return={row['avg_mean_month_return_pct']}% "
                f"avg_month_win_ratio={row['avg_profitable_month_ratio']}"
            )

    print("")
    print("=== 幸存组合 ===")
    if results["survivor_summary"].empty:
        print("no_survivors")
    else:
        for _, row in results["survivor_summary"].iterrows():
            print(
                f"{row['symbol']} + {row['strategy_name']} full={row['full_return_pct']}% "
                f"month_mean={row['mean_month_return_pct']}% "
                f"month_win_ratio={row['profitable_month_ratio']} "
                f"score={row['survivor_score']}"
            )

    print("")
    print(f"results_dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
