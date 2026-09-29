from __future__ import annotations

import argparse
import json
from pathlib import Path

from futures_strategy.config import load_config
from futures_strategy.optimization import (
    DEFAULT_PARAM_GRID,
    DEFAULT_SMALLCAP_SYMBOLS,
    best_params_payload,
    build_param_grid,
    expand_months,
    optimize_across_symbols,
    parse_csv_list,
    parse_float_list,
    render_markdown_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the smallcap 1m short strategy across Binance futures symbols")
    parser.add_argument(
        "--base-config",
        default="configs/binance_smallcap_short_base.yaml",
        help="Base YAML config for risk and fixed strategy settings",
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SMALLCAP_SYMBOLS),
        help="Comma separated Binance native symbols, such as 1000PEPEUSDT,1000BONKUSDT",
    )
    parser.add_argument("--train-months", default="2026-01", help="Comma separated months or ranges like 2026-01,2026-02:2026-03")
    parser.add_argument("--valid-months", default="2026-02", help="Optional validation months")
    parser.add_argument("--market", default="um", choices=["um", "cm"], help="Binance archive market type")
    parser.add_argument("--cache-dir", default="data/binance_archive", help="Folder for monthly CSV cache")
    parser.add_argument("--output-dir", default=None, help="Folder for optimization outputs")
    parser.add_argument("--day-return-thresholds", default=None, help="Comma separated values for intraday_day_return_threshold")
    parser.add_argument("--volume-spike-thresholds", default=None, help="Comma separated values for intraday_volume_spike_threshold")
    parser.add_argument("--sell-volume-ratios", default=None, help="Comma separated values for intraday_sell_volume_ratio")
    parser.add_argument("--upper-wick-ratios", default=None, help="Comma separated values for intraday_upper_wick_ratio")
    parser.add_argument("--drop-from-peak-thresholds", default=None, help="Comma separated values for intraday_drop_from_peak_threshold")
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    symbols = [symbol.upper() for symbol in parse_csv_list(args.symbols, DEFAULT_SMALLCAP_SYMBOLS)]
    train_months = expand_months(args.train_months)
    valid_months = expand_months(args.valid_months)
    split_months = {
        "train": train_months,
        "valid": valid_months,
    }

    grid_values = {
        "intraday_day_return_threshold": parse_float_list(
            args.day_return_thresholds,
            DEFAULT_PARAM_GRID["intraday_day_return_threshold"],
        ),
        "intraday_volume_spike_threshold": parse_float_list(
            args.volume_spike_thresholds,
            DEFAULT_PARAM_GRID["intraday_volume_spike_threshold"],
        ),
        "intraday_sell_volume_ratio": parse_float_list(
            args.sell_volume_ratios,
            DEFAULT_PARAM_GRID["intraday_sell_volume_ratio"],
        ),
        "intraday_upper_wick_ratio": parse_float_list(
            args.upper_wick_ratios,
            DEFAULT_PARAM_GRID["intraday_upper_wick_ratio"],
        ),
        "intraday_drop_from_peak_threshold": parse_float_list(
            args.drop_from_peak_thresholds,
            DEFAULT_PARAM_GRID["intraday_drop_from_peak_threshold"],
        ),
    }
    param_grid = build_param_grid(grid_values)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        train_label = "-".join(train_months) if train_months else "no-train"
        valid_label = "-".join(valid_months) if valid_months else "no-valid"
        output_dir = Path("outputs") / f"smallcap_optimizer_{train_label}__{valid_label}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 优化设置 ===")
    print(f"base_config: {Path(args.base_config).resolve()}")
    print(f"symbols: {symbols}")
    print(f"train_months: {train_months}")
    print(f"valid_months: {valid_months}")
    print(f"grid_size: {len(param_grid)}")
    print(f"output_dir: {output_dir.resolve()}")
    print("")

    results = optimize_across_symbols(
        base_config=base_config,
        symbols=symbols,
        split_months=split_months,
        param_grid=param_grid,
        cache_dir=Path(args.cache_dir),
        market=args.market,
        logger=print,
    )

    symbol_results = results["symbol_results"]
    aggregate_results = results["aggregate_results"]
    selection_table = results["selection_table"]
    payload = best_params_payload(
        selection_table=selection_table,
        aggregate_results=aggregate_results,
        symbol_results=symbol_results,
        split_months=split_months,
        symbols=results["usable_symbols"],
    )

    symbol_results.to_csv(output_dir / "symbol_results.csv", index=False)
    aggregate_results.to_csv(output_dir / "aggregate_results.csv", index=False)
    selection_table.to_csv(output_dir / "selection_table.csv", index=False)
    (output_dir / "best_params.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_markdown_report(payload, selection_table),
        encoding="utf-8",
    )
    (output_dir / "skipped_symbols.json").write_text(
        json.dumps(results["skipped_symbols"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("")
    print("=== 最优结果 ===")
    best_combo = payload.get("best_combo")
    if best_combo is None:
        print("没有可用结果。")
    else:
        print(f"combo_id: {best_combo['combo_id']}")
        print(f"selection_score: {best_combo['selection_score']}")
        for key, value in best_combo["params"].items():
            print(f"{key}: {value}")
        for split_name, metrics in best_combo["split_metrics"].items():
            print(
                f"{split_name}: score={metrics['robustness_score']} "
                f"mean_return={metrics['mean_return_pct']}% "
                f"profitable_ratio={metrics['profitable_ratio']} "
                f"avg_dd={metrics['avg_max_drawdown_pct']}%"
            )

    print("")
    print(f"usable_symbols: {results['usable_symbols']}")
    print(f"skipped_symbols: {len(results['skipped_symbols'])}")
    print(f"results_dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
