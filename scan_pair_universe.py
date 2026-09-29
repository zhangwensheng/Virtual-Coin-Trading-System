from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from futures_strategy.optimization import expand_months
from futures_strategy.pair_optimization import (
    DEFAULT_PAIR_UNIVERSE,
    best_pair_payload,
    build_rolling_windows,
    optimize_pair_universe,
    parse_pair_list,
    render_pair_markdown_report,
)
from futures_strategy.pairs import load_pair_config


def write_best_config(base_config_path: str, best_params: dict[str, float], destination: Path) -> None:
    config = load_pair_config(base_config_path)
    for key, value in best_params.items():
        setattr(config.strategy, key, value)
    config.output.output_dir = str(destination.parent)
    destination.write_text(
        yaml.safe_dump(asdict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a pair-trading strategy across multiple pairs and rolling windows")
    parser.add_argument(
        "--base-config",
        default="configs/binance_pairs_base.yaml",
        help="Base YAML config for pair strategy settings",
    )
    parser.add_argument(
        "--pairs",
        default=",".join(f"{left}:{right}" for left, right in DEFAULT_PAIR_UNIVERSE),
        help="Comma separated pairs, such as BTCUSDT:ETHUSDT,SOLUSDT:ETHUSDT",
    )
    parser.add_argument("--months", default="2025-09:2026-02", help="Months to scan, e.g. 2025-09:2026-02")
    parser.add_argument("--train-window-months", type=int, default=2, help="How many months in each rolling train window")
    parser.add_argument("--valid-window-months", type=int, default=1, help="How many months in each rolling validation window")
    parser.add_argument("--market", default="um", choices=["um", "cm"], help="Binance archive market type")
    parser.add_argument("--cache-dir", default="data/binance_archive", help="Folder for monthly CSV cache")
    parser.add_argument("--output-dir", default=None, help="Folder for scan outputs")
    args = parser.parse_args()

    base_config = load_pair_config(args.base_config)
    pairs = parse_pair_list(args.pairs, DEFAULT_PAIR_UNIVERSE)
    months = expand_months(args.months)
    rolling_windows = build_rolling_windows(months, args.train_window_months, args.valid_window_months)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("outputs") / f"pair_scanner_{months[0]}__{months[-1]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_params = {
        "beta_window": base_config.strategy.beta_window,
        "zscore_window": base_config.strategy.zscore_window,
        "entry_z": base_config.strategy.entry_z,
        "exit_z": base_config.strategy.exit_z,
        "stop_z": base_config.strategy.stop_z,
        "min_correlation": base_config.strategy.min_correlation,
        "correlation_exit_buffer": base_config.strategy.correlation_exit_buffer,
        "max_holding_bars": base_config.strategy.max_holding_bars,
    }

    print("=== 多配对扫描设置 ===")
    print(f"base_config: {Path(args.base_config).resolve()}")
    print(f"pairs: {[f'{left}:{right}' for left, right in pairs]}")
    print(f"months: {months}")
    print(f"rolling_windows: {len(rolling_windows)}")
    print(f"output_dir: {output_dir.resolve()}")
    print("")

    results = optimize_pair_universe(
        base_config=base_config,
        pairs=pairs,
        rolling_windows=rolling_windows,
        param_grid=[base_params],
        cache_dir=Path(args.cache_dir),
        market=args.market,
        logger=print,
    )

    cell_results = results["cell_results"]
    aggregate_results = results["aggregate_results"]
    selection_table = results["selection_table"]
    payload = best_pair_payload(
        selection_table=selection_table,
        aggregate_results=aggregate_results,
        cell_results=cell_results,
        rolling_windows=rolling_windows,
        pairs=results["usable_pairs"],
    )

    cell_results.to_csv(output_dir / "cell_results.csv", index=False)
    aggregate_results.to_csv(output_dir / "aggregate_results.csv", index=False)
    selection_table.to_csv(output_dir / "selection_table.csv", index=False)
    (output_dir / "best_params.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "rolling_windows.json").write_text(
        json.dumps(rolling_windows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_pair_markdown_report(payload, selection_table),
        encoding="utf-8",
    )
    (output_dir / "skipped_symbols.json").write_text(
        json.dumps(results["skipped_symbols"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "skipped_pairs.json").write_text(
        json.dumps(results["skipped_pairs"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_combo = payload.get("best_combo")
    if best_combo is not None:
        write_best_config(args.base_config, best_combo["params"], output_dir / "best_config.yaml")

    print("=== 扫描结果 ===")
    if best_combo is None:
        print("没有得到可用结果。")
    else:
        print(f"selection_score: {best_combo['selection_score']}")
        for key, value in best_combo["params"].items():
            print(f"{key}: {value}")
        for split_name, metrics in best_combo["split_metrics"].items():
            print(
                f"{split_name}: score={metrics['robustness_score']} "
                f"mean_return={metrics['mean_return_pct']}% "
                f"profitable_pairs={metrics['profitable_pair_ratio']} "
                f"avg_dd={metrics['avg_max_drawdown_pct']}%"
            )
    print(f"results_dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
