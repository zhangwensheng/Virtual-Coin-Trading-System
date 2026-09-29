from __future__ import annotations

import argparse
import json
from pathlib import Path

from futures_strategy.factor_daily_scan import parse_config_paths, render_daily_scan_report, run_daily_factor_scan


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan daily Binance factor short candidates")
    parser.add_argument("--config-dir", default=None, help="Directory containing per-symbol YAML configs")
    parser.add_argument("--configs", default=None, help="Comma separated YAML config paths")
    parser.add_argument("--cache-dir", default="data/binance_factor_bundle", help="Dataset cache directory")
    parser.add_argument("--days-back", default=27, type=int, help="Lookback days for refreshed factor datasets")
    parser.add_argument("--end-time", default=None, help="Optional UTC end time, such as 2026-03-19T08:00:00Z")
    parser.add_argument("--no-refresh", action="store_true", help="Use config CSV paths without rebuilding latest factor datasets")
    parser.add_argument("--output-dir", default=None, help="Output folder for scan artifacts")
    args = parser.parse_args()

    config_paths = parse_config_paths(args.config_dir, args.configs)
    result = run_daily_factor_scan(
        config_paths=config_paths,
        refresh_data=not args.no_refresh,
        cache_dir=args.cache_dir,
        days_back=args.days_back,
        end_time=args.end_time,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("outputs") / f"daily_factor_scan_{result.scan_time.strftime('%Y-%m-%d_%H')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result.rows.to_csv(output_dir / "scan_rows.csv", index=False)
    (output_dir / "scan_rows.json").write_text(
        json.dumps(result.rows.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_daily_scan_report(result), encoding="utf-8")

    print("=== 每日筛币结果 ===")
    if result.rows.empty:
        print("no_rows")
    else:
        for _, row in result.rows.iterrows():
            print(
                f"{row['portfolio_rank']}. {row['symbol']} action={row['action']} "
                f"score={row['rank_score']} factor={row['factor_score']} "
                f"funding={row['funding_rate']} oi_change={row['oi_value_change_pct']}%"
            )
    print(f"\n输出目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
