from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from futures_strategy.factor_alert_monitor import (
    append_alerts_jsonl,
    run_alert_scan,
    save_latest_alerts,
)
from futures_strategy.factor_daily_scan import parse_config_paths, render_daily_scan_report


def write_scan_outputs(output_dir: Path, scan_rows) -> None:
    scan_rows.to_csv(output_dir / "scan_rows.csv", index=False)
    (output_dir / "scan_rows.json").write_text(
        json.dumps(scan_rows.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_scan(scan_rows) -> None:
    if scan_rows.empty:
        print("no_rows")
        return
    for _, row in scan_rows.iterrows():
        print(
            f"{row['portfolio_rank']}. {row['symbol']} action={row['action']} "
            f"score={row['rank_score']} factor={row['factor_score']} "
            f"funding={row['funding_rate']} oi_change={row['oi_value_change_pct']}%"
        )


def print_alerts(alerts) -> None:
    if not alerts:
        print("no_new_alerts")
        return
    for alert in alerts:
        print(
            f"ALERT {alert.symbol} reason={alert.reason} action={alert.action} "
            f"prev={alert.previous_action} rank={alert.portfolio_rank} "
            f"score={alert.rank_score} stop={alert.stop_hint}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor factor short alerts")
    parser.add_argument("--config-dir", default=None, help="Directory containing per-symbol YAML configs")
    parser.add_argument("--configs", default=None, help="Comma separated YAML config paths")
    parser.add_argument("--cache-dir", default="data/binance_factor_bundle", help="Dataset cache directory")
    parser.add_argument("--days-back", default=27, type=int, help="Lookback days for refreshed factor datasets")
    parser.add_argument("--end-time", default=None, help="Optional UTC end time for one-off scans")
    parser.add_argument("--top-n", default=2, type=int, help="Top ranked setup entries to alert on")
    parser.add_argument("--interval-seconds", default=300, type=int, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run one scan and exit")
    parser.add_argument("--no-refresh", action="store_true", help="Use config CSV paths without refreshing factors")
    parser.add_argument("--output-dir", default="outputs/factor_alert_monitor", help="Alert monitor output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "monitor_state.json"
    alerts_path = output_dir / "alerts.jsonl"
    latest_alerts_path = output_dir / "latest_alerts.json"
    report_path = output_dir / "latest_report.md"

    config_paths = parse_config_paths(args.config_dir, args.configs)

    while True:
        scan, alerts = run_alert_scan(
            config_paths=config_paths,
            state_path=state_path,
            top_n=args.top_n,
            refresh_data=not args.no_refresh,
            cache_dir=args.cache_dir,
            days_back=args.days_back,
            end_time=args.end_time if args.once else None,
        )

        write_scan_outputs(output_dir, scan.rows)
        report_path.write_text(render_daily_scan_report(scan), encoding="utf-8")
        append_alerts_jsonl(alerts_path, alerts)
        save_latest_alerts(latest_alerts_path, alerts)

        print("=== 当前筛币 ===")
        print_scan(scan.rows)
        print("\n=== 新告警 ===")
        print_alerts(alerts)
        print(f"\n输出目录: {output_dir.resolve()}")

        if args.once:
            break
        time.sleep(max(30, args.interval_seconds))


if __name__ == "__main__":
    main()
