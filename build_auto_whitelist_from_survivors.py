from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build auto-trader whitelist configs from best survivor rows")
    parser.add_argument(
        "--screen-dir",
        default="outputs/universal_short_screen_2024-03__2026-02",
        help="Directory produced by screen_universal_short_archive.py",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Optional source survivor config dir (defaults to <screen-dir>/survivor_configs)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output whitelist dir (defaults to <screen-dir>/auto_whitelist_11_configs)",
    )
    args = parser.parse_args()

    screen_dir = Path(args.screen_dir)
    source_dir = Path(args.source_dir) if args.source_dir else screen_dir / "survivor_configs"
    output_dir = Path(args.output_dir) if args.output_dir else screen_dir / "auto_whitelist_11_configs"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_survivor_path = screen_dir / "best_survivor_per_symbol.csv"
    if not best_survivor_path.exists():
        raise FileNotFoundError(f"best_survivor_per_symbol.csv not found: {best_survivor_path}")
    if not source_dir.exists():
        raise FileNotFoundError(f"survivor config source dir not found: {source_dir}")

    rows = pd.read_csv(best_survivor_path)
    created: list[Path] = []
    for _, row in rows.iterrows():
        symbol = str(row["symbol"]).lower()
        strategy_name = str(row["strategy_name"])
        source_name = f"{symbol}_{strategy_name}.yaml"
        source_path = source_dir / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Survivor config missing: {source_path}")
        target_path = output_dir / source_name
        shutil.copy2(source_path, target_path)
        created.append(target_path)

    print("=== 已生成自动交易 11 币白名单配置 ===")
    print(f"source_dir: {source_dir.resolve()}")
    print(f"output_dir: {output_dir.resolve()}")
    print(f"count: {len(created)}")
    for path in created:
        print(path.name)


if __name__ == "__main__":
    main()
