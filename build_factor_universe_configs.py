from __future__ import annotations

import argparse
import json
from pathlib import Path

from futures_strategy.config import load_config
from futures_strategy.factor_universe import (
    DEFAULT_MIN_LISTING_DAYS,
    DEFAULT_MIN_QUOTE_VOLUME,
    DEFAULT_STRICT_UNIVERSE_SIZE,
    build_shared_symbol_configs,
    refresh_factor_datasets_for_symbols,
    select_strict_short_universe,
)
from futures_strategy.factor_short_optimization import parse_csv_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a strict Binance futures short universe config set")
    parser.add_argument(
        "--base-config",
        default="configs/binance_funding_oi_short_strict_shared.yaml",
        help="Base YAML config for the strict shared short strategy",
    )
    parser.add_argument("--symbols", default=None, help="Optional comma separated native Binance symbols")
    parser.add_argument("--size", default=DEFAULT_STRICT_UNIVERSE_SIZE, type=int, help="Target universe size")
    parser.add_argument(
        "--min-listing-days",
        default=DEFAULT_MIN_LISTING_DAYS,
        type=int,
        help="Only include symbols listed at least this many days",
    )
    parser.add_argument(
        "--min-quote-volume",
        default=DEFAULT_MIN_QUOTE_VOLUME,
        type=float,
        help="Only include symbols whose 24h quote volume exceeds this threshold",
    )
    parser.add_argument("--cache-dir", default="data/binance_factor_bundle", help="Factor dataset cache directory")
    parser.add_argument("--days-back", default=27, type=int, help="Recent lookback days for factor data")
    parser.add_argument("--end-time", default=None, help="Optional UTC end time, such as 2026-03-24T00:00:00Z")
    parser.add_argument("--skip-refresh", action="store_true", help="Do not prebuild factor CSVs")
    parser.add_argument(
        "--output-dir",
        default="outputs/funding_oi_short_strict_liquid_top30",
        help="Folder to write configs and selected-universe manifests",
    )
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    output_dir = Path(args.output_dir)
    config_dir = output_dir / "symbol_configs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.symbols:
        symbols = [symbol.upper() for symbol in parse_csv_list(args.symbols)]
        universe_rows = [{"symbol": symbol} for symbol in symbols]
    else:
        universe_rows = select_strict_short_universe(
            size=max(1, min(int(args.size), 50)),
            min_listing_days=max(0, int(args.min_listing_days)),
            min_quote_volume=max(0.0, float(args.min_quote_volume)),
        )
        symbols = [row["symbol"] for row in universe_rows]

    if not symbols:
        raise RuntimeError("No symbols selected for the strict short universe.")

    csv_paths = None
    if not args.skip_refresh:
        csv_paths = refresh_factor_datasets_for_symbols(
            symbols,
            interval=base_config.exchange.timeframe,
            days_back=args.days_back,
            end_time=args.end_time,
            cache_dir=Path(args.cache_dir),
        )

    config_paths = build_shared_symbol_configs(
        base_config,
        symbols=symbols,
        output_dir=config_dir,
        csv_paths=csv_paths,
        output_name_suffix="funding_oi_short_strict",
    )

    manifest = {
        "base_config": str(Path(args.base_config).resolve()),
        "output_dir": str(output_dir.resolve()),
        "config_dir": str(config_dir.resolve()),
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "days_back": args.days_back,
        "end_time": args.end_time,
        "size": len(symbols),
        "symbols": symbols,
        "rows": universe_rows,
        "config_paths": [str(path.resolve()) for path in config_paths],
    }
    (output_dir / "selected_symbols.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if universe_rows and len(universe_rows[0]) > 1:
        import pandas as pd

        pd.DataFrame(universe_rows).to_csv(output_dir / "selected_symbols.csv", index=False)

    print("=== 严格版大币池配置已生成 ===")
    print(f"config_dir: {config_dir.resolve()}")
    print(f"symbols_count: {len(symbols)}")
    print(f"symbols: {','.join(symbols)}")
    if csv_paths:
        print(f"cache_dir: {Path(args.cache_dir).resolve()}")


if __name__ == "__main__":
    main()
