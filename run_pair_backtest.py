from __future__ import annotations

import argparse
import json
from pathlib import Path

from futures_strategy.pairs import (
    latest_pair_snapshot,
    load_pair_config,
    load_pair_market_data,
    prepare_pair_data,
    run_pair_backtest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pair spread backtest runner")
    parser.add_argument(
        "--config",
        default="configs/binance_btc_eth_pair_5m_2026-01_2026-02.yaml",
        help="Path to pair strategy YAML config",
    )
    args = parser.parse_args()

    config = load_pair_config(args.config)
    frame_a, frame_b = load_pair_market_data(config.pair)
    prepared = prepare_pair_data(frame_a, frame_b, config.strategy)
    result = run_pair_backtest(prepared, config)
    snapshot = latest_pair_snapshot(prepared, config.strategy)
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.trades.to_csv(output_dir / "trades.csv", index=False)
    result.equity_curve.to_csv(output_dir / "equity_curve.csv")
    result.prepared.to_csv(output_dir / "prepared_spread.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "latest_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== 配对回测结果 ===")
    for key, value in result.summary.items():
        print(f"{key}: {value}")

    print("\n=== 最新价差状态 ===")
    for key, value in snapshot.items():
        print(f"{key}: {value}")

    print(f"\n输出目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
