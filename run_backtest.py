from __future__ import annotations

import argparse
import json
from pathlib import Path

from futures_strategy.backtest import run_backtest
from futures_strategy.config import load_config
from futures_strategy.data import load_market_data
from futures_strategy.strategy import latest_signal_snapshot, prepare_market_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto futures strategy backtest runner")
    parser.add_argument("--config", default="configs/binance_btc.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    raw = load_market_data(config.exchange)
    prepared = prepare_market_data(raw, config.strategy)
    result = run_backtest(prepared, config)
    signal = latest_signal_snapshot(prepared, config.strategy)
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.trades.to_csv(output_dir / "trades.csv", index=False)
    result.equity_curve.to_csv(output_dir / "equity_curve.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "latest_signal.json").write_text(
        json.dumps(signal, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== 回测结果 ===")
    for key, value in result.summary.items():
        print(f"{key}: {value}")

    print("\n=== 最新信号 ===")
    for key, value in signal.items():
        print(f"{key}: {value}")

    print(f"\n输出目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
