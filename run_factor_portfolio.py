from __future__ import annotations

import argparse
import json
from pathlib import Path

from futures_strategy.factor_portfolio import (
    load_prepared_symbol_frames,
    override_portfolio_risk,
    run_factor_portfolio_backtest,
)


def parse_config_paths(config_dir: str | None, configs: str | None) -> list[Path]:
    if configs:
        return [Path(item.strip()) for item in configs.split(",") if item.strip()]
    if config_dir:
        return sorted(Path(config_dir).glob("*.yaml"))
    raise ValueError("Please provide --config-dir or --configs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-symbol factor short portfolio backtest")
    parser.add_argument("--config-dir", default=None, help="Directory containing per-symbol YAML configs")
    parser.add_argument("--configs", default=None, help="Comma separated YAML config paths")
    parser.add_argument("--max-positions", default=2, type=int, help="Maximum concurrent positions")
    parser.add_argument(
        "--portfolio-notional-fraction",
        default=0.9,
        type=float,
        help="Total portfolio notional cap, divided across concurrent positions",
    )
    parser.add_argument(
        "--override-risk-per-trade",
        default=None,
        type=float,
        help="Optional override for all symbol configs risk_per_trade",
    )
    parser.add_argument("--output-dir", default="outputs/factor_short_portfolio", help="Portfolio output directory")
    parser.add_argument("--symbol-daily-max-losses", default=None, type=int)
    parser.add_argument("--symbol-daily-loss-limit-pct", default=None, type=float)
    parser.add_argument("--portfolio-daily-loss-limit-pct", default=None, type=float)
    args = parser.parse_args()

    config_paths = parse_config_paths(args.config_dir, args.configs)
    configs, prepared_frames = load_prepared_symbol_frames(config_paths)
    configs = override_portfolio_risk(configs, risk_per_trade=args.override_risk_per_trade)
    result = run_factor_portfolio_backtest(
        configs=configs,
        prepared_frames=prepared_frames,
        max_positions=args.max_positions,
        portfolio_notional_fraction=args.portfolio_notional_fraction,
        symbol_daily_max_losses=args.symbol_daily_max_losses,
        symbol_daily_loss_limit_pct=args.symbol_daily_loss_limit_pct,
        portfolio_daily_loss_limit_pct=args.portfolio_daily_loss_limit_pct,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(output_dir / "trades.csv", index=False)
    result.equity_curve.to_csv(output_dir / "equity_curve.csv")
    result.symbol_summary.to_csv(output_dir / "symbol_summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== 组合回测结果 ===")
    for key, value in result.summary.items():
        print(f"{key}: {value}")

    print("\n=== 分币结果 ===")
    if result.symbol_summary.empty:
        print("no_symbol_trades")
    else:
        for _, row in result.symbol_summary.iterrows():
            print(
                f"{row['symbol']} net={row['net_profit']} trades={row['total_trades']} "
                f"win_rate={row['win_rate_pct']}% pf={row['profit_factor']}"
            )

    print(f"\n输出目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
