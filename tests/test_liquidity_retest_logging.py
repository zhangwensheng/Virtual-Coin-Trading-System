from __future__ import annotations

from pathlib import Path

import pandas as pd

from futures_strategy.liquidity_retest import EventLogger


def make_signal_row(*, status: str, block_reason: str) -> dict:
    return {
        "signal_id": "sig-1",
        "status": status,
        "block_reason": block_reason,
        "symbol": "AAAUSDT",
        "side": "long",
        "signal_time": "2026-01-01T00:00:00Z",
        "fill_time": "",
        "universe_rank": 1,
        "quote_volume_24h": 1_000_000.0,
        "zone_timeframe": "1h",
        "zone_low": 98.0,
        "zone_high": 100.0,
        "zone_score": 4.0,
        "zone_touches": 2,
        "breakout_atr_distance": 1.0,
        "breakout_body_atr": 1.0,
        "breakout_quote_volume_ratio": 1.3,
        "retest_depth_atr": 0.2,
        "retest_wait_bars": 4,
        "confirm_quote_volume_ratio": 1.2,
        "daily_symbol_entries": 0,
        "daily_account_entries": 0,
        "open_positions": 0,
        "cooldown_until": "",
        "planned_entry": 100.0,
        "planned_stop": 99.0,
        "risk_usdt": 50.0,
        "notional_usdt": 5000.0,
        "margin_usdt": 1666.67,
        "leverage": 3.0,
    }


def make_trade_row(*, exit_reason: str) -> dict:
    return {
        "trade_id": "trade-1",
        "signal_id": "sig-1",
        "symbol": "AAAUSDT",
        "side": "long",
        "entry_time": "2026-01-01T00:05:00Z",
        "exit_time": "2026-01-01T00:35:00Z",
        "entry_price": 100.0,
        "exit_price": 101.0,
        "qty": 10.0,
        "notional_usdt": 1000.0,
        "margin_usdt": 333.33,
        "initial_stop": 99.0,
        "final_stop": 100.0,
        "partial_tp_price": 101.5,
        "fee_usdt": 0.0,
        "slippage_bps": 0.0,
        "pnl": 10.0,
        "r_multiple": 1.0,
        "mfe": 1.5,
        "mae": -0.2,
        "bars_held": 6,
        "scaled_out": True,
        "exit_reason": exit_reason,
    }


def test_event_logger_writes_all_replay_files_with_stable_columns(tmp_path: Path) -> None:
    logger = EventLogger(tmp_path, run_id="test-run")
    logger.log_event({"event": "UNIVERSE_UPDATE", "timestamp": "2026-01-01T00:00:00Z", "symbol": "", "side": ""})
    logger.log_signal(make_signal_row(status="blocked", block_reason="BLOCKED_MIN_MARGIN"))
    logger.log_trade(make_trade_row(exit_reason="stop_loss"))
    logger.log_equity({"timestamp": "2026-01-01T00:00:00Z", "equity": 10_000.0, "drawdown_pct": 0.0, "open_positions": 0})
    logger.write_daily_summary(pd.DataFrame([make_trade_row(exit_reason="stop_loss")]))
    logger.write_summary({"total_trades": 1})
    assert {path.name for path in tmp_path.iterdir()} >= {
        "events.jsonl",
        "signals.csv",
        "trades.csv",
        "daily_summary.csv",
        "equity_curve.csv",
        "summary.json",
    }
    signals = pd.read_csv(tmp_path / "signals.csv")
    assert signals.columns[:4].tolist() == ["signal_id", "status", "block_reason", "symbol"]
    trades = pd.read_csv(tmp_path / "trades.csv")
    assert trades.columns[:4].tolist() == ["trade_id", "signal_id", "symbol", "side"]
