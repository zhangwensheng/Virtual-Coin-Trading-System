from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExchangeSettings:
    name: str = "binance"
    symbol: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    limit: int = 1500
    since: str | None = None
    csv_path: str | None = None


@dataclass
class StrategySettings:
    strategy_kind: str = "trend_pullback"
    higher_timeframe: str = "4h"
    htf_fast_ema: int = 50
    htf_slow_ema: int = 200
    htf_adx_period: int = 14
    htf_adx_threshold: float = 18.0
    ltf_ema: int = 20
    rsi_period: int = 14
    rsi_long_min: float = 50.0
    rsi_long_max: float = 64.0
    rsi_short_min: float = 36.0
    rsi_short_max: float = 50.0
    atr_period: int = 14
    atr_stop_mult: float = 1.6
    trail_atr_mult: float = 2.2
    partial_rr: float = 1.2
    partial_close_ratio: float = 0.5
    pullback_atr_tolerance: float = 0.35
    swing_lookback: int = 6
    volume_sma: int = 20
    min_volume_ratio: float = 0.9
    min_atr_pct: float = 0.002
    max_atr_pct: float = 0.03
    max_bars_in_trade: int = 72
    flat_on_new_day: bool = False
    allow_long: bool = True
    allow_short: bool = True
    intraday_fast_ema: int = 9
    intraday_slow_ema: int = 21
    intraday_volume_window: int = 30
    intraday_notional_window: int = 30
    intraday_day_return_threshold: float = 0.08
    intraday_volume_spike_threshold: float = 2.5
    intraday_sell_volume_ratio: float = 1.6
    intraday_vwap_extension_threshold: float = 0.03
    intraday_take_profit_vwap_buffer: float = 0.003
    intraday_upper_wick_ratio: float = 0.3
    intraday_breakdown_lookback: int = 5
    intraday_peak_lookback: int = 60
    intraday_drop_from_peak_threshold: float = 0.015
    intraday_rsi_overbought: float = 72.0
    intraday_rsi_take_profit: float = 35.0
    intraday_min_bars_from_open: int = 30
    majors_fast_ema: int = 21
    majors_slow_ema: int = 55
    majors_volume_window: int = 30
    majors_impulse_window: int = 24
    majors_pullback_window: int = 6
    majors_impulse_threshold: float = 0.008
    majors_pullback_atr_tolerance: float = 0.35
    majors_min_retrace_from_extreme: float = 0.002
    majors_max_retrace_from_extreme: float = 0.012
    majors_vwap_extension_cap: float = 0.01
    majors_volume_ratio_threshold: float = 0.9
    majors_entry_rsi_low: float = 48.0
    majors_entry_rsi_high: float = 62.0
    majors_exit_rsi: float = 45.0
    majors_min_bars_from_open: int = 6
    breakout_fast_ema: int = 21
    breakout_slow_ema: int = 55
    breakout_volume_window: int = 30
    breakout_channel_window: int = 24
    breakout_exit_window: int = 12
    breakout_squeeze_window: int = 36
    breakout_squeeze_lookback: int = 12
    breakout_max_range_pct: float = 0.015
    breakout_atr_compression_mult: float = 0.9
    breakout_min_volume_ratio: float = 1.1
    breakout_rsi_long_min: float = 55.0
    breakout_rsi_long_max: float = 76.0
    breakout_rsi_short_min: float = 24.0
    breakout_rsi_short_max: float = 45.0
    bear_fast_ema: int = 13
    bear_slow_ema: int = 34
    bear_volume_window: int = 24
    bear_bounce_window: int = 12
    bear_impulse_window: int = 48
    bear_breakdown_lookback: int = 3
    bear_exit_window: int = 8
    bear_min_bounce_pct: float = 0.004
    bear_max_bounce_pct: float = 0.03
    bear_min_rejection_wick_ratio: float = 0.22
    bear_min_volume_ratio: float = 1.1
    bear_min_rsi_peak: float = 54.0
    bear_entry_rsi_min: float = 36.0
    bear_entry_rsi_max: float = 58.0
    bear_take_profit_rsi: float = 28.0
    bear_vwap_reclaim_buffer: float = 0.0015
    bear_ema_extension_threshold: float = 0.0015
    bear_min_bars_from_open: int = 6
    enhanced_funding_lookback: int = 3
    enhanced_oi_lookback: int = 6
    enhanced_price_lookback: int = 6
    enhanced_min_funding_rate: float = -0.00002
    enhanced_min_funding_change: float = 0.00001
    enhanced_exit_funding_rate: float = -0.00008
    enhanced_min_oi_change_pct: float = 0.008
    enhanced_exit_oi_change_pct: float = -0.015
    enhanced_min_price_bounce_pct: float = 0.004
    enhanced_max_price_bounce_pct: float = 0.03
    enhanced_entry_min_score: int = 1
    enhanced_risk_step: float = 0.5
    enhanced_max_risk_multiplier: float = 2.5
    reference_fast_ma: int = 7
    reference_slow_ma: int = 20
    reference_volume_window: int = 10
    reference_breakout_lookback: int = 3
    reference_min_volume_ratio: float = 1.8
    reference_confirm_volume_ratio: float = 0.9
    reference_min_body_pct: float = 0.002
    reference_min_extension_pct: float = 0.0015
    reference_min_rejection_wick: float = 0.12
    reference_min_pressure_ratio: float = 0.58
    reference_exit_rsi: float = 60.0
    top_gainers_daily_boll_window: int = 20
    top_gainers_daily_boll_std: float = 2.0
    top_gainers_daily_top_n: int = 5
    top_gainers_daily_min_return_pct: float = 0.06
    top_gainers_daily_max_return_pct: float = 99.0
    top_gainers_intraday_boll_window: int = 20
    top_gainers_intraday_boll_std: float = 2.0
    top_gainers_confirm_boll_window_5m: int = 20
    top_gainers_confirm_boll_std_5m: float = 2.0
    top_gainers_volume_window: int = 20
    top_gainers_min_volume_ratio: float = 1.8
    top_gainers_band_touch_buffer: float = 0.001
    top_gainers_peak_lookback: int = 60
    top_gainers_breakdown_lookback: int = 3
    top_gainers_distribution_lookback: int = 8
    top_gainers_min_vwap_extension_pct: float = 0.02
    top_gainers_min_upper_wick_ratio: float = 0.18
    top_gainers_drop_from_peak_pct: float = 0.006
    top_gainers_exit_rsi: float = 34.0
    top_gainers_min_daily_score: int = 1
    top_gainers_momentum_breakout_lookback: int = 6
    top_gainers_momentum_min_volume_ratio: float = 1.5
    top_gainers_momentum_max_volume_ratio: float = 99.0
    top_gainers_momentum_min_trade_ratio: float = 1.1
    top_gainers_momentum_max_trade_ratio: float = 99.0
    top_gainers_momentum_min_delta_ratio: float = 0.12
    top_gainers_momentum_max_delta_ratio: float = 1.0
    top_gainers_momentum_min_rsi: float = 58.0
    top_gainers_momentum_max_rsi: float = 100.0
    top_gainers_momentum_min_htf_volume_ratio: float = 0.0
    top_gainers_momentum_max_vwap_extension_pct: float = 0.18
    top_gainers_momentum_exit_delta_ratio: float = -0.04
    top_gainers_momentum_min_bars_from_open: int = 0
    top_gainers_momentum_max_bars_from_open: int = 1440
    orderflow_fast_ema: int = 12
    orderflow_slow_ema: int = 48
    orderflow_volume_window: int = 60
    orderflow_trade_window: int = 60
    orderflow_delta_window: int = 12
    orderflow_breakout_lookback: int = 8
    orderflow_min_volume_ratio: float = 1.4
    orderflow_min_trade_ratio: float = 1.15
    orderflow_min_delta_ratio: float = 0.12
    orderflow_min_delta_zscore: float = 0.8
    orderflow_vwap_buffer: float = 0.0005
    orderflow_max_extension_pct: float = 0.0035
    orderflow_exit_delta_ratio: float = 0.03


@dataclass
class RiskSettings:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.005
    leverage: float = 3.0
    max_notional_fraction: float = 0.35
    fee_rate: float = 0.0005
    slippage_bps: float = 2.0
    max_consecutive_losses: int = 2
    cooldown_bars: int = 6


@dataclass
class OutputSettings:
    output_dir: str = "outputs/default_run"


@dataclass
class AppConfig:
    exchange: ExchangeSettings = field(default_factory=ExchangeSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "AppConfig":
        payload = payload or {}
        return cls(
            exchange=ExchangeSettings(**payload.get("exchange", {})),
            strategy=StrategySettings(**payload.get("strategy", {})),
            risk=RiskSettings(**payload.get("risk", {})),
            output=OutputSettings(**payload.get("output", {})),
        )


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AppConfig.from_dict(payload)
