from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from futures_strategy.backtest import run_backtest
from futures_strategy.config import AppConfig
from futures_strategy.strategy import prepare_market_data


class BacktestSmokeTest(unittest.TestCase):
    def test_backtest_runs_on_synthetic_data(self) -> None:
        periods = 900
        index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
        base = 50_000 + np.linspace(0, 4_000, periods)
        wave = 500 * np.sin(np.arange(periods) / 9)
        pullback = 200 * np.sin(np.arange(periods) / 3)
        close = base + wave - pullback
        open_ = np.roll(close, 1)
        open_[0] = close[0] - 20
        high = np.maximum(open_, close) + 80
        low = np.minimum(open_, close) - 80
        volume = 1_000 + (80 * np.sin(np.arange(periods) / 7)) + 20

        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=index,
        )

        config = AppConfig()
        config.strategy.htf_adx_threshold = 5
        config.strategy.min_volume_ratio = 0.0
        config.strategy.max_atr_pct = 0.2
        config.risk.fee_rate = 0.0
        config.risk.slippage_bps = 0.0

        prepared = prepare_market_data(frame, config.strategy)
        result = run_backtest(prepared, config)

        self.assertFalse(prepared.empty)
        self.assertIn("return_pct", result.summary)
        self.assertGreater(len(result.equity_curve), 10)
        self.assertAlmostEqual(
            result.summary["final_equity"],
            round(config.risk.initial_capital + result.trades["pnl"].sum(), 2),
            places=2,
        )

    def test_smallcap_intraday_short_strategy_generates_short_trade(self) -> None:
        periods = 360
        index = pd.date_range("2025-01-01 00:00:00", periods=periods, freq="min", tz="UTC")
        base = np.full(periods, 1.0)
        noise = 0.002 * np.sin(np.arange(periods) / 5)
        close = base + noise
        close[120:150] = np.linspace(1.01, 1.14, 30)
        close[150] = 1.105
        close[151:190] = np.linspace(1.10, 1.02, 39)
        close[190:] = 1.02 + 0.004 * np.sin(np.arange(periods - 190) / 6)

        open_ = np.roll(close, 1)
        open_[0] = 1.0
        high = np.maximum(open_, close) + 0.004
        low = np.minimum(open_, close) - 0.004
        high[150] = 1.17
        low[150] = 1.095
        volume = np.full(periods, 1000.0)
        volume[120:150] = 4500
        volume[150] = 9000
        volume[151:170] = 7000

        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=index,
        )

        config = AppConfig()
        config.exchange.timeframe = "1m"
        config.strategy.strategy_kind = "smallcap_intraday_short"
        config.strategy.allow_long = False
        config.strategy.allow_short = True
        config.strategy.flat_on_new_day = True
        config.strategy.max_bars_in_trade = 120
        config.strategy.atr_stop_mult = 0.8
        config.strategy.trail_atr_mult = 1.2
        config.strategy.partial_rr = 0.8
        config.strategy.partial_close_ratio = 0.5
        config.strategy.intraday_day_return_threshold = 0.05
        config.strategy.intraday_volume_spike_threshold = 2.0
        config.strategy.intraday_sell_volume_ratio = 1.2
        config.strategy.intraday_vwap_extension_threshold = 0.02
        config.strategy.intraday_upper_wick_ratio = 0.2
        config.strategy.intraday_breakdown_lookback = 3
        config.strategy.intraday_peak_lookback = 40
        config.strategy.intraday_drop_from_peak_threshold = 0.01
        config.strategy.intraday_rsi_overbought = 68
        config.strategy.intraday_rsi_take_profit = 40
        config.strategy.intraday_min_bars_from_open = 10
        config.risk.fee_rate = 0.0
        config.risk.slippage_bps = 0.0

        prepared = prepare_market_data(frame, config.strategy)
        result = run_backtest(prepared, config)

        self.assertTrue(prepared["short_signal"].any())
        self.assertFalse(result.trades.empty)
        self.assertTrue((result.trades["side"] == "short").all())

    def test_majors_ltf_strategy_generates_long_trade(self) -> None:
        periods = 420
        index = pd.date_range("2025-01-01 00:00:00", periods=periods, freq="5min", tz="UTC")
        base = 100 + np.linspace(0, 5.5, periods)
        wave = 0.18 * np.sin(np.arange(periods) / 7)
        close = base + wave
        close[170:195] = np.linspace(close[169], close[169] + 1.8, 25)
        close[195:205] = np.linspace(close[194], close[194] - 0.6, 10)
        close[205:220] = np.linspace(close[204], close[204] + 1.2, 15)
        close[220:] = close[220:] + 0.35

        open_ = np.roll(close, 1)
        open_[0] = close[0] - 0.05
        high = np.maximum(open_, close) + 0.05
        low = np.minimum(open_, close) - 0.05
        volume = np.full(periods, 1000.0)
        volume[170:195] = 1800
        volume[195:205] = 1400
        volume[205:215] = 2200

        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=index,
        )

        config = AppConfig()
        config.exchange.timeframe = "5m"
        config.strategy.strategy_kind = "majors_ltf_trend"
        config.strategy.higher_timeframe = "1h"
        config.strategy.htf_fast_ema = 8
        config.strategy.htf_slow_ema = 21
        config.strategy.htf_adx_period = 3
        config.strategy.htf_adx_threshold = 5
        config.strategy.allow_long = True
        config.strategy.allow_short = False
        config.strategy.max_bars_in_trade = 48
        config.strategy.atr_stop_mult = 1.0
        config.strategy.trail_atr_mult = 1.4
        config.strategy.partial_rr = 1.0
        config.strategy.partial_close_ratio = 0.5
        config.strategy.min_atr_pct = 0.0
        config.strategy.max_atr_pct = 0.2
        config.strategy.majors_fast_ema = 12
        config.strategy.majors_slow_ema = 30
        config.strategy.majors_volume_window = 20
        config.strategy.majors_impulse_window = 18
        config.strategy.majors_pullback_window = 5
        config.strategy.majors_impulse_threshold = 0.003
        config.strategy.majors_pullback_atr_tolerance = 0.5
        config.strategy.majors_min_retrace_from_extreme = 0.001
        config.strategy.majors_max_retrace_from_extreme = 0.02
        config.strategy.majors_vwap_extension_cap = 0.02
        config.strategy.majors_volume_ratio_threshold = 0.8
        config.strategy.majors_entry_rsi_low = 45
        config.strategy.majors_entry_rsi_high = 90
        config.strategy.majors_exit_rsi = 45
        config.strategy.majors_min_bars_from_open = 6
        config.risk.fee_rate = 0.0
        config.risk.slippage_bps = 0.0

        prepared = prepare_market_data(frame, config.strategy)
        result = run_backtest(prepared, config)

        self.assertTrue(prepared["long_signal"].any())
        self.assertFalse(result.trades.empty)
        self.assertTrue((result.trades["side"] == "long").all())

    def test_orderflow_imbalance_strategy_generates_long_trade(self) -> None:
        periods = 360
        index = pd.date_range("2025-01-01 00:00:00", periods=periods, freq="min", tz="UTC")
        close = 100 + np.linspace(0, 1.8, periods) + (0.08 * np.sin(np.arange(periods) / 8))
        close[210:235] = np.linspace(close[209], close[209] + 1.6, 25)
        close[235:260] = np.linspace(close[234], close[234] + 0.9, 25)
        close[260:] = close[259] + 0.05 * np.sin(np.arange(periods - 260) / 5) + 0.4

        open_ = np.roll(close, 1)
        open_[0] = close[0] - 0.04
        high = np.maximum(open_, close) + 0.05
        low = np.minimum(open_, close) - 0.05
        volume = np.full(periods, 180.0)
        volume[210:260] = 420.0
        quote_volume = volume * close
        trade_count = np.full(periods, 40.0)
        trade_count[210:260] = 130.0
        taker_buy_base = volume * 0.48
        taker_buy_quote = quote_volume * 0.48
        taker_buy_base[210:260] = volume[210:260] * 0.76
        taker_buy_quote[210:260] = quote_volume[210:260] * 0.76

        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "quote_volume": quote_volume,
                "trade_count": trade_count,
                "taker_buy_base": taker_buy_base,
                "taker_buy_quote": taker_buy_quote,
            },
            index=index,
        )

        config = AppConfig()
        config.exchange.timeframe = "1m"
        config.strategy.strategy_kind = "orderflow_imbalance_1m"
        config.strategy.allow_long = True
        config.strategy.allow_short = False
        config.strategy.max_bars_in_trade = 45
        config.strategy.min_atr_pct = 0.0
        config.strategy.max_atr_pct = 0.2
        config.strategy.atr_stop_mult = 0.8
        config.strategy.trail_atr_mult = 1.1
        config.strategy.partial_rr = 0.8
        config.strategy.partial_close_ratio = 0.5
        config.strategy.orderflow_fast_ema = 8
        config.strategy.orderflow_slow_ema = 21
        config.strategy.orderflow_volume_window = 20
        config.strategy.orderflow_trade_window = 20
        config.strategy.orderflow_delta_window = 6
        config.strategy.orderflow_breakout_lookback = 5
        config.strategy.orderflow_min_volume_ratio = 1.1
        config.strategy.orderflow_min_trade_ratio = 1.1
        config.strategy.orderflow_min_delta_ratio = 0.12
        config.strategy.orderflow_min_delta_zscore = 0.3
        config.strategy.orderflow_vwap_buffer = 0.0
        config.strategy.orderflow_max_extension_pct = 0.02
        config.strategy.orderflow_exit_delta_ratio = 0.02
        config.risk.fee_rate = 0.0
        config.risk.slippage_bps = 0.0

        prepared = prepare_market_data(frame, config.strategy)
        result = run_backtest(prepared, config)

        self.assertTrue(prepared["long_signal"].any())
        self.assertFalse(result.trades.empty)
        self.assertTrue((result.trades["side"] == "long").all())

    def test_reference_volume_reversal_strategy_generates_short_trade(self) -> None:
        periods = 320
        index = pd.date_range("2025-01-01 00:00:00", periods=periods, freq="min", tz="UTC")
        close = 100 - np.linspace(0, 2.2, periods) + (0.06 * np.sin(np.arange(periods) / 7))
        close[170:182] = np.linspace(close[169], close[169] + 1.4, 12)
        close[181] = close[180] + 0.55
        close[182] = close[181] - 1.0
        close[183:205] = np.linspace(close[182], close[182] - 1.8, 22)
        close[205:] = close[204] - 0.04 * np.sin(np.arange(periods - 205) / 5)

        open_ = np.roll(close, 1)
        open_[0] = close[0] + 0.03
        high = np.maximum(open_, close) + 0.05
        low = np.minimum(open_, close) - 0.05
        high[181] = close[181] + 0.25
        volume = np.full(periods, 140.0)
        volume[170:182] = 180.0
        volume[181] = 420.0
        volume[182:190] = 250.0
        quote_volume = volume * close
        taker_buy_quote = quote_volume * 0.52
        taker_buy_quote[181] = quote_volume[181] * 0.74

        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "quote_volume": quote_volume,
                "taker_buy_quote": taker_buy_quote,
            },
            index=index,
        )

        config = AppConfig()
        config.exchange.timeframe = "1m"
        config.strategy.strategy_kind = "reference_volume_reversal_1m"
        config.strategy.allow_long = False
        config.strategy.allow_short = True
        config.strategy.max_bars_in_trade = 20
        config.strategy.min_atr_pct = 0.0
        config.strategy.max_atr_pct = 0.2
        config.strategy.atr_stop_mult = 0.8
        config.strategy.trail_atr_mult = 1.1
        config.strategy.partial_rr = 0.8
        config.strategy.partial_close_ratio = 0.5
        config.strategy.reference_fast_ma = 7
        config.strategy.reference_slow_ma = 20
        config.strategy.reference_volume_window = 10
        config.strategy.reference_breakout_lookback = 3
        config.strategy.reference_min_volume_ratio = 1.6
        config.strategy.reference_confirm_volume_ratio = 0.9
        config.strategy.reference_min_body_pct = 0.002
        config.strategy.reference_min_extension_pct = 0.001
        config.strategy.reference_min_rejection_wick = 0.1
        config.strategy.reference_min_pressure_ratio = 0.6
        config.strategy.reference_exit_rsi = 58
        config.risk.fee_rate = 0.0
        config.risk.slippage_bps = 0.0

        prepared = prepare_market_data(frame, config.strategy)
        result = run_backtest(prepared, config)

        self.assertTrue(prepared["short_signal"].any())
        self.assertFalse(result.trades.empty)
        self.assertTrue((result.trades["side"] == "short").all())

    def test_top_gainers_boll_short_strategy_generates_short_trade(self) -> None:
        periods = 360
        index = pd.date_range("2025-01-02 00:00:00", periods=periods, freq="min", tz="UTC")
        close = np.full(periods, 100.0)
        close[:220] = 100 + 0.08 * np.sin(np.arange(220) / 11)
        close[220:240] = np.linspace(close[219], close[219] + 2.6, 20)
        close[240] = close[239] - 1.2
        close[241:275] = np.linspace(close[240], close[240] - 2.0, 34)
        close[275:] = close[274] + 0.06 * np.sin(np.arange(periods - 275) / 6)

        open_ = np.roll(close, 1)
        open_[0] = close[0] + 0.02
        high = np.maximum(open_, close) + 0.05
        low = np.minimum(open_, close) - 0.05
        high[239] = close[239] + 0.7
        high[240] = close[239] + 0.8
        volume = np.full(periods, 120.0)
        volume[220:240] = 180.0
        volume[240] = 420.0
        volume[241:246] = 260.0

        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "selected_day": True,
                "daily_rank": 1,
                "daily_return_pct": 0.12,
                "daily_close": 112.0,
                "daily_bb_upper": 108.0,
            },
            index=index,
        )

        config = AppConfig()
        config.exchange.timeframe = "1m"
        config.strategy.strategy_kind = "top_gainers_boll_short_1m"
        config.strategy.allow_long = False
        config.strategy.allow_short = True
        config.strategy.max_bars_in_trade = 18
        config.strategy.min_atr_pct = 0.0
        config.strategy.max_atr_pct = 0.2
        config.strategy.atr_stop_mult = 0.9
        config.strategy.trail_atr_mult = 1.0
        config.strategy.partial_rr = 0.6
        config.strategy.partial_close_ratio = 0.5
        config.strategy.top_gainers_daily_top_n = 5
        config.strategy.top_gainers_daily_min_return_pct = 0.05
        config.strategy.top_gainers_intraday_boll_window = 20
        config.strategy.top_gainers_confirm_boll_window_5m = 6
        config.strategy.top_gainers_volume_window = 20
        config.strategy.top_gainers_min_volume_ratio = 1.4
        config.strategy.top_gainers_band_touch_buffer = 0.002
        config.strategy.top_gainers_breakdown_lookback = 3
        config.strategy.top_gainers_exit_rsi = 38
        config.risk.fee_rate = 0.0
        config.risk.slippage_bps = 0.0

        prepared = prepare_market_data(frame, config.strategy)
        result = run_backtest(prepared, config)

        self.assertTrue(prepared["short_signal"].any())
        self.assertFalse(result.trades.empty)
        self.assertTrue((result.trades["side"] == "short").all())

    def test_top_gainers_pump_dump_strategy_generates_long_trade(self) -> None:
        periods = 360
        index = pd.date_range("2025-01-02 00:00:00", periods=periods, freq="min", tz="UTC")
        close = 100 + 0.03 * np.sin(np.arange(periods) / 8)
        close[180:215] = np.linspace(close[179], close[179] + 4.2, 35)
        close[215:245] = np.linspace(close[214], close[214] + 2.8, 30)
        close[245:] = close[244] + 0.08 * np.sin(np.arange(periods - 245) / 5)

        open_ = np.roll(close, 1)
        open_[0] = close[0] - 0.02
        high = np.maximum(open_, close) + 0.06
        low = np.minimum(open_, close) - 0.06
        volume = np.full(periods, 180.0)
        volume[180:245] = 520.0
        quote_volume = volume * close
        trade_count = np.full(periods, 60.0)
        trade_count[180:245] = 180.0
        taker_buy_quote = quote_volume * 0.52
        taker_buy_quote[180:245] = quote_volume[180:245] * 0.72

        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "quote_volume": quote_volume,
                "trade_count": trade_count,
                "taker_buy_quote": taker_buy_quote,
                "selected_day": True,
                "daily_rank": 1,
                "daily_return_pct": 0.12,
                "daily_close": 112.0,
                "daily_bb_upper": 101.0,
            },
            index=index,
        )

        config = AppConfig()
        config.exchange.timeframe = "1m"
        config.strategy.strategy_kind = "top_gainers_pump_dump_1m"
        config.strategy.allow_long = True
        config.strategy.allow_short = False
        config.strategy.max_bars_in_trade = 12
        config.strategy.min_atr_pct = 0.0
        config.strategy.max_atr_pct = 0.2
        config.strategy.atr_stop_mult = 0.75
        config.strategy.trail_atr_mult = 0.9
        config.strategy.partial_rr = 0.7
        config.strategy.top_gainers_daily_min_return_pct = 0.05
        config.strategy.top_gainers_momentum_breakout_lookback = 4
        config.strategy.top_gainers_momentum_min_volume_ratio = 1.2
        config.strategy.top_gainers_momentum_min_trade_ratio = 1.0
        config.strategy.top_gainers_momentum_min_delta_ratio = 0.08
        config.strategy.top_gainers_momentum_min_rsi = 55
        config.strategy.top_gainers_momentum_max_vwap_extension_pct = 0.2
        config.risk.fee_rate = 0.0
        config.risk.slippage_bps = 0.0

        prepared = prepare_market_data(frame, config.strategy)
        result = run_backtest(prepared, config)

        self.assertTrue(prepared["long_signal"].any())
        self.assertFalse(result.trades.empty)
        self.assertTrue((result.trades["side"] == "long").all())

    def test_majors_breakout_squeeze_strategy_generates_long_trade(self) -> None:
        periods = 420
        index = pd.date_range("2025-01-01 00:00:00", periods=periods, freq="5min", tz="UTC")
        base = 100 + np.linspace(0, 4.5, periods)
        close = base.copy()
        close[:220] = 100 + 0.08 * np.sin(np.arange(220) / 7)
        close[220:250] = np.linspace(close[219], close[219] + 0.4, 30)
        close[250:290] = np.linspace(close[249], close[249] + 4.8, 40)
        close[290:] = close[289] + 0.18 * np.sin(np.arange(periods - 290) / 5) + 0.8

        open_ = np.roll(close, 1)
        open_[0] = close[0] - 0.05
        high = np.maximum(open_, close) + 0.06
        low = np.minimum(open_, close) - 0.06
        volume = np.full(periods, 1000.0)
        volume[220:250] = 850
        volume[250:290] = 2400
        volume[290:] = 1500

        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=index,
        )

        config = AppConfig()
        config.exchange.timeframe = "5m"
        config.strategy.strategy_kind = "majors_breakout_squeeze"
        config.strategy.higher_timeframe = "1h"
        config.strategy.htf_fast_ema = 8
        config.strategy.htf_slow_ema = 21
        config.strategy.htf_adx_period = 3
        config.strategy.htf_adx_threshold = 5
        config.strategy.allow_long = True
        config.strategy.allow_short = False
        config.strategy.min_atr_pct = 0.0
        config.strategy.max_atr_pct = 0.2
        config.strategy.atr_stop_mult = 1.0
        config.strategy.trail_atr_mult = 1.4
        config.strategy.partial_rr = 1.0
        config.strategy.partial_close_ratio = 0.5
        config.strategy.breakout_fast_ema = 12
        config.strategy.breakout_slow_ema = 30
        config.strategy.breakout_volume_window = 20
        config.strategy.breakout_channel_window = 18
        config.strategy.breakout_exit_window = 8
        config.strategy.breakout_squeeze_window = 24
        config.strategy.breakout_squeeze_lookback = 8
        config.strategy.breakout_max_range_pct = 0.008
        config.strategy.breakout_atr_compression_mult = 1.0
        config.strategy.breakout_min_volume_ratio = 1.0
        config.strategy.breakout_rsi_long_min = 50
        config.strategy.breakout_rsi_long_max = 100
        config.risk.fee_rate = 0.0
        config.risk.slippage_bps = 0.0

        prepared = prepare_market_data(frame, config.strategy)
        result = run_backtest(prepared, config)

        self.assertTrue(prepared["long_signal"].any())
        self.assertFalse(result.trades.empty)

    def test_majors_bear_rally_short_strategy_generates_short_trade(self) -> None:
        periods = 420
        index = pd.date_range("2025-01-01 00:00:00", periods=periods, freq="5min", tz="UTC")
        close = 100 - np.linspace(0, 10.0, periods) + 0.05 * np.sin(np.arange(periods) / 6)
        close[260:280] = np.linspace(close[259], close[259] + 2.2, 20)
        close[280] = close[279] - 1.8
        close[281:320] = np.linspace(close[280], close[280] - 3.5, 39)

        open_ = np.roll(close, 1)
        open_[0] = close[0] + 0.05
        high = np.maximum(open_, close) + 0.08
        low = np.minimum(open_, close) - 0.08
        high[280] = close[279] + 0.5
        volume = np.full(periods, 1000.0)
        volume[260:280] = 1400
        volume[280] = 2600
        volume[281:290] = 1900

        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=index,
        )

        config = AppConfig()
        config.exchange.timeframe = "5m"
        config.strategy.strategy_kind = "majors_bear_rally_short"
        config.strategy.higher_timeframe = "1h"
        config.strategy.htf_fast_ema = 8
        config.strategy.htf_slow_ema = 21
        config.strategy.htf_adx_period = 3
        config.strategy.htf_adx_threshold = 5
        config.strategy.allow_long = False
        config.strategy.allow_short = True
        config.strategy.min_atr_pct = 0.0
        config.strategy.max_atr_pct = 0.2
        config.strategy.atr_stop_mult = 0.9
        config.strategy.trail_atr_mult = 1.1
        config.strategy.partial_rr = 0.8
        config.strategy.partial_close_ratio = 0.5
        config.strategy.max_bars_in_trade = 18
        config.strategy.bear_fast_ema = 12
        config.strategy.bear_slow_ema = 30
        config.strategy.bear_volume_window = 20
        config.strategy.bear_bounce_window = 10
        config.strategy.bear_impulse_window = 36
        config.strategy.bear_breakdown_lookback = 3
        config.strategy.bear_exit_window = 6
        config.strategy.bear_min_bounce_pct = 0.01
        config.strategy.bear_max_bounce_pct = 0.05
        config.strategy.bear_min_rejection_wick_ratio = 0.1
        config.strategy.bear_min_volume_ratio = 1.0
        config.strategy.bear_min_rsi_peak = 45
        config.strategy.bear_entry_rsi_min = 20
        config.strategy.bear_entry_rsi_max = 80
        config.strategy.bear_take_profit_rsi = 35
        config.strategy.bear_vwap_reclaim_buffer = 0.003
        config.strategy.bear_ema_extension_threshold = 0.0
        config.strategy.bear_min_bars_from_open = 6
        config.risk.fee_rate = 0.0
        config.risk.slippage_bps = 0.0

        prepared = prepare_market_data(frame, config.strategy)
        result = run_backtest(prepared, config)

        self.assertTrue(prepared["short_signal"].any())
        self.assertFalse(result.trades.empty)
        self.assertTrue((result.trades["side"] == "short").all())

    def test_funding_oi_bear_short_strategy_generates_short_trade(self) -> None:
        periods = 360
        index = pd.date_range("2025-01-01 00:00:00", periods=periods, freq="h", tz="UTC")
        base = 120 - np.linspace(0, 35, periods)
        close = base + 0.6 * np.sin(np.arange(periods) / 9)
        close[250:262] = np.linspace(close[249], close[249] + 6.0, 12)
        close[262] = close[261] - 3.5
        close[263:280] = np.linspace(close[262], close[262] - 8.0, 17)

        open_ = np.roll(close, 1)
        open_[0] = close[0] + 0.4
        high = np.maximum(open_, close) + 0.5
        low = np.minimum(open_, close) - 0.5
        volume = np.full(periods, 1000.0)
        volume[250:263] = 1700
        volume[263:272] = 2300

        funding = np.full(periods, -0.00006)
        funding[248:264] = np.linspace(-0.00003, 0.00005, 16)
        funding[264:268] = 0.00003
        funding[268:] = -0.00009

        oi_value = np.linspace(2_000_000, 2_080_000, periods)
        oi_value[248:264] = np.linspace(2_020_000, 2_120_000, 16)
        oi_value[264:268] = np.linspace(2_118_000, 2_130_000, 4)
        oi_value[268:280] = np.linspace(2_110_000, 2_020_000, 12)

        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "funding_rate": funding,
                "oi_value": oi_value,
            },
            index=index,
        )

        config = AppConfig()
        config.exchange.timeframe = "1h"
        config.strategy.strategy_kind = "funding_oi_bear_short"
        config.strategy.higher_timeframe = "4h"
        config.strategy.htf_fast_ema = 24
        config.strategy.htf_slow_ema = 48
        config.strategy.htf_adx_period = 5
        config.strategy.htf_adx_threshold = 5
        config.strategy.ltf_ema = 20
        config.strategy.rsi_short_min = 20
        config.strategy.rsi_short_max = 70
        config.strategy.pullback_atr_tolerance = 0.5
        config.strategy.volume_sma = 20
        config.strategy.min_volume_ratio = 0.8
        config.strategy.min_atr_pct = 0.0
        config.strategy.max_atr_pct = 0.2
        config.strategy.max_bars_in_trade = 24
        config.strategy.allow_long = False
        config.strategy.allow_short = True
        config.strategy.enhanced_funding_lookback = 2
        config.strategy.enhanced_oi_lookback = 4
        config.strategy.enhanced_price_lookback = 2
        config.strategy.enhanced_min_funding_rate = -0.00004
        config.strategy.enhanced_min_funding_change = 0.000005
        config.strategy.enhanced_min_oi_change_pct = 0.005
        config.strategy.enhanced_exit_oi_change_pct = -0.02
        config.strategy.enhanced_min_price_bounce_pct = 0.005
        config.strategy.enhanced_max_price_bounce_pct = 0.08
        config.risk.fee_rate = 0.0
        config.risk.slippage_bps = 0.0

        prepared = prepare_market_data(frame, config.strategy)
        result = run_backtest(prepared, config)

        self.assertTrue(prepared["short_signal"].any())
        self.assertFalse(result.trades.empty)
        self.assertTrue((result.trades["side"] == "short").all())


if __name__ == "__main__":
    unittest.main()
