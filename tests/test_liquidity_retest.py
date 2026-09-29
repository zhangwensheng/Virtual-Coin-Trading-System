from __future__ import annotations

import pandas as pd
import pytest

from futures_strategy.liquidity_retest import (
    LiquidityRetestSettings,
    LiquidityRetestSignalEngine,
    SignalCandidate,
    Zone,
    closed_resample_ohlcv,
    detect_zones,
    select_top_quote_volume_symbols,
    transform_candidate,
)


def make_5m_frame(
    start: str,
    periods: int,
    *,
    close_start: float = 100.0,
    quote_volume: float = 1000.0,
    drift: float = 0.0,
) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    close = [close_start + (i * drift) for i in range(periods)]
    open_ = [close[0], *close[:-1]]
    high = [max(o, c) + 0.2 for o, c in zip(open_, close)]
    low = [min(o, c) - 0.2 for o, c in zip(open_, close)]
    volume = [quote_volume / max(c, 1.0) for c in close]
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": quote_volume,
        },
        index=index,
    )


def make_signal_candidate(*, side: str, entry: float, stop: float) -> SignalCandidate:
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    return SignalCandidate(
        signal_id=f"AAAUSDT-{side}",
        symbol="AAAUSDT",
        side=side,
        signal_time=timestamp,
        zone=Zone(
            side="resistance" if side == "long" else "support",
            timeframe="1h",
            low=98.0,
            high=102.0,
            touches=2,
            score=4.0,
            last_touch_time=timestamp - pd.Timedelta(hours=2),
        ),
        planned_entry=entry,
        planned_stop=stop,
        atr_5m=0.5,
        universe_rank=1,
        quote_volume_24h=1_000_000.0,
    )


def test_mirror_original_long_transforms_direction_stop_and_tp1() -> None:
    settings = LiquidityRetestSettings(
        trade_transform_mode="mirror_original_longs_short_only",
        partial_take_profit_r=1.5,
    )
    raw = make_signal_candidate(side="long", entry=100.0, stop=98.0)

    result = transform_candidate(raw, settings)

    assert result.status == "kept"
    assert result.candidate is not None
    assert result.candidate.side == "short"
    assert result.candidate.planned_stop == pytest.approx(103.0)
    assert result.candidate.planned_tp1 == pytest.approx(98.0)
    assert result.candidate.original_side == "long"
    assert result.candidate.original_stop == pytest.approx(98.0)
    assert result.candidate.original_tp1 == pytest.approx(103.0)


def test_short_only_transform_filters_raw_short_candidate() -> None:
    settings = LiquidityRetestSettings(trade_transform_mode="mirror_original_longs_short_only")
    raw = make_signal_candidate(side="short", entry=100.0, stop=102.0)

    result = transform_candidate(raw, settings)

    assert result.candidate is None
    assert result.status == "filtered"
    assert result.reason == "FILTERED_SHORT_ONLY_SOURCE_SIDE"


def make_hourly_zone_frame(*, include_right_confirmation: bool) -> pd.DataFrame:
    highs = [100, 104, 100, 101, 104.2]
    lows = [96, 98, 97, 98, 98]
    closes = [98, 101, 99, 100, 101]
    if include_right_confirmation:
        highs.append(100)
        lows.append(97)
        closes.append(99)
    index = pd.date_range("2026-01-01", periods=len(highs), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": 1000.0,
            "quote_volume": 100_000.0,
        },
        index=index,
    )


def make_long_breakout_retest_frame() -> pd.DataFrame:
    hourly_close = [
        100,
        102,
        100,
        101,
        102.1,
        100.5,
        101.5,
        106.2,
        105.4,
        106.8,
        108.5,
        109.2,
    ]
    hourly_high = [c + 0.35 for c in hourly_close]
    hourly_low = [c - 0.35 for c in hourly_close]
    hourly_high[1] = 104.0
    hourly_high[4] = 104.2
    hourly_low[8] = 104.05
    hourly_close[8] = 104.45
    hourly_close[9] = 105.2
    hourly_high[9] = 105.35
    hourly_close[10] = 106.0
    hourly_high[10] = 106.4
    return expand_hourly(hourly_close, hourly_high, hourly_low, breakout_hour=7, confirm_hour=10)


def make_short_breakout_retest_frame() -> pd.DataFrame:
    hourly_close = [
        100,
        98,
        100,
        99,
        97.9,
        99.5,
        98.5,
        93.8,
        94.6,
        93.8,
        92.8,
        91.0,
    ]
    hourly_high = [c + 0.35 for c in hourly_close]
    hourly_low = [c - 0.35 for c in hourly_close]
    hourly_low[1] = 96.0
    hourly_low[4] = 95.8
    hourly_high[8] = 95.95
    hourly_close[8] = 95.55
    hourly_close[9] = 94.8
    hourly_low[9] = 94.65
    hourly_close[10] = 94.0
    hourly_low[10] = 93.6
    return expand_hourly(hourly_close, hourly_high, hourly_low, breakout_hour=7, confirm_hour=10)


def expand_hourly(
    hourly_close: list[float],
    hourly_high: list[float],
    hourly_low: list[float],
    *,
    breakout_hour: int,
    confirm_hour: int,
) -> pd.DataFrame:
    rows = []
    index = pd.date_range("2026-01-01", periods=len(hourly_close) * 12, freq="5min", tz="UTC")
    previous_close = hourly_close[0]
    for hour, close_value in enumerate(hourly_close):
        start = previous_close
        for slot in range(12):
            fraction = (slot + 1) / 12
            close = start + ((close_value - start) * fraction)
            open_ = start + ((close_value - start) * (slot / 12))
            high = max(open_, close) + 0.03
            low = min(open_, close) - 0.03
            if slot == 6:
                high = max(high, hourly_high[hour])
                low = min(low, hourly_low[hour])
            quote_volume = 100_000.0
            if hour in {breakout_hour, confirm_hour}:
                quote_volume = 180_000.0
            rows.append(
                {
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": quote_volume / max(close, 1.0),
                    "quote_volume": quote_volume,
                }
            )
        previous_close = close_value
    return pd.DataFrame(rows, index=index)


def replay_until_candidate(
    engine: LiquidityRetestSignalEngine,
    symbol: str,
    frame: pd.DataFrame,
    top_symbols: set[str],
):
    latest = None
    for timestamp in frame.index:
        history = frame.loc[:timestamp]
        candidates = engine.on_bar(symbol, history, top_symbols, timestamp + pd.Timedelta(minutes=5))
        if candidates:
            latest = candidates[-1]
    return latest


def test_closed_resample_ohlcv_excludes_unfinished_1h_bar() -> None:
    frame = make_5m_frame("2026-01-01T00:00:00Z", 14, close_start=100.0, quote_volume=1000.0)
    hourly = closed_resample_ohlcv(frame, "1h", pd.Timestamp("2026-01-01T01:05:00Z"))
    assert hourly.index.tolist() == [pd.Timestamp("2026-01-01T00:00:00Z")]
    assert float(hourly.iloc[0]["quote_volume"]) == 12_000.0


def test_select_top_quote_volume_symbols_uses_only_visible_24h_window() -> None:
    as_of = pd.Timestamp("2026-01-02T00:00:00Z")
    frames = {
        "AAAUSDT": make_5m_frame("2026-01-01T00:00:00Z", 288, quote_volume=100.0),
        "BBBUSDT": make_5m_frame("2026-01-01T00:00:00Z", 288, quote_volume=300.0),
        "CCCUSDT": make_5m_frame("2026-01-01T00:00:00Z", 288, quote_volume=200.0),
        "USDCUSDT": make_5m_frame("2026-01-01T00:00:00Z", 288, quote_volume=900.0),
        "DDDUSDT": make_5m_frame("2026-01-01T00:00:00Z", 288, quote_volume=400.0),
        "EEEUSDT": make_5m_frame("2026-01-01T00:00:00Z", 288, quote_volume=500.0),
        "FFFUSDT": make_5m_frame("2026-01-01T00:00:00Z", 288, quote_volume=600.0),
    }
    selected, rows = select_top_quote_volume_symbols(frames, as_of, LiquidityRetestSettings())
    assert selected == ["FFFUSDT", "EEEUSDT", "DDDUSDT", "BBBUSDT", "CCCUSDT"]
    assert rows.iloc[0]["quote_volume_24h"] == 172_800.0


def test_detect_zones_waits_for_right_side_pivot_confirmation() -> None:
    hourly = make_hourly_zone_frame(include_right_confirmation=False)
    zones = detect_zones(
        hourly,
        "1h",
        hourly.index[-1] + pd.Timedelta(hours=1),
        LiquidityRetestSettings(pivot_left_bars=1, pivot_right_bars=1, zone_min_touches=2),
    )
    assert zones == []


def test_signal_engine_generates_long_and_short_candidates_after_retest_confirmation() -> None:
    settings = LiquidityRetestSettings(
        pivot_left_bars=1,
        pivot_right_bars=1,
        zone_min_touches=2,
        breakout_volume_window=2,
        confirm_volume_window=2,
        breakout_min_quote_volume_ratio=1.05,
        confirm_min_quote_volume_ratio=1.0,
    )
    long_engine = LiquidityRetestSignalEngine(settings)
    short_engine = LiquidityRetestSignalEngine(settings)
    long_candidate = replay_until_candidate(long_engine, "AAAUSDT", make_long_breakout_retest_frame(), {"AAAUSDT"})
    short_candidate = replay_until_candidate(short_engine, "BBBUSDT", make_short_breakout_retest_frame(), {"BBBUSDT"})
    assert long_candidate is not None
    assert long_candidate.side == "long"
    assert short_candidate is not None
    assert short_candidate.side == "short"
