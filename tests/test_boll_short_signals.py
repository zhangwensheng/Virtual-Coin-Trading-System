from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from futures_strategy.boll_short_signals import daily_gate, evaluate_signal, rank_universe


UTC = "UTC"


def _daily(open_today: float = 100.0, close_today: float | None = None) -> pd.DataFrame:
    index = pd.date_range("2026-08-13", periods=29, freq="D", tz=UTC)
    history = [
        {"open": 90.0, "high": 100.0, "low": 80.0, "close": 90.0, "volume": 1_000.0}
        for _ in range(28)
    ]
    today_close = open_today if close_today is None else close_today
    rows = history + [
        {"open": open_today, "high": max(open_today, today_close), "low": min(open_today, today_close), "close": today_close, "volume": 1_000.0}
    ]
    return pd.DataFrame(rows, index=index)


def _minute_rows(start: str = "2026-09-10 00:00", base_count: int = 35) -> pd.DataFrame:
    rows = [
        {"open": 100.0, "high": 101.0, "low": 94.0, "close": 100.0, "volume": 100.0}
        for _ in range(base_count)
    ]
    rows.extend(
        [
            {"open": 100.0, "high": 112.0, "low": 94.0, "close": 111.5, "volume": 300.0},
            {"open": 111.0, "high": 111.8, "low": 106.0, "close": 107.0, "volume": 120.0},
            {"open": 106.2, "high": 107.0, "low": 105.0, "close": 105.5, "volume": 120.0},
        ]
    )
    return pd.DataFrame(rows, index=pd.date_range(start, periods=len(rows), freq="min", tz=UTC))


def _exchange_symbol(symbol: str, base: str, onboard: str = "2026-01-01", **extra):
    row = {
        "symbol": symbol,
        "pair": symbol,
        "contractType": "PERPETUAL",
        "status": "TRADING",
        "baseAsset": base,
        "quoteAsset": "USDT",
        "onboardDate": int(pd.Timestamp(onboard, tz=UTC).timestamp() * 1_000),
        "underlyingType": "COIN",
    }
    row.update(extra)
    return row


def _ticker(symbol: str, change: str, quote_volume: str = "30000000"):
    return {"symbol": symbol, "priceChangePercent": change, "quoteVolume": quote_volume}


def test_daily_gate_uses_previous_28_day_williams_r_range():
    eligible = daily_gate(_daily(open_today=96.0), pd.Timestamp("2026-09-10 12:00:00", tz=UTC))
    rejected = daily_gate(_daily(open_today=90.0), pd.Timestamp("2026-09-10 12:00:00", tz=UTC))

    assert eligible["eligible"] is True
    assert eligible["daily_wr"] == pytest.approx(-20.0)
    assert eligible["daily_wr_period"] == 28
    assert rejected["eligible"] is False
    assert rejected["reason"] == "daily_wr_not_overbought"
    assert rejected["daily_wr"] == pytest.approx(-50.0)


def test_daily_gate_never_uses_future_daily_rows_for_threshold():
    daily = _daily(open_today=100.0)
    daily.loc[pd.Timestamp("2026-09-11", tz=UTC)] = {
        "open": 1.0,
        "high": 1_000_000.0,
        "low": 1.0,
        "close": 1_000_000.0,
        "volume": 1.0,
    }

    first = daily_gate(daily, pd.Timestamp("2026-09-10 12:00:00", tz=UTC))
    daily.loc[pd.Timestamp("2026-09-10", tz=UTC), "high"] = 1_000_000.0
    daily.loc[pd.Timestamp("2026-09-11", tz=UTC), "low"] = -1_000_000.0
    second = daily_gate(daily, pd.Timestamp("2026-09-10 12:00:00", tz=UTC))

    assert first["daily_wr"] == pytest.approx(0.0)
    assert second["daily_wr"] == pytest.approx(0.0)
    assert first["eligible"] is True
    assert second["eligible"] is True


def test_rank_universe_filters_non_crypto_before_ranking_without_backfill():
    symbols = [
        _exchange_symbol("ALPHAUSDT", "ALPHA"),
        _exchange_symbol("BETAUSDT", "BETA"),
        _exchange_symbol("DELTAUSDT", "DELTA"),
        _exchange_symbol("EPSUSDT", "EPS"),
        _exchange_symbol("GAMMAUSDT", "GAMMA"),
        _exchange_symbol("ZETAUSDT", "ZETA"),
        _exchange_symbol("USDCUSDT", "USDC"),
        _exchange_symbol("XAUUSDT", "XAU", underlyingType="COMMODITY"),
        _exchange_symbol("YOUNGUSDT", "YOUNG", onboard="2026-08-01"),
        _exchange_symbol("BAD-NAMEUSDT", "BAD-NAME"),
    ]
    tickers = [
        _ticker("ALPHAUSDT", "50"),
        _ticker("BETAUSDT", "40"),
        _ticker("DELTAUSDT", "30"),
        _ticker("EPSUSDT", "20"),
        _ticker("GAMMAUSDT", "20"),
        _ticker("ZETAUSDT", "10"),
        _ticker("USDCUSDT", "99"),
        _ticker("XAUUSDT", "98"),
        _ticker("YOUNGUSDT", "97"),
        _ticker("BAD-NAMEUSDT", "96"),
    ]

    ranked = rank_universe(
        {"timezone": "UTC", "symbols": symbols},
        tickers,
        pd.Timestamp("2026-09-10 12:00:00", tz=UTC),
    )

    assert [row["symbol"] for row in ranked] == [
        "ALPHAUSDT",
        "BETAUSDT",
        "DELTAUSDT",
        "EPSUSDT",
        "GAMMAUSDT",
        "ZETAUSDT",
    ]


def test_rank_universe_returns_at_most_top_seven_candidates():
    symbols = [_exchange_symbol(f"COIN{i:02d}USDT", f"COIN{i:02d}") for i in range(1, 22)]
    tickers = [_ticker(f"COIN{i:02d}USDT", str(100 - i)) for i in range(1, 22)]

    ranked = rank_universe(
        {"timezone": "UTC", "symbols": symbols},
        tickers,
        pd.Timestamp("2026-09-10 12:00:00", tz=UTC),
    )

    assert len(ranked) == 7
    assert ranked[0]["symbol"] == "COIN01USDT"
    assert ranked[-1] == {
        "symbol": "COIN07USDT",
        "rank": 7,
        "price_change_percent": 93.0,
        "quote_volume": 30_000_000.0,
    }


def test_signal_requires_wr_pump_drop_and_returns_json_safe_payload():
    minutes = _minute_rows()
    now = pd.Timestamp("2026-09-10 00:38:10", tz=UTC)

    result = evaluate_signal("ALPHAUSDT", _daily(), minutes, now, rank=1)
    repeated = evaluate_signal("ALPHAUSDT", _daily(), minutes, now, rank=1)

    assert result["stage"] == "SIGNAL"
    assert result["reason"] == "signal_ready"
    assert result["daily_wr"] == pytest.approx(0.0)
    assert result["minute_wr"] <= -30.0
    assert result["wr_drop"] >= 30.0
    assert result["volume_ratio"] == pytest.approx(3.0)
    assert result["signal"] == {
        "signal_id": repeated["signal"]["signal_id"],
        "symbol": "ALPHAUSDT",
        "signal_time": int(pd.Timestamp("2026-09-10 00:38:00", tz=UTC).timestamp()),
        "pump_time": int(pd.Timestamp("2026-09-10 00:35:00", tz=UTC).timestamp()),
        "reversal_time": int(pd.Timestamp("2026-09-10 00:36:00", tz=UTC).timestamp()),
        "peak": 112.0,
        "atr": pytest.approx(result["signal"]["atr"]),
        "pump_wr": pytest.approx(result["signal"]["pump_wr"]),
        "breakdown_wr": pytest.approx(result["signal"]["breakdown_wr"]),
        "wr_drop": pytest.approx(result["signal"]["wr_drop"]),
    }
    json.dumps(result, allow_nan=False)


def test_unclosed_breakdown_candle_cannot_trigger():
    result = evaluate_signal(
        "ALPHAUSDT",
        _daily(),
        _minute_rows(),
        pd.Timestamp("2026-09-10 00:37:30", tz=UTC),
        rank=1,
    )

    assert result["stage"] == "WAIT_BREAKDOWN"
    assert result["signal"] is None


def test_large_breakdown_body_is_rejected_to_avoid_late_chasing():
    minutes = _minute_rows()
    minutes.loc[pd.Timestamp("2026-09-10 00:37", tz=UTC), "open"] = 107.0
    minutes.loc[pd.Timestamp("2026-09-10 00:37", tz=UTC), "close"] = 105.0

    result = evaluate_signal("ALPHAUSDT", _daily(), minutes, pd.Timestamp("2026-09-10 00:38:10", tz=UTC), rank=1)

    assert result["stage"] == "WAIT_PUMP"
    assert result["reason"] == "waiting_for_pump"
    assert result["signal"] is None


def test_deep_wr_drop_is_rejected_to_avoid_exhausted_breakdowns():
    minutes = _minute_rows()
    minutes.loc[pd.Timestamp("2026-09-10 00:37", tz=UTC), ["open", "high", "low", "close"]] = [
        102.5,
        103.0,
        101.9,
        102.0,
    ]

    result = evaluate_signal("ALPHAUSDT", _daily(), minutes, pd.Timestamp("2026-09-10 00:38:10", tz=UTC), rank=1)

    assert result["stage"] == "WAIT_PUMP"
    assert result["reason"] == "waiting_for_pump"
    assert result["signal"] is None


def test_wr_drop_between_thirty_five_and_fifty_is_accepted_when_wick_is_short():
    minutes = _minute_rows()
    minutes.loc[pd.Timestamp("2026-09-10 00:37", tz=UTC), ["open", "high", "low", "close"]] = [
        105.5,
        107.0,
        104.7,
        104.8,
    ]

    result = evaluate_signal("ALPHAUSDT", _daily(), minutes, pd.Timestamp("2026-09-10 00:38:10", tz=UTC), rank=7)

    assert result["stage"] == "SIGNAL"
    assert 35.0 < result["wr_drop"] <= 50.0


def test_breakdown_with_lower_wick_over_thirty_five_percent_is_rejected():
    minutes = _minute_rows()
    minutes.loc[pd.Timestamp("2026-09-10 00:37", tz=UTC), ["open", "high", "low", "close"]] = [
        105.9,
        107.0,
        103.0,
        105.5,
    ]

    result = evaluate_signal("ALPHAUSDT", _daily(), minutes, pd.Timestamp("2026-09-10 00:38:10", tz=UTC), rank=1)

    assert result["stage"] == "WAIT_PUMP"
    assert result["signal"] is None


def test_utc_six_and_seven_hours_block_new_wr_observations():
    minutes = _minute_rows(start="2026-09-10 06:00")

    result = evaluate_signal("ALPHAUSDT", _daily(), minutes, pd.Timestamp("2026-09-10 06:38:10", tz=UTC), rank=1)

    assert result["stage"] == "WAIT_PUMP"
    assert result["reason"] == "entry_hour_blocked"
    assert result["signal"] is None


def test_late_utc_session_blocks_new_wr_observations():
    minutes = _minute_rows(start="2026-09-10 14:00")

    result = evaluate_signal("ALPHAUSDT", _daily(), minutes, pd.Timestamp("2026-09-10 14:38:10", tz=UTC), rank=1)

    assert result["stage"] == "WAIT_PUMP"
    assert result["reason"] == "late_session_blocked"
    assert result["signal"] is None


@pytest.mark.parametrize(
    "minutes, now",
    [
        (
            pd.concat(
                [
                    _minute_rows().iloc[:36],
                    pd.DataFrame(
                        [
                            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100.0}
                            for _ in range(5)
                        ]
                        + [{"open": 111.0, "high": 111.8, "low": 106.0, "close": 107.0, "volume": 120.0}],
                        index=pd.date_range("2026-09-10 00:36", periods=6, freq="min", tz=UTC),
                    ),
                ]
            ),
            pd.Timestamp("2026-09-10 00:42:05", tz=UTC),
        ),
        (
            pd.concat(
                [
                    _minute_rows().iloc[:36],
                    pd.DataFrame(
                        [{"open": 111.0, "high": 112.1, "low": 106.0, "close": 107.0, "volume": 120.0}],
                        index=pd.DatetimeIndex([pd.Timestamp("2026-09-10 00:36", tz=UTC)]),
                    ),
                ]
            ),
            pd.Timestamp("2026-09-10 00:37:05", tz=UTC),
        ),
        (
            _minute_rows(start="2026-09-09 23:24", base_count=35),
            pd.Timestamp("2026-09-10 00:02:05", tz=UTC),
        ),
    ],
)
def test_timeout_new_high_and_utc_day_boundary_cancel_observation(minutes, now):
    result = evaluate_signal("ALPHAUSDT", _daily(), minutes, now, rank=1)

    assert result["stage"] == "WAIT_PUMP"
    assert result["signal"] is None


def test_signal_older_than_fifteen_seconds_is_refused():
    result = evaluate_signal(
        "ALPHAUSDT",
        _daily(),
        _minute_rows(),
        pd.Timestamp("2026-09-10 00:38:16", tz=UTC),
        rank=1,
    )

    assert result["stage"] != "SIGNAL"
    assert result["reason"] == "signal_expired"
    assert result["signal"] is None


@pytest.mark.parametrize("mutation", ["missing", "infinite", "duplicate"])
def test_invalid_minute_series_is_ineligible_and_json_safe(mutation):
    minutes = _minute_rows()
    if mutation == "missing":
        minutes = minutes.drop(minutes.index[10])
    elif mutation == "infinite":
        minutes.loc[minutes.index[10], "close"] = np.inf
    else:
        minutes = pd.concat([minutes, minutes.iloc[[10]]]).sort_index()

    result = evaluate_signal("ALPHAUSDT", _daily(), minutes, pd.Timestamp("2026-09-10 00:38:10", tz=UTC), rank=1)

    assert result["stage"] == "INELIGIBLE"
    assert result["signal"] is None
    json.dumps(result, allow_nan=False)


def test_rank_and_daily_input_rejections_do_not_emit_non_finite_values():
    daily = _daily()
    daily.loc[pd.Timestamp("2026-09-10", tz=UTC), "open"] = np.inf

    bad_daily = evaluate_signal("ALPHAUSDT", daily, _minute_rows(), pd.Timestamp("2026-09-10 00:38:10", tz=UTC), rank=1)
    accepted_rank = evaluate_signal("ALPHAUSDT", _daily(), _minute_rows(), pd.Timestamp("2026-09-10 00:38:10", tz=UTC), rank=7)
    out_of_rank = evaluate_signal("ALPHAUSDT", _daily(), _minute_rows(), pd.Timestamp("2026-09-10 00:38:10", tz=UTC), rank=8)

    assert bad_daily["stage"] == "INELIGIBLE"
    assert bad_daily["daily_open"] is None
    assert accepted_rank["stage"] == "SIGNAL"
    assert out_of_rank["reason"] == "rank_outside_top_7"
    json.dumps(bad_daily, allow_nan=False)
    json.dumps(out_of_rank, allow_nan=False)


def test_unix_seconds_now_is_supported_by_all_public_functions():
    now_seconds = pd.Timestamp("2026-09-10 00:38:10", tz=UTC).timestamp()
    info = {"symbols": [_exchange_symbol("ALPHAUSDT", "ALPHA")]}

    gate = daily_gate(_daily(), now_seconds)
    ranked = rank_universe(info, [_ticker("ALPHAUSDT", "12.5")], now_seconds)
    evaluated = evaluate_signal("ALPHAUSDT", _daily(), _minute_rows(), now_seconds, rank=1)

    assert gate["eligible"] is True
    assert ranked[0]["symbol"] == "ALPHAUSDT"
    assert evaluated["stage"] == "SIGNAL"


def test_daily_history_must_end_on_the_previous_utc_day():
    daily = _daily()
    daily = daily.drop(pd.Timestamp("2026-09-09", tz=UTC))

    result = daily_gate(daily, pd.Timestamp("2026-09-10 12:00", tz=UTC))

    assert result["eligible"] is False
    assert result["stage"] == "INELIGIBLE"


@pytest.mark.parametrize("field, position", [("open", 28), ("close", 0)])
def test_daily_gate_rejects_non_positive_prices(field, position):
    daily = _daily()
    daily.iloc[position, daily.columns.get_loc(field)] = 0.0

    result = daily_gate(daily, pd.Timestamp("2026-09-10 12:00", tz=UTC))

    assert result["eligible"] is False
    assert result["reason"] == "invalid_daily_data"


def test_minute_ohlc_prices_must_be_positive():
    minutes = _minute_rows()
    minutes.loc[minutes.index[10], ["open", "low"]] = 0.0

    result = evaluate_signal("ALPHAUSDT", _daily(), minutes, pd.Timestamp("2026-09-10 00:38:10", tz=UTC), rank=1)

    assert result["stage"] == "INELIGIBLE"
    assert result["reason"] == "invalid_minute_data"


def test_latest_closed_minute_must_immediately_precede_current_minute():
    stale_minutes = _minute_rows().iloc[:-1]

    result = evaluate_signal(
        "ALPHAUSDT",
        _daily(),
        stale_minutes,
        pd.Timestamp("2026-09-10 00:38:10", tz=UTC),
        rank=1,
    )

    assert result["stage"] == "INELIGIBLE"
    assert result["reason"] == "stale_minute_data"
