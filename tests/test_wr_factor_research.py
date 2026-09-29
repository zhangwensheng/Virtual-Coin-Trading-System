import pandas as pd
import pytest

from scripts.research_wr_factors import entry_features, metrics, replay_exit, strategy_mask


def test_drawdown_includes_starting_capital_and_sorts_exits():
    rows = pd.DataFrame({'entry_time':pd.to_datetime(['2026-01-01','2026-01-02'],utc=True),
                         'exit_time':pd.to_datetime(['2026-01-03','2026-01-02'],utc=True),
                         'net_pnl':[10.,-100.]})
    result = metrics(rows)
    assert result['closed_trade_drawdown_pct'] == pytest.approx(10.)
    assert result['net_pnl'] == -90.


def test_entry_features_ignore_future_prices():
    index = pd.date_range('2026-01-01',periods=150,freq='min',tz='UTC')
    frame = pd.DataFrame(dict(open=100.,high=101.,low=99.,close=100.,volume=100.),index=index)
    row = dict(breakdown_time=index[130],pump_time=index[128],reversal_time=index[129],
               risk_distance=2.,entry_price=100.,atr=1.)
    first = entry_features(frame,row)
    frame.loc[index[131]:,['close','high','volume']] = 10000.
    assert entry_features(frame,row) == first
    assert first['return_15m_pct'] == 0.
    assert first['breakdown_lower_wick_fraction'] == .5


def test_stop_gap_uses_open_instead_of_unavailable_stop_price():
    index = pd.date_range('2026-01-01',periods=31,freq='min',tz='UTC')
    frame = pd.DataFrame(dict(open=100.,high=100.5,low=99.5,close=100.,volume=100.),index=index)
    frame.loc[index[1],['open','high','low','close']] = [105.,106.,104.,105.]
    row = dict(entry_time=index[0],stop_price=102.)
    result = replay_exit(frame,row,breakeven=False)
    assert result['exit_price'] == pytest.approx(105.*1.0002)
    assert result['exit_reason'] == 'STOP'


def test_new_trailing_stop_cannot_trigger_retroactively_on_same_candle():
    index = pd.date_range('2026-01-01',periods=31,freq='min',tz='UTC')
    frame = pd.DataFrame(dict(open=99.,high=99.5,low=98.8,close=99.,volume=100.),index=index)
    frame.loc[index[0],['open','high','low','close']] = [100.,101.,97.,98.]
    frame.loc[index[1],['open','high','low','close']] = [100.,101.,99.,100.]
    result = replay_exit(frame,dict(entry_time=index[0],stop_price=102.),breakeven=True)
    assert result['exit_time'] == index[2]


def test_strategy_mask_uses_top_twenty_wr_30_50_and_short_lower_wick():
    times = pd.to_datetime(
        ['2026-01-01 05:00', '2026-01-01 05:01', '2026-01-01 05:02', '2026-01-01 05:03'],
        utc=True,
    )
    pool = pd.DataFrame(
        {
            'daily_rank': [5, 20, 21, 10],
            'wr_drop': [30, 50, 40, 40],
            'breakdown_lower_wick_fraction': [.10, .35, .10, .36],
            'entry_time': times,
            'pump_time': times - pd.Timedelta(minutes=3),
            'reversal_time': times - pd.Timedelta(minutes=1),
        }
    )

    assert strategy_mask(pool, max_rank=20).tolist() == [True, True, False, False]
