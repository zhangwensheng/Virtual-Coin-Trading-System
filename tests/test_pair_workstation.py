from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from futures_strategy.pair_portfolio import PairPortfolio, PairSettings
from futures_strategy.pair_signals import entry_direction, fit_pair


NOW = 1_800_000_000.0


def model(a='AAAUSDT', b='BBBUSDT'):
    return dict(symbol_a=a, symbol_b=b, beta=1.0, alpha=0.0, mean=0.0,
                std=0.02, z=2.5, previous_z=2.8, correlation=0.9,
                pvalue_1h=0.01, pvalue_4h=0.01)


def quotes(engine, a=100.0, b=100.0, now=NOW, symbols=('AAAUSDT', 'BBBUSDT')):
    engine.update_quotes({symbols[0]: dict(bid=a, ask=a, mark=a, timestamp=now),
                          symbols[1]: dict(bid=b, ask=b, mark=b, timestamp=now)}, now=now)


def portfolio(tmp_path, **changes):
    return PairPortfolio(tmp_path / 'paper.sqlite3', PairSettings(**changes))


def test_pair_close_accounts_for_both_legs_and_all_fees(tmp_path):
    p = portfolio(tmp_path, slippage_bps=0, group_loss_limit=10)
    quotes(p)
    p.set_running(True)
    group = p.open_group(model(), now=NOW)
    # Gross 320 = short A 160 + long B 160; each qty 1.6.
    assert group['legs'][0]['qty'] == pytest.approx(1.6)
    assert group['legs'][0]['side'] == 'SHORT'
    quotes(p, a=99, b=101, now=NOW + 10)
    closed = p.close_group(group['id'], now=NOW + 10)
    assert closed['price_pnl'] == pytest.approx(3.2)
    assert closed['fees'] == pytest.approx(0.32)
    assert closed['net_pnl'] == pytest.approx(2.88)
    assert p.snapshot(now=NOW+10)['account']['equity'] == pytest.approx(1002.88)
    assert p.close_group(group['id'], now=NOW + 10)['net_pnl'] == closed['net_pnl']
    assert len(p.snapshot()['closed_groups']) == 1


def test_close_all_pauses_entries_and_retains_groups_without_fresh_quotes(tmp_path):
    p = portfolio(tmp_path)
    p.set_running(True)
    quotes(p)
    first = p.open_group(model(), now=NOW)
    quotes(p, symbols=('CCCUSDT', 'DDDUSDT'))
    second = p.open_group(model('CCCUSDT', 'DDDUSDT'), now=NOW)
    quotes(p, now=NOW + 100)
    result = p.close_all(now=NOW + 100)
    assert result['closed'] == [first['id']]
    assert result['failed'][0]['id'] == second['id']
    state = p.snapshot(now=NOW + 100)
    assert state['running'] is False
    assert [g['id'] for g in state['groups']] == [second['id']]
    with pytest.raises(ValueError, match='PAUSED'):
        p.open_group(model('EEEUSDT', 'FFFUSDT'), now=NOW+100)


def test_restart_restores_positions_but_never_reuses_quotes_or_auto_starts(tmp_path):
    p = portfolio(tmp_path)
    quotes(p)
    p.set_running(True)
    group = p.open_group(model(), now=NOW)
    restored = portfolio(tmp_path)
    assert len(restored.snapshot()['groups']) == 1
    assert restored.snapshot()['running'] is False
    with pytest.raises(ValueError, match='QUOTE'):
        restored.close_group(group['id'], now=NOW+1)


def test_coin_cannot_be_shared_and_daily_count_is_per_coin(tmp_path):
    p = portfolio(tmp_path, cooldown_seconds=0)
    p.set_running(True)
    for i in range(3):
        quotes(p, now=NOW+i)
        group = p.open_group(model(), now=NOW+i)
        if i == 0:
            with pytest.raises(ValueError, match='SYMBOL_BUSY'):
                p.open_group(model('AAAUSDT', 'CCCUSDT'), now=NOW)
        p.close_group(group['id'], now=NOW+i)
    quotes(p, now=NOW+4)
    with pytest.raises(ValueError, match='DAILY_LIMIT'):
        p.open_group(model(), now=NOW+4)


def test_manual_close_cooldown_and_concurrent_duplicate_are_persisted(tmp_path):
    p = portfolio(tmp_path)
    p.set_running(True)
    quotes(p)
    g = p.open_group(model(), now=NOW)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: p.close_group(g['id'], now=NOW+1), range(2)))
    assert results[0]['net_pnl'] == results[1]['net_pnl']
    with pytest.raises(ValueError, match='COOLDOWN'):
        p.open_group(model(), now=NOW+2)
    assert len(p.snapshot()['closed_groups']) == 1


def test_funding_is_signed_settlement_only_and_idempotent(tmp_path):
    p = portfolio(tmp_path, slippage_bps=0, group_loss_limit=10)
    p.set_running(True)
    quotes(p)
    g = p.open_group(model(), now=NOW)
    p.settle_funding('AAAUSDT', NOW-1, 0.01, 100, now=NOW+10)
    p.settle_funding('AAAUSDT', NOW+5, 0.01, 100, now=NOW+10)
    p.settle_funding('AAAUSDT', NOW+5, 0.01, 100, now=NOW+10)
    p.settle_funding('BBBUSDT', NOW+5, 0.02, 100, now=NOW+10)
    assert p.snapshot(now=NOW+10)['groups'][0]['funding_pnl'] == pytest.approx(-1.6)
    with pytest.raises(ValueError, match='FUTURE'):
        p.settle_funding('AAAUSDT', NOW+20, 0.01, 100, now=NOW+10)
    closed = p.close_group(g['id'], now=NOW+10)
    assert closed['net_pnl'] == pytest.approx(-1.92)


def test_invalid_quote_cannot_liquidate_or_open_a_group(tmp_path):
    p = portfolio(tmp_path)
    p.set_running(True)
    quotes(p)
    g = p.open_group(model(), now=NOW)
    with pytest.raises(ValueError):
        p.update_quotes({'AAAUSDT': dict(bid=-1, ask=0, mark=float('nan'), timestamp=NOW)}, now=NOW)
    with pytest.raises(ValueError, match='QUOTE'):
        p.close_group(g['id'], now=NOW+100)
    assert len(p.snapshot()['groups']) == 1


def test_entry_requires_convergence_and_rejects_extreme_or_invalid_hedge():
    assert entry_direction(2.5, 2.8) == 'SHORT_A_LONG_B'
    assert entry_direction(-2.5, -2.8) == 'LONG_A_SHORT_B'
    assert entry_direction(2.8, 2.5) is None
    assert entry_direction(4.1, 4.3) is None
    assert entry_direction(float('nan'), 2.5) is None


def test_fit_uses_only_closed_historical_bars_and_rejects_missing_history():
    rng = np.random.default_rng(19)
    index = pd.date_range('2026-01-01', periods=800, freq='h', tz='UTC')
    base = 4 + rng.normal(0, .015, 800).cumsum()
    b = pd.DataFrame({'close': np.exp(base)}, index=index)
    a = pd.DataFrame({'close': np.exp(.4 + .8*base + rng.normal(0, .002, 800))}, index=index)
    at = index[-1].timestamp()
    result = fit_pair('A', 'B', a, b, at)
    assert result is not None
    assert result['beta'] == pytest.approx(.8, abs=.03)
    a.loc[index[-1], 'close'] = 9999999  # current unfinished candle must not alter fit
    assert fit_pair('A', 'B', a, b, at)['beta'] == result['beta']
    assert fit_pair('A', 'B', a.iloc[-10:], b.iloc[-10:], at) is None


def test_group_exit_uses_frozen_spread_and_handles_crossing_mean(tmp_path):
    p = portfolio(tmp_path, slippage_bps=0)
    p.set_running(True)
    quotes(p, a=105, b=100)
    g = p.open_group(model(), now=NOW)
    quotes(p, a=99.9, b=100, now=NOW+10)
    result = p.monitor(now=NOW+10)
    assert result[0]['id'] == g['id']
    assert result[0]['exit_reason'] == 'MEAN_REVERSION'


def test_budget_limits_total_two_leg_notional(tmp_path):
    p = portfolio(tmp_path)
    p.set_running(True)
    for a, b in [('A', 'B'), ('C', 'D'), ('E', 'F')]:
        quotes(p, symbols=(a,b))
        p.open_group(model(a,b), now=NOW)
    assert p.snapshot(now=NOW)['account']['gross_notional'] <= 800
    assert p.snapshot(now=NOW)['account']['available'] >= 200


def test_daily_loss_pauses_new_entries(tmp_path):
    p = portfolio(tmp_path, slippage_bps=0, daily_loss_limit=1)
    p.set_running(True)
    quotes(p)
    p.open_group(model(), now=NOW)
    quotes(p, a=102, b=100, now=NOW+10)
    p.monitor(now=NOW+10)
    assert p.snapshot(now=NOW+10)['running'] is False


def test_late_published_funding_is_reconciled_after_close(tmp_path):
    p = portfolio(tmp_path, slippage_bps=0, group_loss_limit=10)
    p.set_running(True)
    quotes(p)
    g = p.open_group(model(), now=NOW)
    p.close_group(g['id'], now=NOW+10)
    p.settle_funding('AAAUSDT', NOW+5, .01, 100, now=NOW+20)
    p.settle_funding('AAAUSDT', NOW+15, .01, 100, now=NOW+20)
    assert p.snapshot()['closed_groups'][0]['net_pnl'] == pytest.approx(1.28)
    assert p.snapshot()['account']['cash'] == pytest.approx(1001.28)


def test_stale_mixed_leg_valuation_never_triggers_permanent_circuit(tmp_path):
    p = portfolio(tmp_path, slippage_bps=0, group_loss_limit=10)
    p.set_running(True)
    quotes(p)
    p.open_group(model(), now=NOW)
    # B falls while A's equally old quote is not updated: do not fix this fake loss into risk state.
    p.update_quotes({'BBBUSDT':dict(bid=50,ask=50,mark=50,timestamp=NOW+100)},now=NOW+100)
    p.monitor(now=NOW+100)
    assert p.snapshot(now=NOW+100)['circuit'] is False
    quotes(p,symbols=('CCCUSDT','DDDUSDT'),now=NOW+100)
    with pytest.raises(ValueError,match='STALE_PORTFOLIO'):
        p.open_group(model('CCCUSDT','DDDUSDT'),now=NOW+100)


def test_closed_funding_backlog_survives_long_shutdown_until_both_legs_checked(tmp_path):
    p = portfolio(tmp_path)
    p.set_running(True)
    quotes(p)
    g=p.open_group(model(),now=NOW)
    p.close_group(g['id'],now=NOW+10)
    restored=portfolio(tmp_path)
    assert set(restored.funding_windows()) == {'AAAUSDT','BBBUSDT'}
    restored.mark_funding_checked('AAAUSDT',now=NOW+5*86400)
    assert 'BBBUSDT' in restored.funding_windows()
    restored.mark_funding_checked('BBBUSDT',now=NOW+5*86400)
    assert restored.funding_windows()=={}


def test_recent_close_remains_in_funding_backlog_for_delayed_publication(tmp_path):
    p = portfolio(tmp_path)
    p.set_running(True)
    quotes(p)
    g=p.open_group(model(),now=NOW)
    p.close_group(g['id'],now=NOW+10)
    for symbol in ('AAAUSDT','BBBUSDT'):
        p.mark_funding_checked(symbol,now=NOW+60)
    assert len(p.funding_windows()) == 2
