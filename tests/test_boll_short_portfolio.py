import time

import pytest

from futures_strategy.boll_short_portfolio import BollPortfolio


NOW = 1789002000.0
RULES = dict(tick_size=.01, step_size=.001, min_qty=.001, min_notional=5, max_qty=10000)


def signal(symbol='AAAUSDT', now=NOW, sid='s1'):
    return dict(symbol=symbol, signal_id=sid, signal_time=now, pump_time=now-180,
                reversal_time=now-120, peak=101., atr=1.)


def quote(p, symbol='AAAUSDT', now=NOW, price=100.):
    p.update_quotes({symbol: dict(bid=price, ask=price+.02, mark=price+.01, timestamp=now)}, now=now)


@pytest.fixture
def p(tmp_path):
    obj = BollPortfolio(tmp_path/'boll.sqlite3')
    obj.set_running(True, now=NOW)
    quote(obj)
    yield obj
    obj.db.close()


def test_open_sizes_by_stop_budget_and_duplicate_signal_cannot_debit_twice(p):
    g = p.open_signal(signal(), RULES, now=NOW)
    assert g['side'] == 'SHORT'
    assert g['entry_price'] == pytest.approx(99.98)
    assert g['stop_price'] == pytest.approx(101.2)
    assert g['planned_risk'] <= 3
    assert g['gross_notional'] <= 320
    before = p.snapshot(now=NOW)['account']['cash']
    with pytest.raises(ValueError, match='SIGNAL_USED'):
        p.open_signal(signal(), RULES, now=NOW)
    assert p.snapshot(now=NOW)['account']['cash'] == before
    assert len(p.snapshot(now=NOW)['positions']) == 1


@pytest.mark.parametrize('change,error', [({'signal_time': NOW-16}, 'SIGNAL_EXPIRED'),
                                        ({'peak': float('nan')}, 'INVALID_SIGNAL'),
                                        ({'atr': 0}, 'INVALID_SIGNAL')])
def test_invalid_signals_cannot_open(p, change, error):
    s = signal(); s.update(change)
    with pytest.raises(ValueError, match=error):
        p.open_signal(s, RULES, now=NOW)
    assert p.snapshot(now=NOW)['account']['cash'] == 1000


def test_stale_quotes_wide_spread_and_unknown_leverage_block_open(p):
    with pytest.raises(ValueError, match='STALE'):
        p.open_signal(signal(now=NOW+6), RULES, now=NOW+6)
    p.update_quotes({'AAAUSDT': dict(bid=100, ask=101, mark=100.5, timestamp=NOW)}, now=NOW)
    with pytest.raises(ValueError, match='SPREAD'):
        p.open_signal(signal(), RULES, now=NOW)
    with pytest.raises(ValueError, match='LEVERAGE'):
        BollPortfolio(p.path.parent/'two.sqlite3', leverage=2)


def test_max_two_positions_and_same_symbol_not_reopened(p):
    p.open_signal(signal(), RULES, now=NOW)
    with pytest.raises(ValueError, match='SYMBOL_BUSY'):
        p.open_signal(signal(sid='s2'), RULES, now=NOW)
    quote(p, 'BBBUSDT')
    p.open_signal(signal('BBBUSDT', sid='b'), RULES, now=NOW)
    quote(p, 'CCCUSDT')
    with pytest.raises(ValueError, match='POSITION_LIMIT'):
        p.open_signal(signal('CCCUSDT', sid='c'), RULES, now=NOW)


def test_close_uses_buy_quote_and_is_idempotent(p):
    g = p.open_signal(signal(), RULES, now=NOW)
    quote(p, price=99, now=NOW+1)
    closed = p.close_position(g['id'], now=NOW+1)
    assert closed['exit_price'] == pytest.approx(99.02*1.0002)
    expected = (99.98-99.02*1.0002)*g['qty']-closed['fees']
    assert closed['net_pnl'] == pytest.approx(expected)
    assert p.close_position(g['id'], now=NOW+2) == closed
    assert p.snapshot(now=NOW+2)['account']['cash'] == pytest.approx(1000+expected)


def test_close_all_pauses_and_retains_stale_position(p):
    g = p.open_signal(signal(), RULES, now=NOW)
    result = p.close_all(now=NOW+6)
    assert result['closed'] == []
    assert result['failed'][0]['id'] == g['id']
    assert p.snapshot(now=NOW+6)['running'] is False
    assert len(p.snapshot(now=NOW+6)['positions']) == 1


def test_stop_and_time_exits_work_while_paused(p):
    g = p.open_signal(signal(), RULES, now=NOW)
    p.set_running(False, now=NOW)
    quote(p, price=102, now=NOW+1)
    p.monitor(now=NOW+1)
    assert p.snapshot(now=NOW+1)['closed_positions'][0]['exit_reason'] == 'STOP'


def test_trailing_never_widens_and_target_can_close(p):
    g = p.open_signal(signal(), RULES, now=NOW)
    quote(p, price=98.5, now=NOW+1)
    p.monitor(now=NOW+1)
    moved = p.snapshot(now=NOW+1)['positions'][0]['stop_price']
    assert moved < g['stop_price']
    quote(p, price=98.8, now=NOW+2)
    p.monitor(now=NOW+2)
    assert p.snapshot(now=NOW+2)['positions'][0]['stop_price'] == moved
    quote(p, price=97, now=NOW+3)
    p.monitor(now=NOW+3)
    assert p.snapshot(now=NOW+3)['closed_positions'][0]['exit_reason'] == 'TARGET'


def test_restart_restores_position_but_pauses_and_consumed_signal_persists(p):
    p.open_signal(signal(), RULES, now=NOW)
    restored = BollPortfolio(p.path)
    try:
        assert restored.snapshot(now=NOW)['running'] is False
        assert len(restored.snapshot(now=NOW)['positions']) == 1
        restored.set_running(True, now=NOW)
        quote(restored)
        with pytest.raises(ValueError, match='SIGNAL_USED'):
            restored.open_signal(signal(), RULES, now=NOW)
    finally:
        restored.db.close()


def test_funding_late_settlement_and_dedup(p):
    g = p.open_signal(signal(), RULES, now=NOW)
    quote(p, now=NOW+100)
    p.close_position(g['id'], now=NOW+100)
    cash = p.snapshot(now=NOW+100)['account']['cash']
    p.settle_funding('AAAUSDT', NOW+60, .001, 100, now=NOW+200)
    p.settle_funding('AAAUSDT', NOW+60, .001, 100, now=NOW+200)
    assert p.snapshot(now=NOW+200)['account']['cash'] == pytest.approx(cash+g['qty']*.1)
    assert p.funding_windows() == {'AAAUSDT': NOW}
    p.mark_funding_checked('AAAUSDT', now=NOW+100+72*3600)
    assert p.funding_windows() == {}


def test_cooldown_daily_limit_and_daily_loss_latch(p):
    for n in range(3):
        t = NOW+n*1900
        quote(p, now=t)
        g = p.open_signal(signal(now=t, sid=str(n)), RULES, now=t)
        p.close_position(g['id'], now=t)
        with pytest.raises(ValueError, match='COOLDOWN|DAILY_LIMIT'):
            p.open_signal(signal(now=t, sid=f'again{n}'), RULES, now=t)
    t = NOW+6000
    quote(p, now=t)
    with pytest.raises(ValueError, match='DAILY_LIMIT'):
        p.open_signal(signal(now=t, sid='four'), RULES, now=t)
    # A fresh second asset gaps through its stop; losses must latch for the day.
    quote(p, 'BBBUSDT', now=t)
    p.open_signal(signal('BBBUSDT', t, 'b'), RULES, now=t)
    quote(p, 'BBBUSDT', now=t+1, price=120)
    p.monitor(now=t+1)
    assert p.snapshot(now=t+1)['risk_paused'] is True
    with pytest.raises(ValueError, match='DAILY_LOSS'):
        p.set_running(True, now=t+1)


def test_failed_audit_write_rolls_back_cash_position_and_signal(p):
    p.db.execute("CREATE TRIGGER reject_open BEFORE INSERT ON events WHEN NEW.kind='OPEN_POSITION' BEGIN SELECT RAISE(ABORT,'disk test'); END")
    with pytest.raises(Exception,match='disk test'):
        p.open_signal(signal(), RULES, now=NOW)
    assert p.snapshot(now=NOW)['account']['cash'] == 1000
    assert p.snapshot(now=NOW)['positions'] == []
    p.db.execute('DROP TRIGGER reject_open')
    assert p.open_signal(signal(),RULES,now=NOW)['symbol'] == 'AAAUSDT'


def test_stale_day_boundary_keeps_baseline_and_time_exit(p):
    from datetime import datetime, timezone, timedelta
    t = datetime(2026,9,10,23,59,tzinfo=timezone(timedelta(hours=8))).timestamp()
    quote(p,now=t)
    p.set_running(True,now=t)
    g=p.open_signal(signal(now=t),RULES,now=t)
    p.monitor(now=t)
    baseline=p.state['day_equity']
    p.monitor(now=t+120)
    assert p.state['day_equity']==baseline
    quote(p,now=t+1801)
    p.monitor(now=t+1801)
    assert p.snapshot(now=t+1801)['closed_positions'][0]['exit_reason']=='TIME_EXIT'
