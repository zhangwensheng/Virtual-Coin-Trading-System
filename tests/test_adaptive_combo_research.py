"""Accounting and causality checks for the isolated research backtester."""
import importlib.util
import unittest

import numpy as np
import pandas as pd


class AdaptiveResearchTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.find_spec('futures_strategy.adaptive_combo_research')
        self.assertIsNotNone(spec, 'The research engine must implement the tested contract')
        from futures_strategy import adaptive_combo_research
        return adaptive_combo_research

    def test_higher_timeframe_not_visible_before_close(self):
        m = self.module()
        ix = pd.date_range('2026-01-01', periods=30, freq='min', tz='UTC')
        f = pd.DataFrame({'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 1.}, index=ix)
        f.loc[ix[14], 'close'] = 110.
        bars = m.closed_bars(f, '15min')
        aligned = bars['close'].reindex(ix, method='ffill')
        self.assertTrue(pd.isna(aligned.iloc[14]))
        self.assertEqual(aligned.iloc[15], 110.)

    def test_stop_beats_target_and_gap_gets_worse_fill(self):
        m = self.module()
        self.assertEqual(m.bar_exit(1, 100, 106, 94, 95, 105), (95, 'stop'))
        self.assertEqual(m.bar_exit(1, 90, 100, 89, 95, 105), (90, 'stop'))
        self.assertEqual(m.bar_exit(-1, 110, 111, 95, 105, 95), (110, 'stop'))

    def test_round_trip_fees_and_slippage_on_both_sides(self):
        m = self.module()
        entry = m.fill_price(100., 1, .0002)
        exit_ = m.fill_price(100., -1, .0002)
        net = 10 * (exit_ - entry) - 10 * (entry + exit_) * .0005
        self.assertAlmostEqual(net, -1.4)

    def test_funding_direction_and_both_legs(self):
        m = self.module()
        self.assertAlmostEqual(m.funding_cashflow(np.array([2., -1.]), np.array([100., 100.]), np.array([.001, .002])), 0.)
        self.assertAlmostEqual(m.funding_cashflow(np.array([2.]), np.array([100.]), np.array([.001])), -.2)

    def test_shared_daily_counts_include_each_pair_leg(self):
        m = self.module()
        self.assertFalse(m.can_enter(['BTC', 'ETH'], {'BTC': 3}, set(), 1))
        self.assertFalse(m.can_enter(['BTC', 'ETH'], {}, {'ETH'}, 1))
        self.assertFalse(m.can_enter(['BTC'], {}, set(), 20))
        self.assertTrue(m.can_enter(['BTC', 'ETH'], {'BTC': 2}, set(), 19))

    def test_features_unchanged_when_future_prices_change(self):
        m = self.module()
        ix = pd.date_range('2026-01-01', periods=6000, freq='min', tz='UTC')
        c = 100 + np.sin(np.arange(len(ix)) / 100)
        f = pd.DataFrame({'open': c, 'high': c + .2, 'low': c - .2, 'close': c, 'volume': 1.}, index=ix)
        baseline = m.prepare_features(f)
        f.iloc[5000:, :4] *= 2
        changed = m.prepare_features(f)
        pd.testing.assert_frame_equal(baseline.iloc[:5000], changed.iloc[:5000])

    def test_account_executes_next_open_and_reconciles_all_costs(self):
        m = self.module()
        ix = pd.date_range('2026-06-01', periods=8, freq='min', tz='UTC')
        f = pd.DataFrame({'open': 100., 'high': 100.1, 'low': 99.9, 'close': 100., 'b_atr': 1.}, index=ix)
        signal = dict(i=0, strategy='trend', symbols=['BTC'], side=1, stop=95., target=115., eligible=True, max_hold=99)
        rates = pd.Series([.001], index=ix[2:3])
        summary, trades, curve, _ = m.run_portfolio({'BTC': f}, [signal], {'BTC': rates}, {}, ix[0], ix[-1] + pd.Timedelta(minutes=1), m.Settings())
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.entry_time.iloc[0], ix[1])
        self.assertLess(trades.funding.iloc[0], 0)
        self.assertGreater(trades.fees.iloc[0], 0)
        self.assertAlmostEqual(summary['final_equity'], 1000 + trades.net_pnl.sum())
        self.assertAlmostEqual(trades.net_pnl.sum(), trades.price_pnl.sum() - trades.fees.sum() + trades.funding.sum())
        self.assertEqual(curve.positions.iloc[-1], 0)

    def test_actual_portfolio_enforces_daily_three_entries(self):
        m = self.module()
        ix = pd.date_range('2026-06-01', periods=180, freq='min', tz='UTC')
        f = pd.DataFrame({'open': 100., 'high': 100.1, 'low': 99.9, 'close': 100., 'b_atr': 1.}, index=ix)
        signals = [dict(i=i, strategy='trend', symbols=['BTC'], side=1, stop=95., target=115., eligible=True, max_hold=1) for i in (0, 30, 60, 90, 120)]
        result = m.run_portfolio({'BTC': f}, signals, {'BTC': pd.Series(dtype=float)}, {}, ix[0], ix[-1] + pd.Timedelta(minutes=1), m.Settings())
        self.assertEqual(result[0]['trades'], 3)

    def test_pair_keeps_two_leg_accounting_and_shared_symbol_lock(self):
        m = self.module()
        ix = pd.date_range('2026-06-01', periods=8, freq='min', tz='UTC')
        f = pd.DataFrame({'open': 100., 'high': 100.01, 'low': 99.99, 'close': 100., 'b_atr': 1.}, index=ix)
        pair = dict(i=0, strategy='pair', symbols=['BTC', 'ETH'], side=-1, eligible=True, alpha=-.022,
                    beta=1., sd=.01, entry_z=2.2, max_hold=99)
        directional = dict(i=1, strategy='trend', symbols=['ETH'], side=1, stop=95., target=115., eligible=True, max_hold=99)
        summary, trades, _, _ = m.run_portfolio({'BTC': f, 'ETH': f}, [pair, directional],
            {'BTC': pd.Series([.001], index=ix[2:3]), 'ETH': pd.Series([.002], index=ix[2:3])},
            {('BTC', 'ETH'): np.ones(len(ix), dtype=bool)}, ix[0], ix[-1] + pd.Timedelta(minutes=1), m.Settings())
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.symbols.iloc[0], 'BTC__ETH')
        self.assertLess(trades.funding.iloc[0], 0)
        self.assertAlmostEqual(summary['capital_reconciliation_error'], 0.)

    def test_gap_loss_latches_account_halt_without_capping_reported_loss(self):
        m = self.module()
        ix = pd.date_range('2026-06-01', periods=60, freq='min', tz='UTC')
        price = np.full(len(ix), 100.)
        price[2:] = 50.
        f = pd.DataFrame({'open': price, 'high': price + .01, 'low': price - .01, 'close': price, 'b_atr': 1.}, index=ix)
        signals = [dict(i=0, strategy='trend', symbols=['BTC'], side=1, stop=99.8, target=101., eligible=True, max_hold=99),
                   dict(i=30, strategy='trend', symbols=['BTC'], side=1, stop=49., target=53., eligible=True, max_hold=99)]
        summary, trades, _, _ = m.run_portfolio({'BTC': f}, signals, {'BTC': pd.Series(dtype=float)}, {},
            ix[0], ix[-1] + pd.Timedelta(minutes=1), m.Settings())
        self.assertTrue(summary['hard_halt'])
        self.assertEqual(len(trades), 1)
        self.assertGreater(summary['max_drawdown_pct'], 20.)
        self.assertLess(summary['final_equity'], 800.)

    def test_optional_exit_rule_does_not_force_half_sale_at_one_r(self):
        m = self.module()
        ix = pd.date_range('2026-06-01', periods=8, freq='min', tz='UTC')
        f = pd.DataFrame({'open':100.,'high':100.1,'low':99.9,'close':100.,'b_atr':1.}, index=ix)
        f.loc[ix[2], 'high'] = 101.1
        signal = dict(i=0,strategy='trend',symbols=['BTC'],side=1,stop=99.,target=104.,eligible=True,
                      max_hold=99,partial_ratio=0.,partial_rr=1.5,trail_mult=2.5)
        result = m.run_portfolio({'BTC':f},[signal],{'BTC':pd.Series(dtype=float)}, {},ix[0],
            ix[-1]+pd.Timedelta(minutes=1),m.Settings(fee=0.,slip=0.))
        self.assertAlmostEqual(result[1].price_pnl.sum(),0.)


if __name__ == '__main__':
    unittest.main()
