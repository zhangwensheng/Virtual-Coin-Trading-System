import importlib.util
import unittest
import numpy as np
import pandas as pd


class ComboFeaturesTests(unittest.TestCase):
    def mod(self):
        self.assertIsNotNone(importlib.util.find_spec('futures_strategy.combo_features'))
        from futures_strategy import combo_features
        return combo_features

    def test_future_does_not_change_entry_features(self):
        m = self.mod()
        ix = pd.date_range('2026-01-01', periods=5000, freq='min', tz='UTC')
        c = 100 + np.sin(np.arange(5000) / 70)
        f = pd.DataFrame(dict(open=c, high=c+.1, low=c-.1, close=c, volume=1., taker_buy_base=.5), index=ix)
        first = m.enrich_features(f)
        f.loc[ix[4500]:, 'close'] *= 1.1
        second = m.enrich_features(f)
        pd.testing.assert_frame_equal(first.iloc[:4500], second.iloc[:4500])

    def test_cost_ratio_uses_stop_distance_not_leverage(self):
        m = self.mod()
        self.assertAlmostEqual(m.cost_risk_ratio(100., 99., .0014), .14)
        self.assertAlmostEqual(m.cost_risk_ratio(100., 99.8, .0014), .7)

    def test_selector_rejects_losing_and_tiny_sample_sleeves(self):
        m = self.mod()
        self.assertEqual(m.select_sleeves([{'name':'a','train_net':10.,'valid_net':-2.,'train_n':100,'valid_n':50},
                                           {'name':'b','train_net':10.,'valid_net':5.,'train_n':3,'valid_n':2}]), [])
        self.assertEqual(m.select_sleeves([{'name':'a','train_net':10.,'valid_net':5.,'train_n':100,'valid_n':50}]), ['a'])

    def test_candidate_signals_do_not_depend_on_future_candles(self):
        m=self.mod()
        from futures_strategy.combo_candidates import generate_candidates
        ix=pd.date_range('2026-01-01',periods=6000,freq='min',tz='UTC')
        x=np.arange(6000)
        c=100*np.exp(.001*x/60+.01*np.sin(x/300)+.002*np.sin(x/7))
        raw=pd.DataFrame(dict(open=c,high=c+.2,low=c-.2,close=c,volume=1.,taker_buy_base=.5),index=ix)
        f=m.enrich_features(raw)
        first=generate_candidates({'BTCUSDT':f,'ETHUSDT':f},[])
        raw.iloc[5000:,:4]*=1.5
        changed=m.enrich_features(raw)
        second=generate_candidates({'BTCUSDT':changed,'ETHUSDT':changed},[])
        fields=('i','strategy','side','stop','target')
        count=0
        for name,signals in first.items():
            before=[tuple(s.get(k) for k in fields) for s in signals if s['i']<5000]
            after=[tuple(s.get(k) for k in fields) for s in second[name] if s['i']<5000]
            self.assertEqual(before,after,name)
            count+=len(before)
        self.assertGreater(count,0)


if __name__ == '__main__':
    unittest.main()
