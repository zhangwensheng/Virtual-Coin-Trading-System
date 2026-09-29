"""Synthetic, temporary PAPER accounts for UI acceptance; never production ledgers."""
import math
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from futures_strategy.pair_portfolio import PairPortfolio
from futures_strategy.boll_short_portfolio import BollPortfolio
from futures_strategy.pair_web import create_server


def main():
    root = Path(tempfile.mkdtemp(prefix='dual-workstation-ui-'))
    pair,boll = PairPortfolio(root/'pair.db'),BollPortfolio(root/'boll.db')
    now = time.time()
    pair.set_running(True); boll.set_running(True)
    prices = {'BTCUSDT':95000.,'ETHUSDT':3500.,'AAAUSDT':100.,'BBBUSDT':100.}
    def quotes():
        now = time.time()
        q = {s:dict(bid=p,ask=p*1.0001,mark=p*1.00005,timestamp=now) for s,p in prices.items()}
        pair.update_quotes(q);boll.update_quotes(q)
    quotes()
    pair.open_group(dict(symbol_a='BTCUSDT',symbol_b='ETHUSDT',beta=1.,
        alpha=math.log(95000)-math.log(3500)-.025,mean=0.,std=.01,z=2.5,previous_z=2.8))
    for name in ['AAAUSDT','BBBUSDT']:
        boll.open_signal(dict(symbol=name,signal_id=name,signal_time=now,pump_time=now-180,
            reversal_time=now-120,peak=101.,atr=1.),
            dict(tick_size=.01,step_size=.001,min_qty=.001,min_notional=5.,max_qty=10000.))
    pair.set_running(False);boll.set_running(False)
    pair.log('SYNTHETIC_UI_ONLY',{'message':'仅用于页面验收，非真实行情或回测'})
    boll.log('SYNTHETIC_UI_ONLY',{'message':'仅用于页面验收，非真实行情或回测'})
    def watch():
        while True:
            quotes(); time.sleep(1)
    threading.Thread(target=watch,daemon=True).start()
    server = create_server(pair,port=8769,boll=boll)
    print(f'SYNTHETIC UI ONLY http://127.0.0.1:8769/ database={root}',flush=True)
    server.serve_forever()


if __name__=='__main__':
    main()
