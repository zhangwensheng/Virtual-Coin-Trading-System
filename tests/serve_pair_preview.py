"""Isolated synthetic account for manual browser acceptance checks; never a live feed."""
import math
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from futures_strategy.pair_portfolio import PairPortfolio, PairSettings
from futures_strategy.pair_web import create_server


def main():
    root = Path(tempfile.mkdtemp(prefix='pair-ui-check-'))
    p = PairPortfolio(root/'paper.sqlite3', PairSettings())
    p.set_running(True)
    values = [('BTCUSDT', 'ETHUSDT', 95000., 3500., 1.1), ('SOLUSDT','AVAXUSDT',145.,24.,.8)]
    now=time.time()
    for a,b,pa,pb,beta in values:
        p.update_quotes({a:dict(bid=pa,ask=pa,mark=pa,timestamp=now),b:dict(bid=pb,ask=pb,mark=pb,timestamp=now)})
        p.open_group(dict(symbol_a=a,symbol_b=b,beta=beta,alpha=math.log(pa)-beta*math.log(pb)-.025,
                          mean=0.,std=.01,z=2.5,previous_z=2.8))
    p.set_running(False)
    p.log('UI_TEST_ONLY', {'message':'隔离的合成测试账户，仅用于验证界面和平仓按钮'})
    def refresh():
        while True:
            now=time.time()
            for i,(a,b,pa,pb,_) in enumerate(values):
                pa*=.997 if i==0 else 1.003
                pb*=1.002 if i==0 else .999
                p.update_quotes({a:dict(bid=pa,ask=pa,mark=pa,timestamp=now),b:dict(bid=pb,ask=pb,mark=pb,timestamp=now)})
            time.sleep(2)
    threading.Thread(target=refresh,daemon=True).start()
    server=create_server(p,port=8766)
    print(f'UI test only: http://127.0.0.1:8766; database: {root}',flush=True)
    server.serve_forever()


if __name__=='__main__':
    main()
