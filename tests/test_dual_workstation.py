import json
import threading
import urllib.error

import pytest

from futures_strategy.pair_portfolio import PairPortfolio
from futures_strategy.boll_short_portfolio import BollPortfolio
from futures_strategy.pair_web import create_server
from test_pair_workstation_http import request


@pytest.fixture
def dual(tmp_path):
    pair = PairPortfolio(tmp_path/'pair.db')
    boll = BollPortfolio(tmp_path/'boll.db')
    server = create_server(pair,port=0,boll=boll)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    yield pair,boll,server,base
    server.shutdown(); server.server_close(); thread.join(timeout=2)
    pair.db.close(); boll.db.close()


def test_boll_running_is_independent_and_overview_does_not_double_count(dual):
    pair,boll,server,base = dual
    response = json.load(request(base,'/api/boll/state'))
    assert response['account']['equity'] == 1000
    result = json.load(request(base,'/api/overview'))
    assert result['paper_initial_capital'] == 2000
    assert result['equity'] == 2000
    request(base,'/api/boll/running',{'running':True},server.token,base).close()
    assert boll.snapshot()['running'] is True
    assert pair.snapshot()['running'] is False
    assert json.load(request(base,'/api/state'))['mode'] == 'PAPER'


def test_live_cannot_be_enabled_even_with_valid_session(dual):
    _,boll,server,base = dual
    with pytest.raises(urllib.error.HTTPError) as err:
        request(base,'/api/boll/mode',{'mode':'LIVE'},server.token,base)
    assert err.value.code == 409
    assert boll.snapshot()['mode'] == 'PAPER'
    assert boll.snapshot()['live_orders_sent'] == 0


def test_new_routes_keep_origin_protection_and_log_is_separate(dual):
    pair,boll,server,base = dual
    boll.log('BOLL_SENTINEL',{'message':'独立日志'})
    with pytest.raises(urllib.error.HTTPError) as err:
        request(base,'/api/boll/running',{'running':True},server.token,'https://example.com')
    assert err.value.code == 403
    assert 'BOLL_SENTINEL' in request(base,'/api/boll/logs.csv').read().decode('utf-8-sig')
    assert 'BOLL_SENTINEL' not in request(base,'/api/logs.csv').read().decode('utf-8-sig')


def test_workstation_close_all_pauses_both(dual):
    pair,boll,server,base = dual
    pair.set_running(True); boll.set_running(True)
    result = json.load(request(base,'/api/workstation/close-all',{},server.token,base))['result']
    assert set(result) == {'pair','boll_short'}
    assert pair.snapshot()['running'] is False
    assert boll.snapshot()['running'] is False
    assert result['boll_short'] == {'closed':[],'failed':[]}


def test_boll_asset_is_served(dual):
    _,_,_,base = dual
    assert request(base,'/boll.js').headers['Content-Type'].startswith('text/javascript')
