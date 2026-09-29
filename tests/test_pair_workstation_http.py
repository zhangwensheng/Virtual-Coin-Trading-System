import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from futures_strategy.pair_portfolio import PairPortfolio, PairSettings
from futures_strategy.pair_web import create_server, account_lease, default_output_dir
from test_pair_workstation import model, quotes


@pytest.fixture
def app(tmp_path):
    engine = PairPortfolio(tmp_path / 'state.sqlite3', PairSettings())
    engine.set_running(True)
    quotes(engine, now=time.time())
    g = engine.open_group(model())
    server = create_server(engine, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield engine, g, server, f'http://127.0.0.1:{server.server_port}'
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(base, path, payload=None, token=None, origin=None):
    headers = {}
    if token:
        headers['X-Pair-Token'] = token
    if origin:
        headers['Origin'] = origin
    data = None if payload is None else json.dumps(payload).encode()
    if data is not None:
        headers['Content-Type'] = 'application/json'
    return urllib.request.urlopen(urllib.request.Request(base+path, data=data, headers=headers), timeout=5)


def test_state_and_single_group_close_are_wired_to_real_portfolio(app):
    engine, g, server, base = app
    state = json.load(request(base, '/api/state'))
    assert state['groups'][0]['id'] == g['id']
    assert len(state['groups'][0]['legs']) == 2
    assert state['mode'] == 'PAPER'
    result = json.load(request(base, '/api/close-group', {'id': g['id']}, state['token'], base))
    assert result['result']['status'] == 'CLOSED'
    assert engine.snapshot()['groups'] == []
    assert len(engine.snapshot()['closed_groups']) == 1


def test_bulk_close_response_reports_stale_group_instead_of_success(app):
    engine, g, server, base = app
    engine.quotes = {}
    result = json.load(request(base, '/api/close-all', {}, server.token, base))
    assert len(result['result']['failed']) == 1
    assert result['result']['closed'] == []
    assert engine.snapshot()['running'] is False


def test_mutations_reject_missing_token_and_cross_origin(app):
    engine, g, server, base = app
    for token, origin in [(None, base), (server.token, 'https://example.org')]:
        with pytest.raises(urllib.error.HTTPError) as error:
            request(base, '/api/close-all', {}, token, origin)
        assert error.value.code == 403
    assert len(engine.snapshot()['groups']) == 1


def test_unknown_group_returns_conflict_and_csv_export_has_audit(app):
    engine, g, server, base = app
    with pytest.raises(urllib.error.HTTPError) as error:
        request(base, '/api/close-group', {'id':'missing'}, server.token, base)
    assert error.value.code == 409
    data = request(base, '/api/logs.csv').read().decode('utf-8-sig')
    assert 'OPEN_GROUP' in data
    assert g['id'] in data


def test_page_and_assets_served_without_exposing_local_files(app):
    _, _, _, base = app
    assert request(base, '/').headers['Content-Type'].startswith('text/html')
    assert request(base, '/app.js').headers['Content-Type'].startswith('text/javascript')
    with pytest.raises(urllib.error.HTTPError) as error:
        request(base, '/../../requirements.txt')
    assert error.value.code == 404


def test_account_lease_prevents_two_servers_writing_the_same_account(tmp_path):
    path = tmp_path / 'account.lock'
    with account_lease(path):
        with pytest.raises(RuntimeError, match='ACCOUNT_IN_USE'):
            with account_lease(path):
                pytest.fail('second writer was allowed')
    with account_lease(path):
        pass


def test_packaged_output_is_outside_temporary_extraction_directory(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys,'frozen',True,raising=False)
    monkeypatch.setattr(sys,'executable',str(tmp_path/'dist'/'PairDesk.exe'))
    assert default_output_dir() == tmp_path/'outputs'/'pair_workstation'
