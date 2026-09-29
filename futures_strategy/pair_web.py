"""Loopback-only browser workstation for the paper pair portfolio."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import io
import json
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from futures_strategy.pair_portfolio import PairPortfolio


ASSETS = Path(__file__).parent / 'web' / 'pairs'


def default_output_dir() -> Path:
    if getattr(sys, 'frozen', False):
        base = Path(sys.executable).resolve().parent
        if base.name.lower() == 'dist':
            base = base.parent
    else:
        base = Path(__file__).resolve().parents[1]
    return base/'outputs'/'pair_workstation'


@contextmanager
def account_lease(path: Path):
    """OS releases this lock even after a crash; two ports cannot share an account."""
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    if path.stat().st_size == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError('ACCOUNT_IN_USE: 此模拟账户已在另一个工作台运行') from exc
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == 'nt':
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def create_server(portfolio, *, port=8765, service=None, boll=None, boll_service=None,
                  listing=None, listing_service=None):
    action_lock = threading.RLock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, code, data, content_type='application/json; charset=utf-8'):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def _host_ok(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, {'error': 'LOCAL_ONLY'})
            if self.path == '/api/boll/state':
                if boll is None:
                    return self._send(503, {'error':'BOLL_NOT_AVAILABLE'})
                state = boll.snapshot()
                state['token'] = self.server.token
                state['market'] = boll_service.snapshot() if boll_service else dict(connection='OFFLINE',message='离线模式',candidates=[])
                return self._send(200,state)
            if self.path == '/api/listing/state':
                if listing is None:
                    return self._send(503, {'error':'LISTING_NOT_AVAILABLE'})
                state = listing.snapshot()
                state['token'] = self.server.token
                state['market'] = listing_service.snapshot() if listing_service else dict(
                    connection='OFFLINE',message='离线模式',candidates=[],universe=[],errors=[])
                return self._send(200,state)
            if self.path == '/api/overview':
                pair_state = portfolio.snapshot()
                boll_state = boll.snapshot() if boll else None
                listing_state = listing.snapshot() if listing else None
                accounts = [pair_state['account']]+([boll_state['account']] if boll_state else [])+([listing_state['account']] if listing_state else [])
                return self._send(200,dict(mode='PAPER',live_available=False,
                    paper_initial_capital=pair_state['settings']['initial_capital']+
                        (boll_state['settings']['initial_capital'] if boll else 0)+
                        (listing_state['settings']['initial_capital'] if listing else 0),
                    equity=sum(a['equity'] for a in accounts),gross_notional=sum(a['gross_notional'] for a in accounts),
                    stale=any(a['stale'] for a in accounts),
                    strategies={'pair':pair_state['account'],'boll_short':boll_state['account'] if boll else None,
                                'new_listing_breakdown':listing_state['account'] if listing else None},
                    strategy_settings={'pair':pair_state['settings'],'boll_short':boll_state['settings'] if boll else None,
                                       'new_listing_breakdown':listing_state['settings'] if listing else None}))
            if self.path == '/api/state':
                state = portfolio.snapshot()
                state['account_path'] = str(portfolio.path.resolve())
                state['token'] = self.server.token
                state['market'] = service.snapshot() if service else dict(connection='OFFLINE', message='离线模式 · 无公共行情', candidates=[], universe=[], errors=[])
                return self._send(200, state)
            if self.path in {'/api/logs.csv','/api/boll/logs.csv','/api/listing/logs.csv'}:
                source = (listing if self.path == '/api/listing/logs.csv' else
                          boll if self.path == '/api/boll/logs.csv' else portfolio)
                if source is None:
                    return self._send(503, {'error':'STRATEGY_NOT_AVAILABLE'})
                output = io.StringIO(newline='')
                writer = csv.writer(output)
                writer.writerow(['seq', 'timestamp', 'kind', 'payload'])
                writer.writerows(source.export_events())
                return self._send(200, output.getvalue().encode('utf-8-sig'), 'text/csv; charset=utf-8')
            allowed = {'/': ('index.html', 'text/html'), '/app.css': ('app.css', 'text/css'),
                       '/app.js': ('app.js', 'text/javascript'), '/boll.js':('boll.js','text/javascript'),
                       '/listing.js':('listing.js','text/javascript'), '/listing.css':('listing.css','text/css')}
            if self.path not in allowed:
                return self._send(404, {'error': 'NOT_FOUND'})
            name, mime = allowed[self.path]
            return self._send(200, (ASSETS/name).read_bytes(), mime+'; charset=utf-8')

        def do_POST(self):
            # Serialize UI actions, especially resume vs. pause-all across accounts.
            with action_lock:
                self._post()

        def _post(self):
            origin = self.headers.get('Origin')
            if (not self._host_ok() or self.headers.get('X-Pair-Token') != self.server.token
                or origin not in {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}):
                return self._send(403, {'error': 'INVALID_ORIGIN_OR_TOKEN'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length < 4096:
                    return self._send(400, {'error': 'INVALID_BODY'})
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    return self._send(400, {'error': 'INVALID_BODY'})
                if self.path.startswith('/api/listing/'):
                    if listing is None:
                        return self._send(503, {'error':'LISTING_NOT_AVAILABLE'})
                    if self.path == '/api/listing/running':
                        if not isinstance(body.get('running'),bool):
                            raise ValueError('INVALID_RUNNING')
                        listing.set_running(body['running'])
                        result = {'running':body['running']}
                    elif self.path == '/api/listing/close-position':
                        if not isinstance(body.get('id'),str):
                            raise ValueError('INVALID_POSITION_ID')
                        result = listing_service.close_position(body['id']) if listing_service else listing.close_position(body['id'])
                    elif self.path == '/api/listing/close-all':
                        result = listing_service.close_all() if listing_service else listing.close_all()
                    elif self.path == '/api/listing/mode':
                        if body.get('mode') != 'PAPER':
                            raise ValueError('LIVE_NOT_AVAILABLE: 实盘执行和保护止损未接入')
                        result = {'mode':'PAPER'}
                    elif self.path == '/api/listing/settings':
                        result = {'settings': listing.set_margin_per_trade(body.get('margin_per_trade'))}
                    else:
                        return self._send(404, {'error':'NOT_FOUND'})
                elif self.path.startswith('/api/boll/'):
                    if boll is None:
                        return self._send(503, {'error':'BOLL_NOT_AVAILABLE'})
                    if self.path == '/api/boll/running':
                        if not isinstance(body.get('running'),bool):
                            raise ValueError('INVALID_RUNNING')
                        boll.set_running(body['running'])
                        result = {'running':body['running']}
                    elif self.path == '/api/boll/close-position':
                        if not isinstance(body.get('id'),str):
                            raise ValueError('INVALID_POSITION_ID')
                        result = boll_service.close_position(body['id']) if boll_service else boll.close_position(body['id'])
                    elif self.path == '/api/boll/close-all':
                        result = boll_service.close_all() if boll_service else boll.close_all()
                    elif self.path == '/api/boll/mode':
                        if body.get('mode') != 'PAPER':
                            raise ValueError('LIVE_NOT_AVAILABLE: 实盘执行与保护测试未完成，不能启用')
                        result = {'mode':'PAPER'}
                    else:
                        return self._send(404, {'error':'NOT_FOUND'})
                elif self.path == '/api/workstation/close-all':
                    portfolio.set_running(False)
                    if boll:
                        boll.set_running(False)
                    if listing:
                        listing.set_running(False)
                    result = {}
                    for name,engine,runner in [('pair',portfolio,service),('boll_short',boll,boll_service),
                                                ('new_listing_breakdown',listing,listing_service)]:
                        if engine is None:
                            continue
                        try:
                            result[name] = runner.close_all() if runner else engine.close_all()
                        except Exception:
                            result[name] = {'closed':[],'failed':[{'id':'ACCOUNT','reason':'平仓未完成，请刷新持仓后重试'}]}
                elif self.path == '/api/close-group':
                    if not isinstance(body.get('id'), str):
                        return self._send(400, {'error': 'INVALID_GROUP_ID'})
                    result = service.close_group(body['id']) if service else portfolio.close_group(body['id'])
                elif self.path == '/api/close-all':
                    result = service.close_all() if service else portfolio.close_all()
                elif self.path == '/api/running':
                    if not isinstance(body.get('running'), bool):
                        return self._send(400, {'error': 'INVALID_RUNNING'})
                    portfolio.set_running(body['running'])
                    result = {'running': body['running']}
                else:
                    return self._send(404, {'error': 'NOT_FOUND'})
                return self._send(200, {'ok': True, 'result': result})
            except (ValueError, KeyError) as exc:
                return self._send(409, {'error': str(exc)})
            except Exception:
                return self._send(500, {'error': 'INTERNAL_ERROR: 请查看服务日志并刷新状态'})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.token = secrets.token_urlsafe(32)
    return server


def run_dashboard():
    parser = argparse.ArgumentParser(description='三策略模拟盘工作台')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--output-dir', default=str(default_output_dir()))
    parser.add_argument('--boll-output-dir', default=None)
    parser.add_argument('--listing-output-dir', default=None)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--offline', action='store_true', help='不连接公共行情，用于离线查看与测试')
    args = parser.parse_args()
    args.boll_output_dir = args.boll_output_dir or str(
        default_output_dir().parent/'boll_short_workstation' if Path(args.output_dir).resolve()==default_output_dir().resolve()
        else Path(args.output_dir)/'boll_short')
    args.listing_output_dir = args.listing_output_dir or str(
        default_output_dir().parent/'new_listing_workstation' if Path(args.output_dir).resolve()==default_output_dir().resolve()
        else Path(args.output_dir)/'new_listing')
    try:
        with account_lease(Path(args.output_dir)/'.account.lock'):
            with account_lease(Path(args.boll_output_dir)/'.account.lock'):
                with account_lease(Path(args.listing_output_dir)/'.account.lock'):
                    _serve(args)
    except RuntimeError as exc:
        if 'ACCOUNT_IN_USE' not in str(exc):
            raise
        import urllib.request
        url = f'http://127.0.0.1:{args.port}'
        try:
            with urllib.request.urlopen(url+'/api/state', timeout=2) as response:
                existing = json.load(response)
            if Path(existing.get('account_path','')).resolve() != (Path(args.output_dir)/'paper.sqlite3').resolve():
                raise ValueError('DIFFERENT_ACCOUNT')
        except Exception:
            raise exc from None
        if not args.no_browser:
            webbrowser.open(url)
        print(f'Pair workstation already running: {url}', flush=True)


def _serve(args):
    from futures_strategy.boll_short_portfolio import BollPortfolio
    from futures_strategy.new_listing_portfolio import NewListingPortfolio
    portfolio = PairPortfolio(Path(args.output_dir)/'paper.sqlite3')
    boll = BollPortfolio(Path(args.boll_output_dir)/'paper.sqlite3')
    listing = NewListingPortfolio(Path(args.listing_output_dir)/'paper.sqlite3')
    service = None
    boll_service = None
    listing_service = None
    if not args.offline:
        from futures_strategy.pair_market_service import PairMarketService
        service = PairMarketService(portfolio)
        from futures_strategy.boll_short_service import BollMarketService
        boll_service = BollMarketService(boll)
        from futures_strategy.new_listing_service import NewListingService
        listing_service = NewListingService(listing)
    server = create_server(portfolio, port=args.port, service=service,boll=boll,boll_service=boll_service,
                           listing=listing,listing_service=listing_service)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Pair workstation (PAPER): {url}', flush=True)
    if service:
        service.start()
        boll_service.start()
        listing_service.start()
    if not args.no_browser:
        threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if service:
            service.stop()
            boll_service.stop()
            listing_service.stop()
        server.server_close()
