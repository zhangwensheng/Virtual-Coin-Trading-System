"""Public Binance data and background supervision for the paper workstation."""
from __future__ import annotations

import itertools
import math
import threading
import time

from futures_strategy.binance_funding_arbitrage_data import (
    BinanceFundingArbitrageDataClient, PERP_BASE_URL, STABLE_BASE_ASSETS,
)
from futures_strategy.pair_signals import fit_pair, signal_snapshot, spread_z


class PublicPairMarket:
    def __init__(self):
        self.client = BinanceFundingArbitrageDataClient(timeout=10)
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(total=2, backoff_factor=.4, status_forcelist=[502, 503, 504], allowed_methods=['GET'])
        self.client.session.mount('https://', HTTPAdapter(max_retries=retry))

    def quotes(self, symbols: list[str]) -> dict:
        if not symbols:
            return {}
        books = self.client._get(PERP_BASE_URL, '/fapi/v1/ticker/bookTicker', {})
        premiums = self.client._get(PERP_BASE_URL, '/fapi/v1/premiumIndex', {})
        book = {r['symbol']: r for r in books}
        mark = {r['symbol']: r for r in premiums}
        result = {}
        for symbol in symbols:
            if symbol not in book or symbol not in mark:
                continue
            a, b = book[symbol], mark[symbol]
            try:
                quote = dict(bid=float(a['bidPrice']), ask=float(a['askPrice']),
                             mark=float(b['markPrice']), timestamp=min(int(a['time']), int(b['time']))/1000,
                             funding_rate=float(b['lastFundingRate']), next_funding=int(b['nextFundingTime'])/1000)
                if (all(math.isfinite(quote[k]) and quote[k] > 0 for k in ['bid','ask','mark','timestamp'])
                    and quote['bid'] <= quote['ask'] and math.isfinite(quote['funding_rate'])):
                    result[symbol] = quote
            except (KeyError, TypeError, ValueError):
                continue
        return result

    def universe(self, now: float) -> list[str]:
        info = self.client._get(PERP_BASE_URL, '/fapi/v1/exchangeInfo', {})
        valid = {r['symbol'] for r in info['symbols'] if r['status'] == 'TRADING'
                 and r['contractType'] == 'PERPETUAL' and r['quoteAsset'] == 'USDT'
                 and r['baseAsset'] not in STABLE_BASE_ASSETS
                 and now-int(r.get('onboardDate', now*1000))/1000 >= 90*86400}
        tickers = self.client.fetch_24h_tickers('perp')
        return sorted((s for s in valid if s in tickers), key=lambda s: -tickers[s]['quote_volume'])[:20]

    def bars(self, symbol: str, interval: str, now: float):
        duration = 31*86400 if interval == '1h' else 3*3600
        return self.client.fetch_perp_klines(symbol, interval, int((now-duration)*1000), int(now*1000))


class PairMarketService:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.status = dict(connection='WAITING', message='正在连接公共行情', last_quote=None,
                           last_scan=None, scanned_pairs=0, universe=[], candidates=[], errors=[])
        self.threads = []

    def snapshot(self):
        import copy
        with self.lock:
            return copy.deepcopy(self.status)

    def start(self):
        for target in [self._scan_loop, self._watch_loop, self._funding_loop]:
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self.threads.append(thread)

    def stop(self):
        self.stop_event.set()

    def refresh_quotes(self, ids: list[str] | None = None):
        state = self.portfolio.snapshot()
        symbols = {l['symbol'] for g in state['groups'] if ids is None or g['id'] in ids for l in g['legs']}
        if symbols:
            incoming = PublicPairMarket().quotes(sorted(symbols))
            self.portfolio.update_quotes(incoming)

    def close_group(self, gid):
        # Refresh outside the portfolio lock; close_group validates both legs again under lock.
        try:
            self.refresh_quotes([gid])
        except Exception as exc:
            self.portfolio.log('QUOTE_REFRESH_FAILED', {'message': str(exc)[:250]})
        return self.portfolio.close_group(gid)

    def close_all(self):
        self.portfolio.set_running(False)
        try:
            self.refresh_quotes()
        except Exception as exc:
            self.portfolio.log('QUOTE_REFRESH_FAILED', {'message': str(exc)[:250]})
        return self.portfolio.close_all()

    def _watch_loop(self):
        market = PublicPairMarket()
        while not self.stop_event.is_set():
            try:
                state = self.portfolio.snapshot()
                with self.lock:
                    candidates = list(self.status['candidates'])
                symbols = {l['symbol'] for g in state['groups'] for l in g['legs']}
                symbols.update(s for c in candidates for s in [c['symbol_a'], c['symbol_b']])
                if symbols:
                    self.portfolio.update_quotes(market.quotes(sorted(symbols)))
                self.portfolio.monitor()
                for candidate in candidates:
                    if not candidate.get('direction') or time.time()-candidate['signal_time'] > 330:
                        continue
                    if not self.portfolio.snapshot()['running']:
                        break
                    try:
                        with self.portfolio.lock:
                            a = self.portfolio._fresh(candidate['symbol_a'], time.time())
                            b = self.portfolio._fresh(candidate['symbol_b'], time.time())
                            z = spread_z(candidate, a['mark'], b['mark'])
                            if z*candidate['z'] <= 0 or not self.portfolio.settings.entry_z <= abs(z) < self.portfolio.settings.stop_z:
                                continue
                            self.portfolio.open_group(dict(candidate, z=z))
                    except ValueError:
                        continue  # Portfolio limits are expected; candidates are still visible.
                with self.lock:
                    if symbols:
                        self.status.update(connection='CONNECTED', message='公共行情已连接 · 模拟成交', last_quote=time.time())
            except Exception as exc:
                self._error('行情更新失败', exc)
            self.stop_event.wait(10)

    def _funding_loop(self):
        client = BinanceFundingArbitrageDataClient(timeout=10)
        while not self.stop_event.is_set():
            since = self.portfolio.funding_windows()
            for symbol, start in since.items():
                if self.stop_event.is_set():
                    return
                try:
                    now = time.time()
                    rows = client.fetch_funding_history(symbol, int(start*1000), int(now*1000))
                    for ts, row in rows.iterrows():
                        self.portfolio.settle_funding(symbol, ts.timestamp(), float(row.funding_rate), float(row.mark_price))
                    self.portfolio.mark_funding_checked(symbol, now=now)
                except Exception as exc:
                    self._error(f'{symbol} 资金费待补记', exc)
            self.stop_event.wait(60)

    def _error(self, message, exc):
        detail = str(exc)[:240]
        with self.lock:
            self.status['errors'] = (self.status['errors'] + [{'timestamp': time.time(), 'message': message}])[-8:]
            self.status.update(connection='DEGRADED', message=message)
        self.portfolio.log('MARKET_ERROR', {'message': message, 'detail': detail})

    def _scan_loop(self):
        market = PublicPairMarket()
        models = []
        last_fit = 0
        universe = []
        while not self.stop_event.is_set():
            now = time.time()
            try:
                if int(now//3600) != int(last_fit//3600):
                    universe = market.universe(now)
                    frames = {}
                    for symbol in universe:
                        if self.stop_event.is_set():
                            return
                        try:
                            frames[symbol] = market.bars(symbol, '1h', now)
                        except Exception as exc:
                            self._error(f'{symbol} 历史数据不可用', exc)
                    models = []
                    for a, b in itertools.combinations(frames, 2):
                        fitted = fit_pair(a, b, frames[a], frames[b], now)
                        if fitted:
                            models.append(fitted)
                    last_fit = now
                symbols = {s for m in models for s in [m['symbol_a'], m['symbol_b']]}
                bars = {s: market.bars(s, '5m', now) for s in symbols}
                candidates = [c for m in models if (c := signal_snapshot(m, bars[m['symbol_a']], bars[m['symbol_b']], now))]
                candidates.sort(key=lambda c: -abs(c['z']))
                with self.lock:
                    self.status.update(last_scan=time.time(), scanned_pairs=len(universe)*(len(universe)-1)//2,
                                       universe=universe, candidates=candidates, connection='CONNECTED',
                                       message=f'已筛选 {len(universe)} 个币 · {len(candidates)} 组通过统计筛选')
                self.portfolio.log('SCAN', {'universe': universe, 'candidates': candidates})
            except Exception as exc:
                self._error('币对扫描失败', exc)
            self.stop_event.wait(max(10, min(60, 300-time.time()%300)))
