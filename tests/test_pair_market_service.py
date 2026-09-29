import time

import pytest

from futures_strategy.pair_market_service import PairMarketService, PublicPairMarket
from futures_strategy.pair_portfolio import PairPortfolio
from test_pair_workstation import model, quotes


def test_public_quotes_use_bid_ask_and_oldest_exchange_timestamp(monkeypatch):
    market = PublicPairMarket()
    def get(base, path, params):
        if path.endswith('bookTicker'):
            return [dict(symbol='BTCUSDT',bidPrice='100',askPrice='101',time=200000)]
        return [dict(symbol='BTCUSDT',markPrice='100.5',time=199000,lastFundingRate='.001',nextFundingTime=300000)]
    monkeypatch.setattr(market.client, '_get', get)
    result = market.quotes(['BTCUSDT','MISSINGUSDT'])
    assert result == {'BTCUSDT':dict(bid=100,ask=101,mark=100.5,timestamp=199,funding_rate=.001,next_funding=300)}


def test_one_invalid_book_does_not_block_other_groups_quotes(monkeypatch):
    market=PublicPairMarket()
    def get(base,path,params):
        if path.endswith('bookTicker'):
            return [dict(symbol='GOOD',bidPrice='100',askPrice='101',time=200000),
                    dict(symbol='HALTED',bidPrice='0',askPrice='0',time=200000)]
        return [dict(symbol=s,markPrice='100.5',time=199000,lastFundingRate='.001',nextFundingTime=300000) for s in ['GOOD','HALTED']]
    monkeypatch.setattr(market.client,'_get',get)
    assert set(market.quotes(['GOOD','HALTED']))=={'GOOD'}


def test_public_universe_excludes_new_listings_and_non_perpetuals(monkeypatch):
    market = PublicPairMarket()
    now = time.time()
    def row(symbol, **overrides):
        return dict(symbol=symbol,baseAsset=symbol,quoteAsset='USDT',status='TRADING',
                    contractType='PERPETUAL',onboardDate=int((now-100*86400)*1000),**overrides)
    new = row('NEW');new['onboardDate']=int((now-3*86400)*1000)
    delivery = row('DELIVERY');delivery['contractType']='CURRENT_QUARTER'
    stable = row('STABLE');stable['baseAsset']='USDC'
    monkeypatch.setattr(market.client,'_get',lambda *args:dict(symbols=[row('AAA'),row('BBB'),new,delivery,stable]))
    monkeypatch.setattr(market.client,'fetch_24h_tickers',lambda _: {s:dict(quote_volume=v) for s,v in [('AAA',100),('BBB',200),('NEW',1000),('DELIVERY',1000),('STABLE',1000)]})
    assert market.universe(now)==['BBB','AAA']


def test_service_bulk_close_pauses_before_network_and_reports_failure(tmp_path, monkeypatch):
    p=PairPortfolio(tmp_path/'paper.sqlite3')
    now=time.time()
    p.set_running(True)
    quotes(p,now=now)
    g=p.open_group(model())
    p.quotes.clear()
    service=PairMarketService(p)
    def failed_refresh(*args):
        assert p.snapshot()['running'] is False
        raise ConnectionError('network unavailable')
    monkeypatch.setattr(service,'refresh_quotes',failed_refresh)
    result=service.close_all()
    assert result['closed']==[]
    assert result['failed'][0]['id']==g['id']
    assert len(p.snapshot()['groups'])==1
