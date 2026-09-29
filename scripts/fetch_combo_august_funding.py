"""Download public funding archives for the previously untested Aug 27-31 slice."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import io
import zipfile
import pandas as pd
import requests


def fetch(symbol):
    dest=Path('data/adaptive_combo_research')/f'{symbol}_funding_2026-08.csv'
    if dest.exists():
        return str(dest)
    session=requests.Session()
    session.trust_env=False
    url=f'https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-2026-08.zip'
    response=session.get(url,timeout=30)
    response.raise_for_status()
    archive=zipfile.ZipFile(io.BytesIO(response.content))
    raw=pd.read_csv(archive.open(archive.namelist()[0]))
    frame=pd.DataFrame({'funding_time':pd.to_datetime(raw.calc_time,unit='ms',utc=True),
                        'funding_rate':raw.last_funding_rate,'source_url':url})
    frame.to_csv(dest,index=False)
    print(symbol,len(frame),flush=True)
    return str(dest)


if __name__=='__main__':
    with ThreadPoolExecutor(5) as pool:
        list(pool.map(fetch,['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT']))
