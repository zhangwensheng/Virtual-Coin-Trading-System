import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

def run_binance_backtest():
    # 初始化币安永续合约接口
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    print("正在获取币安所有USDT永续合约列表...")
    markets = exchange.load_markets()
    
    # 筛选出所有 USDT 永续合约标的
    symbols = [symbol for symbol, market in markets.items() if market['linear'] and market['quote'] == 'USDT' and market['active']]
    
    print(f"共找到 {len(symbols)} 个USDT永续合约，开始下载数据并回测...\n")
    
    all_trades_summary = []
    
    # 参数设置
    leverage = 10.0
    tp_pct = 0.5  # 10倍杠杆下，标的跌50%等于本金盈利5倍 (500%)
    
    for symbol in symbols:
        try:
            # 获取日线数据 (抓取尽可能多的历史数据)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=1000)
            if len(ohlcv) < 30: # 过滤上市时间太短、数据不足的币种
                continue
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 策略核心条件计算
            df['cond_high'] = df['high'] <= df['high'].shift(1)
            df['cond_bear'] = df['close'] < df['open']
            df['signal'] = df['cond_high'] & df['cond_bear']
            
            # 回测模拟状态变量
            in_position = False
            entry_price = 0.0
            stop_loss = 0.0
            take_profit = 0.0
            entry_date = None
            
            for i in range(1, len(df)):
                current_date = df.loc[i, 'datetime']
                c_open = df.loc[i, 'open']
                c_high = df.loc[i, 'high']
                c_low = df.loc[i, 'low']
                c_close = df.loc[i, 'close']
                
                if not in_position:
                    # 检查是否触发开空信号
                    if df.loc[i-1, 'signal']: # 使用前一日收盘确认后的信号在当日开盘或盘中触发
                        in_position = True
                        entry_price = c_open # 假设以当日开盘价做空
                        stop_loss = df.loc[i-1, 'high'] # 止损为信号K最高点
                        take_profit = entry_price * (1 - tp_pct)
                        entry_date = current_date
                else:
                    # 持仓中：检查每日的高低点是否触及止损或止盈
                    hit_tp = c_low <= take_profit
                    hit_sl = c_high >= stop_loss
                    
                    if hit_sl and hit_tp:
                        # 如果同一天极端行情既触及止损又触及止盈，保守起见算作止损
                        pnl_pct = -((stop_loss - entry_price) / entry_price) * leverage * 100
                        all_trades_summary.append({'symbol': symbol, 'type': 'Loss (Both)', 'pnl_%': pnl_pct})
                        in_position = False
                    elif hit_sl:
                        # 触发止损：亏损百分比取决于止损距离与杠杆
                        pnl_pct = -((stop_loss - entry_price) / entry_price) * leverage * 100
                        all_trades_summary.append({'symbol': symbol, 'type': 'Loss', 'pnl_%': pnl_pct})
                        in_position = False
                    elif hit_tp:
                        # 触发5倍止盈 (+500%)
                        all_trades_summary.append({'symbol': symbol, 'type': 'Win', 'pnl_%': 500.0})
                        in_position = False
                        
        except Exception as e:
            # 忽略部分暂停交易或API限制的币种
            continue

    # 输出回测统计结果
    if len(all_trades_summary) > 0:
        trades_df = pd.DataFrame(all_trades_summary)
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['type'] == 'Win'])
        win_rate = (wins / total_trades) * 100
        total_pnl = trades_df['pnl_%'].sum()
        
        print("="*40)
        print("📊 币安合约新币做空策略回测报告")
        print("="*40)
        print(f"总交易次数: {total_trades}")
        print(f"胜率 (Win Rate): {win_rate:.2f}%")
        print(f"总体累计收益率 (考虑10倍杠杆): {total_pnl:.2f}%")
        print("="*40)
    else:
        print("未检测到符合条件的交易，请检查过滤条件。")

if __name__ == "__main__":
    run_binance_backtest()