# 虚拟币合约策略模板

这是一个偏稳健风格的合约策略项目，目标不是“保证暴利”，而是在 **趋势明确时参与、回撤可控、参数可回测** 的前提下尽量提高胜率和收益质量。
现在项目已经内置：

- 多交易所统一历史 K 线抓取器
- 多交易所 WebSocket 实时 K 线订阅器
- 适配当前策略逻辑的实时监控入口
- `1m` 订单流代理策略
- 全市场扫描的 `1m` “日涨幅前五 + 日线破 Boll 上轨 + 1m/5m 顶部分歧做空”主策略
- 全市场实时排名的 `1m` “泵拉动量 + 出货反转”实验策略
- 全市场成交额前五的 `4h/1h` 压力支撑突破回踩 + `5m` 入场双向策略
- 小币种 `1m` 只做空的“放量拉升后衰竭下跌”策略
- 主流币 `5m` 小周期的“1h 趋势过滤 + 5m 回踩续涨/续跌”策略
- 主流币 `1h` 只做空的“4h 空头趋势 + 1h 反弹回抽做空”策略

## 策略思路

- 平台支持：
  - 回测 / 历史数据：`Binance`、`OKX`、`Bybit`、`Bitget`、`Gate`、`KuCoin Futures`、`Deribit`、`Hyperliquid`
  - 实时 WebSocket K 线：`Binance`、`OKX`、`Bybit`、`Bitget`、`Gate`、`Deribit`、`Hyperliquid`
- 标的建议：优先 `BTC/USDT:USDT`、`ETH/USDT:USDT` 这类流动性最好的永续合约
- 周期设计：默认 `1h` 执行，`4h` 做大级别趋势过滤
- 入场逻辑：
  - 4 小时级别处于强趋势
  - 1 小时级别回踩 `EMA20`
  - `RSI` 回落到健康区间后重新转强/转弱
  - 当前 K 线突破上一根高点/低点确认动能
- 风控逻辑：
  - 单笔风险固定为账户权益的 `0.5%`
  - 使用 `ATR` + 摆动高低点设初始止损
  - 到达 `1.2R` 先止盈一半，再用 ATR 移动止损保护利润
  - 连续亏损后自动冷却，减少震荡市回撤

如果你想做更短周期的微观结构策略，项目现在也提供 `1m` 的订单流代理版。它不是逐笔 `L2` 盘口，而是利用 Binance 合约 K 线自带的：

- `quote_volume`
- `trade_count`
- `taker_buy_base`
- `taker_buy_quote`

去构造主动买卖不平衡、成交笔数放大和短线突破确认。

## 先说结论

“高胜率、高回报、低回撤” 三者不能被任何策略长期同时保证。这个项目给你的，是一个更接近实盘可控的起点：

- 胜率：靠趋势过滤 + 回踩确认去提高
- 收益：靠分批止盈 + 让盈利单跑出来
- 回撤：靠固定风险仓位、ATR 止损、亏损冷却去压制

如果你把杠杆、仓位、交易频率拉得很高，回撤一定会上去。

## 安装

```bash
python -m pip install -r requirements.txt
```

## 运行回测

Binance:

```bash
python run_backtest.py --config configs/binance_btc.yaml
```

OKX:

```bash
python run_backtest.py --config configs/okx_btc.yaml
```

Bybit:

```bash
python run_backtest.py --config configs/bybit_btc.yaml
```

Bitget:

```bash
python run_backtest.py --config configs/bitget_btc.yaml
```

Gate:

```bash
python run_backtest.py --config configs/gate_btc.yaml
```

Deribit:

```bash
python run_backtest.py --config configs/deribit_btc.yaml
```

Hyperliquid:

```bash
python run_backtest.py --config configs/hyperliquid_btc.yaml
```

订单流 `1m` 示例：

```bash
python run_backtest.py --config configs/binance_orderflow_1m_base.yaml
```

已验证样例：

```bash
python run_backtest.py --config configs/binance_sol_orderflow_short_1m_2026-02.yaml
```

离线演示:

```bash
python run_backtest.py --config configs/demo.yaml
```

`demo` 只用于验证命令和输出链路是否正常，不代表真实市场收益表现。

回测完成后会在对应输出目录生成：

- `summary.json`
- `trades.csv`
- `equity_curve.csv`
- `latest_signal.json`

## 成交额前五突破回踩策略

这套策略只用于回测和模拟盘信号，不会真实下单。它每小时按 Binance U 本位 USDT 永续的过去 24 小时 `quote_volume` 选前五，`4h/1h` 找支撑压力和有效突破，`5m` 等回踩后二次确认入场，支持做多和做空。

默认限制：

- 同一币种北京时间每天最多开 `3` 单
- 全账户北京时间每天最多开 `6` 单
- 最多同时持仓 `2` 个币种
- 单笔风险 `0.5%`，最低初始保证金 `1 USDT`

回测示例：

```bash
python run_liquidity_retest_backtest.py --config configs/binance_liquidity_retest_5m.yaml --start-month 2026-01 --end-month 2026-02 --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT --output-dir outputs/liquidity_retest_backtest
```

模拟盘单轮信号示例：

```bash
python run_liquidity_retest_paper_trader.py --config configs/binance_liquidity_retest_5m.yaml --output-dir outputs/liquidity_retest_paper --once
```

运行目录会记录 `events.jsonl`、`signals.csv`、`trades.csv`、`daily_summary.csv`、`equity_curve.csv` 和 `summary.json`。后面优化时优先看 `signals.csv` 里的拒绝原因、突破量能、回踩深度，以及 `trades.csv` 里的 MFE/MAE、R 倍数和退出原因。

### 只做空假突破模式

`configs/binance_liquidity_retest_short_only.yaml` 只接收原始上涨突破回踩信号，并同步镜像方向、止损和止盈：原多单止盈变为空单止损，原多单止损变为空单第一止盈。账户从峰值权益回撤达到 `20%` 后停止新开仓，但继续管理已有仓位。

```bash
python run_liquidity_retest_backtest.py --start-month 2026-07 --end-month 2026-08 --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT --config configs/binance_liquidity_retest_short_only.yaml --output-dir outputs/liquidity_retest_short_only_backtest
python optimize_liquidity_retest_short.py --start-month 2026-07 --end-month 2026-08 --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT --config configs/binance_liquidity_retest_short_only.yaml --output-dir outputs/liquidity_retest_short_optimization
python run_liquidity_retest_paper_trader.py --config configs/binance_liquidity_retest_short_only.yaml --output-dir outputs/liquidity_retest_short_only_paper --once
```

优化器使用前20天训练、后10天样本外验证，并保留历史数据作为指标预热。结果包含手续费和滑点。若候选币池来自当前成交额排名，报告会标注幸存者偏差。回测收益不保证复现，模拟盘不会提交真实订单，100%收益只能作为高风险情景目标。

## 统一历史 K 线抓取器

命令：

```bash
python fetch_klines.py --exchange bybit --symbol BTC/USDT:USDT --timeframe 1h --limit 1000 --output data/bybit_btc_1h.csv
```

支持的 `exchange`：

- `binance`
- `okx`
- `bybit`
- `bitget`
- `gate`
- `kucoinfutures`
- `deribit`
- `hyperliquid`

如果你要跑 `1m` 订单流代理策略，建议优先用 Binance 公开归档下载，因为标准 `ccxt fetch_ohlcv()` 只有 `OHLCV` 六列，不带 `trade_count / taker_buy_*`。项目里已经带了月度归档下载器：

```bash
python download_binance_archive.py --symbol SOLUSDT --timeframe 1m --start-month 2026-02 --end-month 2026-02 --output data/solusdt_orderflow_1m_2026-02.csv
```

当前这版订单流策略的建议是：

- 回测先用归档 CSV
- 优先 `SOL / BTC / ETH` 这类流动性高、滑点较可控的合约
- 不要直接把它当成真实 `L2` 逐笔订单流

如果你要跑“全市场涨幅前五出货做空”的主策略，直接用这个入口：

```bash
python run_top_gainers_boll_backtest.py --selection-mode realtime --start-month 2026-02 --end-month 2026-02 --max-positions 1 --output-dir outputs/top_gainers_boll_short_1m_2026-02_main
```

这套策略会先扫描 Binance 全部 USDT 永续合约，按每一分钟的当前日内涨幅找出实时前 `5`，然后只在同时满足这些条件时做空：

- 当前价格高于上一根完整日线计算出来的 `Bollinger Upper`
- `1m` 顶部放量转阴
- `5m` 也出现上轨附近的放量转弱确认
- 价格从当日高点回落并跌破短线结构低点

主配置在 `configs/binance_top_gainers_boll_short_1m_base.yaml`，输出里会包含：

- `daily_selection.csv`
- `symbol_summary.csv`
- `trades.csv`
- `equity_curve.csv`
- `summary.json`

如果要跑更激进的“泵拉动量 + 出货反转”实验策略，用：

```bash
python run_top_gainers_boll_backtest.py --base-config configs/binance_top_gainers_pump_dump_1m_turbo.yaml --selection-mode realtime --start-month 2026-01 --end-month 2026-02 --max-positions 2 --portfolio-notional-fraction 0.95 --output-dir outputs/top_gainers_pump_dump_1m_turbo_2026-01_2026-02
```

这版允许先做多爆拉段，再识别出货段反手做空。它更高频，也更容易被手续费、滑点和过拟合伤到，所以应该先看走步验证结果，再考虑接实盘。

## WebSocket 实时 K 线订阅器

命令：

```bash
python stream_klines.py --exchange bybit --symbol BTC/USDT:USDT --timeframe 1h --closed-only
```

如果想把实时 K 线落盘成 JSONL：

```bash
python stream_klines.py --exchange okx --symbol BTC/USDT:USDT --timeframe 1h --closed-only --output outputs/okx_1h.jsonl
```

当前内置的实时 K 线支持：

- `binance`
- `okx`
- `bybit`
- `bitget`
- `gate`
- `deribit`
- `hyperliquid`

## 实时策略监控

这个入口会先拉一段历史 K 线做预热，然后接收实时已收盘 K 线，并持续输出最新策略信号：

```bash
python run_live_monitor.py --config configs/bybit_btc.yaml
```

输出目录会额外生成：

- `live_klines.jsonl`
- `live_signals.jsonl`
- `latest_signal.json`

## Binance 账户桌面面板

项目现在带了一个 Windows 桌面面板，可以直接看：

- 钱包总资金
- 可用余额
- 未实现盈亏
- 持仓列表
- 挂单列表
- 成交历史
- 收益流水

并且有一个专门的 API 设置页，可以填写 Binance Futures `API Key / API Secret`。

源码入口：

```bash
python run_binance_dashboard.py
```

打包 exe：

```bash
python build_binance_dashboard_exe.py
```

打包完成后可执行文件默认在：

```text
dist/BinanceFuturesDashboard.exe
```

桌面面板会把 API Key 本地保存到当前 Windows 用户目录，并用系统加密能力处理，不会明文写在项目配置里。

建议：

- 只创建“只读”的 Binance Futures API Key
- 不要给提币权限
- 如果只看账户，不需要开交易权限

## 每日筛币与告警监控

如果你已经有利润导向筛出来的币种配置，比如：

- `ADA`
- `HBAR`
- `SOL`

可以先跑一轮“当前该盯谁”的日筛：

```bash
python scan_factor_daily.py --configs outputs/funding_oi_short_optimizer_profit_2026-03-18_universe10/symbol_configs/adausdt_profit.yaml,outputs/funding_oi_short_optimizer_profit_2026-03-18_universe10/symbol_configs/hbarusdt_profit.yaml,outputs/funding_oi_short_optimizer_profit_2026-03-18_universe10/symbol_configs/solusdt_profit.yaml --days-back 27
```

输出目录会生成：

- `scan_rows.csv`
- `scan_rows.json`
- `report.md`

如果你想持续轮询并只在状态切到 `FUNDING_OI_BEAR_SHORT_SETUP` 时报警：

```bash
python run_factor_alert_monitor.py --configs outputs/funding_oi_short_optimizer_profit_2026-03-18_universe10/symbol_configs/adausdt_profit.yaml,outputs/funding_oi_short_optimizer_profit_2026-03-18_universe10/symbol_configs/hbarusdt_profit.yaml,outputs/funding_oi_short_optimizer_profit_2026-03-18_universe10/symbol_configs/solusdt_profit.yaml --top-n 2 --interval-seconds 300 --output-dir outputs/factor_alert_monitor_top3
```

这个监控器会持续更新：

- `monitor_state.json`
- `scan_rows.csv`
- `scan_rows.json`
- `latest_report.md`
- `alerts.jsonl`
- `latest_alerts.json`

其中：

- `alerts.jsonl` 只追加真正的新告警
- `latest_alerts.json` 保存最近一轮新增告警
- `monitor_state.json` 用来避免重复提醒

## 1m 订单流代理策略

策略名：

```text
orderflow_imbalance_1m
```

核心思路：

- 用 `taker_buy_quote` 和 `quote_volume` 计算主动买卖不平衡
- 用 `trade_count` 识别成交密度是否真的放大
- 只在极端不平衡同时发生短线突破时开仓
- 用 `VWAP + EMA + delta 反转` 做退出，不让单子在 `1m` 噪音里拖太久

这套策略当前更适合：

- 高频但不是超高频的 `1m` 回测
- 流动性较好的合约
- 保守的短持仓和较低滑点假设

它当前还不适合：

- 直接替代逐笔成交或盘口级订单流
- 用普通 `OHLCV` 六列 CSV 去跑
- 不加手续费/滑点就盲目放大收益

## 如果交易所 API 连不上

有些网络环境下，Binance / OKX 公共接口可能会超时或被重置。此时直接在配置里填写本地 CSV：

```yaml
exchange:
  name: binance
  symbol: BTC/USDT:USDT
  timeframe: 1h
  csv_path: data/btc_1h.csv
```

CSV 需要包含这些字段：

```text
timestamp,open,high,low,close,volume
```

如果你跑的是 `orderflow_imbalance_1m`，还需要这些额外字段：

```text
quote_volume,trade_count,taker_buy_base,taker_buy_quote
```

`timestamp` 支持 ISO 时间字符串、秒时间戳或毫秒时间戳。

## 交易所符号建议

- `Binance / OKX / Bybit / Bitget / Gate`: 优先用 `BTC/USDT:USDT`
- `Deribit`: 用 `BTC-PERPETUAL`
- `Hyperliquid`: 用 `BTC/USDC:USDC`

对于实时 WebSocket，项目会自动把常见统一符号转换成交易所原生合约名。

## 最值得先调的参数

- `risk_per_trade`: 想压回撤，先从 `0.3%` 到 `0.5%` 调
- `leverage`: 建议先不要超过 `3x`
- `htf_adx_threshold`: 越高越保守，交易次数越少
- `partial_rr`: 越大越追求单笔收益，胜率通常会下降
- `min_volume_ratio`: 越高越强调流动性确认
- `max_atr_pct`: 可过滤极端波动行情

## 实盘建议

- 先只测 `BTC` 和 `ETH`
- 先只做 `1h + 4h`
- 先用半年到两年的历史数据做回测
- 再做滚动参数验证，不要只看单次最优结果
- 最后一定先模拟盘，再小资金实盘

## 20x 只做空策略

如果你要的是更激进、明确偏空头的版本，当前更有边际的一条线不是 `5m` 追信号，而是：

- `4h` 先确认空头趋势
- `1h` 等反弹回抽 `EMA20`
- 只在回抽失败、重新跌破前低时开空
- 用较短的 `ATR` 止损和分批止盈去放大空头行情里的收益

这套配置默认就是 `20x`、只做空、固定高风险仓位，适合做研究，不适合直接当成无脑实盘模板。

推荐命令：

```bash
python run_backtest.py --config configs/binance_btc_bear_pullback_short_1h_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_eth_bear_pullback_short_1h_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_sol_bear_pullback_short_1h_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_xrp_bear_pullback_short_1h_2026-01_2026-02.yaml
```

基础配置：

```text
configs/binance_majors_bear_pullback_short_base.yaml
```

如果你要继续往上加收益，优先调这几个参数：

- `risk_per_trade`
- `htf_adx_threshold`
- `pullback_atr_tolerance`
- `rsi_short_max`
- `max_bars_in_trade`

## 资金费率 + OI + 价格增强版空头策略

这套是最近新增的增强版，只做空，核心是在价格回抽做空的基础上，再看：

- `funding rate` 是否在抬升，说明多头拥挤
- `open interest` 是否在增加，说明不是单纯空头回补
- 价格是否刚经历一段反弹，再次转弱

先构建增强数据集：

```bash
python build_binance_factor_dataset.py --symbols BTCUSDT,ETHUSDT,ADAUSDT,HBARUSDT --interval 1h --days-back 30 --end-time 2026-03-18T00:00:00Z
```

再跑单币回测：

```bash
python run_backtest.py --config configs/binance_btc_funding_oi_short_1h_2026-03-18.yaml
python run_backtest.py --config configs/binance_eth_funding_oi_short_1h_2026-03-18.yaml
python run_backtest.py --config configs/binance_ada_funding_oi_short_1h_2026-03-18.yaml
python run_backtest.py --config configs/binance_hbar_funding_oi_short_1h_2026-03-18.yaml
```

如果你想自动滚动回测并筛出更适合做空的月份和币种：

```bash
python optimize_bear_factors.py --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BCHUSDT,ATOMUSDT,AAVEUSDT,HBARUSDT,DOGEUSDT,ADAUSDT --interval 1h --days-back 30 --end-time 2026-03-18T00:00:00Z --window-days 10 --step-days 5 --min-funding-rates=-0.0001,-0.00005 --min-funding-changes=-0.00002,0 --min-oi-change-pcts=0,0.003 --min-price-bounce-pcts=0,0.003
```

当前 Binance 官方 `openInterestHist` 在 `1h` 粒度下实测最多只有约 `500` 根可稳定拿到，所以这套增强版更适合做“最近几周”的滚动研究，不适合拿来直接覆盖很久以前的月份。

## 小币种 1m 做空策略

这套策略专门针对小币种当日急拉后的衰竭回落，只做空，不追多。

核心过滤：

- 当天已经出现明显拉升
- 拉升阶段伴随放量
- 顶部出现上影线或衰竭迹象
- 随后出现放量阴线、跌破近几根低点
- 进场后不隔夜，默认跨天平仓

推荐配置：

```bash
python run_backtest.py --config configs/binance_1000pepe_short_1m.yaml
```

如果你已经下载了真实归档数据，也可以直接跑：

```bash
python run_backtest.py --config configs/binance_1000pepe_short_1m_2026-02.yaml
python run_backtest.py --config configs/binance_1000bonk_short_1m_2026-02.yaml
```

下载小币种 1 分钟数据示例：

```bash
python download_binance_archive.py --symbol 1000PEPEUSDT --timeframe 1m --start-month 2026-02 --end-month 2026-02 --output data/1000pepeusdt_1m_2026-02.csv
```

## 跨币种回测 + 调参

如果你想边回测边调参，看看这套小币种空头逻辑能不能适应大部分合约币种，可以直接跑批量优化器：

```bash
python optimize_smallcap_short.py --train-months 2026-01 --valid-months 2026-02
```

默认会：

- 自动下载 Binance 公开归档的 1 分钟合约数据
- 覆盖一组小币种合约做批量回测
- 对关键参数做网格搜索
- 按跨币种稳健性而不是单币种收益排序

默认覆盖币种：

- `1000PEPEUSDT`
- `1000BONKUSDT`
- `1000FLOKIUSDT`
- `1000SHIBUSDT`
- `WIFUSDT`
- `ENAUSDT`

## 主流币 5m GPT 调参

如果你想让模型一轮轮看回测结果，再继续提出下一组参数，可以直接跑主流币专用优化器：

```bash
python optimize_majors_gpt.py --train-months 2026-01 --valid-months 2026-02
```

默认会：

- 覆盖 `BTC / ETH / SOL / XRP` 四个主流币合约
- 每轮先提出一组参数，再做跨币种回测
- 用验证集稳健性而不是单币种收益挑选结果
- 输出每一轮的提案、回测结果和最终最优配置

如果当前环境还没有 `OPENAI_API_KEY`，可以先跑本地回退模式验证链路：

```bash
python optimize_majors_gpt.py --dry-run --rounds 2 --candidates-per-round 4
```

如果你已经配置了 `OPENAI_API_KEY`，脚本会默认调用 `gpt-5.4`。你也可以手动指定模型：

```bash
python optimize_majors_gpt.py --model gpt-5.4
```

输出目录会生成：

- `round_01/`, `round_02/` ...：每轮提案、prompt、回测结果
- `best_params.json`
- `best_config.yaml`
- `selection_table.csv`
- `report.md`

## BTC / ETH 配对套利

如果你更想做“同类型主流币对冲”，项目里现在已经补了一套 `BTC / ETH` 的价差均值回归策略。

这套逻辑不是单边追涨杀跌，而是：

- 同时观察 `BTC` 和 `ETH`
- 只在两者短期相关性足够高时出手
- 当价差张口过大并开始收敛时，同时开双向仓位
- `BTC` 过强时：`空 BTC + 多 ETH`
- `ETH` 过强时：`多 BTC + 空 ETH`
- 价差回归、相关性破坏，或持仓超时后一起平掉

运行命令：

```bash
python run_pair_backtest.py --config configs/binance_btc_eth_pair_5m_2026-01_2026-02.yaml
```

输出目录会生成：

- `summary.json`
- `trades.csv`
- `equity_curve.csv`
- `prepared_spread.csv`
- `latest_snapshot.json`

## 多配对滚动扫描

如果你想同时看多个同类配对，而不是只盯 `BTC / ETH`，可以直接跑多配对扫描器：

```bash
python scan_pair_universe.py --pairs BTCUSDT:ETHUSDT,SOLUSDT:ETHUSDT,BNBUSDT:ETHUSDT --months 2025-09:2026-02 --train-window-months 2 --valid-window-months 1
```

它会自动：

- 下载缺失的 Binance 月度归档 `5m` 数据到 `data/binance_archive`
- 构造滚动窗口，比如 `2025-09~2025-10 -> 2025-11`
- 对每个配对分别跑训练集和验证集回测
- 输出跨配对、跨窗口的稳健性报告

输出目录会生成：

- `cell_results.csv`
- `aggregate_results.csv`
- `selection_table.csv`
- `best_params.json`
- `best_config.yaml`
- `report.md`

## 多配对 GPT 调参

如果你想让模型基于滚动窗口结果继续提下一轮参数，可以直接跑：

```bash
python optimize_pairs_gpt.py --pairs BTCUSDT:ETHUSDT,SOLUSDT:ETHUSDT,BNBUSDT:ETHUSDT --months 2025-09:2026-02 --train-window-months 2 --valid-window-months 1
```

当前如果环境里没有 `OPENAI_API_KEY`，也可以先用本地回退模式走完整条链路：

```bash
python optimize_pairs_gpt.py --dry-run --rounds 2 --candidates-per-round 4
```

脚本默认按 `gpt-5.4` 调用 `Responses API`，并要求结构化 JSON 输出。每一轮都会保存：

- `prompt_messages.json`
- `proposals.json`
- `cell_results.csv`
- `aggregate_results.csv`
- `selection_table.csv`
- `round_summary.json`

## 主流币波动收缩突破

如果你想完全换一条路，不做配对，也不做回踩趋势，可以直接试这套“先等波动收缩，再做放量突破”的主流币策略。

核心逻辑：

- 用 `1h` 过滤大方向
- `5m` 观察一段时间的波动压缩
- 只有在压缩后放量突破通道时才入场
- 用通道反向突破和 `EMA` 失守做退出

运行命令：

```bash
python run_backtest.py --config configs/binance_btc_breakout_5m_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_eth_breakout_5m_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_sol_breakout_5m_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_xrp_breakout_5m_2026-01_2026-02.yaml
```

默认会输出到 `outputs/smallcap_optimizer_*`，其中包括：

- `symbol_results.csv`
- `aggregate_results.csv`
- `selection_table.csv`
- `best_params.json`
- `report.md`

如果你想自定义币种或参数范围：

```bash
python optimize_smallcap_short.py \
  --symbols 1000PEPEUSDT,1000BONKUSDT,1000FLOKIUSDT,WIFUSDT \
  --train-months 2026-01 \
  --valid-months 2026-02 \
  --day-return-thresholds 0.03,0.05 \
  --volume-spike-thresholds 1.6,2.0 \
  --upper-wick-ratios 0.2,0.3
```

更建议看验证集和跨币种结果：

- 少看“某一个币收益最高”
- 多看“多数币是否为正收益”
- 多看“平均回撤和最差币种回撤是否可接受”
- 最后再决定是否把最优参数迁移到新币种

## 主流币 5m 小周期策略

这套策略更适合 `BTC / ETH / SOL / XRP` 这类高流动性合约，不追求高频乱打，而是做：

- `1h` 级别趋势过滤
- `5m` 级别回踩 `EMA` 和日内 `VWAP`
- 最近一段必须先有明显脉冲
- 回踩不宜过深，随后用前高/前低突破恢复动能
- 出场交给 `ATR` 止损、分批止盈和信号离场

基础配置：

```bash
python run_backtest.py --config configs/binance_majors_ltf_base.yaml
```

已经准备好的真实数据回测配置：

```bash
python run_backtest.py --config configs/binance_btc_majors_5m_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_eth_majors_5m_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_sol_majors_5m_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_xrp_majors_5m_2026-01_2026-02.yaml
```

下载主流币 `5m` 归档数据示例：

```bash
python download_binance_archive.py --symbol BTCUSDT --timeframe 5m --start-month 2026-01 --end-month 2026-02 --output data/btcusdt_5m_2026-01_2026-02.csv
```

这套主流币策略的当前参数更偏保守：

- `htf_adx_threshold: 18`
- `majors_impulse_threshold: 0.01`
- `majors_fast_ema: 21`
- `majors_slow_ema: 55`
- `partial_rr: 1.2`

如果你继续往这条线做，我建议优先测试：

- `BTC / ETH` 改成 `15m` 执行
- `XRP / SOL` 继续保留 `5m`
- 分别做“只做多”和“双向都做”的拆分回测

## 近两年通用空头筛选

如果你想避免只看最近几周的局部结果，可以直接用 Binance 公开归档 `1h` 合约 K 线，对主流币和次主流币做两年滚动筛选。

当前这条链路会做四件事：

- 下载 `2024-03` 到 `2026-02` 的 Binance USD-M 月度 K 线归档
- 对多套价格型空头策略做整段回测
- 再按自然月切片，统计每个月是否赚钱、回撤如何、是否持续有交易
- 最后只保留跨月份稳定存活的“策略 + 币种”组合

直接运行：

```bash
python screen_universal_short_archive.py --start-month 2024-03 --end-month 2026-02
```

默认会用这些基线策略：

- `configs/binance_universal_short_trend_defensive_1h.yaml`
- `configs/binance_universal_short_trend_aggressive_1h.yaml`
- `configs/binance_universal_short_breakout_1h.yaml`
- `configs/binance_universal_short_bear_rally_1h.yaml`

默认币池是：

- `BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, BNBUSDT, DOGEUSDT, ADAUSDT, LINKUSDT, AVAXUSDT, DOTUSDT, FILUSDT, BCHUSDT, LTCUSDT, TRXUSDT, UNIUSDT, AAVEUSDT, NEARUSDT, FETUSDT, APTUSDT, SUIUSDT, WLDUSDT, 1000PEPEUSDT`

输出目录默认在：

- `outputs/universal_short_screen_2024-03__2026-02`

重点结果文件：

- `strategy_summary.csv`
- `strategy_symbol_summary.csv`
- `monthly_results.csv`
- `survivor_summary.csv`
- `best_survivor_per_symbol.csv`
- `survivor_configs/*.yaml`
- `report.md`

如果要把自动交易直接切到“每币只保留一个幸存策略”的 `11` 币白名单，可以运行：

```bash
python build_auto_whitelist_from_survivors.py --screen-dir outputs/universal_short_screen_2024-03__2026-02
```

默认会生成目录：

- `outputs/universal_short_screen_2024-03__2026-02/auto_whitelist_11_configs`

## 双币配对模拟盘工作台（当前默认页面）

启动：

```bash
python run_binance_dashboard.py
```

浏览器打开 `http://127.0.0.1:8765`。默认本金 1000U、新开仓暂停；点击“启动新开仓”才开始自动模拟开仓。仅接入币安公共行情，不读取 API 密钥，不发送真实订单，不管理旧策略或真实交易所账户持仓。关闭浏览器不会停止本地服务；关闭启动进程会停止所有监控，下次启动恢复数据库中的持仓并暂停新开仓。

页面围绕每组持仓重新设计，展示两个不同币种的多/空方向、数量、开仓价、当前标记价、分别浮盈、双腿合计盈亏、含预计平仓成本的净盈亏、资金费、对冲比例、Z 值与持仓时间。候选币对、平仓记录、净值轨迹和运行日志可直接查看。

- **平仓此组**：同时模拟平掉这组的两条腿；重复请求不重复结算。报价过期时保留持仓并给出错误。
- **一键平仓所有**：先暂停新开仓，再逐组退出，分别显示成功与失败组；失败组可在行情恢复后重试。不会平掉其他策略或交易所账户中的持仓。
- **暂停新开仓**：停止新增仓位，已有仓位的止盈、止损和最长持仓监控继续。
- **日志**：完整 SQLite 事务审计存于 `outputs/pair_workstation/paper.sqlite3`；日志页可导出全部事件 CSV。入场参数、双腿数量与价格、退出原因、资金费和账户权益均记录在内，供后期优化。

策略从 USDT 永续成交额前 20、上市至少 90 天的币中筛选；过去 30 天完整 1h 和 4h 数据须同时通过协整和相关性筛选，用 5m 已完成 K 线确认价差开始收敛。默认入场 |Z|≥2.2 且小于 4；入场锁定模型和两腿数量；回归至 ±0.4、扩大至 ±4、单组计划亏损 3U、持仓 48 小时或账户风控时退出。入场空间还必须覆盖预估往返成本。最多 5 组、不共享币种，每币上海自然日最多开 3 组，平仓冷却 4 小时。两腿合计名义上限 800U、单组上限 320U（绝不超过用户 1000U 上限），按止损距离缩小仓位，并留出至少 200U 开仓资金缓冲。日亏损 10U 停新单、账户回撤 5% 触发退出与停机；跳价可能超过计划止损。

每腿每次手续费 0.05%、额外滑点 2bps，模拟成交使用真实买卖报价。资金费只按公开的实际结算记录记账，延迟发布后会补记到已平仓组；页面显示的是已记账资金费。资金费线程每分钟查询，公开行情/网络故障可能延长补记时间。模拟成交不包含真实撮合队列、部分成交、最小订单精度及逐仓强平过程，不能直接用于实盘。

可选参数：`--port 8765`、`--output-dir outputs/pair_workstation`、`--no-browser`、`--offline`。同一输出目录有进程锁，不能从多个端口同时运行同一模拟账户。旧桌面程序保留为 `python run_binance_dashboard.py --legacy`，其真实账户功能与新页面独立。

Windows 打包运行 `python build_binance_dashboard_exe.py`，当前输出 `dist/DualStrategyDesk.exe`；原 `PairDesk.exe` 和 `BinanceFuturesDashboard.exe` 不覆盖。源码更新不会自动更新已存在的旧 EXE。

验证命令：

```bash
python -m pytest tests/test_pair_workstation.py tests/test_pair_workstation_http.py tests/test_pair_market_service.py -q
```

这次交付是配对模拟引擎与管理页面，**尚未对新组合策略完成历史盈利回测或样本外验证**。原 `run_pair_backtest.py` 是旧单组研究模型，不能当作本次多组协整策略的回测。界面验收使用 `tests/serve_pair_preview.py` 启动独立临时数据库中的合成持仓，不会注入正式模拟账户。

## 新增：BOLL 冲高做空与双策略总览

同一工作台左侧新增“BOLL 冲高做空”和“双策略总览”。原配对账本不变，新策略使用独立的 1000U 模拟本金，两策略初始本金合计 2000U，不互相划转资金。新策略启动时默认暂停新开仓；点击“启动 BOLL 新开仓”后才消费新信号。暂停不停止已有持仓的退出监督。

这次实现的是独立 **PAPER** 策略，不是旧版 `1m/5m` 做空脚本。规则如下：

- 在上市至少 90 天、24 小时成交额至少 2000 万 U 的 USDT 永续中，观察正涨幅前 5 名。
- 今天 UTC 日开盘价必须高于前 20 根已完成日 K 收盘价计算的 BOLL 上轨（20, 2，总体标准差）。当日价格上涨不改变此门槛。
- 等待已收盘 1 分钟阳线放量（相对前 20 根均量至少 2 倍）并收在分钟上轨外；随后 5 根内出现上轨外冲高、收回轨内的阴线，再等 3 根内收盘跌破该阴线低点。破新高、跨 UTC 日、超时或退出候选榜会使旧观察失效。
- 只在确认 K 收盘后 15 秒内尝试开空；报价最多 5 秒旧；启动不补开此前已发生的信号。
- 固定 1 倍模拟，单仓名义最多 320U、最多 2 仓、合计最多 640U，留至少 200U 缓冲；单笔计划风险不超过 `min(3U, 权益×0.3%)`，按手续费和止损滑点缩小数量。
- 每币北京时间每天最多 3 单、平仓后冷却 30 分钟；日净损失达到 10U 暂停新单。止损在观察峰值加缓冲，目标 2R，达到 1R 后使用 1R 跟踪止损，最长持仓 30 分钟。跳价和断网会使实际损失超过计划风险。

BOLL 页面显示候选币、未入场原因、持仓数量/价格/止损/止盈/计划风险/净盈亏、平仓记录和可导出的日志。支持单仓平仓、本策略全部平仓；总览的全局按钮会先暂停两个模拟策略再分别平仓。失败仓位保留并显示失败原因，不会伪造成功。

新日志数据库：`outputs/boll_short_workstation/paper.sqlite3`。原配对数据库仍在 `outputs/pair_workstation/paper.sqlite3`。BOLL 日志包含扫描与拒绝原因、信号、成交、费用、实际公开资金费补记、退出与权益，可从页面或 `/api/boll/logs.csv` 导出。没有持仓也会记录观察过程。

```bash
python run_pair_workstation.py --no-browser
python -m pytest tests -q
```

可以用 `--boll-output-dir` 指定独立的新账户目录；测试时 `--offline --output-dir outputs/isolated_test` 会将 BOLL 账户放到该测试目录下，避免碰正式模拟账本。

**当前边界：实盘切换未接入，界面说明与后端均拒绝 LIVE。** 未实现交易所维持保证金档位、逐仓强平、真实撮合/部分成交等完整仿真，强平价显示不可用、2 倍禁用。停机或断网无法保证及时止损，重启后恢复账本但暂停新增仓位；跨午夜断网时日损基准使用最近可用权益，不是精确午夜快照。尚未完成本策略近 30 天跨币池收益回测、样本外验证或独立代码审查，测试通过不代表盈利。

详细验证和部署状态见 `docs/boll-workstation-acceptance.md`。

## Binance 资金费率套利（旧独立研究方案）

这是一套独立的模拟研究系统，只做正资金费率下的“买入现货 + 做空等值 USDT 永续”，不借币、不跨交易所，也没有真实下单能力。初始模拟本金为 1000 USDT，配对名义总额最多 800 USDT，同时最多 5 组，单组 50–320 USDT；永续腿固定使用 4 倍保证金承载，但不会放大配对名义或收益。

资金费率套利并非无风险。实际结果会受资金费转负、基差扩大、两腿成交不同步、滑点、手续费、流动性、保证金和交易所运行风险影响。历史回测和模拟盘信号不保证未来盈利。

下载 95 天公共数据：

```bash
python download_binance_funding_arbitrage_data.py --days 95 --interval 5m --output-dir data/binance_funding_arbitrage
```

运行近 90 天回测；`summary.json` 会同时提供完整区间和最近 30 天结果：

```bash
python run_funding_arbitrage_backtest.py --config configs/binance_funding_arbitrage.yaml --data-dir data/binance_funding_arbitrage --days 90 --output-dir outputs/binance_funding_arbitrage_3m_2026-08-27
```

执行一轮模拟盘信号：

```bash
python run_funding_arbitrage_paper_trader.py --config configs/binance_funding_arbitrage.yaml --output-dir outputs/binance_funding_arbitrage_paper_2026-08-27 --once
```

持续模拟盘可去掉 `--once`。重点审计文件包括 `candidates.csv`、`orders.csv`、`funding_settlements.csv`、`rebalances.csv`、`trades.csv`、`equity.csv`、`events.jsonl`、`state.json` 和 `summary.json`。后续优化应基于这些日志按币种、费率区间、基差、持仓时间、成本来源和退出原因分组分析，不能靠提高杠杆或忽略交易成本制造结果。
