# Crypto Futures Strategy Template

English | [简体中文](README.md)

This is a conservatively styled crypto futures strategy project. The goal is not "guaranteed outsized profits" — it is to improve win rate and return quality as much as possible while **participating only when the trend is clear, keeping drawdowns controlled, and keeping every parameter backtestable**.

The project currently ships with:

- Unified multi-exchange historical kline fetcher
- Multi-exchange WebSocket realtime kline subscriber
- Live monitoring entry point wired to the current strategy logic
- A `1m` order-flow proxy strategy
- Main strategy: market-wide scan for `1m` "top-5 daily gainers + daily Bollinger upper-band break + 1m/5m topping divergence" shorts
- Experimental strategy: realtime-ranked `1m` "pump momentum + distribution reversal"
- Bidirectional strategy: `4h/1h` support/resistance breakout-retest on the top 5 contracts by turnover + `5m` entry
- Small-cap `1m` short-only "post-spike exhaustion" strategy
- Majors `5m` "1h trend filter + 5m pullback continuation" strategy
- Majors `1h` short-only "4h bear trend + 1h rally-into-EMA" strategy

## Strategy Approach

- Platform support:
  - Backtesting / historical data: `Binance`, `OKX`, `Bybit`, `Bitget`, `Gate`, `KuCoin Futures`, `Deribit`, `Hyperliquid`
  - Realtime WebSocket klines: `Binance`, `OKX`, `Bybit`, `Bitget`, `Gate`, `Deribit`, `Hyperliquid`
- Suggested symbols: prefer the most liquid perpetuals such as `BTC/USDT:USDT` and `ETH/USDT:USDT`
- Timeframe design: default execution on `1h` with a `4h` higher-timeframe trend filter
- Entry logic:
  - Strong trend on the 4-hour timeframe
  - Pullback into `EMA20` on the 1-hour timeframe
  - `RSI` resets into a healthy range, then turns strong/weak again
  - The current candle breaks the previous candle's high/low to confirm momentum
- Risk control:
  - Fixed risk of `0.5%` of account equity per trade
  - Initial stop placed with `ATR` + swing highs/lows
  - Take half off at `1.2R`, then trail with an ATR stop to protect gains
  - Automatic cooldown after consecutive losses to reduce chop-market drawdowns

If you want to explore shorter-timeframe microstructure strategies, the project also provides a `1m` order-flow proxy version. It is not tick-by-tick `L2` order flow; instead it uses fields that ship with Binance futures klines:

- `quote_volume`
- `trade_count`
- `taker_buy_base`
- `taker_buy_quote`

to construct aggressive buy/sell imbalance, trade-count expansion, and short-term breakout confirmation.

## TL;DR

No strategy can sustainably guarantee "high win rate, high return, low drawdown" all at once. What this project gives you is a starting point that is closer to controllable live trading:

- Win rate: improved through trend filtering + pullback confirmation
- Returns: through scaled take-profits + letting winners run
- Drawdown: suppressed through fixed-fractional risk, ATR stops, and loss cooldowns

If you crank leverage, position size, and trading frequency, drawdown will go up — guaranteed.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Running Backtests

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

Order-flow `1m` example:

```bash
python run_backtest.py --config configs/binance_orderflow_1m_base.yaml
```

Verified sample:

```bash
python run_backtest.py --config configs/binance_sol_orderflow_short_1m_2026-02.yaml
```

Offline demo:

```bash
python run_backtest.py --config configs/demo.yaml
```

`demo` only verifies that the command and output pipeline work; it does not represent real-market performance.

After a backtest finishes, the output directory contains:

- `summary.json`
- `trades.csv`
- `equity_curve.csv`
- `latest_signal.json`

## Top-Turnvolume Breakout-Retest Strategy

This strategy is for backtesting and paper signals only — it never places real orders. Every hour it picks the top five Binance USDT perpetuals by trailing-24h `quote_volume`, finds support/resistance and valid breaks on `4h/1h`, waits for a retest with second confirmation on `5m`, and supports both long and short sides.

Default limits:

- At most `3` entries per symbol per Beijing-time day
- At most `6` entries per account per Beijing-time day
- At most `2` symbols held at once
- `0.5%` risk per trade, minimum initial margin `1 USDT`

Backtest example:

```bash
python run_liquidity_retest_backtest.py --config configs/binance_liquidity_retest_5m.yaml --start-month 2026-01 --end-month 2026-02 --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT --output-dir outputs/liquidity_retest_backtest
```

Single-round paper-trading signal example:

```bash
python run_liquidity_retest_paper_trader.py --config configs/binance_liquidity_retest_5m.yaml --output-dir outputs/liquidity_retest_paper --once
```

The run directory records `events.jsonl`, `signals.csv`, `trades.csv`, `daily_summary.csv`, `equity_curve.csv`, and `summary.json`. When optimizing, start with the rejection reasons, breakout volume, and retest depth in `signals.csv`, plus the MFE/MAE, R multiples, and exit reasons in `trades.csv`.

### Short-Only Failed-Breakout Mode

`configs/binance_liquidity_retest_short_only.yaml` only accepts original upside-breakout retest signals and mirrors direction, stop, and targets: the original long take-profit becomes the short stop, and the original long stop becomes the short's first take-profit. Once account drawdown from peak equity reaches `20%`, new entries stop but existing positions keep being managed.

```bash
python run_liquidity_retest_backtest.py --start-month 2026-07 --end-month 2026-08 --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT --config configs/binance_liquidity_retest_short_only.yaml --output-dir outputs/liquidity_retest_short_only_backtest
python optimize_liquidity_retest_short.py --start-month 2026-07 --end-month 2026-08 --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT --config configs/binance_liquidity_retest_short_only.yaml --output-dir outputs/liquidity_retest_short_optimization
python run_liquidity_retest_paper_trader.py --config configs/binance_liquidity_retest_short_only.yaml --output-dir outputs/liquidity_retest_short_only_paper --once
```

The optimizer uses the first 20 days for training and the last 10 days for out-of-sample validation, keeping earlier history as indicator warm-up. Results include fees and slippage. If the candidate pool comes from the current turnover ranking, the report flags survivorship bias. Backtest returns are not guaranteed to reproduce, the paper trader submits no real orders, and a 100% return is only a high-risk scenario target.

## Unified Historical Kline Fetcher

Command:

```bash
python fetch_klines.py --exchange bybit --symbol BTC/USDT:USDT --timeframe 1h --limit 1000 --output data/bybit_btc_1h.csv
```

Supported `exchange` values:

- `binance`
- `okx`
- `bybit`
- `bitget`
- `gate`
- `kucoinfutures`
- `deribit`
- `hyperliquid`

If you want to run the `1m` order-flow proxy strategy, prefer the Binance public archive downloader, because the standard `ccxt fetch_ohlcv()` only returns the six `OHLCV` columns without `trade_count / taker_buy_*`. The project ships a monthly archive downloader:

```bash
python download_binance_archive.py --symbol SOLUSDT --timeframe 1m --start-month 2026-02 --end-month 2026-02 --output data/solusdt_orderflow_1m_2026-02.csv
```

Current recommendations for the order-flow strategy:

- Backtest with archive CSVs first
- Prefer liquid contracts such as `SOL / BTC / ETH` where slippage is more controllable
- Do not treat it as true `L2` tick-by-tick order flow

To run the "top-5 gainers distribution short" main strategy, use this entry point:

```bash
python run_top_gainers_boll_backtest.py --selection-mode realtime --start-month 2026-02 --end-month 2026-02 --max-positions 1 --output-dir outputs/top_gainers_boll_short_1m_2026-02_main
```

This strategy first scans all Binance USDT perpetuals, ranks the realtime top `5` by intraday gain each minute, and only goes short when all of these hold:

- Price is above the `Bollinger Upper` band computed from the last completed daily candle
- The `1m` top shows above-average volume turning bearish
- The `5m` confirms with a volume-backed fade near its upper band
- Price has rolled off the day's high and broken a short-term structure low

The main config is `configs/binance_top_gainers_boll_short_1m_base.yaml`; outputs include:

- `daily_selection.csv`
- `symbol_summary.csv`
- `trades.csv`
- `equity_curve.csv`
- `summary.json`

For the more aggressive "pump momentum + distribution reversal" experimental strategy:

```bash
python run_top_gainers_boll_backtest.py --base-config configs/binance_top_gainers_pump_dump_1m_turbo.yaml --selection-mode realtime --start-month 2026-01 --end-month 2026-02 --max-positions 2 --portfolio-notional-fraction 0.95 --output-dir outputs/top_gainers_pump_dump_1m_turbo_2026-01_2026-02
```

This variant may first long the pump, then flip short when it detects distribution. It is higher frequency and more exposed to fees, slippage, and overfitting — review walk-forward results before considering live trading.

## WebSocket Realtime Kline Subscriber

Command:

```bash
python stream_klines.py --exchange bybit --symbol BTC/USDT:USDT --timeframe 1h --closed-only
```

To persist realtime klines as JSONL:

```bash
python stream_klines.py --exchange okx --symbol BTC/USDT:USDT --timeframe 1h --closed-only --output outputs/okx_1h.jsonl
```

Currently supported realtime feeds:

- `binance`
- `okx`
- `bybit`
- `bitget`
- `gate`
- `deribit`
- `hyperliquid`

## Live Strategy Monitor

This entry point first pulls historical klines for warm-up, then consumes realtime closed candles and continuously emits the latest strategy signals:

```bash
python run_live_monitor.py --config configs/bybit_btc.yaml
```

The output directory additionally contains:

- `live_klines.jsonl`
- `live_signals.jsonl`
- `latest_signal.json`

## Binance Account Desktop Dashboard

The project ships a Windows desktop dashboard that shows:

- Total wallet balance
- Available balance
- Unrealized PnL
- Position list
- Open orders
- Trade history
- Income history

It also has a dedicated API settings page for your Binance Futures `API Key / API Secret`.

Source entry point:

```bash
python run_binance_dashboard.py
```

Build the exe:

```bash
python build_binance_dashboard_exe.py
```

After packaging, the executable is at:

```text
dist/BinanceFuturesDashboard.exe
```

The dashboard stores the API key locally in the current Windows user profile using OS-level encryption — never in plaintext inside project configs.

Recommendations:

- Create a **read-only** Binance Futures API key
- Never grant withdrawal permissions
- Viewing your account does not require trading permissions

## Daily Screening & Alert Monitoring

If you already have profit-screened symbol configs, e.g.:

- `ADA`
- `HBAR`
- `SOL`

first run a "who should I watch today" daily scan:

```bash
python scan_factor_daily.py --configs outputs/funding_oi_short_optimizer_profit_2026-03-18_universe10/symbol_configs/adausdt_profit.yaml,outputs/funding_oi_short_optimizer_profit_2026-03-18_universe10/symbol_configs/hbarusdt_profit.yaml,outputs/funding_oi_short_optimizer_profit_2026-03-18_universe10/symbol_configs/solusdt_profit.yaml --days-back 27
```

The output directory contains:

- `scan_rows.csv`
- `scan_rows.json`
- `report.md`

To keep polling and alert only when state flips to `FUNDING_OI_BEAR_SHORT_SETUP`:

```bash
python run_factor_alert_monitor.py --configs outputs/funding_oi_short_optimizer_profit_2026-03-18_universe10/symbol_configs/adausdt_profit.yaml,outputs/funding_oi_short_optimizer_profit_2026-03-18_universe10/symbol_configs/hbarusdt_profit.yaml,outputs/funding_oi_short_optimizer_profit_2026-03-18_universe10/symbol_configs/solusdt_profit.yaml --top-n 2 --interval-seconds 300 --output-dir outputs/factor_alert_monitor_top3
```

The monitor continuously updates:

- `monitor_state.json`
- `scan_rows.csv`
- `scan_rows.json`
- `latest_report.md`
- `alerts.jsonl`
- `latest_alerts.json`

Where:

- `alerts.jsonl` only appends genuinely new alerts
- `latest_alerts.json` keeps the newest round of alerts
- `monitor_state.json` prevents duplicate notifications

## 1m Order-Flow Proxy Strategy

Strategy name:

```text
orderflow_imbalance_1m
```

Core idea:

- Use `taker_buy_quote` and `quote_volume` to compute aggressive buy/sell imbalance
- Use `trade_count` to verify that trade density is really expanding
- Only enter on extreme imbalance coinciding with a short-term breakout
- Exit via `VWAP + EMA + delta reversal` so positions don't drag in `1m` noise

This strategy currently fits:

- High-frequency (but not ultra-HFT) `1m` backtests
- Liquid contracts
- Conservative short holding periods and low slippage assumptions

It currently does not fit:

- Replacing tick-by-tick or book-level order flow
- Running on plain six-column `OHLCV` CSVs
- Inflating returns by ignoring fees/slippage

## If the Exchange API Is Unreachable

On some networks, Binance / OKX public endpoints may time out or get reset. In that case, point the config at a local CSV:

```yaml
exchange:
  name: binance
  symbol: BTC/USDT:USDT
  timeframe: 1h
  csv_path: data/btc_1h.csv
```

The CSV needs these columns:

```text
timestamp,open,high,low,close,volume
```

For `orderflow_imbalance_1m` you also need:

```text
quote_volume,trade_count,taker_buy_base,taker_buy_quote
```

`timestamp` accepts ISO time strings, seconds, or milliseconds since epoch.

## Exchange Symbol Suggestions

- `Binance / OKX / Bybit / Bitget / Gate`: prefer `BTC/USDT:USDT`
- `Deribit`: use `BTC-PERPETUAL`
- `Hyperliquid`: use `BTC/USDC:USDC`

For realtime WebSocket, the project automatically converts common unified symbols to each exchange's native contract names.

## Parameters Worth Tuning First

- `risk_per_trade`: to compress drawdown, start at `0.3%`–`0.5%`
- `leverage`: keep it at or below `3x` initially
- `htf_adx_threshold`: higher = more conservative, fewer trades
- `partial_rr`: higher = bigger per-trade targets, usually lower win rate
- `min_volume_ratio`: higher = stricter liquidity confirmation
- `max_atr_pct`: filters out extreme-volatility regimes

## Live Trading Recommendations

- Start with `BTC` and `ETH` only
- Start with `1h + 4h` only
- Backtest on six months to two years of history first
- Then run rolling parameter validation — never trust a single best run
- Finally: paper trade first, then go live with small capital

## 20x Short-Only Strategy

If you want a more aggressive, explicitly bearish line, the current best edge is not chasing `5m` signals but:

- Confirm the bear trend on `4h`
- Wait for the rally back into `EMA20` on `1h`
- Only short when the rally fails and price re-breaks the prior low
- Use tighter `ATR` stops and scaled take-profits to amplify bear-leg returns

These configs default to `20x`, short-only, fixed high-risk sizing — suitable for research, not as a plug-and-play live template.

Recommended commands:

```bash
python run_backtest.py --config configs/binance_btc_bear_pullback_short_1h_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_eth_bear_pullback_short_1h_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_sol_bear_pullback_short_1h_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_xrp_bear_pullback_short_1h_2026-01_2026-02.yaml
```

Base config:

```text
configs/binance_majors_bear_pullback_short_base.yaml
```

To push returns further, tune these first:

- `risk_per_trade`
- `htf_adx_threshold`
- `pullback_atr_tolerance`
- `rsi_short_max`
- `max_bars_in_trade`

## Funding-Rate + OI + Price Enhanced Short Strategy

This is the newest enhanced line: short-only, built on the rally-retest short, additionally checking:

- Whether `funding rate` is rising (longs are crowded)
- Whether `open interest` is increasing (not just short covering)
- Whether price just bounced and is rolling over again

Build the enhanced dataset first:

```bash
python build_binance_factor_dataset.py --symbols BTCUSDT,ETHUSDT,ADAUSDT,HBARUSDT --interval 1h --days-back 30 --end-time 2026-03-18T00:00:00Z
```

Then run per-symbol backtests:

```bash
python run_backtest.py --config configs/binance_btc_funding_oi_short_1h_2026-03-18.yaml
python run_backtest.py --config configs/binance_eth_funding_oi_short_1h_2026-03-18.yaml
python run_backtest.py --config configs/binance_ada_funding_oi_short_1h_2026-03-18.yaml
python run_backtest.py --config configs/binance_hbar_funding_oi_short_1h_2026-03-18.yaml
```

To auto-roll backtests and screen out the best short months and symbols:

```bash
python optimize_bear_factors.py --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BCHUSDT,ATOMUSDT,AAVEUSDT,HBARUSDT,DOGEUSDT,ADAUSDT --interval 1h --days-back 30 --end-time 2026-03-18T00:00:00Z --window-days 10 --step-days 5 --min-funding-rates=-0.0001,-0.00005 --min-funding-changes=-0.00002,0 --min-oi-change-pcts=0,0.003 --min-price-bounce-pcts=0,0.003
```

Note: Binance's official `openInterestHist` endpoint reliably returns only about `500` bars at `1h` granularity, so this enhanced line suits rolling research on "recent weeks" — not deep backtests of distant months.

## Small-Cap 1m Short Strategy

This strategy targets the exhaustion fade after an intraday spike in small caps — short only, never chasing longs.

Core filters:

- A clear intraday markup already happened
- The markup came with expanding volume
- The top shows upper wicks or exhaustion signs
- Then a high-volume bearish candle breaks recent lows
- No overnight holds — positions close by end of day by default

Recommended config:

```bash
python run_backtest.py --config configs/binance_1000pepe_short_1m.yaml
```

With real archive data downloaded:

```bash
python run_backtest.py --config configs/binance_1000pepe_short_1m_2026-02.yaml
python run_backtest.py --config configs/binance_1000bonk_short_1m_2026-02.yaml
```

Small-cap 1m data download example:

```bash
python download_binance_archive.py --symbol 1000PEPEUSDT --timeframe 1m --start-month 2026-02 --end-month 2026-02 --output data/1000pepeusdt_1m_2026-02.csv
```

## Cross-Symbol Backtesting + Tuning

To backtest and tune at the same time — checking whether this small-cap short logic adapts across most contract symbols — run the batch optimizer:

```bash
python optimize_smallcap_short.py --train-months 2026-01 --valid-months 2026-02
```

By default it will:

- Auto-download Binance public archive 1m futures data
- Sweep a set of small-cap perpetuals with batch backtests
- Grid-search the key parameters
- Rank by cross-symbol robustness rather than single-symbol returns

Default symbol universe:

- `1000PEPEUSDT`
- `1000BONKUSDT`
- `1000FLOKIUSDT`
- `1000SHIBUSDT`
- `WIFUSDT`
- `ENAUSDT`

## Majors 5m GPT Tuning

To let the model review backtest results round by round and propose the next parameter set, run the majors-specific optimizer:

```bash
python optimize_majors_gpt.py --train-months 2026-01 --valid-months 2026-02
```

By default it will:

- Cover `BTC / ETH / SOL / XRP` majors
- Propose a parameter set each round, then backtest across symbols
- Pick results by validation-set robustness, not single-symbol returns
- Save each round's proposals, backtest results, and the final best config

If `OPENAI_API_KEY` is not set, verify the pipeline in local fallback mode:

```bash
python optimize_majors_gpt.py --dry-run --rounds 2 --candidates-per-round 4
```

With `OPENAI_API_KEY` configured, the script defaults to `gpt-5.4`. You can also specify the model manually:

```bash
python optimize_majors_gpt.py --model gpt-5.4
```

The output directory contains:

- `round_01/`, `round_02/` ...: per-round proposals, prompts, backtest results
- `best_params.json`
- `best_config.yaml`
- `selection_table.csv`
- `report.md`

## BTC / ETH Pair Trading

If you prefer hedging within same-type majors, the project includes a `BTC / ETH` spread mean-reversion strategy.

Instead of chasing one side, this logic:

- Watches `BTC` and `ETH` simultaneously
- Only engages when short-term correlation is high enough
- Opens both legs when the spread blows out and starts converging
- `BTC` too strong: `short BTC + long ETH`
- `ETH` too strong: `long BTC + short ETH`
- Closes both when the spread reverts, correlation breaks, or the hold times out

Run command:

```bash
python run_pair_backtest.py --config configs/binance_btc_eth_pair_5m_2026-01_2026-02.yaml
```

The output directory contains:

- `summary.json`
- `trades.csv`
- `equity_curve.csv`
- `prepared_spread.csv`
- `latest_snapshot.json`

## Multi-Pair Rolling Scan

To watch several same-type pairs instead of only `BTC / ETH`, run the multi-pair scanner:

```bash
python scan_pair_universe.py --pairs BTCUSDT:ETHUSDT,SOLUSDT:ETHUSDT,BNBUSDT:ETHUSDT --months 2025-09:2026-02 --train-window-months 2 --valid-window-months 1
```

It automatically:

- Downloads missing Binance monthly archive `5m` data into `data/binance_archive`
- Builds rolling windows, e.g. `2025-09~2025-10 -> 2025-11`
- Runs train/validation backtests per pair
- Produces cross-pair, cross-window robustness reports

The output directory contains:

- `cell_results.csv`
- `aggregate_results.csv`
- `selection_table.csv`
- `best_params.json`
- `best_config.yaml`
- `report.md`

## Multi-Pair GPT Tuning

To let the model propose the next round of parameters from rolling-window results:

```bash
python optimize_pairs_gpt.py --pairs BTCUSDT:ETHUSDT,SOLUSDT:ETHUSDT,BNBUSDT:ETHUSDT --months 2025-09:2026-02 --train-window-months 2 --valid-window-months 1
```

Without `OPENAI_API_KEY`, run the full pipeline in local fallback mode:

```bash
python optimize_pairs_gpt.py --dry-run --rounds 2 --candidates-per-round 4
```

The script calls the `Responses API` with `gpt-5.4` by default and requires structured JSON output. Each round saves:

- `prompt_messages.json`
- `proposals.json`
- `cell_results.csv`
- `aggregate_results.csv`
- `selection_table.csv`
- `round_summary.json`

## Majors Volatility-Squeeze Breakout

For a completely different path — no pairs, no pullback trends — try this majors strategy: wait for volatility to compress, then trade the high-volume breakout.

Core logic:

- `1h` filters the dominant direction
- `5m` watches for a period of volatility compression
- Enter only on a high-volume break of the compression channel
- Exit on the opposite channel break or an `EMA` loss

Run commands:

```bash
python run_backtest.py --config configs/binance_btc_breakout_5m_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_eth_breakout_5m_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_sol_breakout_5m_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_xrp_breakout_5m_2026-01_2026-02.yaml
```

Output goes to `outputs/smallcap_optimizer_*` by default, including:

- `symbol_results.csv`
- `aggregate_results.csv`
- `selection_table.csv`
- `best_params.json`
- `report.md`

To customize symbols or parameter ranges:

```bash
python optimize_smallcap_short.py \
  --symbols 1000PEPEUSDT,1000BONKUSDT,1000FLOKIUSDT,WIFUSDT \
  --train-months 2026-01 \
  --valid-months 2026-02 \
  --day-return-thresholds 0.03,0.05 \
  --volume-spike-thresholds 1.6,2.0 \
  --upper-wick-ratios 0.2,0.3
```

Prefer reading validation-set and cross-symbol results:

- Care less about "the single symbol with the highest return"
- Care more about "whether most symbols are net positive"
- Care more about "whether average and worst-symbol drawdowns are acceptable"
- Only then decide whether to transfer the best params to new symbols

## Majors 5m Lower-Timeframe Strategy

This strategy suits liquid contracts like `BTC / ETH / SOL / XRP`. It avoids random high-frequency spraying and instead does:

- `1h` trend filtering
- `5m` pullbacks into `EMA` and intraday `VWAP`
- A clear recent impulse is required first
- The pullback must not be too deep, then momentum resumes via a prior high/low break
- Exits via `ATR` stops, scaled take-profits, and signal exits

Base config:

```bash
python run_backtest.py --config configs/binance_majors_ltf_base.yaml
```

Prepared real-data backtest configs:

```bash
python run_backtest.py --config configs/binance_btc_majors_5m_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_eth_majors_5m_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_sol_majors_5m_2026-01_2026-02.yaml
python run_backtest.py --config configs/binance_xrp_majors_5m_2026-01_2026-02.yaml
```

Majors `5m` archive download example:

```bash
python download_binance_archive.py --symbol BTCUSDT --timeframe 5m --start-month 2026-01 --end-month 2026-02 --output data/btcusdt_5m_2026-01_2026-02.csv
```

Current params for this majors line are conservative:

- `htf_adx_threshold: 18`
- `majors_impulse_threshold: 0.01`
- `majors_fast_ema: 21`
- `majors_slow_ema: 55`
- `partial_rr: 1.2`

If you continue this line, suggested next tests:

- Move `BTC / ETH` to `15m` execution
- Keep `XRP / SOL` on `5m`
- Split backtests into "long-only" vs "both directions"

## Two-Year Universal Short Screening

To avoid reading too much into the last few weeks, use Binance public archive `1h` futures klines to run a two-year rolling screen across majors and semi-majors.

This pipeline does four things:

- Downloads Binance USD-M monthly kline archives from `2024-03` to `2026-02`
- Backtests multiple price-based short strategies over the full span
- Slices results by calendar month: profit or not, drawdown, trading activity
- Keeps only "strategy + symbol" combos that survive stably across months

Run directly:

```bash
python screen_universal_short_archive.py --start-month 2024-03 --end-month 2026-02
```

Default baseline strategies:

- `configs/binance_universal_short_trend_defensive_1h.yaml`
- `configs/binance_universal_short_trend_aggressive_1h.yaml`
- `configs/binance_universal_short_breakout_1h.yaml`
- `configs/binance_universal_short_bear_rally_1h.yaml`

Default symbol pool:

- `BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, BNBUSDT, DOGEUSDT, ADAUSDT, LINKUSDT, AVAXUSDT, DOTUSDT, FILUSDT, BCHUSDT, LTCUSDT, TRXUSDT, UNIUSDT, AAVEUSDT, NEARUSDT, FETUSDT, APTUSDT, SUIUSDT, WLDUSDT, 1000PEPEUSDT`

Default output directory:

- `outputs/universal_short_screen_2024-03__2026-02`

Key result files:

- `strategy_summary.csv`
- `strategy_symbol_summary.csv`
- `monthly_results.csv`
- `survivor_summary.csv`
- `best_survivor_per_symbol.csv`
- `survivor_configs/*.yaml`
- `report.md`

To switch auto-trading to the `11`-symbol whitelist keeping one surviving strategy per symbol:

```bash
python build_auto_whitelist_from_survivors.py --screen-dir outputs/universal_short_screen_2024-03__2026-02
```

Default output directory:

- `outputs/universal_short_screen_2024-03__2026-02/auto_whitelist_11_configs`

## Pair Paper-Trading Workstation (Current Default Page)

Start:

```bash
python run_binance_dashboard.py
```

Open `http://127.0.0.1:8765` in a browser. Default principal is 1000 USDT with new entries paused; click "Enable new entries" to start auto paper entries. It only connects to Binance public market data — no API keys are read, no real orders are sent, and it does not manage legacy strategies or real exchange accounts. Closing the browser does not stop the local service; closing the launcher process stops all monitoring, and the next start restores positions from the database with new entries paused.

The page is redesigned around each held group, showing both legs' symbols, long/short direction, size, entry price, current mark price, per-leg unrealized PnL, combined leg PnL, net PnL including estimated closing costs, funding, hedge ratio, Z-score, and holding time. Candidate pairs, closed trades, equity curve, and run logs are directly viewable.

- **Close this group**: simulates closing both legs at once; duplicate requests do not double-settle. If quotes are stale the position is kept and an error is shown.
- **Close all**: pauses new entries first, then exits group by group, reporting successes and failures separately; failed groups can be retried once market data recovers. Never touches positions in other strategies or real exchange accounts.
- **Pause new entries**: stops new positions; take-profit, stop-loss, and max-hold monitoring of existing positions continue.
- **Logs**: the full SQLite transaction audit lives in `outputs/pair_workstation/paper.sqlite3`; the log page can export all events as CSV. Entry params, both legs' sizes and prices, exit reasons, funding, and account equity are all recorded for later optimization.

The strategy screens from USDT perpetuals in the top 20 by turnover with at least 90 days since listing; the past 30 days of complete `1h` and `4h` data must pass both cointegration and correlation filters, and converged-spread onset is confirmed on `5m` closed candles. Default entry requires |Z| ≥ 2.2 and < 4; the entry locks the model and both legs' sizes. Exits at ±0.4 reversion, ±4 blowout, 3 USDT planned loss per group, 48-hour hold, or account risk controls. Entry spread must also cover estimated round-trip costs. At most 5 groups, no shared symbols, at most 3 groups opened per symbol per Shanghai calendar day, 4-hour cooldown after closing. Two-leg combined notional capped at 800 USDT, 320 USDT per group (never above the user's 1000 USDT cap), sized down by stop distance, with at least 200 USDT kept as an entry buffer. Daily loss of 10 USDT halts new entries; 5% account drawdown triggers exits and shutdown. Price gaps can exceed the planned stop.

Fees are 0.05% per leg per side plus 2 bps extra slippage; simulated fills use real bid/ask quotes. Funding is only booked from public actual settlement records; late-published settlements are back-filled to already-closed groups, and the page shows booked funding. The funding thread polls every minute; public-data or network outages may delay back-filling. Simulated fills exclude the real matching queue, partial fills, minimum order precision, and isolated-margin liquidation cascades — not directly transferable to live trading.

Optional flags: `--port 8765`, `--output-dir outputs/pair_workstation`, `--no-browser`, `--offline`. The same output directory is process-locked — the same paper account cannot run from multiple ports at once. The legacy desktop app remains available as `python run_binance_dashboard.py --legacy`; its real-account features are independent of the new page.

For Windows packaging run `python build_binance_dashboard_exe.py`, which currently outputs `dist/DualStrategyDesk.exe`; the original `PairDesk.exe` and `BinanceFuturesDashboard.exe` are not overwritten. Source updates do not auto-update existing older EXEs.

Verification command:

```bash
python -m pytest tests/test_pair_workstation.py tests/test_pair_workstation_http.py tests/test_pair_market_service.py -q
```

This delivery is the pair paper engine and management UI. **No historical profitability backtest or out-of-sample validation has been completed for the new multi-group strategy yet.** The original `run_pair_backtest.py` is the old single-group research model and must not be treated as the backtest for this multi-group cointegration strategy. UI acceptance uses `tests/serve_pair_preview.py`, which spins synthetic positions in an isolated throwaway database and never touches the formal paper account.

## New: BOLL Spike-Short & Dual-Strategy Overview

The same workstation's left panel adds "BOLL spike-short" and a "Dual-strategy overview". The original pair ledger is unchanged; the new strategy runs on its own 1000 USDT paper principal — the two strategies' initial principals total 2000 USDT with no fund transfers between them. The new strategy starts with new entries paused; click "Enable BOLL new entries" to start consuming fresh signals. Pausing does not stop exit supervision of existing positions.

This is an independent **PAPER** strategy, not the old `1m/5m` short scripts. Rules:

- Among USDT perpetuals listed at least 90 days with 24h turnover of at least 20M USDT, watch the top 5 by positive intraday gain.
- Today's UTC-day open price must be above the BOLL upper band (20, 2, population standard deviation) computed from the previous 20 completed daily closes. Intraday price rises do not change this gate.
- Wait for a closed 1m bullish candle with volume at least 2x the 20-bar average closing outside the minute-level upper band; within the next 5 bars, look for a bearish candle that spikes above the band and closes back inside; then wait within 3 bars for a close below that candle's low. New highs, crossing the UTC day, timeout, or dropping off the candidate list invalidate the old observation.
- Attempts to open the short only within 15 seconds of the confirmation candle's close; quotes must be at most 5 seconds stale; previously-missed signals are not back-filled on startup.
- Fixed 1x paper notional: max 320 USDT per position, max 2 positions, 640 USDT combined, at least 200 USDT buffer; planned risk per trade capped at `min(3 USDT, equity × 0.3%)`, with size reduced for fees and stop slippage.
- At most 3 orders per symbol per Beijing-time day, 30-minute cooldown after closing; 10 USDT daily net loss pauses new entries. The stop sits at the observation peak plus buffer, target is 2R, a 1R trailing stop activates at +1R, max hold 30 minutes. Price gaps and network outages can make real losses exceed planned risk.

The BOLL page shows candidates, non-entry reasons, position size/price/stop/target/planned risk/net PnL, closed trades, and exportable logs. Supports closing a single position or all positions of this strategy; the overview's global buttons pause both paper strategies before closing each out. Failed closes are kept and shown with their failure reason — success is never faked.

New log database: `outputs/boll_short_workstation/paper.sqlite3`. The original pair database remains at `outputs/pair_workstation/paper.sqlite3`. BOLL logs cover scans and rejection reasons, signals, fills, fees, actual public funding back-fill, exits, and equity — exportable from the page or via `/api/boll/logs.csv`. Observations are logged even when no position is opened.

```bash
python run_pair_workstation.py --no-browser
python -m pytest tests -q
```

Use `--boll-output-dir` for an independent new-account directory; in tests, `--offline --output-dir outputs/isolated_test` places the BOLL account under that test directory to keep the formal paper ledger untouched.

**Current boundary: live trading is not wired in — both the UI text and the backend reject LIVE.** Exchange maintenance-margin tiers, isolated-margin liquidation, and real matching/partial fills are not fully simulated; liquidation price display is unavailable and 2x mode is disabled. Shutdowns or outages cannot guarantee timely stops; after a restart the ledger is restored but new entries stay paused. Across a midnight outage the daily-loss baseline uses the most recent available equity, not an exact midnight snapshot. The last 30 days of cross-symbol backtesting, out-of-sample validation, and independent code review are not yet complete for this strategy — passing tests does not imply profitability.

See `docs/boll-workstation-acceptance.md` for detailed validation and deployment status.

## Binance Funding-Rate Arbitrage (Legacy Standalone Research)

This is a standalone simulated research system that only does "buy spot + short an equal-value USDT perpetual" under positive funding rates — no coin borrowing, no cross-exchange legs, and no real order capability. Initial paper principal is 1000 USDT, paired notional capped at 800 USDT, at most 5 groups at once, 50–320 USDT per group; the perpetual leg is always carried on 4x margin but this never amplifies pair notional or returns.

Funding-rate arbitrage is not risk-free. Real outcomes are affected by funding flipping negative, basis widening, desynchronized leg fills, slippage, fees, liquidity, margin, and exchange operational risks. Historical backtests and paper signals do not guarantee future profits.

Download 95 days of public data:

```bash
python download_binance_funding_arbitrage_data.py --days 95 --interval 5m --output-dir data/binance_funding_arbitrage
```

Run the ~90-day backtest; `summary.json` provides both the full window and the last 30 days:

```bash
python run_funding_arbitrage_backtest.py --config configs/binance_funding_arbitrage.yaml --data-dir data/binance_funding_arbitrage --days 90 --output-dir outputs/binance_funding_arbitrage_3m_2026-08-27
```

One round of paper-trading signals:

```bash
python run_funding_arbitrage_paper_trader.py --config configs/binance_funding_arbitrage.yaml --output-dir outputs/binance_funding_arbitrage_paper_2026-08-27 --once
```

Drop `--once` for continuous paper trading. Key audit files include `candidates.csv`, `orders.csv`, `funding_settlements.csv`, `rebalances.csv`, `trades.csv`, `equity.csv`, `events.jsonl`, `state.json`, and `summary.json`. Later optimization should be driven by these logs — grouped by symbol, funding band, basis, holding time, cost source, and exit reason — never by raising leverage or ignoring trading costs to manufacture results.
