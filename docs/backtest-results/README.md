# 公开回测结果汇总

本目录是从本地 `outputs/**/summary.json` 提取的公开摘要，只保留收益、回撤、交易次数、胜率、盈亏因子等核心指标。

原始 `outputs/` 目录未纳入 Git，因为其中可能包含本地绝对路径、运行日志、下载审计、模拟盘状态或不适合公开的大体量明细。

- 生成日期：2026-09-29
- 汇总数量：146 个对象型 `summary.json`
- 跳过数量：2 个非标准或无法解析的 `summary.json`
- 完整表格：`summary.csv`

## 收益率排序前 20

| result_id | return_pct | max_drawdown_pct | total_trades | win_rate_pct | profit_factor |
| --- | ---: | ---: | ---: | ---: | ---: |
| top_gainers_pump_dump_1m_turbo_whitelist_2026-01_2026-02 | 47.82 | -5.75 | 216 | 45.83 | 1.73 |
| factor_short_portfolio_top3_p1_r5 | 37.42 | -7.36 | 16 | 68.75 | 4.57 |
| binance_sol_bear_pullback_short_1h_2026-01_2026-02 | 36.31 | -6.78 | 25 | 72 | 3.09 |
| top_gainers_pump_dump_1m_turbo_delta_cap_2026-01_2026-02_risk08 | 36.16 | -3.54 | 161 | 50.31 | 1.94 |
| factor_short_portfolio_top4_p2_r4 | 30.56 | -7.52 | 22 | 63.64 | 3.73 |
| factor_short_portfolio_top3_p1_r4 | 29.27 | -5.92 | 16 | 68.75 | 4.69 |
| binance_eth_bear_pullback_short_1h_2026-01_2026-02 | 28.17 | -11.9 | 23 | 69.57 | 2.35 |
| binance_xrp_bear_pullback_short_1h_2026-01_2026-02 | 27.77 | -5.59 | 20 | 80 | 3.79 |
| top_gainers_pump_dump_1m_turbo_strict_quality_2026-01_2026-02_risk08 | 23.99 | -3.96 | 131 | 51.15 | 1.77 |
| factor_short_portfolio_top4_p2 | 22.16 | -5.27 | 22 | 63.64 | 4.06 |
| factor_short_portfolio_top4_p2_nf12 | 22.16 | -5.27 | 22 | 63.64 | 4.06 |
| factor_short_portfolio_top4_p3 | 21.77 | -5.27 | 22 | 63.64 | 4.02 |
| factor_short_portfolio_top3_p1 | 21.46 | -4.46 | 16 | 68.75 | 4.82 |
| factor_short_portfolio_top3_p1_nf12 | 21.46 | -4.46 | 16 | 68.75 | 4.82 |
| factor_short_portfolio_top3_p2 | 21.22 | -4.46 | 17 | 64.71 | 4.63 |
| top_gainers_pump_dump_1m_sniper_rank2_vwap10_risk08_2026-01_2026-02 | 20.98 | -2.83 | 67 | 56.72 | 2.54 |
| factor_short_portfolio_top3_p3 | 20.84 | -4.46 | 17 | 64.71 | 4.58 |
| factor_short_portfolio_top4_p1 | 20.44 | -5.27 | 19 | 63.16 | 3.96 |
| top_gainers_pump_dump_1m_turbo_delta_cap_2026-01_2026-02_risk08_daily1loss | 18.97 | -3.5 | 132 | 48.48 | 1.6 |
| top_gainers_pump_dump_1m_turbo_delta_cap_2026-01_2026-02_fastcheck | 16.89 | -1.78 | 161 | 50.31 | 1.95 |

## 使用说明

- 这些结果来自历史回测或研究输出，不构成投资建议。
- `return_pct`、`max_drawdown_pct` 等指标直接取自各运行目录的 `summary.json`。
- 未公开原始逐笔交易、权益曲线和行情数据；如需复现，请按 README 中的回测命令重新生成。
- 发布到公开仓库前，请先执行仓库根目录的 `OPEN_SOURCE_CHECKLIST.md` 中的检查命令。
