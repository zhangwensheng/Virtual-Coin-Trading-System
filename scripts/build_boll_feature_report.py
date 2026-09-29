"""Package reviewed feature diagnostics as a native analytical report."""
import json
import sqlite3
import pandas as pd
from analyze_boll_trade_features import OUT

original = pd.read_csv(OUT/'rule_stability.csv')
db = sqlite3.connect(':memory:')
pd.read_csv(OUT/'trade_features.csv').to_sql('trade_features',db,index=False)
sql = """WITH flags AS (
SELECT *, 'late_utc_14_24' AS rule, hour_utc >= 14 AS flag FROM trade_features
UNION ALL SELECT *, 'weak_pump_body_lt_0_65pct', pump_body_pct < 0.65 FROM trade_features
UNION ALL SELECT *, 'large_breakdown_body_gt_0_7pct', breakdown_body_pct > 0.7 FROM trade_features
UNION ALL SELECT *, 'pump_volume_le_5x', volume_ratio <= 5 FROM trade_features
), periods AS (
SELECT *, month AS period FROM flags UNION ALL SELECT *, 'ALL' AS period FROM flags
)
SELECT rule, period, CASE WHEN flag THEN 'feature' ELSE 'complement' END AS [group],
COUNT(DISTINCT day) AS days, COUNT(DISTINCT symbol) AS symbols, COUNT(*) AS n,
SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) AS wins,
100.0 * AVG(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) AS win_pct,
SUM(net_pnl) AS net, AVG(net_pnl) AS avg,
SUM(CASE WHEN net_pnl > 0 THEN net_pnl ELSE 0 END) /
NULLIF(-SUM(CASE WHEN net_pnl < 0 THEN net_pnl ELSE 0 END),0) AS pf
FROM periods GROUP BY rule, period, flag ORDER BY rule, period, flag"""
r = pd.read_sql_query(sql,db)
check = r.merge(original,on=['rule','period','group'],suffixes=('_sql','_pandas'))
assert len(check) == len(r)
assert (check.n_sql == check.n_pandas).all()
assert (abs(check.net_sql-check.net_pandas)<1e-9).all()
title = 'BOLL 冲高做空：亏损订单特征'
source = {'id':'diagnostics','label':'BOLL 回测订单与入场前 K 线特征分析',
          'path':'outputs/boll_short_feature_diagnostics_20260911/rule_stability.csv',
          'query':{'language':'sql','engine':'SQLite','sql':sql,'description':'trade_features.csv 导入内存 SQLite，按入场前特征重新汇总；与原 pandas 分组结果逐组核对一致。',
                   'tables_used':['trade_features'],
                   'filters':['2026年1月至3月；333笔已接受回测订单；特征仅使用入场前已收盘K线'],
                   'metric_definitions':{'平均净盈亏':'组内 net_pnl 总和 / 订单数，单位 USDT，含原回测手续费与滑点',
                                         '胜率':'net_pnl > 0 的订单数 / 组内订单数'}}}
blocks=[]
def md(id,body, sourced=True):
    block={'id':id,'type':'markdown','body':body}
    if sourced: block['sourceId']='diagnostics'
    blocks.append(block)

md('title','# '+title,False)
md('summary','## 核心结论\n\n最值得跟踪的是 **入场时段、冲高力度不足、下跌确认过大**。它们是入场前可见的特征，后三者中时段差异的证据最强。\n\n333 单中 184 单亏损，净亏 70.93U。北京时间 22:00—次日 08:00 的 75 单净亏 72.85U；其余 258 单仅净赚 1.92U。避开亏损组并不等于已获得稳定盈利策略。\n\n冲高阳线实体涨幅不足 0.65%、确认阴线实体跌幅超过 0.7%，各自在三个​​月份都比对照组表现差，但样本不确定性仍较大。建议作为待验证特征记录，暂不把探索阈值直接当成实盘规则。')
md('definition','## 如何读这些结果\n\n样本为 2026 年 1—3 月的 333 笔原始回测订单，涉及 72 个币种、71 个有交易的 UTC 日期。每组均和“其余订单”比较；平均净盈亏已沿用原回测的手续费和滑点。不同特征组可能重叠，亏损金额不能相加。\n\n全部 333 单都成功匹配了冲高、反转、跌破三个阶段，特征只使用入场前已收盘的 K 线。1 月用于初步分桶，2—3 月检查方向；三个​​月份此前已被查看，因此这些检查不是完全未见数据的独立测试。')
md('timing','## 深夜入场的差异跨月、跨币仍存在\n\n北京时间 22:00—次日 08:00（UTC 14:00—24:00）平均每单亏 0.971U，其他时段平均每单赚 0.007U。图中比较每个月的单均净盈亏，避免把交易数量差异误认为质量差异。三个​​月份晚时段都更差；其他时段在 2 月、3 月同样亏损。\n\n去掉全样本亏损最大的五个币后，晚时段仍有 55 单、净亏 50.56U。两种时段均有订单的 33 个币中，26 个晚时段更差。按交易日整组重采样，均值差的探索性 95% 区间为 −1.56 至 −0.40U/单；此区间未修正多重筛选与阈值选择。\n\n这说明差异不只来自最差的几个币，但时段仍可能替代了日内行情状态或排行榜构成。需要用真实滚动 24 小时排行榜的新日志验证。')
timing=r[(r.rule=='late_utc_14_24')&(r.period!='ALL')].copy()
timing['时段']=timing['group'].map({'feature':'北京时间22—次日08点','complement':'北京时间08—22点'})
blocks.append({'id':'timing-plot','type':'chart','chartId':'timing'})
md('price','## 冲高不足与追空过深，是两个不同的风险候选\n\n**冲高力度不足：**冲高阳线从开盘到收盘上涨不足 0.65% 的 102 单，胜率 39.2%，净亏 50.28U，平均亏 0.493U；其余 231 单平均亏 0.089U。可能的解释是普通上涨也被识别为冲高，尚不足以形成衰竭。\n\n**下跌确认过大：**跌破确认阴线从开盘到收盘下跌超过 0.7% 的 92 单，净亏 46.11U，平均亏 0.501U；其余 241 单平均亏 0.103U。可能的解释是等到大阴线收盘再做空，入场已偏晚，容易遭遇反弹。\n\n两种特征跨月都比对照组差。只看北京时间 08—22 点，弱冲高组仍为 −0.177U/单，对照 +0.084U；大阴线组为 −0.431U/单，对照 +0.174U。这降低了纯时段混杂的可能，但不能证明因果。两项按日重采样的差值区间都跨过零，仍属于候选。')
names={'weak_pump_body_lt_0_65pct':'冲高实体涨幅不足0.65%', 'large_breakdown_body_gt_0_7pct':'确认实体跌幅超过0.7%', 'pump_volume_le_5x':'冲高量比不超过5倍'}
comparison=r[(r.period=='ALL')&r.rule.isin(names)].copy()
comparison['特征']=comparison.rule.map(names)
comparison['分组']=comparison['group'].map({'feature':'符合该特征','complement':'其余订单'})
blocks.append({'id':'price-plot','type':'chart','chartId':'price'})
md('weak','## 放量强度有线索，上影线和买入占比还不稳定\n\n冲高量比定义为冲高 K 线成交量 ÷ 之前 20 根 K 线平均量。超过 5 倍的 110 单胜率 50.9%、累计 +2.85U；不超过 5 倍的 223 单胜率 41.7%、累计 −73.78U。强放量可能更接近短时情绪释放，但 1 月的强放量组仍亏 8.78U；不能据此断言 5 倍就是有效阈值。\n\n反转阴线几乎没有上影线的组，在 3 月反而优于对照；主动买入占比阈值从约 55.5% 改为 55% 后差异明显变弱。同币对照也不支持稳定的买入占比方向。暂不采用这两项为硬过滤。\n\n冲高后恰好第 5 分钟入场的 62 单亏 50.14U，但第 6 分钟的 18 单赚 20.28U，缺乏连续性。日线开盘偏离上轨、涨幅名次和止损 ATR 倍数也没有发现稳定的单调关系。')
md('next','## 后续日志应记录这些特征与对照结果\n\n优先记录冲高实体涨幅、冲高量比、确认阴线实体跌幅、北京时间时段，以及 15 分钟涨幅、布林带宽度与扩张速度。每条信号都应带策略版本和入场前快照；成交与被过滤信号都留记录，避免只观察成交单带来的选择偏差。\n\n下一轮先验证时段差异、弱冲高、大阴线三个单独假设；价格幅度还应按 ATR 或布林带宽度归一化，以检验跨币适用性。未来订单才适合检验这些发现是否可用。当前工作保存了分析和逐单特征，尚未把候选阈值应用到运行中的策略。')
md('questions','## 尚待回答的问题\n\n深夜效应来自真实市场规律，还是日内涨幅排名随时间变化？弱冲高在低波动币种里是否仍然“弱”？确认阴线过大时改为等待反抽，是否有足够成交机会并覆盖成本？这些需要新的排名快照和重新撮合才能回答。')
md('caveats','## 证据边界\n\n这是原订单的特征归因，不是调整规则后的完整资金曲线回测。过滤会改变资金占用、持仓冲突、冷却和每日风控，因此不能把剩余订单之和当作策略新收益。\n\n原始回测使用缓存币种按 UTC 当日涨幅近似排名，与模拟盘滚动 24 小时排名不同；币种覆盖、上市时长和流动性准入尚需另行核验。原回测含每边 0.05% 手续费与 2 基点滑点，未模拟资金费、排队、部分成交和强平。\n\n共探索 31 个入场特征，后续做 9 组阈值敏感性检查；阈值与区间均属探索性结果，存在多重比较、相关订单和历史数据选择偏差。不得将本报告解释为盈利承诺。')

charts=[]
for id,dataset,x,color,titlec in [('timing','timing','period','时段','分月单均净盈亏'),('price','comparison','特征','分组','入场特征与单均净盈亏')]:
    charts.append({'id':id,'type':'bar','title':titlec,'dataset':dataset,'sourceId':'diagnostics',
                   'encodings':{'x':{'field':x},'y':{'field':'avg','label':'USDT/单'},'color':{'field':color}},
                   'options':{'grouping':'grouped','showLegend':True}})
artifact={'surface':'report','manifest':{'version':1,'title':title,'blocks':blocks,'charts':charts,'sources':[source]},
          'snapshot':{'version':1,'status':'ready','datasets':{'timing':timing.to_dict('records'),'comparison':comparison.to_dict('records')}},
          'sources':[source]}
(OUT/'artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf-8')
# Supporting notes preserve the report structure and chart contracts without UI clutter.
(OUT/'report_notes.json').write_text(json.dumps({'audience':'product stakeholders','delivery':'mcp-app',
    'structure':'title, summary, definitions, findings, recommendations, open questions, caveats; summary localized to Chinese per user language instruction',
    'charts':[{'id':'timing','question':'Is late-session per-trade loss worse each month?','family':'grouped bar','rows':6,'palette':'hard two-root cap, shared renderer','non_color':'axis labels and legend','zero_baseline':True},
              {'id':'price','question':'Do entry feature groups underperform their complements?','family':'grouped bar','rows':6,'palette':'hard two-root cap, shared renderer','non_color':'axis labels and legend','zero_baseline':True}],
    'repeated_chart_reason':'Both compare same-unit group means; only three monthly points make a trend line unsuitable.'},ensure_ascii=False,indent=2),encoding='utf-8')
print(OUT/'artifact.json')
