const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const source = fs.readFileSync(path.resolve('futures_strategy/web/pairs/boll.js'), 'utf8');

const expectedMappings = {
  entry_hour_blocked: '北京时间 14:00-15:59 暂停新开仓',
  late_session_blocked: '北京时间 22:00 后不再观察新信号',
  waiting_for_pump: '等待 WR 高位放量阳线',
  waiting_for_wr_reversal: '等待 WR 从高位回落',
  waiting_for_breakdown: '等待后续收盘跌破阴线低点',
  daily_wr_not_overbought: '日线 WR 未进入高位区',
  rank_outside_top_7: '已跌出涨幅前 7',
};

for (const [reason, text] of Object.entries(expectedMappings)) {
  assert.match(source, new RegExp(`${reason}:'${text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}'`));
}

console.log('PASS: WR candidate reason codes have Chinese UI mappings');
