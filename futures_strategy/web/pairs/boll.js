'use strict';
(() => {
  const el = s => document.querySelector(s);
  const e = v => String(v ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const n = (v,d=2) => Number.isFinite(v)?v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
  const date = v => v?new Date(v*1000).toLocaleString('zh-CN',{hour12:false,timeZone:'Asia/Shanghai'}):'—';
  const color = v => v>0?'positive':v<0?'negative':'';
  const stages = {INELIGIBLE:'资格未通过',WAIT_PUMP:'等待放量',WAIT_REVERSAL:'等待转弱',WAIT_BREAKDOWN:'等待破低',SIGNAL:'入场信号'};
  const reasonText = {DAILY_OPEN_NOT_ABOVE_UPPER:'旧版BOLL条件未通过',DAILY_GATE_FAILED:'日线WR未到高位',
    WAIT_PUMP:'等待WR高位放量阳线',WAIT_REVERSAL:'等待WR从高位明显回落',WAIT_BREAKDOWN:'等待后续收盘跌破阴线低点',
    WARMUP_NO_REPLAY:'启动前信号只观察，不补开',SIGNAL_EXPIRED:'确认信号已过期',PAUSED:'新开仓暂停',
    OPENED:'已模拟开仓',SIGNAL_USED:'该信号已处理',SYMBOL_BUSY:'本币已有持仓',POSITION_LIMIT:'已达2仓上限',
    DAILY_LIMIT:'本币今日已开3单',COOLDOWN:'平仓后冷却中',DAILY_LOSS:'达到每日亏损暂停阈值',
    STALE_OR_MISSING_QUOTE:'报价过期或缺失',SPREAD_TOO_WIDE:'买卖价差过大',DATA_ERROR:'行情数据不可用',
    CAPITAL_OR_RISK_BUDGET:'余额、最小订单或风险额度不满足',MANUAL_CLOSE:'手动平仓',MANUAL_CLOSE_ALL:'批量平仓',
    STOP:'止损／追踪退出',TARGET:'达到2R目标',TIME_EXIT:'30分钟时间退出'};
  Object.assign(reasonText,{daily_open_not_above_upper:'旧版BOLL条件未通过',daily_wr_not_overbought:'日线 WR 未进入高位区',waiting_for_pump:'等待 WR 高位放量阳线',waiting_for_wr_reversal:'等待 WR 从高位回落',waiting_for_reversal:'等待 WR 从高位明显回落',waiting_for_breakdown:'等待后续收盘跌破阴线低点',entry_hour_blocked:'北京时间 14:00-15:59 暂停新开仓',late_session_blocked:'北京时间 22:00 后不再观察新信号',signal_ready:'已确认破低，待执行风控',signal_expired:'信号已过期',invalid_daily_data:'日线数据无效',missing_current_daily_open:'缺少当日日K数据',insufficient_daily_history:'日线预热不足28根',daily_data_not_continuous:'日线数据不连续',invalid_minute_data:'分钟行情数据无效',insufficient_minute_history:'分钟K预热不足',minute_data_not_continuous:'分钟K存在缺口',stale_minute_data:'最新分钟K未更新',rank_outside_top_7:'已跌出涨幅前 7',RANKING_REENTRY_REQUIRES_NEW_PUMP:'重新上榜，等待新的放量信号'});
  const why = v => reasonText[v] || v || '等待条件';
  let snapshot, token, busy=false, loading=false, lastOk=0, pending=null;
  const api = {
    async read(path){const r=await fetch(path,{signal:AbortSignal.timeout(12000)});const d=await r.json();if(!r.ok)throw Error(d.error||'读取失败');return d;},
    async write(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Pair-Token':token},body:JSON.stringify(body),signal:AbortSignal.timeout(45000)});const d=await r.json();if(!r.ok)throw Error(d.error||'操作失败');return d.result;}
  };
  const metric = (title,value,foot,sign=false) => `<article class="metric"><div class="metric-label">${e(title)}<span>USDT</span></div><div class="metric-value ${sign?color(value):''}">${n(value)}</div><div class="metric-foot">${e(foot)}</div></article>`;
  function eventText(v){const p=v.payload;if(v.kind==='EQUITY')return '净值 '+n(p.equity)+' U';if(v.kind==='SCAN')return '候选 '+p.candidates.length+' 币';if(v.kind==='OPEN_POSITION')return `${p.symbol} · 开空 ${n(p.qty,5)} · 成交 ${n(p.entry_price,5)} · 计划风险 ${n(p.planned_risk)} U`;if(v.kind==='CLOSE_POSITION')return `${p.symbol} · ${why(p.exit_reason)} · 净收益 ${n(p.net_pnl)} U`;return p.message||(p.reason?why(p.reason):JSON.stringify(p));}
  function controls(){
    const stale=Date.now()-lastOk>15000;
    el('#boll-running').disabled=busy||!snapshot||stale||snapshot.risk_paused;
    el('#boll-running').textContent=busy?'处理中…':snapshot?.running?'暂停 WR 新开仓':'启动 WR 新开仓';
    el('#boll-close-all').disabled=busy||!snapshot?.positions.length||stale;
    el('#workstation-close-all').disabled=busy||!token||stale;
    document.querySelectorAll('[data-boll-close]').forEach(b=>b.disabled=busy||stale);
  }
  function card(p){return `<article class="group-card bb-position"><div class="group-header"><div><h3><span class="side short">空</span> ${e(p.symbol)}</h3><small class="muted">1 倍模拟 · 名义 ${n(p.gross_notional)} U</small></div><span class="tag ${p.stale?'warn':''}">${p.stale?'报价过期':'持仓中'}</span></div><div class="bb-position-body"><div><small>入场 / 当前价格</small><strong>${n(p.entry_price,5)} / ${n(p.mark_price,5)}</strong></div><div><small>数量 / 占用保证金</small><strong>${n(p.qty,5)} / ${n(p.initial_margin)} U</strong></div><div><small>${p.trailing?'追踪止损':'初始止损'} / 止盈目标</small><strong>${n(p.stop_price,5)} / ${n(p.target_price,5)}</strong></div><div><small>计划风险 / 持仓时长</small><strong>${n(p.planned_risk)} U / ${n(p.holding_seconds/60,1)} 分钟</strong></div></div><div class="group-total"><div>预计平仓净收益<small>含费用及已结算资金费</small></div><strong class="group-total-value ${color(p.estimated_close_net)}">${n(p.estimated_close_net)}<small>USDT</small></strong></div><div class="group-footer"><span class="muted">${p.close_error?e(why(p.close_error)):'强平价未接入，不能估算'}</span><button class="button danger" data-boll-close="${e(p.id)}">平仓此币 ↗</button></div></article>`;}
  function render(s,o){
    snapshot=s;token=s.token;
    const a=s.account, m=s.market;
    el('#boll-nav-count').textContent=s.positions.length;
    el('#boll-connection').textContent=m.connection==='CONNECTED'?'公开行情在线':m.connection==='OFFLINE'?'离线模式':m.connection==='WAITING'?'连接中':'行情异常';
    el('#boll-metrics').innerHTML=metric('独立账户净值',a.equity,'初始本金 1,000 U')+metric('持仓浮盈',a.unrealized,a.stale?'报价过期 · 参考估值':'可用资金 '+n(a.available)+' U',true)+metric('日内净损益',a.daily_pnl,s.risk_paused?'已触发日亏损暂停':'日损 10 U 暂停新开仓',true)+metric('已用名义本金',a.gross_notional,'保证金 '+n(a.initial_margin)+' U / 最多 2 仓');
    el('#boll-scan-time').textContent=(m.message||'等待扫描')+' · 最近 '+date(m.last_scan);
    el('#boll-candidates').innerHTML=(m.candidates||[]).map(c=>`<tr><td><small>#${e(c.rank)}</small><strong>${e(c.symbol)}</strong></td><td class="positive">${n(c.price_change_percent)}%</td><td>${n(c.daily_wr)}<small>${n(c.minute_wr)}</small></td><td>${n(c.volume_ratio)}×</td><td><span class="tag">${e(stages[c.stage]||c.stage)}</span></td><td class="bb-reason">${e(why(c.reason))}</td></tr>`).join('')||'<tr><td colspan="6" class="empty-cell">暂时没有候选。正在等待有效涨幅榜和已收盘 K 线，不强制开仓。</td></tr>';
    el('#boll-position-count').textContent=s.positions.length+' / 2';
    el('#boll-positions').innerHTML=s.positions.length?s.positions.map(card).join(''):`<div class="empty-state"><div class="empty-symbol">↘</div><h3>当前没有做空持仓</h3><p>${s.running?'正在观察放量冲高后的转弱确认，不因涨得多就直接做空。':'本策略新开仓暂停；启动后独立观察，不影响配对策略。'}</p></div>`;
    el('#boll-history').innerHTML=s.closed_positions.map(p=>`<tr><td>${e(p.symbol)}</td><td>${date(p.opened_at)}<small>${date(p.closed_at)}</small></td><td>${n(p.qty,5)}</td><td>${n(p.fees,4)}</td><td>${n(p.funding_pnl,4)}</td><td class="${color(p.net_pnl)}">${n(p.net_pnl)}</td><td>${e(why(p.exit_reason))}</td></tr>`).join('')||'<tr><td colspan="7" class="empty-cell">暂无已平仓记录</td></tr>';
    el('#boll-logs').innerHTML=s.events.map(v=>`<div class="log-row"><time>${date(v.timestamp)}</time><span class="log-kind">${e(v.kind)}</span><div class="log-detail">${e(eventText(v))}</div></div>`).join('');
    const listingMargin=o.strategy_settings?.new_listing_breakdown?.margin_per_trade;
    el('#listing-overview-margin').textContent=n(Number.isFinite(listingMargin)?listingMargin:100,0);
    el('#overview-metrics').innerHTML=metric('模拟总权益',o.equity,'初始本金 '+n(o.paper_initial_capital)+' U · 合计敞口 '+n(o.gross_notional)+' U')+metric('配对账户',o.strategies.pair.equity,'原账户及历史记录保留')+metric('WR 做空账户',o.strategies.boll_short?.equity,'独立预算')+metric('新币破位账户',o.strategies.new_listing_breakdown?.equity,'独立预算');
    controls();
  }
  async function refreshBoll(){
    if(loading||busy)return;loading=true;
    try{const [s,o]=await Promise.all([api.read('/api/boll/state'),api.read('/api/overview')]);lastOk=Date.now();render(s,o);el('#boll-error').hidden=true;el('#overview-error').hidden=true;}
    catch(err){for(const id of ['#boll-error','#overview-error']){el(id).hidden=false;el(id).textContent='策略状态同步失败：'+err.message+'。页面保留最后状态，过期后禁用操作。';}}
    finally{loading=false;controls();}
  }
  async function perform(path,body){
    if(busy||!token)return;busy=true;controls();
    const out=el(path.includes('/workstation/')?'#overview-result':'#boll-result');
    try{const r=await api.write(path,body);out.hidden=false;out.className='banner';
      if(path.includes('close-all')){const results=path.includes('/workstation/')?Object.entries(r):[['BOLL',r]];out.textContent=results.map(([name,v])=>`${name}：平仓成功 ${v.closed.length}，失败 ${v.failed.length}${v.failed.length?'；'+v.failed.map(f=>f.id+' '+why(f.reason)).join('；'):''}`).join('\n')+'。新开仓已暂停。';if(results.some(([,v])=>v.failed.length))out.classList.add('error');}
      else out.textContent=path.endsWith('/running')?(body.running?'WR 新开仓已启动，等待合格信号。':'WR 新开仓已暂停，持仓退出监控继续。'):'本币已模拟平仓，净收益 '+n(r.net_pnl)+' U。';
    }catch(err){out.hidden=false;out.className='banner error';out.textContent=err.message+'；若操作超时，请刷新确认状态后再操作。';}
    finally{busy=false;await refreshBoll();controls();}
  }
  function confirm(path,body,text){if(busy||!token)return;pending={path,body};el('#boll-confirm-text').textContent=text;el('#boll-confirm').returnValue='cancel';el('#boll-confirm').showModal();}
  el('#boll-running').addEventListener('click',()=>perform('/api/boll/running',{running:!snapshot.running}));
  el('#boll-close-all').addEventListener('click',()=>confirm('/api/boll/close-all',{},`暂停 WR 新开仓并平掉其 ${snapshot.positions.length} 个模拟仓位；不影响配对账户。`));
  el('#workstation-close-all').addEventListener('click',()=>confirm('/api/workstation/close-all',{},'暂停三套策略，并平掉三套账户内全部模拟持仓。不会操作真实账户。'));
  el('#boll-live-info').addEventListener('click',()=>{el('#boll-result').hidden=false;el('#boll-result').textContent='实盘尚未接入：订单对账、交易所保护止损及配对单腿失败处理未完成。当前无法启用，不会发送真实订单。';});
  document.addEventListener('click',event=>{const b=event.target.closest('[data-boll-close]');if(b)confirm('/api/boll/close-position',{id:b.dataset.bollClose},'仅平掉这一个 BOLL 模拟空单，配对策略不受影响。');});
  el('#boll-confirm').addEventListener('close',()=>{const op=pending;pending=null;if(el('#boll-confirm').returnValue==='confirm'&&op)perform(op.path,op.body);});
  setInterval(refreshBoll,5000);setInterval(controls,1000);refreshBoll();
})();
