'use strict';
const $ = (s) => document.querySelector(s);
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (n, digits=2) => Number.isFinite(n) ? n.toLocaleString('en-US',{minimumFractionDigits:digits,maximumFractionDigits:digits}) : '—';
const signed = (n, digits=2) => Number.isFinite(n) ? (n>0?'+':'')+num(n,digits) : '—';
const tone = n => n>0?'positive':n<0?'negative':'';
const coin = s => s.replace(/USDT$/, '');
const stamp = t => t ? new Date(t*1000).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false}) : '—';
const price = n => num(n,n>=100?2:n>=1?4:6);
const reasons = {MANUAL_CLOSE:'手动整组平仓',MANUAL_CLOSE_ALL:'一键全部平仓',MEAN_REVERSION:'价差回归',SPREAD_STOP:'价差扩大止损',GROUP_LOSS_LIMIT:'单组亏损止损',TIME_EXIT:'持仓超时',ACCOUNT_DRAWDOWN:'账户回撤停机'};
let state = null, busy = false, fetching = false, operation = null, lastSuccess = 0, timer;
function errorText(text){
  const s=String(text);
  const mappings=[['STALE_OR_MISSING_QUOTE','报价缺失或已过期，持仓保留；等待行情恢复后重试'],['GROUP_NOT_FOUND','该持仓不存在，请刷新'],['DRAWDOWN_CIRCUIT','账户已触发回撤停机，当前不能恢复新开仓'],['PAUSED_OR_RISK','新开仓已暂停或受到风控限制'],['INVALID_ORIGIN_OR_TOKEN','会话已变化，请刷新页面后重试']];
  return mappings.find(([key])=>s.includes(key))?.[1]||s;
}
function toast(text){$('#toast').textContent=text;$('#toast').hidden=false;clearTimeout(timer);timer=setTimeout(()=>$('#toast').hidden=true,4500)}
function switchPage(page){
  document.body.dataset.strategy=['boll','listing','overview'].includes(page)?page:'pair';
  document.querySelectorAll('.page-panel').forEach(p=>p.hidden=p.id!==`page-${page}`);
  document.querySelectorAll('.nav').forEach(b=>b.classList.toggle('active',b.dataset.page===page));
  $('#page-label').textContent={positions:'配对持仓',watch:'币对观察',history:'平仓记录',logs:'运行日志',boll:'WR 冲高做空',listing:'新币破位做空',overview:'三策略总览'}[page];
}
function controls(){
  $('#toggle-running').disabled=busy||!state||Boolean(state.circuit)||Date.now()-lastSuccess>15000;
  $('#toggle-running').textContent=busy?'处理中…':state?.running?'暂停新开仓':'启动新开仓';
  $('#close-all').disabled=busy||!state?.groups.length||Date.now()-lastSuccess>15000;
  document.querySelectorAll('[data-close]').forEach(b=>b.disabled=busy||Date.now()-lastSuccess>15000);
}
function metric(id,value,sign=false,suffix=''){
  const el=$(id);el.textContent=(sign?signed(value):num(value))+suffix;el.classList.remove('positive','negative');if(sign&&tone(value))el.classList.add(tone(value));
}
function groupCard(g){
  const a=g.legs[0],b=g.legs[1], pnl=g.estimated_close_net;
  const stale=g.stale, z=g.z, left=Number.isFinite(z)?Math.max(1,Math.min(99,(z+4)/8*100)):50;
  return `<article class="group-card" data-group-id="${esc(g.id)}"><div class="group-header"><div class="pair-name"><div class="pair-coins"><span class="coin">${esc(coin(a.symbol).slice(0,2))}</span><span class="coin">${esc(coin(b.symbol).slice(0,2))}</span></div><div><h3>${esc(coin(a.symbol))}<span>/</span>${esc(coin(b.symbol))}</h3><div class="group-subtitle">双永续对冲 · ${num(g.gross_notional)} U</div></div></div><span class="tag ${stale||g.close_error?'warn':''}">${stale?'报价待更新':g.close_error?'平仓待重试':'持仓中'}</span></div>
  <div class="legs">${g.legs.map(l=>`<div class="leg"><div class="leg-identity"><span class="side ${l.side==='LONG'?'long':'short'}">${l.side==='LONG'?'多':'空'}</span><strong>${esc(coin(l.symbol))}</strong><small>数量 ${num(l.qty,6)}</small></div><div class="leg-price">${price(l.mark_price)}<small>入场 ${price(l.entry_price)}</small></div><div class="leg-pnl ${tone(l.pnl)}">${signed(l.pnl)}<small>${l.stale?'参考浮盈 · 报价过期':'浮动盈亏 / U'}</small></div></div>`).join('')}</div>
  <div class="group-total"><div>预计平仓净盈亏<small>含已记资金费，扣除开平仓费用与滑点</small></div><div class="group-total-value ${tone(pnl)}">${signed(pnl)}<small>USDT${stale?' · 暂不可估值':''}</small></div></div>
  <div class="z-section"><div class="z-heading"><span>价差偏离 · Z 值</span><strong>${signed(g.entry_z)} → ${signed(z)}</strong></div><div class="z-track" role="img" aria-label="当前 Z 值 ${num(z)}，回归目标正负 ${state.settings.exit_z}">${Number.isFinite(z)?`<i class="z-marker" style="left:${left}%"></i>`:''}</div><div class="z-labels"><span>−${state.settings.stop_z.toFixed(1)} 止损</span><span>回归区 ±${state.settings.exit_z}</span><span>+${state.settings.stop_z.toFixed(1)} 止损</span></div></div>
  <details class="group-details"><summary>费用、对冲比例与时间</summary><div class="detail-grid"><div>开仓费用 <b>${num(g.entry_fee,4)} U</b></div><div>预计平仓费 <b>${num(g.estimated_exit_fee,4)} U</b></div><div>已记资金费 <b>${signed(g.funding_pnl,4)} U</b></div><div>对冲比例 β <b>${num(g.model.beta,3)}</b></div><div>最长持仓 <b>${state.settings.max_hold_hours} h</b></div><div>计划止损 <b>${num(state.settings.group_loss_limit)} U</b></div></div><p style="margin-top:10px">开仓 ${stamp(g.opened_at)}<br>组号 ${esc(g.id)}</p></details>
  ${g.close_error?`<p class="close-error">${esc(errorText(g.close_error))}</p>`:''}<div class="group-footer"><span class="muted">◷ 已持有 ${num(g.holding_hours,1)} 小时</span><button class="button danger" data-close="${esc(g.id)}" aria-label="平仓 ${esc(coin(a.symbol))} 和 ${esc(coin(b.symbol))} 整组">平仓此组 ↗</button></div></article>`;
}
function render(){
  const expanded = new Set([...document.querySelectorAll('.group-card:has(details[open])')].map(e=>e.dataset.groupId));
  const focusedClose = document.activeElement?.dataset?.close;
  const a=state.account,s=state.settings,m=state.market;
  metric('#equity',a.equity);$('#available').textContent=num(a.available)+' U';metric('#unrealized',a.unrealized,true);$('#realized').textContent=signed(a.realized)+' U';$('#realized').className=tone(a.realized);metric('#gross',a.gross_notional);metric('#drawdown',a.stale?null:a.drawdown*100,false,a.stale?'':'%');
  $('#max-gross').textContent=num(s.max_gross,0)+' U';$('#drawdown-limit').textContent=num(s.max_drawdown*100)+'%';$('#allocation-bar').style.width=Math.min(100,a.gross_notional/s.max_gross*100)+'%';$('#valuation-status').textContent=a.stale?'· 报价过期，参考估值':'';
  $('#group-count').textContent=`${state.groups.length} / ${s.max_groups}`;$('#nav-count').textContent=state.groups.length;
  $('#engine-status').textContent=state.circuit?'回撤停机':state.running?'开仓运行中':'新开仓暂停';$('#engine-status').className='status-pill'+(state.running?' running':'');
  $('#risk-label').textContent=state.circuit?'已触发风控':'风控监控';
  $('#connection').className='connection'+(m.connection==='CONNECTED'?'':' bad');$('#connection').innerHTML=`<i class="dot"></i>${m.connection==='CONNECTED'?'公共行情在线':m.connection==='OFFLINE'?'离线模式':m.connection==='WAITING'?'连接行情中':'行情异常'}`;
  $('#groups').innerHTML=state.groups.length?state.groups.map(groupCard).join(''):`<div class="empty-state"><div class="empty-symbol">∥</div><h3>当前没有配对持仓</h3><p>${state.running?'正在等待符合条件的币对。价差足够偏离并开始收敛后，系统才会模拟开仓。':'新开仓已暂停。启动后，系统会筛选关系稳定的币对，等待价差偏离与收敛信号。'}<br>最多 ${s.max_groups} 组 · 每币每日最多 ${s.daily_coin_limit} 次 · 不强制凑满仓位</p><button class="button secondary" data-go="watch">查看候选币对 →</button></div>`;
  document.querySelectorAll('.group-card').forEach(e=>{if(expanded.has(e.dataset.groupId))e.querySelector('details').open=true});
  if(focusedClose){[...document.querySelectorAll('[data-close]')].find(e=>e.dataset.close===focusedClose)?.focus({preventScroll:true})}
  const candidates=m.candidates||[];
  $('#radar').innerHTML=candidates.length?candidates.slice(0,3).map(c=>`<div class="radar-row"><div><strong>${esc(coin(c.symbol_a))} / ${esc(coin(c.symbol_b))}</strong><br><small>${c.direction?'收敛信号 · 待风控检查':'等待入场条件'}</small></div><span class="radar-z">Z ${signed(c.z)}</span></div>`).join(''):'<div class="radar-empty">暂无通过统计筛选的币对</div>';
  $('#scan-summary').textContent=m.message;$('#scan-time').textContent='最近扫描 '+stamp(m.last_scan);
  $('#candidates').innerHTML=candidates.length?candidates.map(c=>`<tr><td>${esc(coin(c.symbol_a))} / ${esc(coin(c.symbol_b))}</td><td>${signed(c.z)}</td><td>${signed(c.previous_z)}</td><td>${num(c.correlation,3)}</td><td>${num(c.pvalue_1h,4)} / ${num(c.pvalue_4h,4)}</td><td>${num(c.beta,3)}</td><td><span class="tag ${c.direction?'':'warn'}">${Date.now()/1000-c.signal_time>330?'信号过期':c.direction?'收敛信号 · 待风控':'观察中'}</span></td></tr>`).join(''):'<tr><td colspan="7" class="empty-cell">尚无合格币对。行情扫描完成后会自动更新，不会用相关性不足的币对凑数。</td></tr>';
  $('#history').innerHTML=state.closed_groups.length?state.closed_groups.map(g=>`<tr><td>${g.legs.map(l=>esc(coin(l.symbol))).join(' / ')}<small>${esc(g.id)}</small></td><td>${stamp(g.closed_at)}</td><td class="${tone(g.price_pnl)}">${signed(g.price_pnl)}</td><td>${signed(g.funding_pnl,4)}</td><td>${num(g.fees,4)}</td><td class="${tone(g.net_pnl)}">${signed(g.net_pnl)}</td><td>${esc(reasons[g.exit_reason]||g.exit_reason)}</td></tr>`).join(''):'<tr><td colspan="7" class="empty-cell">暂无平仓记录。手动和自动平仓都会记录在这里。</td></tr>';
  $('#logs').innerHTML=state.events.map(e=>`<div class="log-row"><time>${stamp(e.timestamp)}</time><span class="log-kind">${esc(e.kind)}</span><div class="log-detail">${esc(eventDescription(e))}</div></div>`).join('');
  drawEquity(state.equity_history);
  $('#last-update').textContent='最后同步 '+stamp(state.timestamp)+' · 每 5 秒刷新';
  const marketErrors=m.connection==='DEGRADED'?(m.errors||[]).filter(e=>Date.now()/1000-e.timestamp<180):[];
  $('#error-banner').hidden=!marketErrors.length&&!state.circuit;
  if(!$('#error-banner').hidden)$('#error-banner').textContent=state.circuit?'账户已触发回撤停机。系统将尝试退出持仓；请检查仍未平仓的组。':marketErrors.at(-1).message+'。新鲜行情不足时不会模拟成交。';
  controls();
}
function eventDescription(e){const p=e.payload;if(p.message)return p.message+(p.detail?'：'+p.detail:'');if(e.kind==='EQUITY')return `净值 ${num(p.equity)} U · ${p.group_count} 组持仓`;if(e.kind==='SCAN')return `已扫描 ${(p.universe||[]).length} 个币 · ${(p.candidates||[]).length} 组通过筛选`;if(e.kind==='OPEN_GROUP'||e.kind==='CLOSE_GROUP')return `${p.legs.map(l=>coin(l.symbol)).join(' / ')} · ${e.kind==='OPEN_GROUP'?'开仓 '+num(p.gross_notional)+' U':(reasons[p.exit_reason]||p.exit_reason)+' · 净收益 '+signed(p.net_pnl)+' U'}`;return JSON.stringify(p)}
function drawEquity(history){
  if(history.length<2){$('#equity-chart').textContent='等待至少两个真实净值记录';return}
  const values=history.map(x=>x.equity),min=Math.min(...values),max=Math.max(...values),pad=Math.max(.5,(max-min)*.15),low=min-pad,high=max+pad;
  const points=values.map((v,i)=>`${i/(values.length-1)*600},${110-(v-low)/(high-low)*95}`).join(' ');
  $('#equity-chart').innerHTML=`<svg viewBox="0 0 600 125" preserveAspectRatio="none" role="img" aria-label="净值由 ${num(values[0])} 变为 ${num(values.at(-1))} USDT"><path d="M0 15H600 M0 62H600 M0 110H600" stroke="#26342f" stroke-dasharray="3 5" fill="none"/><polygon points="0,125 ${points} 600,125" fill="#9ee6be" opacity=".045"/><polyline points="${points}" fill="none" stroke="#a1cbb2" stroke-width="1.8" vector-effect="non-scaling-stroke"/></svg>`;
  $('#chart-time').textContent=stamp(history[0].timestamp).split(' ')[1]+' — '+stamp(history.at(-1).timestamp).split(' ')[1];
}
async function refresh(){
  if(fetching||busy)return;fetching=true;
  try{const response=await fetch('/api/state',{signal:AbortSignal.timeout(10000)});if(!response.ok)throw new Error('服务暂不可用');state=await response.json();lastSuccess=Date.now();render()}
  catch(e){$('#error-banner').hidden=false;$('#error-banner').textContent='无法同步模拟账户，请确认本地服务仍在运行。当前显示的是最后一次状态。';$('#connection').className='connection bad';$('#connection').innerHTML='<i class="dot"></i>连接中断';controls()}
  finally{fetching=false}
}
async function mutate(path,body){
  if(busy)return;busy=true;controls();
  try{
    const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Pair-Token':state.token},body:JSON.stringify(body),signal:AbortSignal.timeout(45000)});
    const data=await response.json();if(!response.ok)throw new Error(data.error||'操作失败');
    if(path==='/api/close-all'){
      const r=data.result;$('#operation-result').hidden=false;$('#operation-result').className='banner'+(r.failed.length?' error':'');$('#operation-result').textContent=`已暂停新开仓。平仓成功 ${r.closed.length} 组，失败 ${r.failed.length} 组。`+(r.failed.length?'\n'+r.failed.map(f=>`${f.id}：${errorText(f.reason)}`).join('\n'):'');
    }else if(path==='/api/close-group'){toast('整组已平仓 · 净收益 '+signed(data.result.net_pnl)+' U')}
    else{toast(body.running?'已启动新开仓；等待合格信号':'新开仓已暂停，已有持仓仍受退出监控')}
  }catch(e){$('#operation-result').hidden=false;$('#operation-result').className='banner error';$('#operation-result').textContent=e.name==='TimeoutError'?'操作响应超时，结果尚不确定。请刷新持仓和日志确认后再重试。':errorText(e.message)}
  finally{busy=false;controls();await refresh()}
}
function confirmClose(id){
  if(!state||busy)return;operation=id?{id}:{};
  const g=state.groups.find(g=>g.id===id);if(id&&!g)return;
  $('#confirm-title').textContent=id?'平仓这一组？':'平仓所有配对持仓？';
  $('#confirm-copy').textContent=id?`将同时平掉 ${g.legs.map(l=>`${coin(l.symbol)} ${l.side==='LONG'?'多单':'空单'}`).join(' 与 ')}，并进入 ${state.settings.cooldown_seconds/3600} 小时冷却。`:`将先暂停新开仓，再平掉当前所有模拟持仓（目前 ${state.groups.length} 组）。操作后不会自动恢复开仓。`;
  const net=id?g.estimated_close_net:state.groups.every(g=>g.estimated_close_net!==null)?state.groups.reduce((a,g)=>a+g.estimated_close_net,0):null;
  $('#confirm-detail').textContent=`预计净盈亏  ${signed(net)} USDT`;
  $('#confirm-dialog').returnValue='cancel';
  $('#confirm-dialog').showModal();
}
document.addEventListener('click',e=>{
  const nav=e.target.closest('[data-page]');if(nav)switchPage(nav.dataset.page);
  const go=e.target.closest('[data-go]');if(go)switchPage(go.dataset.go);
  const close=e.target.closest('[data-close]');if(close)confirmClose(close.dataset.close);
});
$('#confirm-dialog').addEventListener('close',()=>{if($('#confirm-dialog').returnValue==='confirm'&&operation){const op=operation;operation=null;mutate(op.id?'/api/close-group':'/api/close-all',op)}});
$('#close-all').addEventListener('click',()=>confirmClose(null));
$('#toggle-running').addEventListener('click',()=>{if(state)mutate('/api/running',{running:!state.running})});
$('#refresh').addEventListener('click',refresh);
$('#rules-button').addEventListener('click',()=>{
  if(!state)return;const s=state.settings;const rows=[['策略类型','不同币种 · 双永续配对'],['关系筛选','过去 30 天 · 1h / 4h 协整'],['币种范围','24h 成交额前 20 · 上市 ≥90 天'],['入场阈值',`|Z| ≥ ${s.entry_z}，5m 开始收敛`],['退出条件',`回归 ±${s.exit_z} / 偏离 ±${s.stop_z} / ${s.max_hold_hours}h`],['组合名义本金',`最大 ${s.max_gross} U · 单组 ≤${s.group_gross} U`],['账户备用金',`${s.reserve} U · 按 1 倍名义占用预算`],['最多持仓',`${s.max_groups} 组 · 币种不重复`],['开仓频率',`每币每天 ${s.daily_coin_limit} 次 · 冷却 ${s.cooldown_seconds/3600}h`],['计划亏损限制',`单组 ${s.group_loss_limit} U / 每日 ${s.daily_loss_limit} U`],['账户停机回撤',`${num(s.max_drawdown*100)}%`],['成本假设',`每腿每次费率 ${num(s.fee_rate*100,3)}% + ${s.slippage_bps} bps 滑点`],['估值说明','资金费每分钟查询，延迟结算会补记'],['研究限制','未模拟撮合队列、部分成交与强平过程']];
  $('#rules-content').innerHTML=rows.map(([a,b])=>`<div class="rule-row"><span>${esc(a)}</span><b>${esc(b)}</b></div>`).join('');$('#rules-dialog').showModal();
});
$('#close-rules').addEventListener('click',()=>$('#rules-dialog').close());
setInterval(()=>{$('#clock').textContent=new Date().toLocaleTimeString('zh-CN',{hour12:false,timeZone:'Asia/Shanghai'});controls()},1000);
setInterval(refresh,5000);refresh();
