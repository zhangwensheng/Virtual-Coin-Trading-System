'use strict';
(() => {
  const el=s=>document.querySelector(s);
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const n=(v,d=2)=>Number.isFinite(v)?v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
  const date=v=>v?new Date(v*1000).toLocaleString('zh-CN',{hour12:false,timeZone:'Asia/Shanghai'}):'—';
  const tone=v=>v>0?'positive':v<0?'negative':'';
  const reasons={WAIT_BREAK:'等待首次收盘破前日低点',LISTING_AGE:'不在上市 1–30 天内',MISSING_PREVIOUS_DAY:'缺少前一日日 K',INVALID_DAILY:'前一日日 K 无效',PREVIOUS_DAY_NOT_GREEN:'前一日日 K 未收阳',INCOMPLETE_MINUTES:'当日分钟 K 不完整',INVALID_MINUTES:'分钟 K 数据无效',FIRST_BREAK_ALREADY_PASSED:'当天首次破位已过去',LATE_UTC_SIGNAL:'破位时间晚于 UTC 17:02',INSUFFICIENT_DRAWDOWN:'从当日高点回撤不足 3%',BREAK_TOO_DEEP:'破位深度超过 0.263%',SIGNAL_READY:'信号符合，等待风控',SIGNAL_EXPIRED:'信号已过期',WARMUP_NO_REPLAY:'启动前的信号不补开',PAUSED:'新开仓暂停',OPENED:'已模拟开仓',DAILY_LIMIT:'本币今日已开仓',SIGNAL_USED:'当日信号已处理',POSITION_LIMIT:'已达 5 仓上限',SYMBOL_BUSY:'本币已有持仓',DAILY_LOSS:'当日亏损达停机阈值',STALE_OR_MISSING_QUOTE:'报价缺失或过期',SPREAD_TOO_WIDE:'买卖价差过大',CAPITAL_OR_RISK_BUDGET:'余额或最小下单额不满足',DATA_ERROR:'行情获取失败',STOP:'价格触及止损',TARGET:'价格触及止盈',TIME_EXIT:'持仓满 24 小时',MANUAL_CLOSE:'手动平仓',MANUAL_CLOSE_ALL:'批量平仓'};
  const why=v=>reasons[v]||v||'等待条件';
  const metric=(title,value,foot,sign=false)=>`<article class="metric"><div class="metric-label">${esc(title)}<span>USDT</span></div><div class="metric-value ${sign?tone(value):''}">${n(value)}</div><div class="metric-foot">${esc(foot)}</div></article>`;
  let snapshot,token,busy=false,loading=false,lastOk=0,pending=null;
  function controls(){
    const stale=Date.now()-lastOk>15000;
    el('#listing-running').disabled=busy||!snapshot||stale||snapshot.risk_paused;
    el('#listing-running').textContent=busy?'处理中…':snapshot?.running?'暂停新币新开仓':'启动新币新开仓';
    el('#listing-close-all').disabled=busy||!snapshot?.positions.length||stale;
    const input=el('#listing-margin-input'),save=el('#listing-margin-save');
    const next=Number(input.value),current=Number(snapshot?.settings?.margin_per_trade);
    save.disabled=busy||!snapshot||stale||!Number.isFinite(next)||next<1||next>1000||Math.abs(next-current)<0.0001;
    document.querySelectorAll('[data-listing-close]').forEach(b=>b.disabled=busy||stale);
  }
  function card(p){return `<article class="group-card bb-position"><div class="group-header"><div><h3><span class="side short">空</span> ${esc(p.symbol)}</h3><small class="muted">10 倍模拟 · 名义 ${n(p.gross_notional)} U · 保证金 ${n(p.initial_margin)} U</small></div><span class="tag ${p.stale?'warn':''}">${p.stale?'报价过期':'持仓中'}</span></div><div class="bb-position-body"><div><small>入场 / 当前价格</small><strong>${n(p.entry_price,6)} / ${n(p.mark_price,6)}</strong></div><div><small>数量 / 上市天数</small><strong>${n(p.qty,5)} / ${n(p.signal.listing_age_days)} 天</strong></div><div><small>止损 / 止盈</small><strong>${n(p.stop_price,6)} / ${n(p.target_price,6)}</strong></div><div><small>破位深度 / 持仓时长</small><strong>${n(p.signal.break_depth_pct*100,3)}% / ${n(p.holding_seconds/3600,1)} 小时</strong></div></div><div class="group-total"><div>预计平仓净收益<small>含模拟手续费和滑点</small></div><strong class="group-total-value ${tone(p.estimated_close_net)}">${n(p.estimated_close_net)}<small>USDT</small></strong></div><div class="group-footer"><span class="muted">${p.close_error?esc(why(p.close_error)):'强平价未模拟'}</span><button class="button danger" data-listing-close="${esc(p.id)}">平仓此币 ↗</button></div></article>`;}
  function render(s){
    snapshot=s;token=s.token;const a=s.account,m=s.market;
    el('#listing-nav-count').textContent=s.positions.length;
    el('#listing-connection').textContent=m.connection==='CONNECTED'?'公开行情在线':m.connection==='OFFLINE'?'离线模式':m.connection==='WAITING'?'连接中':'行情异常';
    const margin=Number(s.settings.margin_per_trade),notional=margin*Number(s.settings.leverage);
    el('#listing-margin-label').textContent=n(margin,0);
    el('#listing-notional-label').textContent=n(notional,0);
    el('#listing-overview-margin').textContent=n(margin,0);
    if(document.activeElement!==el('#listing-margin-input'))el('#listing-margin-input').value=String(margin);
    el('#listing-metrics').innerHTML=metric('独立账户净值',a.equity,'初始本金 1,000 U')+metric('持仓浮盈',a.unrealized,a.stale?'报价过期 · 参考估值':'可用资金 '+n(a.available)+' U',true)+metric('日内净损益',a.daily_pnl,s.risk_paused?'达到 25 U 日亏损停机':'日亏损达 25 U 暂停新开仓',true)+metric('已用名义仓位',a.gross_notional,'保证金 '+n(a.initial_margin)+' U / 最多 5 仓');
    el('#listing-scan-time').textContent=(m.message||'等待扫描')+' · 最近 '+date(m.last_scan);
    el('#listing-candidates').innerHTML=(m.candidates||[]).map(c=>`<tr><td><strong>${esc(c.symbol)}</strong></td><td>${n(c.age_days,0)}</td><td>${n(c.prev_low,6)}</td><td>${n(c.break_depth_pct*100,3)}%</td><td>${n(c.session_drawdown_pct*100,2)}%</td><td class="bb-reason">${esc(why(c.reason))}</td></tr>`).join('')||'<tr><td colspan="6" class="empty-cell">当前没有上市 1–30 天的合格新币，或正在等待交易所行情。</td></tr>';
    el('#listing-position-count').textContent=s.positions.length+' / '+s.settings.max_positions;
    el('#listing-positions').innerHTML=s.positions.length?s.positions.map(card).join(''):`<div class="empty-state"><div class="empty-symbol">⇣</div><h3>当前没有新币空单</h3><p>${s.running?'观察前一日日 K 收阳的新上市合约，等待当天首次合格破位。':'本策略新开仓暂停；启动后独立扫描与记账。'}</p></div>`;
    el('#listing-history').innerHTML=s.closed_positions.map(p=>`<tr><td>${esc(p.symbol)}</td><td>${date(p.opened_at)}<small>${date(p.closed_at)}</small></td><td>${n(p.qty,5)}</td><td>${n(p.fees,4)}</td><td>${n(p.funding_pnl,4)}</td><td class="${tone(p.net_pnl)}">${n(p.net_pnl)}</td><td>${esc(why(p.exit_reason))}</td></tr>`).join('')||'<tr><td colspan="7" class="empty-cell">暂无已平仓记录</td></tr>';
    el('#listing-logs').innerHTML=s.events.map(v=>{const p=v.payload;const description=v.kind==='SCAN'?`扫描 ${p.candidates.length} 币 · ${p.candidates.map(c=>c.symbol+' '+why(c.reason)).join('；')}`:v.kind==='EQUITY'?`账户净值 ${n(p.equity)} U · 持仓 ${p.position_count} 币`:v.kind==='OPEN_POSITION'?`${p.symbol} · 开空 ${n(p.qty,5)} · 名义 ${n(p.gross_notional)} U`:v.kind==='CLOSE_POSITION'?`${p.symbol} · ${why(p.exit_reason)} · 净收益 ${n(p.net_pnl)} U`:v.kind==='FUNDING'?`${p.symbol} · 资金费补记 ${p.payments.length} 笔`:v.kind==='RESUME'?'已启动新开仓':v.kind==='PAUSE'?'已暂停新开仓':p.message||p.reason&&why(p.reason)||JSON.stringify(p);return `<div class="log-row"><time>${date(v.timestamp)}</time><span class="log-kind">${esc(v.kind)}</span><div class="log-detail">${esc(description)}</div></div>`}).join('');
    controls();
  }
  async function read(){if(loading||busy)return;loading=true;try{const response=await fetch('/api/listing/state',{signal:AbortSignal.timeout(12000)});if(!response.ok)throw Error('读取失败');const data=await response.json();lastOk=Date.now();render(data);el('#listing-error').hidden=true;}catch(err){el('#listing-error').hidden=false;el('#listing-error').textContent='新币策略状态同步失败：'+err.message+'。操作已暂停，请检查服务。';}finally{loading=false;controls();}}
  async function write(path,body){if(busy||!token)return;busy=true;controls();const out=el('#listing-result');try{const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Pair-Token':token},body:JSON.stringify(body),signal:AbortSignal.timeout(45000)});const data=await response.json();if(!response.ok)throw Error(data.error||'操作失败');out.hidden=false;out.className='banner';if(path.endsWith('close-all')){const r=data.result;out.textContent=`已暂停新开仓；平仓成功 ${r.closed.length}，失败 ${r.failed.length}。`+(r.failed.length?r.failed.map(f=>f.id+'：'+why(f.reason)).join('；'):'');if(r.failed.length)out.classList.add('error');}else if(path.endsWith('/running'))out.textContent=body.running?'新币策略已启动，等待首次合格破位。':'新币策略新开仓已暂停，持仓退出监控继续。';else if(path.endsWith('/settings'))out.textContent=`每笔保证金已更新为 ${n(data.result.settings.margin_per_trade,0)} U，只影响之后的新开仓。`;else out.textContent='本币已模拟平仓，净收益 '+n(data.result.net_pnl)+' U。';}catch(err){out.hidden=false;out.className='banner error';out.textContent=err.message+'；如操作超时，请刷新确认后重试。';}finally{busy=false;await read();controls();}}
  function confirm(path,body,message){if(busy||!token)return;pending={path,body};el('#boll-confirm-text').textContent=message;el('#boll-confirm').returnValue='cancel';el('#boll-confirm').showModal();}
  el('#listing-running').addEventListener('click',()=>snapshot&&write('/api/listing/running',{running:!snapshot.running}));
  el('#listing-close-all').addEventListener('click',()=>confirm('/api/listing/close-all',{},`暂停新币策略，并平掉其 ${snapshot.positions.length} 个模拟空单。其他策略不受影响。`));
  el('#listing-margin-input').addEventListener('input',controls);
  el('#listing-margin-save').addEventListener('click',()=>write('/api/listing/settings',{margin_per_trade:Number(el('#listing-margin-input').value)}));
  document.addEventListener('click',event=>{const b=event.target.closest('[data-listing-close]');if(b)confirm('/api/listing/close-position',{id:b.dataset.listingClose},'仅平掉这一个新币模拟空单，其他策略不受影响。');});
  el('#boll-confirm').addEventListener('close',()=>{const op=pending;pending=null;if(el('#boll-confirm').returnValue==='confirm'&&op)write(op.path,op.body);});
  setInterval(read,5000);setInterval(controls,1000);read();
})();
