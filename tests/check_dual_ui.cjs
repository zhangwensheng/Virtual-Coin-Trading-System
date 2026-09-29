// Run only against the isolated synthetic server in serve_dual_preview.py.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
(async()=>{
  const browser = await chromium.launch({headless:true,executablePath:process.env.DUAL_TEST_BROWSER||undefined});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1050}});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto('http://127.0.0.1:8769/');
    await page.getByRole('button',{name:'WR 冲高做空',exact:false}).click();
    await page.locator('.bb-position').first().waitFor();
    assert.equal(await page.locator('.bb-position').count(),2);
    const out=path.resolve('outputs/dual_ui_smoke/screenshots');fs.mkdirSync(out,{recursive:true});
    await page.screenshot({path:path.join(out,'boll-desktop.png'),fullPage:true});
    await page.locator('[data-boll-close]').first().click();
    await page.locator('#boll-confirm').getByRole('button',{name:'取消',exact:true}).click();
    assert.equal(await page.locator('.bb-position').count(),2);
    await page.locator('[data-boll-close]').first().click();
    await page.locator('#boll-confirm').getByRole('button',{name:'确认',exact:true}).click();
    await page.waitForFunction(()=>document.querySelectorAll('.bb-position').length===1);
    const pair = await (await page.request.get('http://127.0.0.1:8769/api/state')).json();
    assert.equal(pair.groups.length,1);
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(out,'boll-mobile.png'),fullPage:true});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
    await page.getByRole('button',{name:'平本策略全部仓位',exact:true}).click();
    await page.locator('#boll-confirm').getByRole('button',{name:'确认',exact:true}).click();
    await page.waitForFunction(()=>document.querySelectorAll('.bb-position').length===0);
    await page.getByRole('button',{name:'双策略总览',exact:false}).click();
    await page.getByRole('button',{name:'平掉两策略全部模拟持仓',exact:true}).click();
    await page.locator('#boll-confirm').getByRole('button',{name:'确认',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('#overview-result').textContent.includes('pair'));
    const after = await (await page.request.get('http://127.0.0.1:8769/api/state')).json();
    assert.equal(after.groups.length,0);assert.equal(after.running,false);
    assert.deepEqual(errors,[]);
    console.log('PASS: desktop/mobile, cancel, BOLL single close isolation, BOLL bulk close, global close; no JS errors');
    console.log(out);
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
