/* Reconcile every capped browser trajectory with independently verified outputs. */
const fs=require('fs'),vm=require('vm'),zlib=require('zlib'),assert=require('assert'),path=require('path');
const root=path.resolve(__dirname,'..'),html=fs.readFileSync(path.join(root,'reports/capped_kelly.html'),'utf8');
const payload=html.match(/<script id="cappedData" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements={};
function element(id){if(!elements[id])elements[id]={value:'',checked:false,textContent:'',innerHTML:'',addEventListener(){}};return elements[id];}
element('cappedData').textContent=payload;element('portfolio').value='bounded_core_plus_fade1';
element('sample').value='all';element('model').value='odds_confidence';element('policy').value='full';
const sandbox={document:{getElementById:element},window:{},console};vm.createContext(sandbox);vm.runInContext(script,sandbox);
const viewer=sandbox.window.cappedViewer;
function csv(text){const rows=text.trim().split('\n').map(line=>line.split(',')),header=rows.shift();return rows.map(row=>Object.fromEntries(header.map((key,i)=>[key,row[i]])));}
function close(actual,expected,steps){const bound=Number.EPSILON*(steps+1)*Math.max(1,Math.abs(expected));assert(Math.abs(actual-expected)<=bound,`${actual} != ${expected}`);}
const pathRows=csv(zlib.gunzipSync(fs.readFileSync(path.join(root,'results/capped_kelly_paths.csv.gz'))).toString());
const groups=new Map();for(const row of pathRows){const key=[row.portfolio,row.sample,row.model,row.policy,row.cap_name].join('|');if(!groups.has(key))groups.set(key,[]);groups.get(key).push(row);}
for(const [key,expected] of groups){
 const parts=key.split('|'),cap=parts.pop(),policy=parts.pop();
 const result=viewer.calculate(viewer.data[parts.join('|')],viewer.multipliers[policy],viewer.caps[cap]);
 assert.equal(result.wealth.length,expected.length);
 expected.forEach((row,i)=>{close(result.wealth[i],Number(row.wealth),expected.length);close(result.drawdown[i],Number(row.drawdown),expected.length);if(i)close(result.stake[i-1],Number(row.stake),expected.length);});
}
const summaries=csv(fs.readFileSync(path.join(root,'results/capped_kelly_scenarios.csv'),'utf8'));
for(const row of summaries){
 const context=viewer.data[[row.portfolio,row.sample,row.model].join('|')],multiplier=viewer.multipliers[row.policy];
 const result=viewer.calculate(context,multiplier,viewer.caps[row.cap_name]),base=viewer.calculate(context,multiplier,null);
 close(result.finalWealth,Number(row.final_wealth),result.wealth.length);close(result.maxDD,Number(row.max_drawdown),result.wealth.length);
 close(1-result.finalWealth/base.finalWealth,Number(row.terminal_wealth_sacrificed_fraction),result.wealth.length);
 close(base.maxDD-result.maxDD,Number(row.drawdown_reduction),result.wealth.length);
 assert.equal(result.cappedBets,Number(row.capped_bets));assert.equal(result.bets,Number(row.bets));
}
let states=0;for(const key of Object.keys(viewer.data)){
 const [portfolio,sample,model]=key.split('|');element('portfolio').value=portfolio;element('sample').value=sample;element('model').value=model;
 for(const policy of Object.keys(viewer.multipliers))for(const logarithmic of [false,true]){
  element('policy').value=policy;element('logScale').checked=logarithmic;viewer.update();
  for(const id of ['wealthChart','drawdownChart']){assert(element(id).innerHTML.includes('<svg'));assert(!/NaN|Infinity/.test(element(id).innerHTML));}
  assert(element('viewerSummary').innerHTML.includes('Bets reduced'));states++;
 }
}
console.log(JSON.stringify({status:'PASS',paths_reconciled:groups.size,paired_metrics_reconciled:summaries.length,rendered_control_states:states},null,2));
