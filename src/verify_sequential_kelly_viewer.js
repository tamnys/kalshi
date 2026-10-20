/* Check every viewer state against the saved Python trajectories and frontiers. */
const fs=require('fs'),vm=require('vm'),zlib=require('zlib'),assert=require('assert');
const path=require('path'),root=path.resolve(__dirname,'..');
const html=fs.readFileSync(path.join(root,'reports/sequential_kelly.html'),'utf8');
const payload=html.match(/<script id="sequentialData" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements={};
function element(id){
 if(!elements[id])elements[id]={value:'',checked:false,textContent:'',innerHTML:'',addEventListener(){},setCustomValidity(message){this.validityMessage=message;}};
 return elements[id];
}
element('sequentialData').textContent=payload;
element('portfolio').value='bounded_core_plus_fade1';element('sample').value='all';element('model').value='odds_confidence';element('customFraction').value='.5';
const sandbox={document:{getElementById:element},window:{},console};
vm.createContext(sandbox);vm.runInContext(script,sandbox);
const viewer=sandbox.window.sequentialViewer;
function csv(text){const rows=text.trim().split('\n').map(line=>line.split(','));const header=rows.shift();return rows.map(row=>Object.fromEntries(header.map((key,i)=>[key,row[i]])));}
function close(actual,expected,steps){
 const bound=Number.EPSILON*(steps+1)*Math.max(1,Math.abs(expected));
 assert(Math.abs(actual-expected)<=bound,`${actual} != ${expected} (roundoff ${bound})`);
}
const rows=csv(zlib.gunzipSync(fs.readFileSync(path.join(root,'results/sequential_kelly_paths.csv.gz'))).toString());
const grouped=new Map();
for(const row of rows){const key=[row.portfolio,row.sample,row.model,row.policy].join('|');if(!grouped.has(key))grouped.set(key,[]);grouped.get(key).push(row);}
let pathsChecked=0,statesChecked=0;
for(const [key,expected] of grouped){
 const parts=key.split('|'),policy=parts.pop(),context=viewer.data[parts.join('|')];
 const result=viewer.calculate(context,policy);
 assert.equal(result.wealth.length,expected.length);
 expected.forEach((row,i)=>{close(result.wealth[i],Number(row.wealth),expected.length);close(result.drawdown[i],Number(row.drawdown),expected.length);if(i)close(result.stake[i-1],Number(row.stake),expected.length);});
 pathsChecked++;
}
const frontier=csv(fs.readFileSync(path.join(root,'results/sequential_kelly_frontiers.csv'),'utf8'));
for(const row of frontier){
 const result=viewer.calculate(viewer.data[[row.portfolio,row.sample,row.model].join('|')],'custom',Number(row.fraction));
 close(result.finalWealth,Number(row.final_wealth),result.wealth.length);
 close(result.maxDD,Number(row.max_drawdown),result.wealth.length);
 close(result.maxStake,Number(row.max_stake),result.wealth.length);
}
for(const key of Object.keys(viewer.data)){
 const [portfolio,sample,model]=key.split('|');element('portfolio').value=portfolio;element('sample').value=sample;element('model').value=model;
 for(const logarithmic of [false,true])for(const extras of [false,true]){
  element('logScale').checked=logarithmic;element('showOracle').checked=extras;element('showCustom').checked=extras;
  viewer.update();
  for(const id of ['wealthChart','drawdownChart']){assert(element(id).innerHTML.includes('<svg'));assert(!/NaN|Infinity/.test(element(id).innerHTML));}
  assert(element('viewerSummary').innerHTML.includes('Largest individual stake'));statesChecked++;
 }
}
element('customFraction').value='1.1';viewer.update();assert(element('customFraction').validityMessage);
element('customFraction').value='0';viewer.update();assert.equal(element('customFraction').validityMessage,'');
console.log(JSON.stringify({status:'PASS',python_paths_reconciled:pathsChecked,custom_frontier_points:frontier.length,rendered_control_states:statesChecked},null,2));
