// Run the generated script against a minimal DOM; no external dependencies.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(__dirname,'../reports/lower_confidence.html'),'utf8');
const payload=source.match(/<script id="trajectory-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const code=source.match(/<script>([\s\S]*?)<\/script>/)[1];
const data=JSON.parse(payload);
const elements={};
for(const name of ['trajectory-data','rule','sample','cadence','model','chart','metrics'])elements[name]={value:'',textContent:'',innerHTML:'',listeners:{},addEventListener(type,handler){this.listeners[type]=handler;}};
elements['trajectory-data'].textContent=payload;
elements.rule.value='fade_C1_all';elements.sample.value='all';elements.cadence.value='card';elements.model.value='odds_confidence';
const context=vm.createContext({document:{getElementById:name=>elements[name]}});
vm.runInContext(code,context);
let controls=0,curves=0;
for(const [key,rows] of Object.entries(data.scenarios)){
 const vals=key.split('|');['rule','sample','cadence','model'].forEach((name,i)=>elements[name].value=vals[i]);
 elements.rule.listeners.change();
 const drawn=[...elements.chart.innerHTML.matchAll(/<polyline[^>]*points="([^"]+)"/g)];
 assert.equal(drawn.length,rows.filter(x=>x.feasible).length);
 assert.equal(rows.length,3);
 for(const [,points] of drawn){assert.equal(points.split(' ').length,188);assert.ok(!/NaN|Infinity|undefined/.test(points));curves++;}
 assert.equal((elements.metrics.innerHTML.match(/<tr>/g)||[]).length,4);
 for(const row of rows){
  if(row.feasible){assert.equal(row.wealth.length,188);assert.equal(row.wealth[0],1);assert.ok(Math.abs(row.wealth.at(-1)-1-row.growth)<1e-7);}
  else assert.ok(elements.metrics.innerHTML.includes('Unavailable'));
 }
 controls++;
}
assert.equal(controls,Object.keys(data.labels).length*2*2*3);
assert.ok(data.scenarios['fade_C1_50plus|all|card|odds_confidence'][0].growth>.22);
assert.ok(data.scenarios['fade_C2_all|all|card|odds_confidence'][0].growth<0);
console.log(`PASS: ${controls} control combinations, ${curves} finite 188-point curves; unavailable allocations explicitly shown`);
