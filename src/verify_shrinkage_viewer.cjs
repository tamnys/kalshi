/* Exercise the report's own rendering code for every available control state. */
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const root=path.resolve(__dirname,'..');
const html=fs.readFileSync(path.join(root,'reports/shrinkage_kelly.html'),'utf8');
const payload=html.match(/<script id="shrinkage-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const data=JSON.parse(payload),elements={};
function element(id){return elements[id]??=( {value:'',textContent:'',innerHTML:'',addEventListener(){}} );}
element('shrinkage-data').textContent=payload;
for(const [id,v] of Object.entries({portfolio:'bounded_core_plus_fade1',sample:'all',policy:'full',cap:'uncapped',method:'shrink_confidence',scale:'linear'}))element(id).value=v;
const context={document:{getElementById:element,querySelectorAll(){return [];}}};
vm.createContext(context);vm.runInContext(script,context);
let states=0,curves=0;
for(const [key,values] of Object.entries(data.curves)){
 const result=data.cases[key];assert.equal(values.length,188);
 let peak=values[0],maxDD=0;
 values.forEach(v=>{assert(v>0&&Number.isFinite(v));peak=Math.max(peak,v);maxDD=Math.max(maxDD,1-v/peak);});
 assert.equal(values.at(-1),Number(result.final_wealth));
 assert(Math.abs(maxDD-Number(result.max_drawdown))<=Number.EPSILON*values.length);curves++;
 const [portfolio,sample,model,policy,cap]=key.split('|');
 if(!model.startsWith('shrink'))continue;
 for(const scale of ['linear','log']){
  for(const [id,v] of Object.entries({portfolio,sample,method:model,policy,cap,scale}))element(id).value=v;
  context.render();
  for(const id of ['wealth','drawdown']){assert(element(id).innerHTML.includes('<svg'));assert(!/NaN|Infinity|undefined/.test(element(id).innerHTML));}
  assert(element('metrics').innerHTML.includes(Number(result.final_wealth).toFixed(3)+'×'));
  assert(element('metrics').innerHTML.includes(data.labels[model]));states++;
 }
}
const links=[...html.matchAll(/href="([^"]+)"/g)].map(m=>m[1]);
let localLinks=0;
for(const link of links){if(!/^https?:|^#/.test(link)){assert(fs.existsSync(path.resolve(root,'reports',link)),link);localLinks++;}}
const result={status:'PASS',curves_reconciled:curves,rendered_control_states:states,local_links:localLinks,
 note:'Executes the exact embedded viewer script with a minimal DOM; this does not replace a real-browser visual inspection.'};
fs.writeFileSync(path.join(root,'results/shrinkage_viewer_verification.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
