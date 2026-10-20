/* Reconcile all browser curves with independently verified cash trajectories. */
const fs=require('fs'),vm=require('vm'),zlib=require('zlib'),assert=require('assert'),path=require('path');
const root=path.resolve(__dirname,'..'),html=fs.readFileSync(path.join(root,'reports/confidence23.html'),'utf8');
const elements={};
function element(id){if(!elements[id])elements[id]={value:'',textContent:'',innerHTML:'',addEventListener(){}};return elements[id];}
for(const id of ['policy-data','model-labels'])element(id).textContent=html.match(new RegExp('<script id="'+id+'" type="application/json">([\\s\\S]*?)</script>'))[1];
element('strategy').value='C2_back_under37';element('sample').value='all';element('cap').value='1';
const sandbox={document:{getElementById:element},console};vm.createContext(sandbox);
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],sandbox);
function close(a,b,n){assert(Math.abs(a-b)<=Number.EPSILON*(n+1)*Math.max(1,Math.abs(b)),`${a} != ${b}`);}
const raw=zlib.gunzipSync(fs.readFileSync(path.join(root,'results/confidence23_paths.csv.gz'))).toString().trim().split('\n');
const columns=raw.shift().split(','),groups=new Map();
for(const line of raw){const r=line.split(','),key=r.slice(0,5).join('|');if(!groups.has(key))groups.set(key,[]);groups.get(key).push(Number(r[columns.indexOf('wealth')]));}
for(const [key,expected] of groups){const fields=key.split('|'),cap=Number(fields.pop()),fraction=Number(fields.pop());const actual=sandbox.pathFor(fields.join('|'),fraction,cap);assert.equal(actual.wealth.length,expected.length);expected.forEach((w,i)=>close(actual.wealth[i],w,expected.length));}
const summary=fs.readFileSync(path.join(root,'results/confidence23_kelly.csv'),'utf8').trim().split('\n'),header=summary.shift().split(',');
for(const line of summary){const values=line.split(','),r=Object.fromEntries(header.map((h,i)=>[h,values[i]])),actual=sandbox.pathFor([r.id,r.sample,r.model].join('|'),Number(r.fraction),Number(r.cap));
 close(actual.final,Number(r.final_wealth),actual.wealth.length);close(actual.dd,Number(r.max_drawdown),actual.wealth.length);close(actual.maxStake,Number(r.max_stake),actual.wealth.length);assert.equal(actual.bets,Number(r.bets));}
let states=0;
for(const key of Object.keys(JSON.parse(element('policy-data').textContent))){const [id,sample,model]=key.split('|');element('strategy').value=id;element('sample').value=sample;sandbox.optionsFor();assert(element('model').innerHTML.includes('value="'+model+'"'));element('model').value=model;
 for(const cap of [1,.05,.1,.15,.2]){element('cap').value=String(cap);sandbox.draw();assert(element('chart').innerHTML.includes('<polyline'));assert(!/NaN|Infinity/.test(element('chart').innerHTML));assert(element('metrics').innerHTML.includes('Largest stake'));states++;}}
console.log(JSON.stringify({status:'PASS',paths_reconciled:groups.size,summary_rows:summary.length,rendered_control_states:states},null,2));
