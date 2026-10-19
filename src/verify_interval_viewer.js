// Exercise the generated report's actual script without external packages.
// This checks data/control/render behavior, not browser layout or accessibility.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname, '../reports/interval_trajectories.html'), 'utf8');
const payload = source.match(/<script id="analysis-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script = source.match(/<script>([\s\S]*?)<\/script>/)[1];
const data = JSON.parse(payload);
const element = () => ({value: '', textContent: '', innerHTML: '', children: [], listeners: {},
  appendChild(child) { this.children.push(child); },
  addEventListener(name, handler) { this.listeners[name] = handler; }});
const elements = Object.fromEntries(['analysis-data', 'category', 'prefix', 'category-detail', 'trajectory', 'path-stats'].map(k => [k, element()]));
elements['analysis-data'].textContent = payload;
elements.prefix.value = '3';
const context = vm.createContext({document: {getElementById: id => elements[id], createElement: element}});
vm.runInContext(script, context);
assert.equal(elements.category.children.length, data.candidates.length);
let states = 0;
for (const candidate of data.candidates) {
  elements.category.value = candidate.interval_id;
  for (const prefix of ['3', '24']) {
    elements.prefix.value = prefix;
    elements.category.listeners.change();
    elements.prefix.listeners.change();
    const curves = [...elements.trajectory.innerHTML.matchAll(/<polyline[^>]*points="([^"]+)"/g)];
    assert.equal(curves.length, 3);
    for (const [, points] of curves) {
      assert.equal(points.split(' ').length, 188);
      assert.ok(!/NaN|Infinity|undefined/.test(points));
    }
    assert.equal((elements['path-stats'].innerHTML.match(/<tr>/g) || []).length, 4);
    assert.ok(elements['category-detail'].textContent.includes(candidate.wins + '/' + candidate.n));
    states++;
  }
}
for (const [prefix, expected] of [['3', '-7.2%'], ['24', '10.2%']]) {
  elements.category.value = 'core_all'; elements.prefix.value = prefix;
  elements.prefix.listeners.change();
  assert.ok(elements['path-stats'].innerHTML.includes(expected));
}
console.log(`PASS: ${states} category/training control states; three complete finite paths per state; early-versus-later training results`);
