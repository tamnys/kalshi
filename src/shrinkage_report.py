"""Standalone report and interactive sequential paths for market shrinkage."""
import csv
import gzip
import html
import json
from collections import defaultdict
from shrinkage_backtest import ROOT, MODELS, LABELS, PORTFOLIOS, read_csv

NAMES = {'core_bounded': 'High-confidence core (C4–5, 35–81¢)',
         'bounded_core_plus_fade1': 'Core + opposing C1 favorites',
         'bounded_core_plus_narrow_fade1': 'Core + narrow C1 opposing interval',
         'core_fade1_plus_fade2': 'Core + C1 and C2 opposing positions',
         'all_disjoint_candidates': 'All non-overlapping candidates',
         'two_unbeaten_intervals': 'Two retrospectively unbeaten intervals'}


def pct(x):
    return f'{100*float(x):.1f}%'


def table(head, rows):
    cells = lambda tag, values: ''.join(f'<{tag}>{html.escape(str(v))}</{tag}>' for v in values)
    return '<div class="table-wrap"><table><thead><tr>'+cells('th', head)+'</tr></thead><tbody>'+''.join('<tr>'+cells('td', r)+'</tr>' for r in rows)+'</tbody></table></div>'


def run():
    out = ROOT/'results'
    cases = read_csv(out/'shrinkage_scenarios.csv')
    scores = read_csv(out/'shrinkage_scores.csv')
    focal = read_csv(out/'shrinkage_focal_stakes.csv')
    lookup = {(r['portfolio'], r['sample'], r['model'], r['policy'], r['cap_name']): r for r in cases}
    def case(portfolio, model, policy='full', cap='uncapped', sample='all'):
        return lookup[(portfolio, sample, model, policy, cap)]
    def result(r):
        return f"{float(r['final_wealth']):.2f}× / {pct(r['max_drawdown'])}"
    portfolio_table = table(['Strategy', 'Original: wealth / drawdown', 'Shrink confidence, earliest start',
                             'Shrink confidence, established warmup', 'Shrink original, established warmup'],
        [[NAMES[p], *[result(case(p, m)) for m in ('original', 'shrink_confidence', 'shrink_confidence_warm', 'shrink_original_warm')]] for p in PORTFOLIOS])
    core = 'bounded_core_plus_fade1'
    core_table = table(['Kelly fraction', 'Cap', 'Original: wealth / drawdown',
                       'Shrink confidence, established warmup', 'Shrink original, established warmup'],
        [[policy.title(), 'None' if cap == 'uncapped' else cap[-2:]+'%',
          *[result(case(core, m, policy, cap)) for m in ('original', 'shrink_confidence_warm', 'shrink_original_warm')]]
         for policy in ('quarter', 'half', 'full') for cap in ('uncapped', 'cap_05', 'cap_10', 'cap_15', 'cap_20')])
    accuracy = table(['Probability estimate', 'Later 161: log loss ↓', 'All available 182: log loss ↓',
                      'Recovered-only 65: log loss ↓'],
        [[LABELS[m], *[f"{float(next(r['log_loss'] for r in scores if r['model'] == m and r['sample'] == s and r['scope'] == scope)):.4f}"
                       for s, scope in [('all', 'legacy_161'), ('all', 'common_available'), ('observed', 'common_available')]]]
         for m in ('market', *MODELS)])
    focal_table = table(['Model', 'C1 opposing bet: probability / full Kelly', 'C5 backing bet: probability / full Kelly'],
        [[LABELS[m], *[next(f"{pct(r['side_probability'])} / {pct(r['full_kelly'])}" for r in focal
                           if r['sample'] == 'all' and r['model'] == m and int(r['fight_index']) == step)
                       for step in (147, 162)]] for m in MODELS])
    recovered = table(['Strategy', 'Original: wealth / drawdown / bets', 'Shrink confidence: wealth / drawdown / bets',
                       'Shrink original: wealth / drawdown / bets'],
        [[NAMES[p], *[result(case(p, m, sample='observed'))+' / '+case(p, m, sample='observed')['bets']
                     for m in ('original', 'shrink_confidence', 'shrink_original')]] for p in PORTFOLIOS])
    curves = defaultdict(list)
    with gzip.open(out/'shrinkage_paths.csv.gz', 'rt', newline='') as f:
        for r in csv.DictReader(f):
            key = '|'.join(r[k] for k in ('portfolio', 'sample', 'model', 'policy', 'cap_name'))
            curves[key].append(float(r['wealth']))
    payload = json.dumps(dict(curves=curves, cases={'|'.join(k): v for k, v in lookup.items()}, labels=LABELS), separators=(',', ':'))
    options = lambda entries: ''.join(f'<option value="{html.escape(k)}">{html.escape(v)}</option>' for k, v in entries)
    controls = ''.join(f'<label>{label}<select id="{key}">{options(entries)}</select></label>' for label, key, entries in [
        ('Strategy', 'portfolio', list(NAMES.items())),
        ('Price history', 'sample', [('all', 'Recovered + reconstructed'), ('observed', 'Recovered only: fitting and trading')]),
        ('Kelly fraction', 'policy', [('full', 'Full Kelly'), ('half', 'Half Kelly'), ('quarter', 'Quarter Kelly')]),
        ('Per-fight cap', 'cap', [('uncapped', 'No cap'), ('cap_05', '5%'), ('cap_10', '10%'), ('cap_15', '15%'), ('cap_20', '20%')]),
        ('Shrinkage comparison', 'method', [(m, LABELS[m]) for m in MODELS if m.startswith('shrink')]),
        ('Wealth scale', 'scale', [('linear', 'Linear'), ('log', 'Logarithmic')])])
    body = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Market shrinkage and sequential Kelly</title><style>
body{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:auto;padding:28px;background:#f7faf8;color:#203b35}h1,h2{line-height:1.25}a{color:#176c60}.note{background:#e6f0ea;padding:18px;border-left:4px solid #247766}.table-wrap{overflow:auto;margin:22px 0}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{padding:10px;border-bottom:1px solid #dce6df;text-align:left}th{background:#e6eee8}.muted{color:#52695e;font-size:14px}.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}label{font-size:14px}select{display:block;width:100%;padding:9px;font:inherit;background:white;border:1px solid #a9c3b5;border-radius:5px}svg{width:100%;height:auto;background:white;border:1px solid #dce6df}.legend span{display:inline-block;margin:8px 20px 8px 0}.chart{margin:18px 0}code{overflow-wrap:anywhere}@media(max-width:650px){body{padding:16px}h1{font-size:28px}}
</style></head><body><p><a href="interval_trajectories.html">Interval viewer</a> · <a href="capped_kelly.html">Existing cap comparison</a> · <a href="confidence_signal.html">Confidence signal</a></p>
<h1>Shrinking toward the market removes most of the apparent edge</h1><p class="muted">October 24, 2026 · 1,080 sequential paths · six existing strategies · quarter, half and full Kelly · requested 5%, 10%, 15%, 20% caps</p>
<div class="note"><strong>The result depends materially on when we start judging the model's forecasts.</strong> Learning from the earliest identifiable forecasts leads to no bets across the six leading strategies in the full price history. Starting shrinkage learning after the previously established 24-fight training prefix retains some bets, reduces the two focal stakes, and substantially reduces historical growth. Neither version demonstrates better probability accuracy than the market baseline.</div>
<h2>What changed</h2><p>Before each fight, estimate how much of the base model's adjustment deserves to survive. A retention of zero means using the market quote; one means keeping the full adjustment. The amount is chosen by the log score of <strong>strictly earlier forecasts made before their outcomes</strong>, across all confidence levels. It is not chosen using betting returns, the current result, or subsequent outcomes.</p>
<p><code>logit(new probability) = logit(market quote) + retention × [logit(base probability) − logit(market quote)]</code></p>
<p>The primary base model fixes the market log-odds slope at one and adds an intercept plus a confidence adjustment. Both adjustments shrink together toward zero. A second test shrinks the existing odds-plus-confidence model, including its fitted price-slope deviation. These are predictive shrinkage rules; they do not claim to be a fully Bayesian prior/posterior analysis.</p>
<p>The early-start test uses the existing earliest identifiable forecasting start. The warmup sensitivity excludes the first 24 decisive fights only from <em>shrinkage-learning scores</em>; those fights continue to train the base models. This is the prior study's established comparison prefix, not a newly optimized cutoff. No prior shrinkage scores means retaining zero adjustment. The initial choice can matter when the sample is small.</p>
<h2>Growth and drawdown across the leading strategies</h2><p>Full Kelly, no cap, recovered plus reconstructed prices. Each cell gives ending bankroll divided by initial bankroll, followed by maximum peak-to-trough drawdown. 1.00× means no net gain. Every fight settles before sizing the next.</p>''' + portfolio_table + '''
<p>The no-bet result is not a profitable strategy or a risk-adjusted victory. It says that this broad, earlier-forecast accuracy criterion did not earn the model enough weight to clear trading costs. It also gives up every historical gain. The retained warmup results should not be promoted to a preferred strategy solely because they are more profitable.</p>
<h2>The two losses behind the 50.5% drawdown</h2><p>These are the original C1 opposing position at step 147 and C5 backing position at step 162. Entries give probability that the purchased side wins, then the full-Kelly fraction of current bankroll. Half Kelly halves these fractions before any cap.</p>''' + focal_table + '''
<p>With the established warmup, shrinking the original model reduces the two full-Kelly stakes from 13.77% and 42.55% to 2.06% and 11.37%. Shrinking the confidence-only adjustment produces 7.22% and 23.67%. The early-start versions skip both bets. All changes are learned from earlier forecasts, not retroactively from these losses.</p>
<p>The whole-history maximum drawdown can move to a different period. In the core-plus-C1 portfolio, warmup shrinkage of the original model has a 36.8% drawdown earlier in the history, while warmup confidence shrinkage has a 30.2% drawdown extending beyond the second focal loss. Reducing these two stakes does not guarantee a uniformly smoother path.</p>
<h2>Does prediction accuracy improve?</h2>''' + accuracy + '''
<p>On the familiar 161-fight comparison, early-start shrinkage exactly matches the market's 0.6043 log loss because its retained adjustment is zero throughout that window. Warmup shrinkage of the original model scores 0.6112 and warmup confidence shrinkage 0.6092, both worse than the market. The latter is also slightly worse than its own unshrunk confidence-only base (0.6077). Lower is better.</p>
<p>The approximate 95% paired whole-card bootstrap interval for warmup-original improvement over the original model is −0.0143 to +0.0192 log-loss units per fight. For warmup-confidence improvement over its own base, it is −0.0125 to +0.0104. These span improvement and deterioration. They condition on the saved forecast sequences and do not remove model-selection or strategy-selection uncertainty. <a href="https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf">Gneiting and Raftery</a> describe logarithmic and quadratic scores for evaluating probabilistic forecasts.</p>
<h2>Recovered Kalshi quotes only</h2><p>Full Kelly, no cap; both model fitting and betting use recovered quotes. The two learning-start versions coincide here because the recovered-price forecast history begins after the established warmup.</p>''' + recovered + '''
<p>For the core plus C1-fade portfolio, shrunk confidence ends at 1.35× with 3.7% drawdown—but places only three bets. Shrinking the original model ends at 1.65× with zero recorded drawdown from a single winning bet, which risks 59.4% of bankroll. That is not reliable evidence of a safer profitable system. Learned shrinkage can temporarily retain a large adjustment, so it is not a replacement for an explicit cap.</p>
<h2>Sequential trajectories and the requested caps</h2><p>Compare the original model, the unshrunk confidence-only base, and a selected shrinkage rule. The x-axis includes every recorded fight, including skipped positions. Curves start at one unit of bankroll. The tables retain exact saved results; all chart points use the same saved trajectories.</p>''' + controls + '''
<div id="legend" class="legend"></div><div id="metrics"></div><div class="chart" id="wealth"></div><div class="chart" id="drawdown"></div>
<h2>Core plus C1 fades: fraction and cap tradeoffs</h2>''' + core_table + '''
<p>These caps are the previously requested limits, not optimized new recommendations. For example, warmup confidence shrinkage with full Kelly and a 10% cap ends at 1.37× with 17.7% drawdown; the original model under that same cap ends at 3.57× with 19.0% drawdown. At half Kelly without a cap, the corresponding figures are 1.12× / 15.6% and 2.86× / 26.7%.</p>
<h2>Interpretation and limits</h2><p>The experiment weakens the case for trusting aggressive unshrunk Kelly probabilities. It does not establish that there is no useful local signal. One global shrinkage factor is selected for probability accuracy across all confidence levels; a profitable subgroup could coexist with poor global calibration. Investigating such a subgroup is another model-selection exercise, not a reason to overrule this result after seeing profits.</p>
<p>All six category definitions remain retrospectively selected. Quoted asks are probability benchmarks, not known fair probabilities. The reconstructed-price mapping remains retrospective. Recovered quotes are historical pre-card snapshots and do not establish executable fight-time entries. Existing modeled fees, spread, draw settlement, and the user's immediate sequential settlement convention are unchanged. No new data or untouched holdout has been created.</p>
<p>The practical finding is that market shrinkage can suppress implausibly large model adjustments, but this sample does not support a stable shrinkage setting that preserves the earlier aggressive growth while establishing superior forecasts. The differing learning-start results are themselves evidence of uncertainty.</p>
<p>Downloads: <a href="../results/shrinkage_scenarios.csv">all scenario results</a> · <a href="../results/shrinkage_paths.csv.gz">full paths</a> · <a href="../results/shrinkage_scores.csv">prediction scores</a> · <a href="../results/shrinkage_paired_scores.csv">paired uncertainty</a> · <a href="../results/shrinkage_later_periods.csv">later-period resets</a> · <a href="research_log.md">research record</a>.</p>
<script id="shrinkage-data" type="application/json">''' + payload + '''</script>
<script>
const DATA=JSON.parse(document.getElementById('shrinkage-data').textContent);
const COLORS=['#7b4f9b','#a36d16','#147d68'];
function dd(v){let peak=v[0];return v.map(x=>{peak=Math.max(peak,x);return 1-x/peak;});}
function chart(series,title,percent,log){
 const w=1000,h=300,left=70,right=20,top=38,bottom=42;
 const transform=x=>log?Math.log(x):x;
 const all=series.flatMap(s=>s.values.map(transform));
 let lo=Math.min(0,...all),hi=Math.max(...all);
 if(percent){lo=0;hi=Math.max(hi,.01);}else if(hi===lo){hi=lo+1;}
 const X=i=>left+i/(series[0].values.length-1)*(w-left-right);
 const Y=v=>h-bottom-(transform(v)-lo)/(hi-lo)*(h-top-bottom);
 let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${title}"><text x="${left}" y="23" font-size="17">${title}</text>`;
 for(let j=0;j<=4;j++){const v=lo+(hi-lo)*j/4,y=h-bottom-j/4*(h-top-bottom);const shown=log?Math.exp(v):v;svg+=`<line x1="${left}" x2="${w-right}" y1="${y}" y2="${y}" stroke="#e4ebe7"/><text x="${left-8}" y="${y+4}" text-anchor="end" font-size="12">${percent?(100*shown).toFixed(0)+'%':shown.toFixed(2)+'×'}</text>`;}
 series.forEach((s,j)=>{const points=s.values.map((v,i)=>X(i).toFixed(2)+','+Y(v).toFixed(2)).join(' ');svg+=`<polyline fill="none" stroke="${COLORS[j]}" stroke-width="2.5" points="${points}"/>`;});
 [0,47,94,140,187].forEach(i=>{svg+=`<text x="${X(i)}" y="${h-22}" text-anchor="middle" font-size="12">${i}</text>`;});
 return svg+`<text x="${w/2}" y="${h-3}" text-anchor="middle" font-size="12">Settled fight index</text></svg>`;
}
function render(){
 const get=id=>document.getElementById(id).value;
 const models=['original','confidence_offset',get('method')];
 const series=models.map(m=>{const key=[get('portfolio'),get('sample'),m,get('policy'),get('cap')].join('|');return {model:m,values:DATA.curves[key],metrics:DATA.cases[key]};});
 document.getElementById('legend').innerHTML=models.map((m,j)=>`<span style="color:${COLORS[j]}">● ${DATA.labels[m]}</span>`).join('');
 document.getElementById('wealth').innerHTML=chart(series,'Bankroll relative to starting bankroll',false,get('scale')==='log');
 document.getElementById('drawdown').innerHTML=chart(series.map(s=>({...s,values:dd(s.values)})),'Drawdown from the previous peak',true,false);
 document.getElementById('metrics').innerHTML='<div class="table-wrap"><table><tr><th>Model</th><th>Ending bankroll</th><th>Max drawdown</th><th>Largest stake</th><th>Bets</th></tr>'+series.map(s=>`<tr><td>${DATA.labels[s.model]}</td><td>${Number(s.metrics.final_wealth).toFixed(3)}×</td><td>${(100*s.metrics.max_drawdown).toFixed(1)}%</td><td>${(100*s.metrics.max_stake).toFixed(1)}%</td><td>${s.metrics.bets}</td></tr>`).join('')+'</table></div>';
}
document.getElementById('portfolio').value='bounded_core_plus_fade1';
document.querySelectorAll('select').forEach(s=>s.addEventListener('change',render));render();
</script></body></html>'''
    (ROOT/'reports/shrinkage_kelly.html').write_text(body)
    print('Saved reports/shrinkage_kelly.html with', len(curves), 'complete curves', flush=True)


if __name__ == '__main__':
    run()
