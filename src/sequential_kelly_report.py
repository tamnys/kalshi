"""Standalone sequential strategy comparison and interactive fight trajectories."""
import html
import json
import numpy as np
import pandas as pd
from audit import ROOT
from analyze import plt
from report import table, pct, image
from aggressive_growth_report import LABELS, MODELS
from sequential_kelly import context_inputs

FOCUS = ['core_bounded', 'bounded_core_plus_fade1', 'bounded_core_plus_narrow_fade1',
         'core_fade1_plus_fade2', 'all_disjoint_candidates', 'two_unbeaten_intervals']
SHORT = {**LABELS, 'core_bounded': 'High-confidence core',
         'bounded_core_plus_fade1': 'Core + C1 favorite fades',
         'bounded_core_plus_narrow_fade1': 'Core + C1 fades at 25–70¢',
         'core_fade1_plus_fade2': 'Core + C1 favorite fades + C2 fades',
         'all_disjoint_candidates': 'All five candidate groups',
         'two_unbeaten_intervals': 'Two narrow undefeated intervals'}
POLICIES = {'quarter':'Quarter Kelly', 'half':'Half Kelly', 'full':'Full Kelly',
            'learned_from_prior':'Fraction learned from earlier outcomes',
            'hindsight_best_feasible':'Best fixed fraction in hindsight'}


def run():
    scenarios = pd.read_csv(ROOT/'results/sequential_kelly_scenarios.csv')
    paths = pd.read_csv(ROOT/'results/sequential_kelly_paths.csv.gz')
    frozen = pd.read_csv(ROOT/'results/sequential_kelly_frozen.csv')
    bootstrap = pd.read_csv(ROOT/'results/sequential_kelly_bootstrap.csv')
    meta = json.loads((ROOT/'results/sequential_kelly_summary.json').read_text())
    lookup = scenarios.set_index(['portfolio', 'sample', 'model', 'policy'])

    def get(name, sample='all', model='odds_confidence', policy='half'):
        return lookup.loc[(name, sample, model, policy)]

    def wealth_dd(row):
        return f'{row.final_wealth:,.3f}× / {pct(row.max_drawdown)}'

    def fixed_table(sample):
        rows = []
        for name in FOCUS:
            choices = [get(name, sample, policy=p) for p in ('quarter', 'half', 'full')]
            rows.append([SHORT[name], int(choices[0].bets), *map(wealth_dd, choices), pct(choices[2].max_stake)])
        return table(rows, ['Strategy', 'Bets', '¼ Kelly: wealth / drawdown',
                            '½ Kelly: wealth / drawdown', 'Full Kelly: wealth / drawdown', 'Largest full-Kelly stake'])

    model_rows, learned_rows, bootstrap_rows = [], [], []
    for sample in ('all', 'observed'):
        for name in FOCUS:
            model_rows.append([SHORT[name], sample,
                               *[wealth_dd(get(name, sample, model)) for model in MODELS]])
            learned = get(name, sample, policy='learned_from_prior')
            learned_rows.append([SHORT[name], sample, wealth_dd(get(name, sample)),
                                 wealth_dd(get(name, sample, policy='full')), wealth_dd(learned),
                                 int(learned.bets), pct(learned.max_stake)])
            row = bootstrap[bootstrap.portfolio.eq(name) & bootstrap['sample'].eq(sample)
                            & bootstrap.model.eq('odds_confidence') & bootstrap.policy.eq('half')].iloc[0]
            bootstrap_rows.append([SHORT[name], sample, f'{row.wealth_p025:,.3f}–{row.wealth_p975:,.3f}×',
                                   pct(row.empirical_loss_frequency, 2) if row.has_recorded_loss else 'No losing bet to resample',
                                   pct(row.drawdown_p95) if row.has_recorded_loss else 'Uninformative zero',
                                   int(row.recorded_cards_with_positions)])
    models_table = table(model_rows, ['Strategy', 'Price sample', *[v+': wealth / drawdown' for v in MODELS.values()]])
    learned_table = table(learned_rows, ['Strategy', 'Price sample', 'Fixed half: wealth / DD',
                                       'Fixed full: wealth / DD', 'Learned fraction: wealth / DD',
                                       'Learned-policy bets', 'Largest learned-policy stake'])
    bootstrap_table = table(bootstrap_rows, ['Strategy', 'Price sample', 'Central 95% resampled wealth range',
                                            'Resampled loss frequency', '95th percentile max drawdown', 'Cards with positions'])
    frozen_rows = []
    for name in ('bounded_core_plus_fade1', 'all_disjoint_candidates'):
        for sample in ('all', 'observed'):
            for split in meta['splits']:
                subset = frozen[frozen.portfolio.eq(name) & frozen['sample'].eq(sample)
                                & frozen.model.eq('odds_confidence') & frozen.split.eq(split)].set_index('policy')
                prior, half = subset.loc['prior_fitted'], subset.loc['half']
                frozen_rows.append([SHORT[name], sample, split, int(prior.training_opportunities),
                                    pct(prior.fraction, 2), wealth_dd(prior), wealth_dd(half), int(prior.bets)])
    frozen_table = table(frozen_rows, ['Strategy', 'Price sample', 'Test begins', 'Earlier opportunities',
                                     'Fraction fitted earlier', 'Later wealth / DD', 'Same-window half Kelly', 'Later bets'])
    all_table = table([[SHORT[r.portfolio], r.sample, MODELS[r.model], POLICIES[r.policy],
                       pct(r.fraction, 2) if pd.notna(r.fraction) else 'Changes after each fight',
                       f'{r.final_wealth:,.3f}×', pct(r.max_drawdown), int(r.bets), pct(r.max_stake)]
                      for r in scenarios.itertuples()],
                     ['Strategy', 'Sample', 'Probability model', 'Sizing policy', 'Kelly multiplier',
                      'Final wealth', 'Max drawdown', 'Bets', 'Largest individual stake'])

    full, contexts = context_inputs()
    payload = {}
    for context, f, returns, _, _ in contexts:
        mask = np.logical_and.reduce([paths[k].eq(v).to_numpy() for k, v in context.items()])
        learned = paths[mask & paths.policy.eq('learned_from_prior')].sort_values('fight_index')
        payload['|'.join(context.values())] = dict(f=f.tolist(), z=(f*returns).tolist(),
            learned=learned.multiplier.to_numpy()[1:].tolist(),
            optimum=float(get(context['portfolio'], context['sample'], context['model'], 'hindsight_best_feasible').fraction))
    payload_json = json.dumps(payload, separators=(',', ':'), allow_nan=False)
    options = ''.join(f'<option value="{key}"'+(' selected' if key == 'bounded_core_plus_fade1' else '')+f'>{html.escape(value)}</option>' for key, value in LABELS.items())
    model_options = ''.join(f'<option value="{key}">{value}</option>' for key, value in MODELS.items())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {'quarter':'#8b9c40', 'half':'#168278', 'full':'#2870b9', 'learned_from_prior':'#b45b28'}
    for ax, sample in zip(axes, ('all', 'observed')):
        for policy, color in colors.items():
            p = paths[paths.portfolio.eq('bounded_core_plus_fade1') & paths['sample'].eq(sample)
                      & paths.model.eq('odds_confidence') & paths.policy.eq(policy)].sort_values('fight_index')
            ax.plot(p.fight_index, p.wealth, color=color, label=POLICIES[policy],
                    linestyle='--' if policy == 'learned_from_prior' else '-')
        ax.axhline(1, color='#666', lw=.6)
        ax.set(title='All prices' if sample == 'all' else 'Recovered quotes only',
               xlabel='Recorded fight index', ylabel='Bankroll / starting bankroll')
        ax.grid(alpha=.15); ax.legend(fontsize=7)
    fig.suptitle('Core + C1 favorite fades: settle each fight before the next bet')
    fig.tight_layout(); fig.savefig(ROOT/'reports/figures/sequential_kelly.png', dpi=170); plt.close(fig)

    content = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sequential fight-by-fight Kelly strategies</title><style>
body{{font:16px/1.65 system-ui,sans-serif;max-width:1240px;margin:auto;padding:28px;background:#f7faf8;color:#203b35}}h1,h2{{line-height:1.25}}a{{color:#176c60}}.note{{padding:18px;background:#e6f0ea;border-left:4px solid #247766}}.muted,figcaption{{font-size:14px;color:#52695e}}.table-wrap{{overflow-x:auto;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}th,td{{padding:10px;border-bottom:1px solid #dce6df;text-align:left;vertical-align:top}}th{{background:#e6eee8}}.controls{{display:flex;flex-wrap:wrap;gap:18px;margin:18px 0}}label{{display:block}}select,input{{font:inherit;max-width:100%;padding:5px}}select{{width:100%}}svg,img{{width:100%;height:auto}}.chart{{background:white;border:1px solid #dce6df;border-radius:6px;margin:16px 0}}figure{{margin:24px 0}}details{{margin:24px 0}}code{{background:#e5ede7;padding:2px 5px}}@media(max-width:650px){{body{{padding:16px}}h1{{font-size:27px}}}}
</style></head><body>
<p><a href="final_report.html">Main report</a> · <a href="interval_trajectories.html">Interval viewer</a> · <a href="lower_confidence.html">Confidence 1–3</a> · <a href="aggressive_growth.html">Aggressive-growth explanation</a></p>
<h1>Sequential fight-by-fight Kelly strategies</h1><p class="muted">October 19, 2026 · {meta['scenarios']} strategies · {meta['frozen_comparisons']:,} later-period sizing comparisons · {meta['bootstrap_comparisons']} card-resampling sensitivities</p>
<p><strong>Requested stake caps:</strong> the <a href="capped_kelly.html">5%, 10%, 15% and 20% cap comparison</a> shows the growth sacrificed and drawdown reduced across the leading portfolios, with complete capped trajectories.</p>
<div class="note"><strong>One fight settles before the next bet.</strong> The next stake uses the updated bankroll and a forecast fitted only to earlier settled results. The pooled high-confidence core plus confidence-1 favorite fades produces <strong>{get('bounded_core_plus_fade1').final_wealth:.3f}× at half Kelly</strong>, with <strong>{pct(get('bounded_core_plus_fade1').max_drawdown)} maximum drawdown</strong>. Adding all five candidate groups gives <strong>{get('all_disjoint_candidates').final_wealth:.3f}× at half Kelly</strong>, with the same historical maximum drawdown. Recovered-price and model checks show that more aggressive sizing is not consistently better.</div>
<p>All bankrolls start at 1. A result of 2.858× means $1,000 would become approximately $2,858 under these assumptions. Drawdown is the largest decline from an earlier bankroll peak. Each stake percentage is the maximum all-in amount risked on one fight as a share of that strategy's bankroll immediately before the fight. Half Kelly can still involve a large stake; <a href="aggressive_growth.html">the 27.5% example is explained here</a>.</p>
<h2>Strategies and full-history results</h2>
<p>The <strong>high-confidence core</strong> backs confidence 4–5 at YES asks of 35–81¢. <strong>C1 favorite fades</strong> buy the opposite side when the original confidence-1 pick's YES ask is at least 50¢. The narrower alternative fades original asks of 25–70¢. <strong>C2 fades</strong> use the previously discovered original-pick ask interval of approximately 36.644–62¢. The <strong>five-group combination</strong> adds confidence-2 backing at 16–36¢ and confidence-3 backing at 77 to below 86¢ to the core and those two fade groups. All constituent groups are disjoint, with at most one position per fight.</p>
<p>The <strong>two undefeated intervals</strong> combine confidence 3–5 backing at 35 to below 40¢ with confidence-1 fading at original asks of 25–70¢. Their complete qualifying samples were 8/8 and 6/6; Kelly places fewer bets because it waits for a usable estimate and skips estimated nonpositive edges.</p>
<p>These tables use the pooled odds/confidence probability model. “All” combines recovered and reconstructed Kalshi quotes. Fractions are existing comparison points, not a selected risk tolerance.</p>
{fixed_table('all')}
<p>The five-group combination reaches {get('all_disjoint_candidates', policy='full').final_wealth:.3f}× at full Kelly. That is a substantial historical gain, but its maximum drawdown is {pct(get('all_disjoint_candidates', policy='full').max_drawdown)} and its largest individual stake is {pct(get('all_disjoint_candidates', policy='full').max_stake)}. The confidence-2 underdog addition includes only one executed bet in the full-history pooled model. The equal drawdowns across several combinations reflect this particular sequence, not a general guarantee that extra categories add no risk.</p>
<h2 id="drawdown">What caused the 50.5% drawdown?</h2>
<p>Two consecutive executed bets lost, with no other positions between them in these three broader portfolios. First, opposing a confidence-1 favorite lost <strong>13.7683%</strong> of the then-current bankroll. Next, backing a confidence-5 pick lost <strong>42.5530%</strong> of the remaining bankroll. The combined decline is <code>1 − (1 − 0.137683) × (1 − 0.425530) = 50.4625%</code>. Each stake was sized after the previous fight settled.</p>
<p>For the core plus C1 favorite fades, an illustrative $1,000 initial bankroll had grown to a peak of <strong>$8,563.80</strong>, then fell to <strong>$7,384.71</strong> after the first loss and <strong>$4,242.30</strong> after the second. It later finished at $5,903.24. The 50.5% is a loss from the prior peak, not from the initial $1,000. The five-group portfolio followed the same percentage decline from $21,116.57 to $10,460.63 and eventually finished at $16,474.13.</p>
<p>The main driver was the second stake: its fitted win probability was 83.05%, versus a 70.50¢ all-in cost, implying 42.55% estimated full Kelly. Both losing trades used recovered quotes; their probability estimates were fitted on the full price history, including reconstructed prices. Half Kelly reduced the two stakes to 6.88% and 21.28%, giving a combined <strong>26.70%</strong> decline. Quarter Kelly reduced it to <strong>13.71%</strong>. The narrower 25–70¢ C1 fade rule excluded the first loss at an 84¢ original-pick ask, leaving a 42.55% full-Kelly decline from the second loss alone. That exclusion was selected retrospectively.</p>
<p class="muted">The <a href="../results/sequential_drawdown_attribution.json">saved attribution</a> reproduces the decline by multiplying the remaining bankroll fractions. Identifying fight details are retained in the local private analysis files.</p>
<h2>What survives using recovered Kalshi quotes only?</h2>
<p>Here both probability fitting and trading use only recovered quotes. This changes the training sample as well as which fights are bet; it is not merely deletion of reconstructed-price trades from a fixed model. The same categories were selected using the wider study, so this is a source-sensitivity check rather than an untouched validation sample.</p>
{fixed_table('observed')}
<p>For the core plus favorite fades, full Kelly falls from {get('bounded_core_plus_fade1', policy='full').final_wealth:.3f}× in the full price history to {get('bounded_core_plus_fade1', 'observed', policy='full').final_wealth:.3f}× with recovered quotes only, with {pct(get('bounded_core_plus_fade1', 'observed', policy='full').max_drawdown)} drawdown. Half Kelly returns {get('bounded_core_plus_fade1', 'observed').final_wealth:.3f}×. The narrow 25–70¢ C1 fade combined with the core is stronger in both price samples, but it excludes a losing fade just above the selected upper boundary.</p>
<h2>Explore the complete fight trajectories</h2>
<p>Choose any of the twenty portfolios, three probability models and two price samples. Each plot includes all {meta['recorded_rows']} recorded rows, including skipped fights. The custom multiplier recomputes the entire path, including stake sizes, using the same prior-only forecasts.</p>
<div class="controls"><label>Portfolio<select id="portfolio">{options}</select></label><label>Price sample<select id="sample"><option value="all">All: recovered + reconstructed</option><option value="observed">Recovered quotes only</option></select></label><label>Probability model<select id="model">{model_options}</select></label></div>
<div class="controls"><label><input type="checkbox" id="logScale"> Logarithmic bankroll scale</label><label><input type="checkbox" id="showOracle"> Include best fixed fraction in hindsight</label><label><input type="checkbox" id="showCustom"> Include custom fraction</label><label>Custom Kelly multiplier <input id="customFraction" type="number" min="0" max="1" step="any" value="0.5"></label></div>
<p id="contextLabel" class="muted"></p><div id="viewerSummary" class="table-wrap"></div>
<h3>Bankroll after each settlement</h3><div id="wealthChart" class="chart"></div><h3>Drawdown from the previous peak</h3><div id="drawdownChart" class="chart"></div>
<p class="muted">Blue: full Kelly. Green: half Kelly. Olive: quarter Kelly. Orange dashed: fraction learned only from earlier outcomes. The optional purple line uses hindsight; the black line uses your custom multiplier. These plots assume the saved price is obtainable when each next bet is placed.</p>
<h2>Does learning the Kelly fraction from earlier results help?</h2>
<p>The new learned-fraction policy chooses a common multiplier in [0,1] that maximizes log growth over strictly earlier opportunities, then uses that multiplier for the next fight. It starts with zero when there is no evidence or the objective is flat. It can learn from a prior observable result even if it skipped the bet. Probability forecasts also continue updating after each settlement. This is an earlier-data rule for sizing; the candidate categories themselves remain retrospective.</p>
{learned_table}
<p>The core plus favorite fades reaches <strong>{get('bounded_core_plus_fade1', policy='learned_from_prior').final_wealth:.3f}×</strong> with the learned fraction in the full price history, but only <strong>{get('bounded_core_plus_fade1', 'observed', policy='learned_from_prior').final_wealth:.3f}×</strong> using recovered quotes. The five-group combination similarly moves from {get('all_disjoint_candidates', policy='learned_from_prior').final_wealth:.3f}× to {get('all_disjoint_candidates', 'observed', policy='learned_from_prior').final_wealth:.3f}×. Optimizing the multiplier repeatedly from a small prior sample is not consistently better than fixing a fraction.</p>
{image('sequential_kelly.png','Each fight settles before the next stake is calculated. Both panels use the same portfolio and probability-model family; the right panel fits and trades only recovered quotes.')}
<h2>Fit a fraction earlier, then hold it fixed</h2>
<p>The next check fits the multiplier before the two natural date splits already used in the research, then freezes it throughout the later period. Probability forecasts still update fight by fight. Every result below restarts at 1 at the stated test date; compare it with the half-Kelly result over that same window, not with whole-history wealth.</p>
{frozen_table}
<p>The full-price-history estimates choose full Kelly for these broader combinations, and that fraction grows the later bankroll. Before October 10, 2026, the recovered-price core/favorite-fade history instead chooses approximately 49.41% of Kelly; it produces 1.159× over the next five bets, almost the same as fixed half Kelly. The five-group version chooses approximately 46.14%, producing 1.206× versus 1.221× at fixed half Kelly. There are no recovered-price positive-Kelly opportunities before the earlier split, so its fitted fraction stays zero. A no-bet result there is lack of sizing evidence, not proof that the category cannot profit.</p>
<h2>How sensitive are the results to the recorded cards?</h2>
<p>For each fixed-fraction strategy, {meta['bootstrap_replicates']:,} replicates draw {meta['bootstrap_cards_per_replica']} whole cards with replacement, preserving the fight order within each sampled card and settling every fight sequentially. The model forecasts and category choices are held fixed. The ranges below are empirical resampling ranges; the loss frequencies are <strong>not probabilities of future loss, not search-adjusted significance, and not a model-refitting experiment</strong>.</p>
{bootstrap_table}
<p>For the full-price-history core plus favorite fades at half Kelly, the central resampled range is 1.056–7.107× and 1.96% of replicates finish below the starting bankroll. With recovered quotes only, the range becomes 0.592–6.084× and the loss frequency rises to 21.66%. Adding both confidence-2 fades and the other candidate groups raises full-history growth but does not uniformly improve the recovered-price resampling distribution. Repeated cards can reproduce the same favorable outcomes; they do not provide new evidence.</p>
<p>The undefeated categories cannot generate a losing bet in this empirical bootstrap, so their zero drawdown and loss-frequency estimates are uninformative about unseen losses. At category-offset full Kelly, their largest actual allocation is 88.9% of bankroll. A single loss at such a stake would remove 88.9% immediately, regardless of the zero historical drawdown.</p>
<h2>How much does the probability model matter?</h2>
<p>These are all half-Kelly results. The pooled model treats confidence as an ordinal predictor alongside price; the separate-confidence model estimates distinct confidence effects; the within-category model learns each chosen category's probability adjustment from its own earlier outcomes.</p>
{models_table}
<p>The core plus the narrower C1 fades remains profitable in all six half-Kelly model/price combinations, ranging from 1.653× to 15.484×. That consistency is a reason to retain it as a candidate, while the large spread in growth and drawdown shows that stake sizing is uncertain. The original 25–70¢ fade category contains only six qualifying wins and was discovered retrospectively.</p>
<p>The two undefeated intervals are the extreme case: category-offset half Kelly reaches 187.927× and full Kelly 6,595.711× in the full price history, but only 3.635× and 9.224× on four recovered-price positions. These are mathematically consistent historical paths. They do not establish the unusually high probabilities needed to make their large stakes optimal in future fights.</p>
<h2>What the comparisons support</h2>
<p><strong>The simpler aggressive candidate</strong> remains confidence-4–5 backing at 35–81¢ plus confidence-1 favorite fades, using one pooled probability model. Quarter and half Kelly show the growth/drawdown tradeoff without pretending that a preferred risk tolerance was supplied. Full Kelly is fundable sequentially and wins on the full price history, but is substantially worse than half Kelly in the recovered-price sensitivity.</p>
<p><strong>The stronger narrow candidate in these comparisons</strong> replaces the broad favorite fades with C1 fades at original asks of 25–70¢. It improves both price-sample results and stays profitable across the tested half-Kelly model variants. Its sharp, retrospectively chosen boundary needs genuinely new outcomes before it should dominate confidence in the strategy.</p>
<p><strong>The five-group combination is the more aggressive exploratory portfolio.</strong> At pooled half Kelly it gives 5.027× over 49 bets in the full price history and 1.709× over 18 recovered-price bets. The extra groups raise historical growth, but their benefits vary with the source and model. They do not establish a superior future strategy. The undefeated pair remains a separate high-upside hypothesis whose extreme growth should not dictate probability estimates.</p>
<p>Re-estimating the multiplier after every fight is not supported as an automatic improvement by these checks. Maintain one bankroll, update the model only from settled outcomes, price the actual purchased side including fees, and avoid overlapping positions. Kelly's growth objective and a drawdown constraint are distinct, as discussed in <a href="https://arxiv.org/abs/1603.06183">Busseti, Ryu and Boyd's risk-constrained Kelly study</a>. The present results show the tradeoffs without inventing a drawdown target or stake cap.</p>
<h2>Scope, accounting and reproducibility</h2>
<p>The existing audited dataset contains 185 decisive fights, one draw and one cancellation across 15 cards. Chronology follows the audited ESPN fight order. Missing Kalshi prices remain reconstructed; forecasts do not establish executable later-fight quotes or market depth. Opposite-side asks use one minus the YES bid, following <a href="https://docs.kalshi.com/getting_started/orderbook_responses">Kalshi's reciprocal orderbook pricing</a>. No live bets were placed.</p>
<p>All 600 paths were independently reconstructed through cash spent, contract quantities and settlement payouts. The 480 previously computed scenarios reconcile with earlier outputs. Every learned-fraction decision was checked against strictly earlier returns, and frozen-fraction optimality and resampling distributions were separately checked. Earlier search-adjusted results still do not establish a statistically reliable profitable category; additional sizing comparisons cannot turn the same retrospective sample into independent evidence.</p>
<details><summary>All 600 whole-history strategy summaries</summary>{all_table}</details>
<p>Saved data: <a href="../results/sequential_kelly_scenarios.csv">strategy summaries</a> · <a href="../results/sequential_kelly_paths.csv.gz">complete paths</a> · <a href="../results/sequential_kelly_frontiers.csv">fraction frontiers</a> · <a href="../results/sequential_kelly_frozen.csv">all 1,200 frozen comparisons</a> · <a href="../results/sequential_kelly_bootstrap.csv">resampling results</a> · <a href="../results/sequential_kelly_verification.json">verification</a> · <a href="research_log.md">research record</a></p>
<script id="sequentialData" type="application/json">{payload_json}</script>
<script>'''+JAVASCRIPT+'''</script></body></html>'''
    (ROOT/'reports/sequential_kelly.html').write_text(content)
    print('Wrote reports/sequential_kelly.html', flush=True)


JAVASCRIPT = r'''
const data=JSON.parse(document.getElementById('sequentialData').textContent);
const names={quarter:'Quarter Kelly',half:'Half Kelly',full:'Full Kelly',learned_from_prior:'Learned from earlier outcomes',hindsight_best_feasible:'Best fixed fraction in hindsight',custom:'Custom fraction'};
const colors={quarter:'#8b9c40',half:'#168278',full:'#2870b9',learned_from_prior:'#b45b28',hindsight_best_feasible:'#7c4ba6',custom:'#182d25'};
const number=x=>x.toLocaleString('en-US',{maximumFractionDigits:3});
const percent=x=>(100*x).toFixed(1)+'%';
function calculate(context,policy,custom=.5){
 const fixed={quarter:.25,half:.5,full:1,hindsight_best_feasible:context.optimum,custom};
 const wealth=[1],drawdown=[0],stake=[],multiplier=[];let peak=1,bets=0,maxStake=0,maxDD=0;
 for(let i=0;i<context.f.length;i++){
  const m=policy==='learned_from_prior'?context.learned[i]:fixed[policy];
  const f=m*context.f[i];multiplier.push(m);stake.push(f);if(f>0)bets++;maxStake=Math.max(maxStake,f);
  const w=wealth[wealth.length-1]*(1+m*context.z[i]);wealth.push(w);peak=Math.max(peak,w);
  const dd=1-w/peak;drawdown.push(dd);maxDD=Math.max(maxDD,dd);
 }
 return {wealth,drawdown,stake,multiplier,bets,maxStake,maxDD,finalWealth:wealth[wealth.length-1]};
}
function svgChart(series,field,logarithmic){
 const width=1000,height=280,left=76,right=24,top=20,bottom=42;
 const transform=v=>logarithmic?Math.log10(v):v;
 let low=field==='drawdown'||!logarithmic?0:Math.min(...series.flatMap(s=>s.result[field].map(transform)));
 let high=Math.max(...series.flatMap(s=>s.result[field].map(transform)));
 if(high===low)high=low+1;
 const last=series[0].result[field].length-1;
 const X=i=>left+i/last*(width-left-right),Y=v=>height-bottom-(transform(v)-low)/(high-low)*(height-top-bottom);
 let body='';
 for(const fraction of [0,.5,1]){
  const value=low+(high-low)*fraction,y=height-bottom-fraction*(height-top-bottom);
  const label=field==='drawdown'?percent(value):number(logarithmic?10**value:value)+'×';
  body+=`<line x1="${left}" y1="${y}" x2="${width-right}" y2="${y}" stroke="#dce6df"/><text x="${left-10}" y="${y+4}" text-anchor="end" font-size="12" fill="#52695e">${label}</text>`;
  const ix=Math.round(last*fraction),x=X(ix);
  body+=`<text x="${x}" y="${height-14}" text-anchor="middle" font-size="12" fill="#52695e">${ix}</text>`;
 }
 for(const seriesItem of series){
  const dash=['learned_from_prior','hindsight_best_feasible','custom'].includes(seriesItem.policy)?' stroke-dasharray="7 4"':'';
  const points=seriesItem.result[field].map((v,i)=>`${X(i).toFixed(2)},${Y(v).toFixed(2)}`).join(' ');
  body+=`<polyline points="${points}" fill="none" stroke="${colors[seriesItem.policy]}" stroke-width="2"${dash}/>`;
 }
 return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${field==='wealth'?'Bankroll':'Drawdown'} after each fight settlement">${body}</svg>`;
}
function update(){
 const portfolio=document.getElementById('portfolio').value,sample=document.getElementById('sample').value,model=document.getElementById('model').value;
 const context=data[[portfolio,sample,model].join('|')];
 const input=document.getElementById('customFraction');let custom=Number(input.value);
 if(!Number.isFinite(custom)||custom<0||custom>1){input.setCustomValidity('Use a fractional-Kelly multiplier from 0 to 1.');return;}
 input.setCustomValidity('');
 const policies=['quarter','half','full','learned_from_prior'];
 if(document.getElementById('showOracle').checked)policies.push('hindsight_best_feasible');
 if(document.getElementById('showCustom').checked)policies.push('custom');
 const series=policies.map(policy=>({policy,result:calculate(context,policy,custom)}));
 document.getElementById('contextLabel').textContent='Each stake uses the current settled bankroll. '+(sample==='observed'?'Training and trading use recovered quotes only.':'Training and trading use recovered and reconstructed prices.');
 let rows=series.map(s=>{const r=s.result;const suffix=s.policy==='custom'?` (${percent(custom)} of Kelly)`:s.policy==='hindsight_best_feasible'?` (${percent(context.optimum)} of Kelly)`:'';
  return `<tr><td style="color:${colors[s.policy]}">${names[s.policy]}${suffix}</td><td>${number(r.finalWealth)}×</td><td>${percent(r.maxDD)}</td><td>${percent(r.maxStake)}</td><td>${r.bets}</td></tr>`;}).join('');
 document.getElementById('viewerSummary').innerHTML='<table><thead><tr><th>Sizing</th><th>Final bankroll</th><th>Max drawdown</th><th>Largest individual stake</th><th>Bets</th></tr></thead><tbody>'+rows+'</tbody></table>';
 document.getElementById('wealthChart').innerHTML=svgChart(series,'wealth',document.getElementById('logScale').checked);
 document.getElementById('drawdownChart').innerHTML=svgChart(series,'drawdown',false);
}
for(const id of ['portfolio','sample','model','logScale','showOracle','showCustom','customFraction'])document.getElementById(id).addEventListener('change',update);
window.sequentialViewer={data,calculate,update,svgChart};
update();
'''


if __name__ == '__main__':
    run()
