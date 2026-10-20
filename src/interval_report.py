"""Standalone follow-up report with selectable, complete fight-level trajectories."""
import base64
import html
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/mma-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit import ROOT
from report import table, pct


def label(r):
    c = str(int(r.confidence_low)) if r.confidence_low == r.confidence_high else f'{int(r.confidence_low)}–{int(r.confidence_high)}'
    known = {'core_all': 'all prices', 'core_favorites': 'above 50¢', 'core_underdogs': 'below 50¢',
             'readable_band': '35¢ to below 40¢', 'core_above_35': '35¢ and above'}
    band = known.get(r.interval_id, f'{100*r.price_low:.3f}–{100*r.price_high:.3f}¢')
    return f'Confidence {c}, {band}'


def run():
    meta = json.loads((ROOT / 'results/interval_trajectory_summary.json').read_text())
    details = pd.read_csv(ROOT / 'results/interval_selected_details.csv')
    kelly = pd.read_csv(ROOT / 'results/interval_kelly_selected.csv')
    paths = pd.read_csv(ROOT / 'results/interval_bankroll_paths.csv.gz')
    leaders = pd.read_csv(ROOT / 'results/interval_confidence_leaders.csv')
    risk = pd.read_csv(ROOT / 'results/interval_kelly_return_drawdown_frontier.csv')
    sensitivity = pd.read_csv(ROOT / 'results/interval_price_sensitivity.csv')
    details['label'] = [label(r) for r in details.itertuples()]
    by_id = details.set_index('interval_id')
    primary = kelly[(kelly.training_prefix == meta['earliest_identifiable_prefix']) & (kelly.kelly_fraction == .25)].set_index('interval_id')
    primary_risk = risk[(risk.training_prefix == meta['earliest_identifiable_prefix']) & (risk.kelly_fraction == .25)]
    best = primary_risk.sort_values('final_wealth').iloc[-1].interval_id
    low = leaders[(leaders.confidence_low == 1) & (leaders.confidence_high == 2)].iloc[0].interval_id
    featured = list(dict.fromkeys([best, 'core_above_35', 'readable_band', low, 'core_all']))
    rows = []
    for iid in featured:
        d = by_id.loc[iid]; k = primary.loc[iid]
        rows.append([html.escape(d.label), f'{int(d.wins)}/{int(d.n)}', pct(d.roi), int(k.bets),
                     pct(k.bankroll_return), pct(k.max_drawdown), f'{d.both_directions_adjusted_p:.3f}'])
    main_table = table(rows, ['Confidence and Kalshi ask', 'Flat-stake wins / bets', 'Flat-stake net ROI',
                              'Kelly bets', 'Quarter-Kelly bankroll return', 'Maximum drawdown', 'Search-adjusted p*'])
    comp = kelly[kelly.interval_id.isin([best, 'core_above_35', 'readable_band', 'core_all']) &
                 kelly.training_prefix.eq(meta['earliest_identifiable_prefix'])]
    comparison = table([[html.escape(by_id.loc[r.interval_id, 'label']), pct(r.kelly_fraction, 0),
                         int(r.bets), pct(r.bankroll_return), pct(r.max_drawdown), pct(r.max_stake)]
                        for r in comp.itertuples()], ['Category', 'Kelly fraction', 'Bets', 'Bankroll return',
                                                       'Max drawdown', 'Largest single stake / bankroll'])
    leadrows = []
    for r in leaders.itertuples():
        d = by_id.loc[r.interval_id]; k = primary.loc[r.interval_id]
        leadrows.append([html.escape(d.label), f'{int(d.wins)}/{int(d.n)}', pct(d.roi),
                         pct(k.bankroll_return), pct(k.max_drawdown), f'{d.both_directions_adjusted_p:.3f}'])
    leadtable = table(leadrows, ['Strongest standardized signal per confidence range', 'Wins / bets', 'Flat-stake ROI',
                                'Quarter-Kelly return', 'Max drawdown', 'Search-adjusted p*'])
    riskrows = []
    for r in primary_risk.itertuples():
        d = by_id.loc[r.interval_id]
        riskrows.append([html.escape(d.label), int(d.n), pct(d.roi), int(r.bets), pct(r.bankroll_return), pct(r.max_drawdown)])
    risktable = table(riskrows, ['Historical quarter-Kelly return/drawdown frontier', 'Flat-stake N', 'Flat-stake ROI',
                                'Kelly bets', 'Bankroll return', 'Max drawdown'])

    # A standard plotting artifact accompanies the interactive chart.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ['#166a63', '#2c62aa', '#b06a17', '#7358a6', '#ad3d48']
    for iid, color in zip(featured, colors):
        g = paths[(paths.interval_id == iid) & (paths.training_prefix == 3) & (paths.kelly_fraction == .25)]
        axes[0].plot(g.fight_index, g.wealth, label=by_id.loc[iid, 'label'], color=color)
        dd = 1 - g.wealth / np.maximum.accumulate(g.wealth)
        axes[1].plot(g.fight_index, 100 * dd, color=color)
    axes[0].axhline(1, color='#67707c', linewidth=.7)
    axes[0].set(xlabel='Recorded fight index (all 187 rows)', ylabel='Bankroll / initial bankroll', title='Quarter Kelly: complete trajectory')
    axes[1].set(xlabel='Recorded fight index (all 187 rows)', ylabel='Drawdown from prior peak (%)', title='Drawdown along the same trajectory')
    fig.legend(loc='lower center', bbox_to_anchor=(.5, -.04), ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, .15, 1, 1))
    plot = ROOT / 'reports/figures/interval_trajectories.png'
    fig.savefig(plot, dpi=180, bbox_inches='tight'); plt.close(fig)
    embedded = base64.b64encode(plot.read_bytes()).decode()

    payload = dict(candidates=details[['interval_id', 'label', 'n', 'wins', 'roi', 'selection']].to_dict('records'),
                   paths={}, stats={})
    for (iid, prefix, fraction), g in paths.groupby(['interval_id', 'training_prefix', 'kelly_fraction']):
        key = f'{iid}|{prefix}|{fraction}'
        payload['paths'][key] = g.sort_values('fight_index').wealth.round(10).tolist()
    for r in kelly.itertuples():
        key = f'{r.interval_id}|{r.training_prefix}|{r.kelly_fraction}'
        payload['stats'][key] = dict(bets=int(r.bets), growth=r.bankroll_return, drawdown=r.max_drawdown, max_stake=r.max_stake)
    data = json.dumps(payload, allow_nan=False).replace('</', '<\\/')
    best_detail = by_id.loc[best]
    adverse = sensitivity[(sensitivity.interval_id == best) & (sensitivity.residual_quantile == .975)].iloc[0]
    broader = by_id.loc['core_above_35']
    broader_adverse = sensitivity[(sensitivity.interval_id == 'core_above_35') & (sensitivity.residual_quantile == .975)].iloc[0]
    source = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Confidence, odds intervals and full fight-level Kelly trajectories</title>
<style>body{{margin:0;background:#f3f6f8;color:#172b3e;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1180px;margin:28px auto;background:white;padding:36px;border-radius:12px}}h1{{font-size:30px;line-height:1.25}}h2{{margin-top:34px;color:#164c68}}.note{{font-size:14px;color:#526274}}.callout{{padding:18px;border-left:4px solid #166a63;background:#edf7f5}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}}td,th{{border-bottom:1px solid #dbe3e9;padding:9px;text-align:left}}th{{background:#edf3f7}}select{{max-width:100%;padding:9px;font:inherit}}label{{display:block;font-weight:600;margin:14px 0 5px}}svg,img{{width:100%;height:auto}}.legend{{display:flex;gap:18px;flex-wrap:wrap}}.swatch{{display:inline-block;width:22px;height:3px;margin:0 6px 4px 0}}a{{color:#07548c}}summary{{cursor:pointer;font-weight:600}}code{{background:#f1f4f6}}@media(max-width:650px){{main{{margin:0;padding:18px}}h1{{font-size:25px}}}}@media print{{main{{padding:0}}.controls{{display:none}}}}</style></head><body><main>
<h1>Confidence, odds intervals and full fight-level Kelly trajectories</h1>
<p class="note">October 19, 2026 · Follow-up to the <a href="final_report.html">full audit and statistical report</a> · <a href="https://github.com/tamnys/kalshi">Code and saved analyses</a></p>
<p><strong>Additional validation:</strong> see the <a href="prospective_validation.html">earlier-data selection and Kelly robustness report</a> for rules selected from past results, seven probability-estimation comparisons, allocations fixed at pre-card quote time, and uncertainty from 10,000 model refits. The fixed high-confidence filter remains promising; adaptive selection and stake sizing are much less stable.</p>
<p><strong>Confidence 1, 2 and 3 separately:</strong> the <a href="lower_confidence.html">lower-confidence and opposite-pick analysis</a> adds confidence-specific intervals, NO-side pricing, search-adjusted uncertainty and interactive fight-by-fight Kelly paths. Opposing confidence-1 favorites is the clearest additional candidate.</p>
<p><strong>Combining the strategies:</strong> the <a href="aggressive_growth.html">aggressive-growth comparison</a> evaluates a single bankroll across discovered categories, settling each fight and reinvesting before the next bet. Individual Kelly stakes and model sensitivity are explained; simultaneous pre-card funding is a separate comparison.</p>
<p><strong>Expanded sequential analysis:</strong> the <a href="sequential_kelly.html">600-strategy Kelly comparison</a> adds an interactive custom-fraction viewer, sizing learned only from earlier fights, later-period tests, and whole-card resampling sensitivities.</p>
<p><strong>Per-fight stake caps:</strong> the <a href="capped_kelly.html">5%, 10%, 15% and 20% cap comparison</a> measures ending wealth sacrificed and drawdown reduced, always sizing from the current settled bankroll.</p>
<p><strong>Further confidence-2/3 research:</strong> the <a href="confidence23.html">focused investigation and sequential viewer</a> examines broader price rules, confidence-3 opposing picks, and whether earlier-data selection or calibration can recover their apparent edge.</p>
<p><strong>How much signal is in confidence?</strong> The <a href="confidence_signal.html">October 20 assessment</a> separates the confidence scale's ranking information from the evidence for the discovered C2/C3 betting rules.</p>
<p><strong>Yes—several confidence/price combinations have substantially higher historical ROI.</strong> The most useful new candidate is confidence 4–5 at 35–81¢: {int(best_detail.wins)} wins in {int(best_detail.n)} fights, {pct(best_detail.roi)} net flat-stake ROI, and {pct(primary.loc[best, 'bankroll_return'])} quarter-Kelly bankroll growth with {pct(primary.loc[best, 'max_drawdown'])} maximum drawdown. These are exploratory historical results; the interval was selected after seeing the entire sample.</p>
<div class="callout"><strong>Practical interpretation:</strong> confidence 4–5 at 35¢ and above is a simpler candidate for future tracking than treating 81¢ as an established cutoff. It retains {pct(broader.roi)} flat-stake ROI across {int(broader.n)} fights and {pct(primary.loc['core_above_35','bankroll_return'])} quarter-Kelly bankroll growth. Neither rule has independent validation. Keep the 35–40¢ confidence 3–5 pocket as a separate hypothesis, and never stack two positions when overlapping rules select the same fight.</div>
{main_table}
<p class="note">Prices are entry asks per $1 win payout, before the modeled order fee. Exact observed interval endpoints are inclusive; 35¢ to below 40¢ explicitly excludes 40¢. Flat-stake ROI uses equal all-in cash risk per qualifying decisive fight. Kelly stakes vary and may be zero, so bankroll return and flat-stake ROI are different quantities. All returns include the same modeled Kalshi taker fee as the main analysis.</p>
<p class="note">* These p-values adjust the flat-stake return statistic for the complete confidence/price search and the previously examined backing/fading directions. They are not significance tests for the additionally selected Kelly performance. Category and Kelly optimization both use the full history; no category is statistically established after search adjustment.</p>
<h2>What was tested</h2>
<p>Every contiguous confidence interval and every distinct observed-price interval was examined: {meta['total_intervals']:,} backing combinations. All {meta['positive_intervals']:,} positive-return combinations were simulated at quarter, half, and full Kelly using both initial-estimation conventions, producing {meta['simulation_scenarios']:,} scenarios. There is no minimum-sample filter and no top-N exclusion. The maximum raw ROI is a single winning 16¢ bet (+490.3%); that is not a dependable category. The three-win confidence 1–2 pocket at 16–23¢ is also too small to treat its +374.4% as a repeatable expectation.</p>
<p>The viewer contains {meta['displayed_candidates']} candidates: the {meta['roi_sample_frontier_intervals']} ROI/sample-size frontier intervals, confidence-range leaders, Kelly return/drawdown frontier intervals, and broad references. Frontier means no alternative is better on both displayed dimensions. It is a descriptive comparison, not an adjustment for overfitting. All other positive combinations remain in the complete results file.</p>
<h2>Whole-trajectory viewer</h2>
<p>Start at bankroll 1 and advance after every source row in chronological fight order. All 187 rows are represented: 185 decisive fights, one draw and one canceled fight. Zero-stake rows remain flat. The draw uses a 50¢ contract payout if a position is selected. Forecasts use only earlier decisive outcomes.</p>
<div class="controls"><label for="category">Confidence/price combination</label><select id="category"></select><label for="prefix">Initial training period</label><select id="prefix"><option value="3">3 decisive fights — earliest identifiable odds-plus-confidence model</option><option value="24">24 decisive fights — previous six-model comparison</option></select></div>
<p id="category-detail" class="note"></p><div class="legend"><span><i class="swatch" style="background:#166a63"></i>Quarter Kelly</span><span><i class="swatch" style="background:#2c62aa"></i>Half Kelly</span><span><i class="swatch" style="background:#b06a17"></i>Full Kelly</span></div>
<svg id="trajectory" viewBox="0 0 960 360" role="img" aria-label="Fight-by-fight bankroll trajectories"></svg><div id="path-stats" class="table-wrap"></div>
<p class="note">Whole-trajectory Kelly requires an estimated win probability. It does not assume the eventual category win rate was known at the beginning. The Firth logistic model uses market log odds and confidence, refitted after each decisive fight. It first has full column rank after three observations; before that, stakes are zero. Every bet uses <code>fraction × max(0, (estimated probability − all-in cost)/(1 − all-in cost))</code> of the then-current bankroll. Price reconstruction still uses the full paired-price dataset and pre-card quotes are assumed obtainable later; these are not verified historical fills.</p>
<h2>Fraction choice and the early-data effect</h2>{comparison}
<p><strong>The broad confidence-4–5 quarter-Kelly result changes from +10.2% with the earlier 24-fight training prefix to −7.2% when estimation starts as soon as the three-parameter model is identifiable.</strong> The extra early losing bet risks 15.8% of bankroll even at quarter Kelly because the sparse fitted model assigns a large edge. Full Kelly on the broad pool loses 68.3% in this version. This is evidence of estimation risk; selecting the longer warm-up because it produces a better backtest would be another hindsight decision. The 35¢-and-above candidates exclude that fight, but their threshold is itself discovered retrospectively.</p>
<p>A zero observed drawdown for a tiny winning subset is not evidence of zero future risk. The 35–40¢ pocket places only five positive-estimated-edge Kelly bets out of its eight qualifying flat-stake wins. No user risk tolerance has been supplied, so the report compares stake fractions rather than declaring one optimal.</p>
<img src="data:image/png;base64,{embedded}" alt="Complete quarter-Kelly bankroll and drawdown trajectories for featured candidates">
<h2>Reliability of the new candidates</h2>
<p>For confidence 4–5 at 35–81¢, the nominal card-cluster 95% ROI interval is {pct(best_detail.cluster_ci_low)} to {pct(best_detail.cluster_ci_high)}. The common-shift posterior gives {pct(best_detail.posterior_prob_positive)} probability of positive expected ROI. Both are conditional, unadjusted summaries after category discovery. The search-adjusted flat-stake p-value is 1.000 after accounting for both directions.</p>
<p>Removing its largest winner leaves {pct(best_detail.remove_best_roi)} ROI; omitting any one card leaves at least {pct(best_detail.leave_card_out_min_roi)}. The {int(best_detail.api_n)} fights with observed Kalshi asks return {pct(best_detail.observed_roi)}. After adding the empirical adverse reconstruction error to estimated prices and reselecting the interval, {int(adverse.n)} fights remain with {pct(adverse.roi)} ROI. For the simpler 35¢-and-above rule, the corresponding stressed ROI is {pct(broader_adverse.roi)} across {int(broader_adverse.n)} fights.</p>
<p>These checks make the filtered candidates worth tracking, but they reuse the same outcomes. Actual signal timestamps, complete pick coverage, available depth, fees at order size, individual fight start times, and settlement availability remain unverified. The main report’s frozen-before-card sensitivity illustrates why the positive sequential paths cannot establish executable historical profits.</p>
<h2>Return versus drawdown</h2>{risktable}
<p class="note">Every row here was selected using the observed quarter-Kelly trajectory. The apparently best tradeoff will generally look worse on new data; the table is not an independent performance forecast.</p>
<details><summary>Strongest return signal within every confidence range</summary>{leadtable}</details>
<h2>Saved work</h2><p>The <a href="../results/interval_kelly_all_positive.csv.gz">complete positive-interval simulation summaries</a>, <a href="../results/interval_selected_details.csv">selected-category statistics</a>, <a href="../results/interval_bankroll_paths.csv.gz">displayed bankroll trajectories</a>, <a href="../results/interval_roi_frontier.csv">ROI/sample-size frontier</a>, and <a href="../results/interval_price_sensitivity.csv">price-error sensitivity</a> are saved alongside the code. Complete trajectories for every positive interval are saved locally as compressed arrays under <code>artifacts/private/all_positive_interval_paths_prefix_*.npz</code>. They can be regenerated from the supplied workbook.</p>
<script id="analysis-data" type="application/json">{data}</script>
<script>
const data=JSON.parse(document.getElementById('analysis-data').textContent);
const category=document.getElementById('category'), prefix=document.getElementById('prefix');
data.candidates.sort((a,b)=>a.label.localeCompare(b.label)).forEach(c=>{{const option=document.createElement('option');option.value=c.interval_id;option.textContent=c.label+' · N='+c.n;category.appendChild(option)}});
category.value={json.dumps(best)};
const percent=x=>(100*x).toFixed(1)+'%';
function render(){{
 const c=data.candidates.find(x=>x.interval_id===category.value);
 document.getElementById('category-detail').textContent=c.wins+'/'+c.n+' flat-stake wins; net ROI '+percent(c.roi)+'. Selection: '+c.selection+'.';
 const fractions=['0.25','0.5','1.0'], colors=['#166a63','#2c62aa','#b06a17'];
 const series=fractions.map(f=>data.paths[c.interval_id+'|'+prefix.value+'|'+f]);
 const max=Math.max(1,...series.flat())*1.05, width=960,height=360,left=60,right=20,top=20,bottom=48;
 const sx=i=>left+i/(series[0].length-1)*(width-left-right),sy=v=>height-bottom-v/max*(height-top-bottom);
 let svg='<rect width="960" height="360" fill="white"/>';
 for(let t=0;t<=4;t++){{let value=max*t/4,y=sy(value);svg+='<line x1="'+left+'" x2="'+(width-right)+'" y1="'+y+'" y2="'+y+'" stroke="#e1e7ed"/><text x="'+(left-8)+'" y="'+(y+4)+'" text-anchor="end" font-size="12">'+value.toFixed(2)+'×</text>';}}
 [0,Math.round((series[0].length-1)/2),series[0].length-1].forEach(t=>{{svg+='<text x="'+sx(t)+'" y="'+(height-bottom+22)+'" text-anchor="middle" font-size="12">'+t+'</text>';}});
 svg+='<line x1="'+left+'" x2="'+(width-right)+'" y1="'+sy(1)+'" y2="'+sy(1)+'" stroke="#596773" stroke-dasharray="4 4"/>';
 series.forEach((values,j)=>{{const points=values.map((v,i)=>sx(i)+','+sy(v)).join(' ');svg+='<polyline fill="none" stroke="'+colors[j]+'" stroke-width="2.4" points="'+points+'"/>';}});
 svg+='<text x="480" y="354" text-anchor="middle" font-size="13">Fight index — skipped and canceled rows included</text>';
 document.getElementById('trajectory').innerHTML=svg;
 let rows='<table><thead><tr><th>Kelly fraction</th><th>Bets</th><th>Bankroll return</th><th>Max drawdown</th><th>Largest stake</th></tr></thead><tbody>';
 fractions.forEach((f,i)=>{{const s=data.stats[c.interval_id+'|'+prefix.value+'|'+f];rows+='<tr><td>'+['Quarter','Half','Full'][i]+'</td><td>'+s.bets+'</td><td>'+percent(s.growth)+'</td><td>'+percent(s.drawdown)+'</td><td>'+percent(s.max_stake)+'</td></tr>';}});
 document.getElementById('path-stats').innerHTML=rows+'</tbody></table>';
}}
category.addEventListener('change',render);prefix.addEventListener('change',render);render();
</script></main></body></html>'''
    (ROOT / 'reports/interval_trajectories.html').write_text(source)
    print(ROOT / 'reports/interval_trajectories.html')


if __name__ == '__main__':
    run()
