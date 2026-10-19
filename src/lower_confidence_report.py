"""Standalone results and interactive low-confidence bankroll trajectories."""
import html
import json
import numpy as np
import pandas as pd
from audit import ROOT
from analyze import plt
from report import table, pct, image

LABELS = {
 'back_C3_35-under40':'Back C3: 35 to <40¢',
 'back_C3_pick_1594':'Back C3: 76.8126–85.8497¢',
 'back_C3_76-under86':'Back C3: 76 to <86¢',
 'back_C3_77-under86':'Back C3: 77 to <86¢',
 'back_C2_pick_94':'Back C2: 16–36¢',
 'back_C2_15-under37':'Back C2: 15 to <37¢',
 'back_C1_all':'Back C1: all prices',
 'fade_C1_all':'Fade C1: all prices',
 'fade_C1_pick_35':'Fade C1: pick ask 25–70¢',
 'fade_C1_25-under71':'Fade C1: pick ask 25 to <71¢',
 'fade_C1_50plus':'Fade C1: pick ask ≥50¢',
 'fade_C2_all':'Fade C2: all prices',
 'fade_C2_pick_411':'Fade C2: pick ask 36.6440–62¢',
}
MODELS={'odds_confidence':'Pooled odds + confidence','confidence_categories':'Separate confidence levels','category_offset':'Selected-category offset'}
MAIN=['fade_C1_all','fade_C1_50plus','fade_C1_pick_35','back_C2_pick_94','back_C3_pick_1594','back_C3_35-under40','fade_C2_pick_411','back_C1_all','fade_C2_all']


def label(r):
    if r.id in LABELS:return LABELS[r.id]
    side='Back' if r.side=='back' else 'Fade'
    bounds='all prices' if r.price_low==0 and r.price_high==1 else f'{100*r.price_low:.4f} to '+('' if r.high_inclusive else '<')+f'{100*r.price_high:.4f}¢'
    axis='NO ask' if r.axis=='purchase' else 'pick ask'
    return f'{side} C{int(r.confidence)}: {axis} {bounds} (N={int(r.n)})'


def run():
    d=pd.read_csv(ROOT/'results/lower_confidence_candidates.csv')
    d['label']=[label(r) for r in d.itertuples()]
    detail=d.set_index('id');sens=pd.read_csv(ROOT/'results/lower_confidence_sensitivity.csv')
    k=pd.read_csv(ROOT/'results/lower_confidence_kelly.csv')
    meta=json.loads((ROOT/'results/lower_confidence_summary.json').read_text())
    error=json.loads((ROOT/'results/lower_confidence_price_errors.json').read_text())
    paths=pd.read_csv(ROOT/'results/lower_confidence_paths.csv.gz')
    observed=sens[sens.scenario.eq('observed')].set_index('id')
    broad=['back_C1_all','back_C2_all','back_C3_all','fade_C1_all','fade_C2_all']
    broadtable=table([[detail.loc[i,'label'],f'{int(detail.loc[i,"wins"])}/{int(detail.loc[i,"n"])}',pct(detail.loc[i,'roi']),
                      f'{pct(detail.loc[i,"cluster_ci_low"])} to {pct(detail.loc[i,"cluster_ci_high"])}',
                      f'{int(observed.loc[i,"wins"])}/{int(observed.loc[i,"n"])}',pct(observed.loc[i,'roi'])] for i in broad],
                     ['Action','Wins / fights','Net flat ROI','Approx. card-cluster 95% interval','Observed-quote wins / fights','Observed-quote ROI'])
    main=table([[detail.loc[i,'label'],f'{int(detail.loc[i,"wins"])}/{int(detail.loc[i,"n"])}',pct(detail.loc[i,'roi']),
                 int(detail.loc[i,'api_n']),pct(observed.loc[i,'roi']),f'{detail.loc[i,"nominal_p"]:.4f}',
                 f'{detail.loc[i,"within_family_p"]:.4f}',f'{detail.loc[i,"study_adjusted_p"]:.3f}'] for i in MAIN],
               ['Retrospective rule','Wins / fights','Net flat ROI','Observed-quote fights','Observed-only ROI','Nominal p','Within-level interval-adjusted p','Whole-investigation adjusted p'])
    conditional=table([[detail.loc[i,'label'],pct(detail.loc[i,'posterior_prob_positive']),
                       f'{pct(detail.loc[i,"posterior_roi_low"])} to {pct(detail.loc[i,"posterior_roi_high"])}',
                       f'{pct(detail.loc[i,"win_rate_exact_low"])} to {pct(detail.loc[i,"win_rate_exact_high"])}'] for i in MAIN],
                      ['Rule','Model-conditional P(positive ROI)','Model-conditional 95% ROI interval','Exact binomial 95% win-rate interval'])
    quarter=k[k.fraction.eq(.25)&k.cadence.eq('card')]
    q=quarter.set_index(['id','sample','model'])
    ktable=table([[detail.loc[i,'label'],pct(q.loc[(i,'all','odds_confidence'),'final_wealth']-1),
                  pct(q.loc[(i,'all','odds_confidence'),'max_drawdown']),int(q.loc[(i,'all','odds_confidence'),'bets']),
                  pct(q.loc[(i,'all','confidence_categories'),'final_wealth']-1),pct(q.loc[(i,'all','category_offset'),'final_wealth']-1),
                  pct(q.loc[(i,'observed','odds_confidence'),'final_wealth']-1),int(q.loc[(i,'observed','odds_confidence'),'bets'])] for i in MAIN],
                 ['Rule','Pooled quarter-Kelly growth','Pooled drawdown','Pooled bets','Separate-level model growth','Category-offset growth','Observed-only pooled growth','Observed-only bets'])
    comparison=['fade_C1_all','fade_C1_50plus','fade_C1_pick_35','fade_C1_25-under71','back_C2_pick_94','back_C2_15-under37','back_C3_pick_1594','back_C3_77-under86','back_C3_76-under86','back_C3_35-under40','fade_C2_pick_411']
    def scenario(i,name):return sens[(sens.id==i)&(sens.scenario==name)].iloc[0]
    stress=table([[detail.loc[i,'label'],pct(detail.loc[i,'roi']),pct(scenario(i,'omit_largest_winner').roi),
                   pct(sens[sens.id.eq(i)&sens.scenario.str.startswith('omit_card_')].roi.min()),
                   pct(scenario(i,'reselect_pick_error_0.025_purchase_adverse').roi),
                   pct(scenario(i,'reselect_pick_error_0.975_purchase_adverse').roi)] for i in comparison],
                 ['Rule','Baseline flat ROI','Remove largest winner','Worst leave-one-card-out ROI','Reselect: lower pick-price error','Reselect: higher pick-price error'])
    frozen=pd.read_csv(ROOT/'results/lower_confidence_frozen.csv')
    fr=pd.read_csv(ROOT/'results/lower_confidence_frozen_rules.csv')
    fq=frozen[frozen.fraction.eq(.25)].merge(fr,on=['sample','confidence','side','axis','split','criterion'])
    ft=table([[r.sample,r.side+' C'+str(r.confidence),r.axis,r.split,r.criterion,
               f'{100*r.price_low:.2f} to <{100*r.price_high:.2f}¢',int(r.training_n),f'{r.flat_wins}/{r.flat_n}',
               pct(r.flat_roi),pct(r.final_wealth-1),int(r.bets)] for r in fq.itertuples()],
             ['Price sample','Action','Price axis','Freeze before','Training objective','Deployed interval','Training N','Later wins / fights','Later flat ROI','Later quarter-Kelly growth','Kelly bets'])
    allstats=table([[r.label,r.reason,int(r.n),int(r.wins),pct(r.roi),int(r.api_n),
                    f'{r.within_family_p:.4f}',f'{r.study_adjusted_p:.3f}'] for r in d.itertuples()],
                  ['Candidate','Why displayed','N','Wins','Net flat ROI','Observed N','Within-family p','Whole-study p'])
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    plotids=['fade_C1_all','fade_C1_50plus','back_C2_pick_94','back_C3_pick_1594']
    for ax,i in zip(axes.flat,plotids):
        for sample,style in [('all','-'),('observed','--')]:
            for model,color in [('odds_confidence','#16665d'),('confidence_categories','#4667a6'),('category_offset','#b36b14')]:
                p=paths[(paths.id==i)&paths['sample'].eq(sample)&paths.cadence.eq('card')&paths.model.eq(model)&paths.fraction.eq(.25)]
                assert len(p)==meta['source_rows']+1
                ax.plot(p.fight_index,p.wealth,linestyle=style,color=color,label=MODELS[model]+(' / observed only' if sample=='observed' else ''))
        ax.axhline(1,color='#999',linewidth=.6);ax.set_title(LABELS[i]);ax.set_ylabel('Bankroll / initial bankroll');ax.grid(alpha=.15)
    for ax in axes[-1]:ax.set_xlabel('Recorded fight index (all 187 rows)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8)
    fig.suptitle('Quarter Kelly: cash reserved before each card; equity settles after every fight')
    fig.tight_layout(rect=(0,.1,1,.96));fig.savefig(ROOT/'reports/figures/lower_confidence_paths.png',dpi=160);plt.close(fig)
    # Aggregate path data only; no fighter names or proprietary input rows.
    path_lookup={key:np.round(g.sort_values('fight_index').wealth.to_numpy(),7).tolist() for key,g in paths.groupby(['id','sample','cadence','model','fraction'],sort=False)}
    scenarios={}
    for r in k.itertuples():
        key='|'.join([r.id,r.sample,r.cadence,r.model]);pathkey=(r.id,r.sample,r.cadence,r.model,r.fraction)
        scenarios.setdefault(key,[]).append(dict(fraction=r.fraction,feasible=bool(r.feasible),
             growth=float(r.final_wealth-1) if r.feasible else None,drawdown=float(r.max_drawdown) if r.feasible else None,
             bets=int(r.bets),max_stake=float(r.max_stake),wealth=path_lookup.get(pathkey)))
    opts=MAIN+[i for i in d.id if i not in MAIN]
    options=''.join(f'<option value="{html.escape(i)}">{html.escape(detail.loc[i,"label"])}</option>' for i in opts)
    meta_js=dict(scenarios=scenarios,labels=detail.label.to_dict())
    body=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Confidence 1–3 and opposing picks</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#19332f;background:#f7faf8;margin:auto;padding:28px;max-width:1250px}}h1,h2{{line-height:1.2}}h1{{font-size:2.3rem}}a{{color:#12675e}}.note{{background:#e7f1ed;border-left:4px solid #28776b;padding:18px}}.table-wrap{{overflow:auto;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}td,th{{padding:10px;border-bottom:1px solid #d9e3de;text-align:left;vertical-align:top}}th{{background:#e6eee9}}figure{{margin:25px 0}}img{{max-width:100%;height:auto}}figcaption,.small{{font-size:13px;color:#52655f}}select{{padding:8px;max-width:100%;margin:4px}}label{{display:inline-block;margin:5px}}svg{{background:white;width:100%;height:auto;border:1px solid #d8e2dc}}details{{margin:20px 0}}code{{background:#e9efeb;padding:2px 5px}}</style></head><body>
<p><a href="final_report.html">Main report</a> · <a href="interval_trajectories.html">All-confidence interval viewer</a> · <a href="prospective_validation.html">Earlier-data and execution checks</a></p>
<h1>Confidence 1–3: backing and opposing picks</h1><p>Analysis dated October 19, 2026. Same corrected source: 187 rows, 185 decisive outcomes, one draw and one canceled fight. Prices combine recovered pre-card quotes and explicitly reconstructed prices.</p>
<div class="note"><strong>The clearest new candidate is to oppose confidence-1 picks, especially when the original pick is a favorite.</strong> This is supported by positive broad-category returns and by several earlier-outcome probability models. Confidence-2 underdogs are a secondary candidate. Confidence-3 favorites have an interesting undefeated interval, but its apparent edge depends on the boundaries. None establishes a profitable strategy after the full investigation's search correction.</div>
<h2>What the broad confidence levels say</h2>{broadtable}
<p>Flat-return interval statistics use decisive fights; bankroll trajectories also include the draw's 50¢ settlement. “Fade” means buy NO on the selected fighter. All interval prices below refer to the <strong>original pick's YES ask</strong> unless labeled NO ask. The NO purchase itself is priced at <code>1 − YES bid</code>, then charged its own fee. This preserves the bid/ask spread, following <a href="https://docs.kalshi.com/getting_started/orderbook_responses">Kalshi's orderbook documentation</a>.</p>
<h2>Intervals worth examining</h2>{main}
<p>The complete extension evaluates {meta['intervals']:,} intervals across seven confidence/direction/price-axis families. {meta['positive_intervals']:,} have positive historical ROI. No minimum sample size is used. The {meta['candidate_rules']} displayed rules include the ROI/sample-size frontier, the largest standardized return, longest all-win intervals, and broad or rounded comparisons. Every interval remains in <a href="../results/lower_confidence_intervals.csv">the complete scan</a>.</p>
<ul><li><strong>Confidence 1, opposite side:</strong> all 11 fades returned +47.4%, and fading original favorites (YES ask ≥50¢) returned +80.6% over seven fights. The sharper 25–70¢ pick-ask interval won 6/6, with NO purchase asks ranging from 32 to 76¢. It returned +132.2%; four observed-price fights alone returned +120.8%. Widening to below 71¢ includes a losing fade at a 70.3269¢ original pick ask and reduces ROI to +99.0%. The broader favorite rule avoids relying on the exact 70¢ cut, although it still was examined after seeing the data.</li>
<li><strong>Confidence 2, backing:</strong> 16–36¢ returned +134.8%, winning 5/8. Four observed-price fights returned +182.4%; removing the largest winner leaves +84.0%. The broader 15 to below 37¢ band returns +87.8% over ten fights. This is promising on equal-stake returns, but model-dependent Kelly results weaken the case for aggressive sizing.</li>
<li><strong>Confidence 3, backing:</strong> the exact 76.8126–85.8497¢ interval won 12/12 and returned +20.8%, including five observed-price wins. A practical 77 to below 86¢ filter has 11/11 wins and +20.2% ROI. However, 76 to below 86¢ includes three additional fights, two losses, and falls to +5.3% overall and −7.8% on observed quotes. The 35 to below 40¢ interval returns +153.2%, but consists of only three reconstructed-price wins and no observed quotes. All of those narrow-band wins occurred in 2025.</li>
<li><strong>Confidence 1, backing:</strong> broad ROI is −39.1%. Its best ROI interval isolates one 23¢ winner; another interval has only two high-price wins. There is no persuasive backing category here.</li>
<li><strong>Confidence 2, opposite side:</strong> fading every pick loses 13.1%. The broader middle-price pocket, original YES ask 36.6440–62¢, wins 13/18 and returns +26.8%, including +40.0% across eight observed-price fights. Its approximate card-cluster 95% ROI interval is −10.2% to +63.8%, and its within-level search-adjusted p-value is about 1.0. It is a secondary hypothesis, not a reason to fade all confidence-2 picks.</li></ul>
<h2>How much statistical support is there?</h2>
<p>The confidence-1 25–70¢ fade has an exact nominal all-win probability of 0.0079 under independent fee-adjusted break-even outcomes. Accounting for every interval <em>within confidence-1 fading alone</em> gives approximately 0.0336 (simulation standard error about 0.0018). This would be interesting for a genuinely prespecified confidence-1-only investigation. Here, the broader confidence/price/direction search was already conducted, so that smaller family cannot be treated as the original research scope. No candidate passes the wider correction.</p>
<p>The whole-study column uses a Bonferroni union across the prior complete backing scan, prior complete opposing scan, and the new direct-NO-ask scan. The lowest adjusted p-value anywhere in this extension is about 0.606, for the NO-price description of the same confidence-1 six-win sample. An adjusted value of 1 does not prove no edge; it means this search has not separated that candidate from chance with the stated test. Simulations assume independent fights and known costs. Dependence, price reconstruction and retrospective research choices remain limitations.</p>
{conditional}
<p>These posterior summaries use the existing Jeffreys/Firth common log-odds-shift model within each fixed category and independent outcomes. They <strong>do not correct for choosing the category after seeing results</strong>. Thus the six-win fade's 99.85% model-conditional positive-edge probability is not a 99.85% probability that this discovered strategy works. Exact binomial win-rate intervals assume a common win probability, whereas odds vary. All-win samples cannot produce a trustworthy empirical return-variance interval; those cluster intervals are omitted. Twelve favorite wins are less surprising than six cheaper NO wins: the confidence-3 favorite run has break-even all-win probability 0.104.</p>
<h2>Fractional Kelly over every fight</h2>
<p>For probability <code>p</code> and all-in cost <code>c</code>, the bankroll fraction at risk is <code>λ × max(0, (p − c)/(1 − c))</code>. For a fade, use the probability the pick loses and the actual NO cost. Taker cost is modeled as <code>c = ask + 0.07 × ask × (1 − ask)</code>, using multiplier one; the <a href="https://kalshi.com/docs/kalshi-fee-schedule.pdf">published fee schedule</a> specifies order-level rounding. These scalable simulations omit quantity rounding and fill-depth constraints. Draws use the ledger's 50¢ settlement; canceled fights are skipped.</p>
<p>The main comparison fixes forecasts and cash allocations before each card, then updates equity after every fight, valuing unsettled positions at cost. It never reinvests a result before an already-placed card position. Forecasts use earlier settled outcomes only, starting when the model is identifiable. Observed-only means both training and trading use recovered quotes only. Category choice remains retrospective in all fixed-rule simulations.</p>
{ktable}
<p>The pooled model treats confidence as an ordered numerical signal. The separate-level model allows each confidence level its own intercept. The category-offset model learns a market-relative probability adjustment using only earlier results in the chosen category. The latter can place large stakes after very few category wins. A displayed 0% with zero bets means the model found no positive priced edge, not that an active strategy broke even.</p>
{image('lower_confidence_paths.png','Quarter-Kelly paths at every recorded fight. Solid lines include reconstructed prices; dashed lines fit and trade observed quotes only. These are historical paths, not future-risk bands.')}
<h2>Inspect all candidate trajectories</h2>
<p>Each displayed path has 188 points: the initial bankroll plus all 187 recorded rows. Quarter, half and full Kelly are comparisons, not a chosen risk tolerance. An unavailable curve means the proposed before-card positions require more cash than the bankroll; the simulator does not borrow or silently rescale them.</p>
<div><label>Rule <select id="rule">{options}</select></label><label>Prices <select id="sample"><option value="all">Observed + reconstructed</option><option value="observed">Observed only: train and trade</option></select></label><label>Probabilities <select id="model"><option value="odds_confidence">Pooled odds + confidence</option><option value="confidence_categories">Separate confidence levels</option><option value="category_offset">Selected-category offset</option></select></label><label>Timing <select id="cadence"><option value="card">Allocate before card; settle every fight</option><option value="fight">Refit and compound after every fight</option></select></label></div>
<svg id="chart" viewBox="0 0 1050 430" role="img" aria-label="Historical bankroll trajectories"></svg><div id="metrics"></div>
<p class="small">The sequential alternative assumes pre-card quotes remain obtainable at each later fight, which cannot be established from this data. Historical zero drawdown for an all-winning selected band is not a future risk estimate.</p>
<h2>Boundary, influence and price-error sensitivity</h2>{stress}
<p>Reconstruction error is measured with complete-card cross-validation of the 44 independent sportsbook/Kalshi pairs. YES-ask errors have empirical 2.5th/97.5th percentiles of −2.72/+2.95¢. The separately fitted NO-price reconstruction has mean absolute error {100*error['no_loco_mae']:.2f}¢, RMSE {100*error['no_loco_rmse']:.2f}¢ and corresponding error percentiles −3.83/+3.71¢. The table shifts imputed YES prices and reselects categories; opposing positions also receive the adverse NO cost shift. Observed quotes stay fixed. These are sensitivity scenarios, not joint confidence intervals. The CSV also includes fixed-membership cost stresses and all calendar blocks.</p>
<p>The confidence-3 35–40¢ band falls to −18.5% in the adverse upward-price/reselection scenario. In contrast, the confidence-2 16–36¢ backing band and confidence-1 fade remain positive under the displayed price stresses. That establishes numerical resilience to those errors, not independent profitability.</p>
<h2>Rules chosen before the later outcomes</h2>
<p>To reduce hindsight in interval choice, each confidence/direction/price-axis family was searched using only data before July 19, 2025 and October 10, 2026, the existing calendar splits. Highest ROI and highest standardized return are shown, with no minimum N. Bounds for unseen prices are training-price midpoints, with outer endpoints 0 and 1. The selected rule stays fixed; pooled probabilities update using earlier cards. These tests were designed during this investigation and are not a preregistered holdout.</p>
<p>The pre-July confidence-1 fade rule selected by standardized return already favored original pick asks below 67.61¢. It later won 3/4, with +46.2% flat ROI and +6.0% quarter-Kelly growth. The pre-October version later split 1/2, with +32.9% flat ROI but −3.2% Kelly growth because stake sizes differed. Other lower-confidence frozen rules often selected a single earlier winner and then had no later qualifying fights. The observed-only confidence-3 favorite rule chosen before October lost 34.1% at equal stakes over four later fights; its pooled Kelly model placed no bets.</p>
<details><summary>All {len(fq)} frozen-rule quarter-Kelly evaluations</summary>{ft}</details>
<h2>What I would track next</h2>
<p><strong>First: opposing confidence-1 favorites, with the all-price fade as a broad reference.</strong> It has positive results across the three probability estimators and both price-provenance samples. At quarter Kelly with pre-card pooled sizing, fading all confidence-1 picks grows 14.4% with 12.8% drawdown; restricting to favorites grows 22.8% with 3.4% drawdown. Half Kelly changes those to 27.6%/24.5% and 48.8%/6.1%, respectively. The observed-only favorite-fade quarter-Kelly path grows 37.9% with 13.0% drawdown. All of these samples are small. Full Kelly can overcommit the observed-only card bankroll.</p>
<p><strong>Second: back confidence-2 underdogs around 16–36¢.</strong> Its flat-stake gains survive removing the largest winner and the tested price errors, but the pooled pre-card model takes just one reconstructed-history bet and returns +9.5%; observed-only fitting/trading loses 2.0% over two bets. Category-offset quarter Kelly returns +56.5%, but widening to 15 to below 37¢ changes it to −16.8%. This is a reason to retain a pooled estimate and record new picks, not extrapolate a 134.8% expected ROI.</p>
<p><strong>Confidence 3 is a weaker extension:</strong> track the 77 to below 86¢ favorite band, but do not treat its 11/11 record as a reliable win-probability estimate. The pooled pre-card model barely bets it. The three-win 35–40¢ band has no observed-price evidence. <strong>For confidence 2 fades</strong>, the middle-price pocket merits tracking, while blanket fading does not. There is no supported confidence-1 backing recommendation.</p>
<p>Use the actual purchase quote, require estimated probability above the fee-adjusted cost, and update estimates only after settlement. Do not infer an optimal Kelly fraction from the best historical curve. A unique risk-optimal fraction still needs a drawdown preference and more evidence about probability error. The larger six-win category-offset bets are especially exposed to estimation and selection error.</p>
<details><summary>All {len(d)} displayed candidates and broad comparisons</summary>{allstats}</details>
<p>Saved artifacts: <a href="../results/lower_confidence_candidates.csv">candidate statistics</a>, <a href="../results/lower_confidence_kelly.csv">all Kelly summaries</a>, <a href="../results/lower_confidence_paths.csv.gz">complete candidate paths</a>, <a href="../results/lower_confidence_sensitivity.csv">sensitivity details</a>, <a href="../results/lower_confidence_frozen.csv">earlier-data results</a>, and <a href="research_log.md">research record</a>. Proprietary source rows and intermediate forecasts remain local.</p>
<script id="trajectory-data" type="application/json">{json.dumps(meta_js,separators=(',',':'),allow_nan=False)}</script>
<script>
const DATA=JSON.parse(document.getElementById('trajectory-data').textContent);
const colors=['#16665d','#4667a6','#b36b14'];
const fmt=x=>x===null?'Unavailable':(100*x).toFixed(1)+'%';
function render(){{
 const key=['rule','sample','cadence','model'].map(x=>document.getElementById(x).value).join('|');
 const rows=DATA.scenarios[key];const valid=rows.filter(r=>r.feasible);
 const values=valid.flatMap(r=>r.wealth);let lo=values.length?Math.min(...values):.9,hi=values.length?Math.max(...values):1.1;
 const margin=Math.max(.025,(hi-lo)*.08);lo=Math.max(0,lo-margin);hi+=margin;
 const X=i=>65+i*955/187,Y=v=>375-(v-lo)*340/(hi-lo);
 let svg='<title>'+DATA.labels[document.getElementById('rule').value]+'</title>';
 for(let j=0;j<=4;j++){{let v=lo+(hi-lo)*j/4;svg+='<line x1="65" x2="1020" y1="'+Y(v)+'" y2="'+Y(v)+'" stroke="#dde5e0"/><text x="55" y="'+(Y(v)+4)+'" text-anchor="end" font-size="12">'+v.toFixed(2)+'×</text>';}}
 [0,50,100,150,187].forEach(i=>svg+='<text x="'+X(i)+'" y="399" text-anchor="middle" font-size="12">'+i+'</text>');
 svg+='<text x="530" y="422" text-anchor="middle" font-size="12">Recorded fight index</text>';
 rows.forEach((r,j)=>{{if(r.feasible)svg+='<polyline fill="none" stroke="'+colors[j]+'" stroke-width="2.5" points="'+r.wealth.map((v,i)=>X(i).toFixed(2)+','+Y(v).toFixed(2)).join(' ')+'"/>';}});
 document.getElementById('chart').innerHTML=svg;
 document.getElementById('metrics').innerHTML='<div class="table-wrap"><table><tr><th>Kelly fraction</th><th>Bankroll growth</th><th>Maximum drawdown</th><th>Bets</th><th>Largest individual allocation</th></tr>'+rows.map((r,j)=>'<tr><td style="color:'+colors[j]+'">'+fmt(r.fraction)+'</td><td>'+fmt(r.growth)+'</td><td>'+fmt(r.drawdown)+'</td><td>'+r.bets+'</td><td>'+fmt(r.max_stake)+'</td></tr>').join('')+'</table></div>';
}}
['rule','sample','cadence','model'].forEach(x=>document.getElementById(x).addEventListener('change',render));render();
</script></body></html>'''
    (ROOT/'reports/lower_confidence.html').write_text(body)
    print('Wrote reports/lower_confidence.html',len(body),'bytes',flush=True)


if __name__=='__main__':run()
