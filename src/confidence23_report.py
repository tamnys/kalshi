"""Standalone focused C2/C3 report, with sequential bankroll comparison."""
import html
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit import ROOT
from report import table, pct, image

MODELS={'pooled':'Pooled odds + confidence','level_offset':'Within-level price offset',
        'level_calibration':'Within-level price slope','category_offset':'Within selected category'}
LABELS={
 'C2_back_under37':'C2: back below 37¢', 'C2_back_16_to_36':'C2: back 16–36¢',
 'C2_back_15_under37':'C2: back 15 to <37¢','C2_fade_middle':'C2: oppose 36.644–62¢',
 'C3_back_77_under86':'C3: back 77 to <86¢','C3_back_76_under86':'C3: back 76 to <86¢',
 'C3_back_35_under40':'C3: back 35 to <40¢','C3_fade_40_under50':'C3: oppose 40 to <50¢',
 'C3_fade_65_under70':'C3: oppose 65 to <70¢','C3_fade_68_under70':'C3: oppose 68 to <70¢',
 'C3_fade_best_one_cut':'C3: oppose below 76.4063¢',
}
FOCUS=['C2_back_under37','C2_back_16_to_36','C2_fade_middle','C3_fade_best_one_cut',
       'C3_fade_65_under70','C3_fade_68_under70','C3_fade_40_under50','C3_back_77_under86']


def run():
    out=ROOT/'results'
    d=pd.read_csv(out/'confidence23_candidates.csv').set_index('id')
    s=pd.read_csv(out/'confidence23_sensitivity.csv')
    k=pd.read_csv(out/'confidence23_kelly.csv')
    scores=pd.read_csv(out/'confidence23_scores.csv')
    adaptive=pd.read_csv(out/'confidence23_adaptive_flat.csv')
    meta=json.loads((out/'confidence23_summary.json').read_text())
    inputs=json.loads((ROOT/'artifacts/private/confidence23_policy_inputs.json').read_text())
    labels=dict(LABELS)
    for key,r in d.iterrows():
        labels.setdefault(key,f'C{r.confidence}: '+('back' if r.side=='back' else 'oppose')+f' [{100*r.price_low:g}, {100*r.price_high:g}'+(']' if r.high_inclusive else ')')+'¢')
    for c in (2,3):
        for model in ('pooled','level_offset','level_calibration'):
            labels[f'C{c}_model_{model}']=f'C{c}: model chooses either side · {MODELS[model]}'
        labels[f'C{c}_adaptive_one_cut']=f'C{c}: learn side + one cutoff from prior fights'
        labels[f'C{c}_adaptive_interval']=f'C{c}: learn side + two bounds from prior fights'
    def ci(r):
        return 'Not estimable from this sample' if pd.isna(r.cluster_ci_low) else f'{pct(r.cluster_ci_low)} to {pct(r.cluster_ci_high)}'
    primary=table([[labels[key],f'{int(d.loc[key].wins)}/{int(d.loc[key].n)}',pct(d.loc[key].roi),ci(d.loc[key]),
                    f'{pct(d.loc[key].observed_roi)} ({int(d.loc[key].api_n)} fights)',f'{d.loc[key].nominal_p:.3f}',
                    f'{d.loc[key].within_level_side_p:.3f}',f'{d.loc[key].study_adjusted_p:.3f}'] for key in FOCUS],
                  ['Rule: original pick YES ask','Wins / fights','Flat net ROI','Approx. 95% card-cluster interval*',
                   'Recovered prices only','Nominal p','Within level + side p','Study-adjusted p'])
    broad=table([[f'C{c}',pct(d.loc[f'C{c}_back_all'].roi),pct(d.loc[f'C{c}_fade_all'].roi),int(d.loc[f'C{c}_back_all'].n)] for c in (2,3)],
                ['Confidence','Back every pick','Oppose every pick','Decisive fights'])
    robustness=[]
    for key in FOCUS:
        row=d.loc[key];ss=s[(s.id==key)&(s.scenario=='period')]
        periods=[f'{pct(r.roi)} ({int(r.n)})' if r.n else 'No fights' for r in ss.itertuples()]
        stress=s[(s.id==key)&(s.scenario=='price_error_reselect')].roi
        robustness.append([labels[key],pct(row.drop_best_roi),pct(row.leave_card_min_roi),*periods,f'{pct(stress.min())} to {pct(stress.max())}'])
    sensitivity=table(robustness,['Rule','Remove largest win','Worst single-card omission','Spring 2025: ROI (N)',
                                 'Later 2025: ROI (N)','October 2026: ROI (N)','Reconstructed-price stress ROI'])
    scoretable=table([[f'C{r.confidence}',r.sample,MODELS[r.model],int(r.n),f'{r.log_loss:.4f}',f'{r.market_log_loss:.4f}',
                      f'{r.log_loss_difference:+.4f}',f'{r.difference_ci_low:+.4f} to {r.difference_ci_high:+.4f}'] for r in scores.itertuples()],
                     ['Confidence','Prices','Model','Scored fights','Model log loss ↓','Quote log loss ↓','Difference ↓','Approx. 95% interval'])
    adaptivetable=table([[labels[r.id],r.sample,int(r.bets),pct(r.roi),ci(r)] for r in adaptive.itertuples()],
                        ['Learn rule using earlier fights','Prices','Decisive bets','Flat net ROI','Approx. 95% interval'])
    def get(key,sample,model,fraction=.25,cap=1.):
        return k[(k.id==key)&(k['sample']==sample)&(k.model==model)&(k.fraction==fraction)&(k.cap==cap)].iloc[0]
    rows=[]
    for key in FOCUS:
        for sample in ('all','observed'):
            cells=[]
            for model in MODELS:
                r=get(key,sample,model)
                cells.append('No bets' if not r.bets else f'{r.final_wealth:.3f}× / {pct(r.max_drawdown)} ({int(r.bets)})')
            rows.append([labels[key],sample,*cells])
    sizing=table(rows,['Rule','Prices',*[f'{v}: wealth / DD (bets)' for v in MODELS.values()]])
    # Two views of the same evidence: hindsight price cutoffs and actual prior-
    # outcome learning. Every cutoff plotted already exists in the prior scan.
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,c in enumerate((2,3)):
        side='back' if c==2 else 'fade'
        scan=pd.read_csv(out/('all_interval_search.csv' if side=='back' else 'fade_interval_search.csv'))
        scan=scan[(scan.confidence_low==c)&(scan.confidence_high==c)]
        scan=scan[scan.price_low==scan.price_low.min()].sort_values('price_high')
        axes[0,col].plot(100*scan.price_high,100*scan.roi,color='#176c60')
        axes[0,col].axhline(0,color='#777',lw=.8)
        axes[0,col].set(title=f'C{c}: {side} all picks up to each price cutoff',xlabel='Maximum original YES ask (¢)',ylabel='Historical flat ROI (%)')
        for model in ('pooled','level_offset','level_calibration'):
            context=next(r for r in inputs if r['id']==f'C{c}_model_{model}' and r['sample']=='all')
            wealth=np.r_[1.,np.cumprod(1+.25*np.array(context['full_kelly'])*np.array(context['returns']))]
            axes[1,col].plot(wealth,label=MODELS[model])
        for shape in ('one_cut','interval'):
            context=next(r for r in inputs if r['id']==f'C{c}_adaptive_{shape}' and r['sample']=='all')
            wealth=np.r_[1.,np.cumprod(1+.25*np.array(context['full_kelly'])*np.array(context['returns']))]
            axes[1,col].plot(wealth,ls='--',label='Earlier-data '+shape.replace('_',' '))
        axes[1,col].axhline(1,color='#777',lw=.8)
        axes[1,col].set(title=f'C{c}: quarter Kelly, learn only from earlier fights',xlabel='Recorded fight index',ylabel='Bankroll / starting bankroll')
        axes[1,col].legend(fontsize=8)
        for ax in axes[:,col]:ax.grid(alpha=.15)
    fig.suptitle('Historical price pockets are easier to find than profitable earlier-data rules')
    fig.tight_layout();fig.savefig(ROOT/'reports/figures/confidence23.png',dpi=170);plt.close(fig)
    payload={'|'.join((r['id'],r['sample'],r['model'])):dict(f=r['full_kelly'],r=r['returns']) for r in inputs}
    strategies=list(dict.fromkeys(r['id'] for r in inputs))
    options=''.join(f'<option value="{key}"'+(' selected' if key=='C2_back_under37' else '')+f'>{html.escape(labels[key])}</option>' for key in strategies)
    alltable=table([[labels[key],int(r.n),int(r.wins),pct(r.roi),ci(r),f'{r.study_adjusted_p:.3f}'] for key,r in d.iterrows()],
                   ['Rule','Fights','Wins','ROI','Approx. 95% interval','Study-adjusted p'])
    body=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Confidence 2 and 3: extractable edge or selected price pockets?</title><style>
body{{font:16px/1.65 system-ui,sans-serif;max-width:1320px;margin:auto;padding:28px;background:#f7faf8;color:#203b35}}h1,h2{{line-height:1.25}}a{{color:#176c60}}.note{{background:#e6f0ea;padding:18px;border-left:4px solid #247766}}.table-wrap{{overflow-x:auto;margin:22px 0}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dce6df;text-align:left;vertical-align:top}}th{{background:#e6eee8}}.controls{{display:flex;flex-wrap:wrap;gap:18px;margin:18px 0}}label{{display:block}}select{{font:inherit;padding:5px;max-width:100%;width:100%}}svg,img{{width:100%;height:auto}}svg{{background:white;border:1px solid #dce6df}}.muted,figcaption{{color:#52695e;font-size:14px}}figure{{margin:24px 0}}details{{margin:24px 0}}code{{background:#e5ede7;padding:2px 5px}}@media(max-width:650px){{body{{padding:16px}}h1{{font-size:27px}}}}
</style></head><body><p><a href="interval_trajectories.html">Interval viewer</a> · <a href="lower_confidence.html">Original lower-confidence study</a> · <a href="capped_kelly.html">Bankroll caps</a></p>
<h1>Confidence 2 and 3: additional candidates, limited evidence of a learnable edge</h1><p class="muted">October 19, 2026 · 45 C2 and 73 C3 decisive fights · {meta['rules']} fixed rules · {meta['scenarios']:,} sequential bankroll comparisons</p>
<div class="note"><strong>There are useful hypotheses in both levels.</strong> C2 cheap picks and middle-price fades remain interesting after widening boundaries and separating time periods. C3 opposing picks provide a stronger new research direction than its narrow undefeated backing band. But the earlier-data discovery procedures and general probability models lose money: none establishes a deployable advantage at a reliable, search-adjusted confidence level.</div>
<p>Confidence 1, 4 and 5 also remain hypotheses rather than proven edges. We should apply the same evidential standard to every level. The test is whether the system adds information beyond the price after spread and fees, not whether a subset has a high win rate.</p>
<h2>Broad backing or fading does not solve it</h2>{broad}
<p>Opposing means buying NO on the selected fighter. NO ask is <code>1 − YES bid</code>, not <code>1 − YES ask</code>, consistent with <a href="https://docs.kalshi.com/getting_started/orderbook_responses">Kalshi's orderbook mechanics</a>. Each side pays its own modeled fee. All prices defining rules below refer to the <strong>original pick's YES ask</strong>.</p>
<h2>The most informative price rules</h2>{primary}
<p class="muted">*These intervals assume the category is fixed and use approximate card-cluster inference. They do not correct for discovering the category. All-win/all-loss samples do not supply a defensible empirical outcome-variance interval. The nominal p-value tests fixed-category net profit against independent fee-adjusted break-even outcomes. Within-level/side p adjusts for all intervals in that confidence and direction; study-adjusted p retains the existing wider backing, fading and direct-NO-price search scope. An adjusted value of 1 is insufficient evidence, not proof of no edge.</p>
<h2>What changed for confidence 2?</h2>
<p><strong>The cheap-pick hypothesis survives a more honest lower boundary.</strong> Backing 16–36¢ gives +134.8% across eight fights, but excludes nearby losers. Removing the lower cutoff and backing everything below 37¢ gives +70.7% across eleven fights, including +125.9% over five recovered-price observations. Removing its largest winner still leaves +28.8%; all three existing calendar blocks remain positive. This is a better broad tracking hypothesis than interpreting the 16¢ boundary as real precision. Its nominal p-value is about 0.100 and its approximate interval is −86.3% to +227.7%, so the available evidence is weak despite the attractive return.</p>
<p><strong>Opposing C2 picks at roughly 36.6–62¢ is a separate candidate.</strong> It wins 13/18 for +26.8%, stays positive after omitting any card, and returns +16.2%, +27.1% and +47.8% in the three existing calendar blocks. Its eight recovered-price fights return +40.0%. The approximate 95% interval is −10.2% to +63.8%; no full-search support is established. The broad cheap-pick and middle-fade definitions overlap around 36.6–37¢ and must not both be applied to the same fight. No combined portfolio is claimed here.</p>
<h2>What changed for confidence 3?</h2>
<p><strong>Fading moderate favorites is worth more attention.</strong> Original YES asks 65 to below 70¢ produce seven opposing wins in twelve fights, +65.8% net ROI, +90.4% on six recovered-price fights, and +53.9% after removing the most favorable card. Its nominal p-value is about 0.034, but the within-level fading search adjustment is about 0.988. Narrowing to 68–70¢ gives +141.8% and a positive nominal confidence interval; it also shrinks to six observations. That narrower result is not stronger independent evidence.</p>
<p><strong>A broader C3 fade offers a larger-sample hypothesis.</strong> The strongest single-cut fade selects original YES asks below 76.4063¢: historically, all observed asks at most 76¢. It returns +19.8% across 58 decisive fights, +20.3% on 21 recovered-price fights, and at least +13.9% after any one card is omitted. All calendar blocks are positive. The 76.4063¢ number is the midpoint between adjacent recorded prices, not a precisely established economic threshold. The approximate interval is −5.3% to +44.9%, and sizing results are mixed.</p>
<p><strong>Fading 40–50¢ C3 picks wins 7/7 for +70.4%.</strong> Only two of those decisive fights have recovered quotes. The fixed-band all-win null probability is 2.43%, rising to about 0.826 after searching C3 fading intervals. A separate draw in this band receives its 50¢ settlement in the bankroll paths. Both this pocket and the 65–70¢ pocket have no qualifying decisive fights in October 2026, limiting evidence of recent persistence.</p>
<p>The existing 77 to below 86¢ C3 backing band remains 11/11, but expensive favorites can produce winning streaks without useful edge. Its all-win null probability is 13.3%. Widening to 76 to below 86¢ adds four fights including two losses, cutting ROI to +5.3% and recovered-price ROI to −7.8%. It is a lower-priority hypothesis than the opposing-pick candidates.</p>
<h2>Time, influence and price sensitivity</h2>{sensitivity}
<p>The price stress uses the previously measured cross-card reconstruction errors: YES asks move −2.72¢ or +2.95¢ for reconstructed rows; opposing positions also receive an adverse +3.71¢ NO-cost move. Rule membership is recomputed. These are sensitivity scenarios, not joint confidence bounds. Omitting a card retains the original rule; it does not replay discovery without that card. Every nested one-boundary expansion to another observed price is saved in <a href="../results/confidence23_boundary_expansions.csv">the boundary-expansion data</a>.</p>
<h2>Can earlier fights identify the pattern?</h2>
<p>For each new fight, the discovery procedure searches only earlier settled fights in that confidence level and chooses both direction and the interval with the greatest profit divided by break-even standard deviation. One version allows a single price cutoff; the other allows two endpoints. There is no tuned minimum sample. New prices use midpoints between training prices. Equal-stake outcomes below evaluate whether the selected rule transfers before introducing Kelly sizing.</p>
{adaptivetable}
<p>C2's single-cut procedure earns +6.7% over thirteen decisive bets, while the full interval procedure loses 12.1%. C3 loses 17.8% and 40.6%, respectively. Recovered-only C2 selection loses all three bets. These procedures are examples designed during the investigation, not a proof that every possible learner fails. Their failure does show that the attractive fixed bands have not translated into reliable discovery from past observations.</p>
<h2>Do smooth probability models add information?</h2>
<p>The within-level offset fits <code>logit(p) = logit(YES ask) + a</code>; the price-slope model fits <code>logit(p) = a + b × logit(YES ask)</code>. Both use Firth estimation to keep small-sample or all-win fits finite, using only previous C2 or previous C3 fights. The pooled reference uses all earlier confidence levels. Comparison observations are shared within each confidence/source sample and begin as soon as all compared models are identifiable.</p>
{scoretable}
<p>Every fitted model has worse point-estimate log loss than the unadjusted quote baseline in these comparisons. Most uncertainty intervals include zero; the result does not establish that all models are intrinsically worse. It does mean this test provides no positive evidence that these forms extract useful additional information. A localized edge can coexist with a weak overall score, and the quote baseline is not known truth.</p>
{image('confidence23.png','Top: historical returns under every one-sided price cutoff in the existing exhaustive scan. Bottom: sequential quarter-Kelly wealth using only earlier outcomes; all five learning paths finish below their starting bankroll in each confidence level. The rules themselves were designed during this investigation.')}
<h2>Sequential Kelly on the fixed candidate bands</h2>
<p>Every cell below uses quarter Kelly and reports final bankroll / initial bankroll, maximum drawdown and bets. The four probability methods make different choices; “No bets” is not successful trading. The selected-category method learns only inside the already-discovered interval, so it retains hindsight in category membership. Recovered-only scenarios refit and trade only recovered quotes.</p>
{sizing}
<p>The 65–70¢ C3 fade is profitable in every active quarter-Kelly model/source comparison; the pooled full-price-history version places no bets. Its full-history local estimates end around 1.079–1.085×, while recovered-only variants end around 1.122–1.404×. This supports retaining it as a research candidate, but all comparisons reuse the same selected fights. C2 cheap picks have much less stable recovered-only sizing: the three whole-confidence/pooled models lose, while selected-category sizing can profit.</p>
<h2>Inspect complete fight-by-fight paths</h2>
<p>The viewer includes all fixed rules, smooth model policies and adaptive searches. Caps are the user's previously requested 5%, 10%, 15% and 20% of the current settled bankroll, applied after fractional Kelly. They are not limits on cumulative drawdown. “All prices” includes reconstruction; all paths assume the historical pre-card quote can be reused at sequential entry.</p>
<div class="controls"><label>Strategy<select id="strategy">{options}</select></label><label>Prices<select id="sample"><option value="all">All prices</option><option value="observed">Recovered only: train and trade</option></select></label><label>Probability model<select id="model"></select></label><label>Per-fight cap<select id="cap"><option value="1">Uncapped</option><option value="0.05">5%</option><option value="0.1">10%</option><option value="0.15">15%</option><option value="0.2">20%</option></select></label></div>
<svg id="chart" viewBox="0 0 1100 420" role="img" aria-label="Sequential fractional Kelly bankroll paths"></svg><div id="metrics"></div>
<h2>What deserves the next observations?</h2>
<p>For C2, retain cheap-pick backing below 37¢ and opposing the middle-price band as separate hypotheses. For C3, retain opposing 65–70¢ and the broader below-about-76¢ comparison. The narrower 68–70¢ and undefeated 40–50¢ fades are exploratory refinements. None has established selection-adjusted confidence sufficient to treat the fitted probabilities as known. The modest confidence-4–5 and small confidence-1 results do not justify relaxing this standard.</p>
<p>Record new picks before outcomes, actual executable prices, fees, model version and skipped trades. Freeze each candidate and any overlap-resolution rule before evaluating the next observations. Measure flat net returns and probability accuracy separately from sequential bankroll growth. More profitable combinations of this same historical sample would not supply the missing independent evidence; choosing many configurations is the backtest-selection problem discussed by <a href="https://scholarworks.wmich.edu/math_pubs/40/">Bailey and coauthors</a>.</p>
<details><summary>All {len(d)} fixed rules and broad comparisons</summary>{alltable}</details>
<p>Saved data: <a href="../results/confidence23_candidates.csv">candidate statistics</a>, <a href="../results/confidence23_sensitivity.csv">time and price sensitivity</a>, <a href="../results/confidence23_scores.csv">forecast scores</a>, <a href="../results/confidence23_adaptive_flat.csv">earlier-data discovery results</a>, <a href="../results/confidence23_kelly.csv">all Kelly summaries</a>, <a href="../results/confidence23_paths.csv.gz">complete wealth paths</a>, and <a href="research_log.md">research record</a>.</p>
<script id="policy-data" type="application/json">{json.dumps(payload,separators=(',',':'),allow_nan=False)}</script>
<script id="model-labels" type="application/json">{json.dumps(MODELS)}</script>
<script>
const DATA=JSON.parse(document.getElementById('policy-data').textContent);
const MODELS=JSON.parse(document.getElementById('model-labels').textContent);
function pathFor(key,fraction,cap){{
 const d=DATA[key],wealth=[1];let max=1,dd=0,bets=0,maxStake=0;
 for(let i=0;i<d.f.length;i++){{const s=Math.min(fraction*d.f[i],cap),w=wealth[wealth.length-1]*(1+s*d.r[i]);wealth.push(w);max=Math.max(max,w);dd=Math.max(dd,1-w/max);if(s>0)bets++;maxStake=Math.max(maxStake,s);}}
 return {{wealth,dd,bets,maxStake,final:wealth[wealth.length-1]}};
}}
function optionsFor(){{
 const prefix=document.getElementById('strategy').value+'|'+document.getElementById('sample').value+'|';
 const el=document.getElementById('model'),previous=el.value;
 const options=Object.keys(DATA).filter(k=>k.startsWith(prefix)).map(k=>k.split('|')[2]);
 el.innerHTML=options.map(m=>'<option value="'+m+'">'+MODELS[m]+'</option>').join('');
 el.value=options.includes(previous)?previous:options[0];draw();
}}
function draw(){{
 const key=['strategy','sample','model'].map(id=>document.getElementById(id).value).join('|'),cap=Number(document.getElementById('cap').value);
 const paths=[.25,.5,1].map(f=>pathFor(key,f,cap)),colors=['#176c60','#376db5','#b46f19'];
 const lo=Math.min(1,...paths.flatMap(p=>p.wealth)),hi=Math.max(1,...paths.flatMap(p=>p.wealth)),span=hi-lo||1;
 const x=i=>65+1000*i/(paths[0].wealth.length-1),y=w=>355-305*(w-lo)/span;
 let svg='<line x1="65" y1="'+y(1)+'" x2="1065" y2="'+y(1)+'" stroke="#999" stroke-dasharray="4 4"/>';
 for(let j=0;j<=4;j++){{let w=lo+span*j/4;svg+='<text x="58" y="'+(y(w)+4)+'" text-anchor="end" fill="#52695e" font-size="12">'+w.toFixed(2)+'×</text>';}}
 paths.forEach((p,j)=>{{svg+='<polyline fill="none" stroke="'+colors[j]+'" stroke-width="2" points="'+p.wealth.map((w,i)=>x(i)+','+y(w)).join(' ')+'"/>';svg+='<text x="'+(70+330*j)+'" y="25" fill="'+colors[j]+'">'+['Quarter Kelly','Half Kelly','Full Kelly'][j]+'</text>';}});
 svg+='<text x="65" y="382">0</text><text x="1038" y="382">'+(paths[0].wealth.length-1)+'</text><text x="470" y="405">Recorded fight index</text>';
 document.getElementById('chart').innerHTML=svg;
 document.getElementById('metrics').innerHTML='<div class="table-wrap"><table><tr><th>Fraction</th><th>Final bankroll</th><th>Max drawdown</th><th>Largest stake</th><th>Bets</th></tr>'+paths.map((p,j)=>'<tr><td>'+['Quarter','Half','Full'][j]+'</td><td>'+p.final.toFixed(3)+'×</td><td>'+(100*p.dd).toFixed(1)+'%</td><td>'+(100*p.maxStake).toFixed(1)+'%</td><td>'+p.bets+'</td></tr>').join('')+'</table></div>';
}}
document.getElementById('strategy').addEventListener('change',optionsFor);document.getElementById('sample').addEventListener('change',optionsFor);
document.getElementById('model').addEventListener('change',draw);document.getElementById('cap').addEventListener('change',draw);optionsFor();
</script></body></html>'''
    (ROOT/'reports/confidence23.html').write_text(body)
    print('Saved reports/confidence23.html')


if __name__=='__main__':run()
