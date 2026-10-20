"""Explain capped sequential growth and drawdown, with complete interactive paths."""
import html
import json
import numpy as np
import pandas as pd
from audit import ROOT
from analyze import plt
from report import table, pct, image
from sequential_kelly import context_inputs, FIXED
from sequential_kelly_report import SHORT
from aggressive_growth_report import MODELS
from capped_kelly import LEADING, CAPS


CAP_LABELS = {key: 'Uncapped' if value is None else f'{value:.0%} cap' for key, value in CAPS.items()}
PRIMARY = ('bounded_core_plus_fade1', 'bounded_core_plus_narrow_fade1', 'all_disjoint_candidates')


def run():
    d = pd.read_csv(ROOT/'results/capped_kelly_scenarios.csv')
    paths = pd.read_csv(ROOT/'results/capped_kelly_paths.csv.gz')
    meta = json.loads((ROOT/'results/capped_kelly_summary.json').read_text())
    lookup = d.set_index(['portfolio','sample','model','policy','cap_name'])

    def get(name, policy='full', cap='uncapped', sample='all', model='odds_confidence'):
        return lookup.loc[(name, sample, model, policy, cap)]

    def wealth_dd(row):
        return f'{row.final_wealth:,.3f}× / {pct(row.max_drawdown)}'

    def comparison(sample, policy):
        return table([[CAP_LABELS[cap], *[wealth_dd(get(name, policy, cap, sample)) for name in LEADING]] for cap in CAPS],
                     ['Individual stake limit', *[SHORT[name]+': wealth / DD' for name in LEADING]])

    sacrifice_rows = []
    for name in PRIMARY:
        for cap in CAPS:
            if cap == 'uncapped':
                continue
            r = get(name, cap=cap)
            sacrifice_rows.append([SHORT[name], CAP_LABELS[cap], f'{r.final_wealth:.3f}×',
                                   pct(r.terminal_wealth_sacrificed_fraction), pct(r.net_profit_sacrificed_fraction),
                                   f'{100*r.drawdown_reduction:.2f} pp', pct(r.drawdown_reduction_fraction),
                                   f'{int(r.capped_bets)}/{int(r.bets)}'])
    sacrifice_table = table(sacrifice_rows, ['Strategy', 'Limit', 'Ending wealth', 'Ending wealth sacrificed',
                                            'Net profit sacrificed', 'Drawdown reduction', 'Relative DD reduction', 'Bets reduced'])
    model_rows = []
    for name in LEADING:
        for sample in ('all','observed'):
            for model in MODELS:
                base = get(name, sample=sample, model=model)
                cap = get(name, cap='cap_10', sample=sample, model=model)
                model_rows.append([SHORT[name], sample, MODELS[model], wealth_dd(base), wealth_dd(cap),
                                   pct(cap.final_wealth/base.final_wealth-1), f'{100*cap.drawdown_reduction:.2f} pp'])
    model_table = table(model_rows, ['Strategy','Price sample','Probability model','Uncapped full Kelly: wealth / DD',
                                    '10%-capped full Kelly: wealth / DD','Ending wealth change','DD reduction'])
    all_table = table([[SHORT[r.portfolio],r.sample,MODELS[r.model],r.policy,CAP_LABELS[r.cap_name],
                       f'{r.final_wealth:,.3f}×',pct(r.max_drawdown),pct(r.terminal_wealth_sacrificed_fraction),
                       f'{100*r.drawdown_reduction:.2f} pp',int(r.capped_bets),int(r.bets)] for r in d.itertuples()],
                     ['Strategy','Sample','Model','Kelly fraction','Cap','Ending wealth','Max DD',
                      'Ending wealth sacrificed','DD reduction','Bets reduced','Bets'])
    full, contexts = context_inputs()
    payload = {'|'.join(c.values()): dict(f=f.tolist(), returns=r.tolist()) for c,f,r,_,_ in contexts if c['portfolio'] in LEADING}
    options = ''.join(f'<option value="{name}"'+(' selected' if name == 'bounded_core_plus_fade1' else '')+f'>{html.escape(SHORT[name])}</option>' for name in LEADING)
    model_options = ''.join(f'<option value="{key}">{label}</option>' for key,label in MODELS.items())
    colors = {'uncapped':'#666b70','cap_05':'#387ab4','cap_10':'#178270','cap_15':'#b87a25','cap_20':'#8059a4'}
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    for row, sample in enumerate(('all','observed')):
        for col, policy in enumerate(('full','half')):
            ax = axes[row,col]
            for cap,color in colors.items():
                s=paths[paths.portfolio.eq('bounded_core_plus_fade1') & paths['sample'].eq(sample)
                        & paths.model.eq('odds_confidence') & paths.policy.eq(policy) & paths.cap_name.eq(cap)].sort_values('fight_index')
                ax.plot(s.fight_index,s.wealth,label=CAP_LABELS[cap],color=color,linestyle='--' if cap=='uncapped' else '-')
            ax.axhline(1,color='#999',lw=.6);ax.grid(alpha=.15)
            ax.set(title=f'{policy.title()} Kelly · '+('all prices' if sample=='all' else 'recovered quotes only'),
                   xlabel='Recorded fight index',ylabel='Bankroll / starting bankroll')
            ax.legend(fontsize=8)
    fig.suptitle('Core + C1 favorite fades: caps applied to the current settled bankroll')
    fig.tight_layout();fig.savefig(ROOT/'reports/figures/capped_kelly.png',dpi=170);plt.close(fig)
    data_json=json.dumps(payload,separators=(',',':'),allow_nan=False)
    r10,r20=get('bounded_core_plus_fade1',cap='cap_10'),get('bounded_core_plus_fade1',cap='cap_20')
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sequential Kelly with per-fight bankroll caps</title><style>
body{{font:16px/1.65 system-ui,sans-serif;max-width:1320px;margin:auto;padding:28px;background:#f7faf8;color:#203b35}}h1,h2{{line-height:1.25}}a{{color:#176c60}}.note{{background:#e6f0ea;padding:18px;border-left:4px solid #247766}}.table-wrap{{overflow-x:auto;margin:22px 0}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dce6df;text-align:left;vertical-align:top}}th{{background:#e6eee8}}.controls{{display:flex;flex-wrap:wrap;gap:18px;margin:18px 0}}label{{display:block}}select{{font:inherit;padding:5px;max-width:100%;width:100%}}svg,img{{width:100%;height:auto}}.chart{{background:white;border:1px solid #dce6df;border-radius:6px;margin:18px 0}}.muted,figcaption{{color:#52695e;font-size:14px}}figure{{margin:24px 0}}details{{margin:24px 0}}code{{background:#e5ede7;padding:2px 5px}}@media(max-width:650px){{body{{padding:16px}}h1{{font-size:27px}}}}
</style></head><body><p><a href="sequential_kelly.html">Sequential strategy comparison</a> · <a href="interval_trajectories.html">Interval viewer</a> · <a href="aggressive_growth.html">Growth and Kelly explanation</a></p>
<h1>What do 5%, 10%, 15% and 20% bet caps cost?</h1><p class="muted">October 19, 2026 · {meta['scenarios']} scenarios · six leading portfolios · quarter, half and full Kelly · two price samples · three probability models</p>
<div class="note"><strong>The caps sharply reduce historical drawdowns, and sometimes improve ending wealth.</strong> For the pooled core plus C1 favorite fades, full Kelly with a 10% cap ends at <strong>{r10.final_wealth:.3f}×</strong> with <strong>{pct(r10.max_drawdown)}</strong> drawdown, versus 5.903× and 50.5% uncapped. A 20% cap retains <strong>{r20.final_wealth:.3f}×</strong> with <strong>{pct(r20.max_drawdown)}</strong> drawdown. With recovered quotes only, all four caps improve that portfolio's full-Kelly ending wealth.</div>
<p>Each bet uses <code>current settled bankroll × min(Kelly multiplier × estimated full-Kelly fraction, cap)</code>. The cap includes modeled fees. Smaller calculated bets remain smaller; this is not a strategy that bets the cap on every fight. Each result settles before the next stake. The probability forecasts, price inputs, category memberships and Kelly multipliers are identical between each capped/uncapped pair.</p>
<p>The high-confidence <strong>core</strong> backs confidence 4–5 at 35–81¢. The broad C1 fade opposes original picks priced at least 50¢; the narrower version uses original asks of 25–70¢. The five-group combination adds C2 backing at 16–36¢, C3 backing at 77 to below 86¢ and C2 fades at approximately 36.644–62¢ to the core/broad-C1-fade portfolio. The two undefeated intervals combine C3–5 backing at 35 to below 40¢ and C1 fades at 25–70¢. All constituent groups are disjoint.</p>
<h2>Full Kelly: growth versus drawdown</h2><p>These use the pooled odds/confidence model and all prices, including reconstructed quotes. Every cell is <strong>ending bankroll multiplier / maximum drawdown</strong>. A 3.566× result turns an illustrative $1,000 into $3,566.</p>
{comparison('all','full')}
<p>A 10% cap reduces the core/favorite-fade ending bankroll by <strong>{pct(r10.terminal_wealth_sacrificed_fraction)}</strong>, while reducing maximum drawdown by <strong>{100*r10.drawdown_reduction:.2f} percentage points</strong> ({pct(r10.drawdown_reduction_fraction)} relative reduction). At 20%, the ending-bankroll sacrifice is only <strong>{pct(r20.terminal_wealth_sacrificed_fraction)}</strong>, but drawdown is still {pct(r20.max_drawdown)}. The five-group 20%-capped version retains 13.999× from an uncapped 16.474×, with the same 34.7% historical drawdown.</p>
<h2>Half Kelly: a different tradeoff</h2>
{comparison('all','half')}
<p>At half Kelly, a 15% cap sacrifices less than 1% of ending wealth in both the core/favorite-fade and five-group portfolios: 2.838× versus 2.858×, and 4.990× versus 5.027×. Their drawdown falls from 26.7% to 22.2%. A 10% cap trims more growth but reduces drawdown further, to 18.3%. The narrow C1-fade portfolio's 15% and 20% caps do not reduce its historical maximum drawdown, even though they reduce some individual bets.</p>
<p>In this full-price-history pooled comparison, <strong>full Kelly capped at 10% beats uncapped half Kelly on both ending wealth and maximum drawdown</strong> for the core/favorite-fade, narrower core/fade and five-group portfolios. That is a useful historical result; the source check below shows why it is not a universal ranking.</p>
<details><summary>Quarter Kelly results</summary>{comparison('all','quarter')}<p>The main three pooled portfolios never risk more than 13.8% at quarter Kelly, so 15% and 20% caps do not change those paths. A 10% cap has a small effect; a 5% cap reduces more of the bets.</p></details>
<h2>How much growth is sacrificed?</h2>
<p>Ending-wealth sacrifice uses the entire final bankroll as its denominator. Net-profit sacrifice uses only profit above the initial bankroll. For example, dropping from 5.903× to 3.566× sacrifices 39.6% of ending wealth, or 47.7% of net profit. A negative sacrifice means the cap improved the result.</p>
{sacrifice_table}
<h2>What changes when only recovered quotes are used?</h2>
<p>Both training and betting now use recovered Kalshi quotes only. This changes the probability estimates as well as the eligible fights. The following full-Kelly results use the pooled model:</p>
{comparison('observed','full')}
<p>For the core/favorite-fade combination, the uncapped result is 1.163× with 79.1% drawdown. At a 10% cap it becomes <strong>1.556× with 19.0% drawdown</strong>; at 20%, <strong>1.810× with 36.0% drawdown</strong>. Here the cap improves both growth and drawdown. The five-group combination behaves similarly. The narrow core/fade portfolio gives up substantial growth when capped, because its uncapped recovered-price result contains a different mix of large winning and losing bets.</p>
<p>Recovered-price half-Kelly results:</p>{comparison('observed','half')}
<p>For core/favorite fades, half Kelly with a 10% cap gives 1.464× versus 1.602× uncapped: 8.6% less ending wealth, while drawdown falls from 45.8% to 19.0%. A 20% cap retains 1.587×, sacrificing only 0.9% of ending wealth while reducing drawdown to 36.0%. Thus the more attractive cap depends on how much drawdown reduction matters to the bankroll owner.</p>
<details><summary>Recovered-price quarter-Kelly results</summary>{comparison('observed','quarter')}</details>
<h2>Compare every capped trajectory</h2>
<p>All 540 saved paths include every recorded fight. Select a portfolio, probability model, price source and Kelly multiplier. The five curves compare exactly the requested caps with the uncapped baseline.</p>
<div class="controls"><label>Portfolio<select id="portfolio">{options}</select></label><label>Price source<select id="sample"><option value="all">All prices: recovered + reconstructed</option><option value="observed">Recovered quotes only</option></select></label><label>Probability model<select id="model">{model_options}</select></label><label>Kelly multiplier<select id="policy"><option value="full">Full Kelly</option><option value="half">Half Kelly</option><option value="quarter">Quarter Kelly</option></select></label><label><input id="logScale" type="checkbox"> Logarithmic bankroll scale</label></div>
<div id="viewerSummary" class="table-wrap"></div><h3>Bankroll after each settlement</h3><div id="wealthChart" class="chart"></div><h3>Drawdown from the previous peak</h3><div id="drawdownChart" class="chart"></div>
<p class="muted">Grey dashed: uncapped; blue: 5%; green: 10%; orange: 15%; purple: 20%. A positive ending-wealth change means the cap improved growth. “DD reduction” is in percentage points. “Bets reduced” counts positions whose calculated Kelly stake exceeded the cap.</p>
{image('capped_kelly.png','Core plus C1 favorite fades: every stake is recalculated from the current settled bankroll. Price source and base Kelly multiplier both materially affect the cap tradeoff.')}
<h2>The worst drawdown can move</h2>
<p>The maximum drawdown is recomputed over the <strong>whole trajectory</strong>, not just the two losses behind the earlier 50.5% decline. For the core/favorite-fade full-Kelly strategy, a 20% cap reduces that original two-loss episode to 31.0%, but another earlier episode becomes the maximum at <strong>34.7%</strong>. At a 15% cap, the whole-history maximum is 27.75%. For the five-group strategy, a 5% cap leaves a 12.47% maximum decline in a later window.</p>
<p><strong>A per-bet cap is not a maximum-drawdown limit.</strong> Two consecutive full losses at 10% each reduce bankroll by 19%; repeated losses can reduce it further. Capping changes both winning and losing stakes, which is why more generous caps do not always produce higher terminal wealth. For example, in the full-price-history core/favorite-fade half-Kelly results, the 15% cap ends slightly above the 20% cap and has a lower maximum drawdown.</p>
<h2>Probability-model sensitivity</h2>
<p>The table below compares full Kelly with a 10% cap across every tested model and price source. The core/favorite-fade combination remains profitable in all six variants, at 1.417–3.634×, with drawdowns from 15.6% to 35.0%. The cap reduces sensitivity to the largest estimated stakes, but the pooled model's 19.0% drawdown is not a bound across other estimates or future outcomes.</p>
<details><summary>All model/source checks at the 10% cap</summary>{model_table}</details>
<h2>What these tests suggest</h2>
<p><strong>For a substantial reduction in the observed drawdowns, 10%-capped full Kelly is a candidate worth tracking.</strong> It retains 3.566×, 4.079× and 6.509× in the pooled full-price core/favorite-fade, narrow-fade and five-group portfolios, with 19.0%, 19.0% and 21.7% drawdowns. Recovered-price results are much smaller, although the broader portfolios improve on uncapped full Kelly. This is a tradeoff comparison, not a user-selected risk budget.</p>
<p><strong>For preserving more of the aggressive full-history growth, the 20% cap does that.</strong> It retains approximately 94%, 89% and 85% of ending wealth for those three portfolios, but historical drawdowns remain around 35%. <strong>For a modest adjustment to half Kelly, the 15% cap</strong> retains nearly all full-history wealth in the broad core/fade and five-group combinations while removing some of the largest individual positions. Its source-sensitivity results are shown separately.</p>
<p>The 5% cap delivers the smallest drawdowns among these comparisons, at the cost of much more of the favorable full-history growth. No cap is declared universally optimal: the user's four caps are tested as requested, and no preferred drawdown tolerance is assumed. The undefeated intervals still have no historical losing bets to reveal their future drawdown risk, even when capped.</p>
<h2>Evidence and reproducibility</h2>
<p>All 540 paths were independently reconstructed by spending cash, buying the corresponding contract quantity and receiving the recorded payout before sizing the next fight. All 108 uncapped results reconcile to the preceding sequential analysis. The verification checks 80,784 individual current-bankroll cap applications. Categories and probability models remain retrospective research choices; reconstructed prices and pre-card quotes do not establish later executable fills. Caps do not remove selection bias or establish future profitability.</p>
<details><summary>All 540 scenario summaries</summary>{all_table}</details>
<p>Saved data: <a href="../results/capped_kelly_scenarios.csv">scenario metrics and paired changes</a> · <a href="../results/capped_kelly_paths.csv.gz">complete trajectories</a> · <a href="../results/capped_kelly_summary.json">assumptions</a> · <a href="../results/capped_kelly_verification.json">verification</a> · <a href="research_log.md">research record</a></p>
<script id="cappedData" type="application/json">{data_json}</script><script>'''+JAVASCRIPT+'''</script></body></html>'''
    (ROOT/'reports/capped_kelly.html').write_text(text)
    print('Wrote reports/capped_kelly.html',flush=True)


JAVASCRIPT=r'''
const data=JSON.parse(document.getElementById('cappedData').textContent);
const caps={uncapped:null,cap_05:.05,cap_10:.1,cap_15:.15,cap_20:.2};
const labels={uncapped:'Uncapped',cap_05:'5% cap',cap_10:'10% cap',cap_15:'15% cap',cap_20:'20% cap'};
const colors={uncapped:'#666b70',cap_05:'#387ab4',cap_10:'#178270',cap_15:'#b87a25',cap_20:'#8059a4'};
const multipliers={quarter:.25,half:.5,full:1};
const number=x=>x.toLocaleString('en-US',{maximumFractionDigits:3});
const percent=x=>(100*x).toFixed(1)+'%';
const signed=x=>(x>0?'+':'')+percent(x);
function calculate(context,multiplier,cap){
 const wealth=[1],drawdown=[0],stake=[];let peak=1,maxDD=0,maxStake=0,cappedBets=0,bets=0;
 for(let i=0;i<context.f.length;i++){
  const nominal=multiplier*context.f[i],f=cap===null?nominal:Math.min(nominal,cap);
  stake.push(f);if(f>0)bets++;if(f<nominal)cappedBets++;maxStake=Math.max(maxStake,f);
  const next=wealth[wealth.length-1]*(1+f*context.returns[i]);wealth.push(next);peak=Math.max(peak,next);
  const dd=1-next/peak;drawdown.push(dd);maxDD=Math.max(maxDD,dd);
 }
 return {wealth,drawdown,stake,maxDD,maxStake,cappedBets,bets,finalWealth:wealth[wealth.length-1]};
}
function chart(series,field,logarithmic){
 const width=1000,height=280,left=76,right=24,top=20,bottom=42;
 const transform=x=>logarithmic?Math.log10(x):x;
 const values=series.flatMap(s=>s.result[field].map(transform));
 let low=logarithmic?Math.min(...values):0,high=Math.max(...values);if(low===high)high=low+1;
 const last=series[0].result[field].length-1,X=i=>left+i/last*(width-left-right),Y=v=>height-bottom-(transform(v)-low)/(high-low)*(height-top-bottom);
 let body='';for(const fraction of [0,.5,1]){
  const v=low+(high-low)*fraction,y=height-bottom-fraction*(height-top-bottom);
  const label=field==='drawdown'?percent(v):number(logarithmic?10**v:v)+'×';
  body+=`<line x1="${left}" y1="${y}" x2="${width-right}" y2="${y}" stroke="#dce6df"/><text x="${left-10}" y="${y+4}" text-anchor="end" font-size="12" fill="#52695e">${label}</text>`;
  const i=Math.round(last*fraction);body+=`<text x="${X(i)}" y="${height-14}" text-anchor="middle" font-size="12" fill="#52695e">${i}</text>`;
 }
 for(const s of series){const points=s.result[field].map((v,i)=>`${X(i).toFixed(2)},${Y(v).toFixed(2)}`).join(' ');body+=`<polyline points="${points}" fill="none" stroke="${colors[s.cap]}" stroke-width="2"${s.cap==='uncapped'?' stroke-dasharray="7 4"':''}/>`;}
 return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${field==='wealth'?'Bankroll':'Drawdown'} after each fight settlement">${body}</svg>`;
}
function update(){
 const key=['portfolio','sample','model'].map(id=>document.getElementById(id).value).join('|');
 const context=data[key],multiplier=multipliers[document.getElementById('policy').value];
 const series=Object.entries(caps).map(([name,cap])=>({cap:name,result:calculate(context,multiplier,cap)}));
 const base=series[0].result;
 const rows=series.map(s=>{const r=s.result,change=r.finalWealth/base.finalWealth-1,profit=base.finalWealth>1?(r.finalWealth-base.finalWealth)/(base.finalWealth-1):null;
  return `<tr><td style="color:${colors[s.cap]}">${labels[s.cap]}</td><td>${number(r.finalWealth)}×</td><td>${signed(change)}</td><td>${profit===null?'—':signed(profit)}</td><td>${percent(r.maxDD)}</td><td>${(100*(base.maxDD-r.maxDD)).toFixed(2)} pp</td><td>${percent(r.maxStake)}</td><td>${r.cappedBets}/${r.bets}</td></tr>`;}).join('');
 document.getElementById('viewerSummary').innerHTML='<table><thead><tr><th>Cap</th><th>Ending bankroll</th><th>Ending wealth change</th><th>Net profit change</th><th>Max drawdown</th><th>DD reduction</th><th>Largest stake</th><th>Bets reduced</th></tr></thead><tbody>'+rows+'</tbody></table>';
 document.getElementById('wealthChart').innerHTML=chart(series,'wealth',document.getElementById('logScale').checked);
 document.getElementById('drawdownChart').innerHTML=chart(series,'drawdown',false);
}
for(const id of ['portfolio','sample','model','policy','logScale'])document.getElementById(id).addEventListener('change',update);
window.cappedViewer={data,caps,multipliers,calculate,update};update();
'''


if __name__=='__main__':
    run()
