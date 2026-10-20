"""Separate confidence ranking, price-relative information and trading edge.

Reuses the saved 10,000 whole-card refits and 161 earlier-outcome forecasts.
No new interval search, resampling precision choice or stake rule is introduced.
"""
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit import ROOT
from kelly import designs, fit_firth
from confidence23 import interval_mean
from report import pct, table, image


PAIRS=[('confidence_offset','market_offset'),('odds_confidence','odds_only'),
       ('odds_confidence_interaction','odds_only'),('confidence_categories','odds_only'),
       ('odds_confidence','market_probability')]
NAMES={'confidence_offset':'Price offset + confidence','market_offset':'Price offset',
       'odds_confidence':'Fitted odds + confidence','odds_only':'Fitted odds only',
       'odds_confidence_interaction':'Odds × confidence interaction',
       'confidence_categories':'Separate confidence categories','market_probability':'Unadjusted quote'}


def run():
    out=ROOT/'results'
    full=pd.read_csv(ROOT/'data/cleaned/kalshi_fights.csv')
    d=full[full.y.notna()].copy()
    by_level=d.groupby('confidence').agg(n=('y','size'),wins=('y','sum'),win_rate=('y','mean'),
        mean_ask=('kalshi_price','mean'),mean_all_in_cost=('kalshi_cost','mean'),roi=('kalshi_return','mean')).reset_index()
    by_level['win_rate_minus_mean_ask']=by_level.win_rate-by_level.mean_ask
    by_level.to_csv(out/'confidence_signal_levels.csv',index=False)
    prior=pd.read_csv(ROOT/'artifacts/private/prequential_predictions.csv')
    prior=prior[prior.odds_confidence.notna()].copy()
    y=prior.y.to_numpy()
    def loss(p):return -y*np.log(p)-(1-y)*np.log1p(-p)
    paired=[]
    for model,baseline in PAIRS:
        p=prior[model].to_numpy();q=prior[baseline].to_numpy()
        delta=loss(q)-loss(p)
        improvement,low,high=interval_mean(delta,prior.date)
        paired.append(dict(model=model,baseline=baseline,n=len(prior),cards=prior.date.nunique(),
            model_log_loss=loss(p).mean(),baseline_log_loss=loss(q).mean(),log_loss_improvement=improvement,
            improvement_ci_low=low,improvement_ci_high=high,brier_improvement=np.mean((q-y)**2-(p-y)**2)))
    paired=pd.DataFrame(paired)
    paired.to_csv(out/'confidence_signal_paired_scores.csv',index=False)
    saved=np.load(ROOT/'artifacts/private/calibration_bootstrap_coefficients.npz')
    beta=saved['fitted'];coefficients=saved['coefficients'];slopes=coefficients[:,2]
    sensitivity=[]
    for sample in ('all','observed'):
        g=d if sample=='all' else d[d.price_source.eq('api_precard_ask')]
        for lo,hi in [(1,5),(2,5),(1,4),(2,4)]:
            h=g[g.confidence.between(lo,hi)]
            x,offset=designs(h)['odds_confidence']
            b=fit_firth(x,h.y.to_numpy(),offset)
            sensitivity.append(dict(sample=sample,confidence_low=lo,confidence_high=hi,n=len(h),
                intercept=b[0],logit_price=b[1],confidence_slope=b[2],per_level_odds_ratio=np.exp(b[2])))
    pd.DataFrame(sensitivity).to_csv(out/'confidence_signal_sensitivity.csv',index=False)
    influence=pd.read_csv(out/'calibration_card_influence.csv')
    summary=dict(fights=len(d),cards=int(d.date.nunique()),bootstrap_refits=len(slopes),
        confidence_logodds_slope=float(beta[2]),per_level_odds_ratio=float(np.exp(beta[2])),
        odds_ratio_p025=float(np.exp(np.quantile(slopes,.025))),odds_ratio_p975=float(np.exp(np.quantile(slopes,.975))),
        bootstrap_positive_slope_frequency=float(np.mean(slopes>0)),
        leave_card_out_slope_min=float(influence.confidence.min()),leave_card_out_slope_max=float(influence.confidence.max()),
        illustrative_probability_change_from_50pct=float(expit(beta[2])-.5),
        paired_forecasts=len(prior),
        notes=['The confidence coefficient is fitted without imposing a positive sign; equally spaced log-odds effects are a model assumption.',
               'Bootstrap frequencies are conditional resampling stability, not posterior probabilities of true edge.',
               'Price averages are quoted/reconstructed asks, not established fair probabilities; win-rate differences are descriptive, not ROI.',
               'Paired forecast differences use the original common 24-fight prefix and identical later observations.',
               'Improvement is baseline loss minus model loss: positive is better.',
               'Extreme-level exclusions are sensitivity checks, not independently validated models.',
               'No untouched future outcomes are available. C2/C3 intervals retain their earlier search-selection limitations.'])
    (out/'confidence_signal_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Direct reconciliation against existing aggregates and an independently
    # accumulated cluster variance for each new paired comparison.
    cats=pd.read_csv(out/'categories.csv').set_index('category')
    for r in by_level.itertuples():
        old=cats.loc[f'C{r.confidence} / all_prices']
        for column in ('n','wins','win_rate','roi'):assert np.isclose(getattr(r,column),old[column])
    old_summary=json.loads((out/'calibration_uncertainty_summary.json').read_text())
    assert np.isclose(summary['odds_ratio_p025'],np.exp(old_summary['coefficient_p025'][2]))
    assert np.isclose(summary['odds_ratio_p975'],np.exp(old_summary['coefficient_p975'][2]))
    for r in paired.itertuples():
        a=prior[r.model].to_numpy();b=prior[r.baseline].to_numpy()
        delta=np.where(y==1,np.log(a/b),np.log((1-a)/(1-b)))
        sums=[sum(delta[prior.date.to_numpy()==day]-delta.mean()) for day in prior.date.unique()]
        se=np.sqrt(len(sums)/(len(sums)-1)*sum(v*v for v in sums))/len(delta)
        assert np.isclose(delta.mean(),r.log_loss_improvement)
        assert np.isclose(delta.mean()-t.ppf(.975,len(sums)-1)*se,r.improvement_ci_low)
        assert np.isclose(delta.mean()+t.ppf(.975,len(sums)-1)*se,r.improvement_ci_high)
    verification=dict(status='PASS',level_aggregates=len(by_level),paired_score_comparisons=len(paired),cached_refits=len(slopes),sensitivity_fits=len(sensitivity))
    (out/'confidence_signal_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    # A raw trend and price-adjusted coefficient interval are different claims.
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    axes[0].plot(by_level.confidence,100*by_level.win_rate,'o-',label='Observed win rate')
    axes[0].plot(by_level.confidence,100*by_level.mean_ask,'s--',label='Mean YES ask (not fair probability)')
    for r in by_level.itertuples():axes[0].annotate(f'{int(r.wins)}/{r.n}',(r.confidence,100*r.win_rate),xytext=(0,9),textcoords='offset points',ha='center')
    axes[0].set(xticks=[1,2,3,4,5],xlabel='System confidence',ylabel='Percent',title='Higher confidence has higher historical correctness')
    axes[0].legend(fontsize=8,loc='lower right');axes[0].grid(alpha=.15)
    odds=np.exp(slopes);ordered=np.sort(odds)
    axes[1].plot(ordered,np.arange(1,len(ordered)+1)/len(ordered),color='#176c60')
    axes[1].axvline(1,color='#777',ls='--',label='No price-adjusted confidence effect')
    axes[1].axvspan(summary['odds_ratio_p025'],summary['odds_ratio_p975'],color='#176c60',alpha=.1,label='Central 95% of card refits')
    axes[1].set(xlabel='Win-odds multiplier per confidence level',ylabel='Cumulative fraction of card resamples',title='Positive direction, uncertain effect size')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.15)
    fig.tight_layout();fig.savefig(ROOT/'reports/figures/confidence_signal.png',dpi=170);plt.close(fig)
    levels=table([[int(r.confidence),f'{int(r.wins)}/{r.n}',pct(r.win_rate),pct(r.mean_ask),f'{100*r.win_rate_minus_mean_ask:+.1f} pp',pct(r.roi)] for r in by_level.itertuples()],
                 ['Confidence','Wins / fights','Win rate','Mean quoted YES ask','Win rate minus mean ask','Flat net ROI'])
    comparisons=table([[NAMES[r.model],NAMES[r.baseline],f'{r.model_log_loss:.4f}',f'{r.baseline_log_loss:.4f}',f'{r.log_loss_improvement:+.4f}',
                        f'{r.improvement_ci_low:+.4f} to {r.improvement_ci_high:+.4f}'] for r in paired.itertuples()],
                      ['Model','Comparison','Model log loss ↓','Comparison log loss ↓','Improvement ↑','Approx. 95% paired card-cluster interval'])
    sensitivity_table=table([[r['sample'],f"{r['confidence_low']}–{r['confidence_high']}",r['n'],f"{r['confidence_slope']:.3f}",f"{r['per_level_odds_ratio']:.3f}×"] for r in sensitivity],
                            ['Prices','Included confidence levels','Fights','Confidence slope','Win-odds multiplier per level'])
    body=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>How much signal is in the confidence scale?</title><style>
body{{font:16px/1.65 system-ui,sans-serif;max-width:1150px;margin:auto;padding:28px;background:#f7faf8;color:#203b35}}h1,h2{{line-height:1.25}}a{{color:#176c60}}.note{{background:#e6f0ea;padding:18px;border-left:4px solid #247766}}.table-wrap{{overflow-x:auto;margin:22px 0}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dce6df;text-align:left;vertical-align:top}}th{{background:#e6eee8}}img{{max-width:100%}}figcaption,.muted{{color:#52695e;font-size:14px}}figure{{margin:24px 0}}@media(max-width:650px){{body{{padding:16px}}h1{{font-size:27px}}}}
</style></head><body><p><a href="confidence23.html">C2/C3 candidates and trajectories</a> · <a href="interval_trajectories.html">Interval viewer</a> · <a href="final_report.html">Main report</a></p>
<h1>The confidence scale is more credible than the precise C2/C3 betting bands</h1><p class="muted">October 20, 2026 · 185 decisive fights across 15 cards · existing forecasts and card refits reused</p>
<div class="note"><strong>Keep confidence as a potentially useful ranking feature; keep the discovered C2/C3 rules experimental.</strong> Higher confidence tracks higher correctness, and the fitted relationship remains positive after controlling for price. The effect size is uncertain, and the incremental predictive improvement on later fights is small and statistically unresolved. This is stronger evidence for a broad ranking signal than for a profitable exact interval.</div>
<h2>Are the C2/C3 strategies worth using?</h2>
<p>They are worth tracking under fixed definitions on new picks. The present evidence does not justify treating them as established profit sources or trusting their fitted Kelly stakes. My assessment of the exact bands is low confidence: none has convincing support after the broader search adjustment, most favorable cells are small, and the tested earlier-data selection methods generally lose. Positive returns under several models or card omissions reuse the same chosen outcomes.</p>
<p>The C3 65–70¢ fade illustrates the distinction: +65.8% historical ROI, a nominal p-value near 0.034, but about 0.988 after searching all C3 fade intervals. The C2 below-37¢ backing rule has +70.7% ROI but only eleven fights and an approximate interval from −86.3% to +227.7%. These are research candidates, not probabilities of future profit. The numerical labels “97.9% posterior edge probability” or a positive narrow-band confidence interval would condition on a category selected in hindsight and cannot establish that the discovered strategy works.</p>
<p>If the objective is a disciplined betting strategy, I would keep these rules in prospective tracking rather than add them to the funded core solely because of their backtests. Someone deliberately choosing live experimentation would be accepting uncertain edge; the dataset does not determine a justified experimental bankroll amount. Caps address stake concentration and do not validate a prediction. Confidence 1 and 4–5 strategies also lack independent confirmation.</p>
<h2>Does higher confidence mean more correct picks?</h2>{levels}
<p><strong>Descriptively, yes:</strong> the hit rate rises at every confidence level. The small 1.5-percentage-point increase from C2 to C3 is much less persuasive than the broad separation between low and high confidence. C1 has only eleven observations and C5 only seven, so their extreme rates are especially uncertain.</p>
<p>Raw correctness and tradable edge are different quantities. C3 wins slightly more often than C2, but its picks have much higher average asks: 60.1¢ versus 51.5¢. That helps explain why C3's better raw hit rate coexists with a worse betting return. Mean ask is a descriptive price benchmark, not a known fair win probability; mean win rate minus mean ask is not ROI. Actual ROI uses each fight's own cost.</p>
<h2>How much information remains after accounting for odds?</h2>
<p>The existing pooled model fits <code>logit(win probability) = intercept + price slope × logit(YES ask) + confidence slope × (confidence − 3)</code>. Its confidence slope is <strong>{beta[2]:.3f}</strong>, equivalent to multiplying win odds by <strong>{summary['per_level_odds_ratio']:.2f}</strong> per level at the same price. Around a 50% starting win probability, that point estimate is roughly <strong>{100*summary['illustrative_probability_change_from_50pct']:.1f} percentage points</strong> per level. This is an illustration of the fitted effect, not a calibrated conversion of every confidence label into a win probability.</p>
<p>The 10,000 previously saved whole-card refits put the central 95% range for that odds multiplier at <strong>{summary['odds_ratio_p025']:.2f} to {summary['odds_ratio_p975']:.2f}</strong>. It includes no additional effect, represented by 1.0. The coefficient is positive in <strong>{pct(summary['bootstrap_positive_slope_frequency'])}</strong> of those refits. That is encouraging directional stability, <strong>not a {pct(summary['bootstrap_positive_slope_frequency'])} probability that there is true trading edge</strong>. The model, prices, observed cards and retrospective research choices remain conditioning assumptions.</p>
<p>Omitting any one card leaves a positive confidence slope, from {summary['leave_card_out_slope_min']:.3f} to {summary['leave_card_out_slope_max']:.3f}. Thus one card does not explain the whole pattern. The positive sign is estimated rather than imposed, but the model assumes equal log-odds increments between levels.</p>
{image('confidence_signal.png','Left: observed correctness rises with confidence; labels show wins and sample sizes. Quoted asks provide context rather than true probabilities. Right: the existing whole-card bootstrap distribution of the fitted odds multiplier; the central 95% range includes no additional confidence effect.')}
<h2>Is the pattern entirely driven by C1 and C5?</h2>{sensitivity_table}
<p>No single endpoint is necessary for a positive fitted direction. With both C1 and C5 removed, the all-price slope falls from 0.296 to 0.148, about half as large, while remaining positive. Recovered-only fits also retain positive slopes. These exclusions are influence checks, not new independent evidence or confidence intervals. Recovered-only observations also cover a different mix of fights and dates.</p>
<p>A smooth positive overall slope does not mean every adjacent level adds an equal amount of information. In particular, C2 slightly exceeds its average quote while C3 falls below its quote. The confidence scale may be a noisy ranking device with an uneven middle rather than a precise probability scale.</p>
<h2>Does adding confidence improve predictions made from earlier outcomes?</h2>
<p>This paired comparison holds the 161 later fights and the original 24-fight training prefix fixed. Each forecast was fitted only to earlier outcomes. Log loss evaluates the probability forecast rather than just whether it chose the winning side; lower is better. Brier and logarithmic scoring are standard ways to evaluate probabilistic predictions, as discussed by <a href="https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf">Gneiting and Raftery</a>.</p>
{comparisons}
<p>Adding linear confidence reduces the fitted odds-only model's log loss from 0.6175 to 0.6140. The improvement is 0.0035 per fight, with an approximate paired 95% interval of −0.0139 to +0.0209. The direction is encouraging but the uncertainty spans both deterioration and improvement. The simpler offset comparison tells a similar story. More flexible confidence representations do not improve the point estimate.</p>
<p>The unadjusted quote baseline still scores better at 0.6043. Therefore both statements can be true: confidence slightly helps a fitted odds model in this sample, and the resulting model does not yet beat the market-price benchmark. Neither proves a narrow profitable subset impossible. The whole investigation is retrospective, so these historical forecast checks are not an untouched prospective test.</p>
<h2>How I would use the evidence</h2>
<p>Retain the confidence label and its ordinal relationship as a candidate input. Estimate its weight conservatively rather than converting C5 directly into a high win probability or assuming C3 must beat C2 relative to price. Track the fixed C2/C3 hypotheses on new, timestamped picks, including actual YES/NO quotes, fees, skipped trades and system-version changes. Evaluate their returns separately from whether confidence improves probability accuracy.</p>
<p>My current judgment is that the scale probably carries some ranking information, but the degree of incremental information beyond prices is uncertain. The evidence for the exact C2/C3 betting boundaries is materially weaker. There is no defensible single percentage probability of genuine edge from these analyses. Choosing many profitable-looking historical configurations is the selection problem described by <a href="https://scholarworks.wmich.edu/math_pubs/40/">Bailey and coauthors</a>.</p>
<p>Saved outputs: <a href="../results/confidence_signal_levels.csv">level statistics</a>, <a href="../results/confidence_signal_paired_scores.csv">paired forecast comparisons</a>, <a href="../results/confidence_signal_sensitivity.csv">endpoint/source sensitivity</a>, <a href="../results/confidence_signal_summary.json">effect-size summary</a>, and <a href="research_log.md">research record</a>.</p>
</body></html>'''
    (ROOT/'reports/confidence_signal.html').write_text(body)
    print(json.dumps(summary,indent=2))
    print(json.dumps(verification))


if __name__=='__main__':run()
