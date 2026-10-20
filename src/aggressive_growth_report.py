"""Explain retrospective growth with sequential settlement and reinvestment."""
import html,json
import pandas as pd
from audit import ROOT
from analyze import plt
from report import table,pct,image

LABELS={
 'core_broad':'Back C4–5 ≥35¢', 'core_bounded':'Back C4–5 35–81¢', 'c4_bounded':'Back C4 35–81¢',
 'fade1_favorites':'Fade C1 favorites', 'fade1_all':'Fade all C1', 'fade1_narrow':'Fade C1 at pick ask 25–70¢',
 'back35_narrow':'Back C3–5 at 35 to <40¢', 'back2_underdogs':'Back C2 at 16–36¢',
 'back3_favorites':'Back C3 at 77 to <86¢', 'fade2_middle':'Fade C2 middle-price picks',
 'broad_core_plus_fade1':'Broad high-confidence core + C1 favorite fades',
 'bounded_core_plus_fade1':'35–81¢ high-confidence core + C1 favorite fades',
 'bounded_core_plus_all_fade1':'35–81¢ core + all C1 fades',
 'bounded_core_plus_narrow_fade1':'35–81¢ core + narrow C1 fades',
 'core_fade1_plus_back2':'Core + C1 favorite fades + C2 underdogs',
 'core_fade1_plus_back3':'Core + C1 favorite fades + C3 favorites',
 'core_fade1_plus_fade2':'Core + C1 favorite fades + C2 middle-price fades',
 'all_disjoint_candidates':'All five disjoint candidate groups',
 'two_unbeaten_intervals':'The two narrow undefeated intervals',
 'narrow_back_plus_broad_fade':'Narrow C3–5 backing + C1 favorite fades',
}
MODELS={'odds_confidence':'Pooled odds/confidence','confidence_categories':'Separate confidence levels','category_offset':'Within-category offset'}


def run():
    d=pd.read_csv(ROOT/'results/aggressive_growth.csv')
    meta=json.loads((ROOT/'results/aggressive_growth_summary.json').read_text())
    paths=pd.read_csv(ROOT/'results/aggressive_growth_paths.csv.gz')
    broad=d[d['sample'].eq('all')&d.cadence.eq('fight')&d.model.eq('odds_confidence')&d.choice.eq('half')]
    broad_table=table([[LABELS[r.portfolio],f'{r.final_wealth:.3f}×' if r.feasible else 'Insufficient cash',pct(r.max_drawdown),r.bets,pct(r.max_stake)] for r in broad.itertuples()],
                      ['Portfolio','Half-Kelly terminal bankroll','Max drawdown','Positions','Largest individual allocation'])
    risk=d[d.portfolio.eq('bounded_core_plus_fade1')&d['sample'].eq('all')&d.cadence.eq('fight')&d.model.eq('odds_confidence')&d.feasible]
    risk_table=table([[r.choice.replace('_',' '),pct(r.fraction,2),f'{r.final_wealth:.3f}×',pct(r.final_wealth-1),pct(r.max_drawdown),pct(r.max_stake),f'{r.worst_single_win_reversal_wealth:.3f}×',pct(r.worst_single_win_reversal_drawdown)] for r in risk.itertuples()],
                     ['Sizing','Kelly fraction','Terminal bankroll','Growth','Max drawdown','Largest allocation','Worst single-win reversal: terminal bankroll','Worst single-win reversal: drawdown'])
    sensitivity=d[d.portfolio.eq('bounded_core_plus_fade1')&d.cadence.eq('fight')&d.choice.eq('half')]
    model_table=table([[r['sample'],MODELS[r.model],f'{r.final_wealth:.3f}×',pct(r.max_drawdown),int(r.bets),pct(r.max_stake)] for _,r in sensitivity.iterrows()],
                      ['Price sample','Probability estimator','Half-Kelly terminal bankroll','Max drawdown','Positions','Largest allocation'])
    chosen=[('odds_confidence','quarter'),('odds_confidence','half'),('odds_confidence','full'),('category_offset','quarter'),('category_offset','half'),('category_offset','full')]
    rows=[]
    for model,choice in chosen:
        r=d[d.portfolio.eq('two_unbeaten_intervals')&d['sample'].eq('all')&d.cadence.eq('fight')&d.model.eq(model)&d.choice.eq(choice)].iloc[0]
        rows.append([MODELS[model],choice.replace('_',' '),pct(r.fraction,2),f'{r.final_wealth:.3f}×' if r.feasible else 'Insufficient cash',
                     int(r.bets),pct(r.max_stake),pct(r.max_drawdown)])
    narrow_table=table(rows,['Probability estimator','Sizing','Kelly fraction','Terminal bankroll','Positions','Largest individual allocation','Historical drawdown'])
    timing_labels={'fight':'Sequential settlement / reinvestment','card':'Simultaneous pre-card allocation'}
    comparison=d[d.portfolio.eq('bounded_core_plus_fade1')&d['sample'].eq('all')&d.model.eq('odds_confidence')&d.choice.isin(['quarter','half','full'])]
    timing_table=table([[timing_labels[r.cadence],r.choice,f'{r.final_wealth:.3f}×' if r.feasible else 'Insufficient simultaneous cash',pct(r.max_drawdown) if r.feasible else '—'] for r in comparison.itertuples()],
                       ['Timing assumption','Sizing','Terminal bankroll','Max drawdown'])
    all_table=table([[LABELS[r.portfolio],r.sample,timing_labels[r.cadence],MODELS[r.model],r.choice.replace('_',' '),pct(r.fraction,2),
                     f'{r.final_wealth:.3f}×' if r.feasible else 'Insufficient cash',pct(r.max_drawdown),int(r.bets),pct(r.max_stake)] for r in d.itertuples()],
                    ['Portfolio','Price sample','Timing','Probability model','Sizing','Fraction','Terminal bankroll','Max drawdown','Positions','Largest allocation'])
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for name,color in [('core_bounded','#879390'),('bounded_core_plus_fade1','#14766b'),('core_fade1_plus_back2','#426caa')]:
        s=paths[paths.portfolio.eq(name)&paths['sample'].eq('all')&paths.cadence.eq('fight')&paths.model.eq('odds_confidence')&paths.choice.eq('half')]
        assert len(s)==188
        axes[0].plot(s.fight_index,s.wealth,label=LABELS[name],color=color)
    for model,choice,label,color in [('odds_confidence','full','Pooled model, full Kelly','#426caa'),('category_offset','quarter','Category model, quarter Kelly','#14766b'),('category_offset','full','Category model, full Kelly','#ad6521')]:
        s=paths[paths.portfolio.eq('two_unbeaten_intervals')&paths['sample'].eq('all')&paths.cadence.eq('fight')&paths.model.eq(model)&paths.choice.eq(choice)]
        assert len(s)==188
        axes[1].plot(s.fight_index,s.wealth,label=label,color=color)
    axes[0].set_title('Broader portfolios at half Kelly');axes[1].set_title('Two undefeated intervals: logarithmic scale');axes[1].set_yscale('log')
    for ax in axes:
        ax.axhline(1,color='#777',lw=.6);ax.set(xlabel='Recorded fight index',ylabel='Bankroll / initial bankroll');ax.grid(alpha=.15);ax.legend(fontsize=7,loc='upper left')
    fig.suptitle('Sequential betting: settle each fight, update bankroll, size the next bet')
    fig.tight_layout();fig.savefig(ROOT/'reports/figures/aggressive_growth.png',dpi=170);plt.close(fig)
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Retrospective aggressive bankroll growth</title><style>
body{{font:16px/1.6 system-ui,sans-serif;max-width:1200px;margin:auto;padding:28px;color:#203b35;background:#f7faf8}}h1,h2{{line-height:1.2}}a{{color:#176c60}}.note{{padding:18px;background:#e7f0eb;border-left:4px solid #247766}}.table-wrap{{overflow:auto;margin:20px 0}}table{{border-collapse:collapse;background:white;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dce6df;text-align:left;vertical-align:top}}th{{background:#e6eee8}}img{{max-width:100%;height:auto}}figure{{margin:24px 0}}figcaption{{font-size:13px;color:#536b62}}details{{margin-top:24px}}code{{background:#e4ebe5;padding:2px 4px}}</style></head><body>
<p><a href="final_report.html">Main report</a> · <a href="interval_trajectories.html">Interval viewer</a> · <a href="lower_confidence.html">Confidence 1–3 and fades</a></p>
<h1>What would have grown the bankroll aggressively?</h1><p>October 19, 2026 · Retrospective comparison of the categories already discovered. No new interval mining.</p>
<div class="note"><strong>Every primary comparison follows the requested sequence: bet one fight, settle it, update the bankroll, then size the next bet.</strong> Backing confidence 4–5 at 35–81¢ and opposing confidence-1 favorites grows one bankroll to <strong>2.858× at half Kelly</strong>, with <strong>26.7% maximum drawdown</strong>. Full Kelly is fundable in this sequential model and produces <strong>5.903×</strong> with <strong>50.5% drawdown</strong>. The largest hindsight gains come from the two narrow undefeated categories; those extreme results depend heavily on selecting winning streaks after seeing outcomes.</div>
<p><strong>Timing correction:</strong> the earlier version made simultaneous pre-card funding primary. Its financing ceiling does not apply when each fight settles before the next bet. The saved analyses already contain both conventions; this report now presents sequential results first and retains pre-card funding as a separately labeled sensitivity.</p>
<h2>What is being compared</h2>
<p>Twenty single-rule and combined portfolios, three probability estimators, two price samples, two execution conventions, and four sizing choices produce {meta['scenario_count']} scenarios. The {meta['feasible_scenarios']} feasible trajectories each include all 187 recorded rows plus the initial bankroll. The candidates come from the earlier analyses. These results identify winners <em>within this comparison</em>, not a mathematical optimum over every conceivable strategy.</p>
<p>Each portfolio uses one bankroll, at most one position per fight, and a common Kelly fraction across its constituent categories. The probability-model family is also common across constituents. All primary tables and plots refit probabilities after earlier fights settle and reinvest the updated bankroll. Each stake is a fraction of the bankroll immediately before that fight. There is no requirement to finance every selected fight on a card simultaneously.</p>
<p>Prices are Kalshi purchase asks with modeled taker fees; missing prices remain reconstructed. For an opposite-side purchase, NO ask is one minus the YES bid, preserving the spread under <a href="https://docs.kalshi.com/getting_started/orderbook_responses">Kalshi's documented reciprocal pricing</a>. Intervals described for fades refer to the original pick's YES ask. These are scalable bankroll ratios, not evidence that large dollar amounts could have been filled.</p>
<p>The sequential scenario assumes each result settles in time to fund the next fight and that the saved or reconstructed price remains obtainable when that next bet is placed. Actual execution depth and intra-card quote changes are unavailable. The primary comparison accepts the requested settlement assumption; the historical prices remain proxies for execution.</p>
<h2>What a 27.5% half-Kelly stake means</h2>
<p><strong>It means risking 27.5% on one fight.</strong> Half Kelly halves the model's calculated optimal bankroll fraction; it does not impose a small fixed percentage. For a binary contract with all-in cost <code>c</code> and estimated win probability <code>p</code>, full Kelly is <code>max(0, (p − c) / (1 − c))</code>. Multiply that fraction by the chosen Kelly multiplier and the current bankroll to obtain the all-in stake.</p>
<p>At the largest half-Kelly position in the full-history pooled combination, the ask is 77¢, modeled all-in cost is 78.2397¢, and the estimated win probability is 90.2151%. That produces 55.0334% full Kelly and <strong>27.5167% half Kelly</strong>. With $1,000 immediately before this fight, the illustrative all-in stake is $275.17. A loss leaves $724.83; a win leaves $1,076.53 under the scalable fee approximation. The next stake is calculated from whichever new balance results and the next fight's updated estimate. The 90.2% is a model forecast, not an independently established win probability.</p>
<p><strong>Why can estimated full Kelly be as large as 55%?</strong> The model assumes a substantial advantage: about 90.2% probability against a 78.24% break-even probability. An independent maximization of expected log bankroll reproduces the 55.0% stake. The questionable input is the probability estimate, not an extra card allocation or a doubling error. Full and half Kelly use the same forecast but their own current bankrolls, so their dollar stakes need not be in a two-to-one ratio.</p>
<p>The forecast used <strong>97 earlier fights, including only four confidence-5 picks (three wins)</strong>. It combines odds and confidence across all levels through a fitted logistic model; it is not an observed 90.2% win rate in confidence 5. Refitting from strictly earlier outcomes reproduces the forecast. The small confidence-5 sample and model dependence do not establish the precision needed to endorse a 55% individual stake.</p>
<p>Holding that contract's cost fixed, an illustrative true win probability of 80% would imply <strong>8.1% full Kelly / 4.0% half Kelly</strong>; 85% would imply <strong>31.1% / 15.5%</strong>; the fitted 90.2% implies <strong>55.0% / 27.5%</strong>. At the 78.24% break-even probability, the stake is zero. These are sensitivity examples, not alternative calibrated forecasts or risk targets. The saved <a href="../results/largest_stake_diagnostic.json">stake diagnostic</a> records the inputs, prior sample sizes and independent calculation.</p>
<p>For win/loss settlement, the recursion is <code>B_next = B × (1 − f + f × y / c)</code>, where <code>f</code> is the all-in bankroll fraction, <code>y = 1</code> for a win and <code>y = 0</code> for a loss. We reset the bankroll base after each fight while retaining earlier outcomes for probability estimation. Current bankroll and the model's current estimates provide the state for the next decision.</p>
<h2>Choice 1: a broader aggressive combination</h2>
<p>The two components address different signals: confidence-4–5 picks perform better after excluding cheap prices, while confidence-1 favorites have been unusually poor picks and therefore profitable to oppose. They never bet the same fight. Combining them improves the historical growth without adding another overlapping stake on an existing position.</p>
<p>The simplest version uses confidence 4–5 at any ask ≥35¢ and fades confidence-1 favorites. At half Kelly it returns <strong>2.696×</strong> with 26.1% drawdown. Using the retrospectively stronger upper boundary of 81¢ improves terminal wealth to <strong>2.858×</strong>, with 26.7% drawdown. The upper cut contributes less historical growth than adding the confidence-1 fades: the bounded backing core alone produces 2.041×.</p>
{risk_table}
<p>The largest individual stake rises from <strong>13.8%</strong> at quarter Kelly to <strong>27.5%</strong> at half Kelly and <strong>55.0%</strong> at full Kelly, always measured against the bankroll immediately before that fight. Full Kelly is also the retrospective wealth-maximizing multiplier within the tested fractional-Kelly domain [0,1] for this pooled, full-history combination. It requires no simultaneous card funding.</p>
<p>The quarter-to-half increase raises terminal growth from +76.0% to +185.8%, while historical drawdown rises from 13.7% to 26.7%. Full Kelly reaches +490.3% but experiences a 50.5% peak-to-trough decline. Sequential compounding removes the pre-card funding ceiling; it does not remove the effect of a large individual loss. Without a user-selected drawdown preference, these are tradeoffs rather than a unique recommended fraction. The distinction between growth optimization and drawdown control is formalized in <a href="https://arxiv.org/abs/1603.06183">Busseti, Ryu and Boyd's risk-constrained Kelly study</a>.</p>
<h2>Does that combination depend on one model or reconstructed quotes?</h2>
{model_table}
<p>Five of the six half-Kelly versions remain profitable. Observed-only pooled fitting and trading gives <strong>1.602× over thirteen positions</strong>, with 45.8% drawdown. The observed-only category-offset version finishes at 0.964×. Larger full-history gains from category-based estimates come with larger drawdowns. The filters were chosen using the complete historical study, so these source and model checks are not untouched validation samples.</p>
<p>In the observed-only pooled version, quarter Kelly gives 1.390× with 24.6% drawdown, half Kelly 1.602× with 45.8%, and full Kelly 1.163× with 79.1%. More aggressive sizing therefore does not consistently produce more terminal wealth across the price samples. The full-history optimum is not a dependable estimate of the best future fraction.</p>
<h2>Choice 2: concentrate on the two undefeated categories</h2>
<ol><li>Back confidence 3–5 at YES asks from <strong>35¢ to below 40¢</strong>: eight wins from eight qualifying fights.</li><li>Oppose confidence-1 picks whose original YES ask was <strong>25–70¢</strong>: six wins from six qualifying fights; actual NO asks ranged from 32 to 76¢.</li></ol>
<p>The categories are disjoint. A category-offset model learns each category's excess win probability from earlier category outcomes only. Because every later qualifying result was a win, its estimated advantage and bankroll both grow. This converts a small winning streak into an enormous compounding outcome.</p>
{narrow_table}
<p>With category-offset estimates, sequential quarter Kelly gives <strong>18.615×</strong>, half Kelly <strong>187.927×</strong>, and full Kelly <strong>6,595.711×</strong>, across twelve positions after initial learning. All three can be funded fight by fight. The largest individual fractions are 22.2%, 44.4% and 88.9%, respectively. Full Kelly maximizes retrospective wealth within [0,1] because every executed bet in these selected categories won.</p>
<p>With pooled probabilities, the same categories give <strong>2.803× at half Kelly and 7.000× at full Kelly</strong>, across nine positions. Much of the extreme growth therefore comes from the local probability estimator's interpretation of the small undefeated sample. The very large category-offset results are mathematically consistent with twelve winning bets carrying large bankroll allocations. Their zero-loss history and retrospective selection make them fragile evidence for future betting, even while accepting sequential settlement.</p>
<p>Using only observed quotes for fitting and trading leaves four category-model positions. Quarter Kelly produces 2.029× and full Kelly 9.224×. The much smaller sample cannot establish that the all-history extreme growth rate persists. All eight narrow backing wins and all six narrow fade wins were used in discovering the filters, so forecast fitting on earlier outcomes does not remove the selection bias.</p>
<p>Zero historical drawdown here means no chosen bet lost. It is not a bound on future drawdown. A loss at the largest full-Kelly stake would remove <strong>88.9% of the then-current bankroll</strong>; at half Kelly the corresponding loss is 44.4%. The mathematical growth figures are backtest results, but the probability error that matters most is not measured by their zero-loss paths.</p>
{image('aggressive_growth.png','Every curve includes all recorded fights and reinvests only after each preceding fight settles. The right panel uses a logarithmic wealth axis. Categories and model choices are retrospective.')}
<h2>Would adding more categories have helped?</h2>
{broad_table}
<p>Adding confidence-2 underdogs to the bounded core/fade combination improves pooled half-Kelly wealth from 2.858× to 3.856×. However, this increase comes from <strong>one additional position</strong>; observed-only wealth decreases from 1.602× to 1.539×. That single full-history win does not establish a reliable second growth engine.</p>
<p>Adding confidence-2 middle-price fades gives 3.611× in the full history and 1.780× in the observed-only pooled case. Confidence-3 favorites raise full-history wealth to 2.950× and add no observed-only bets. All five groups together give <strong>5.027× at half Kelly</strong>, with 26.7% drawdown; observed-only wealth is 1.709× with 45.8% drawdown. This combined strategy is fundable sequentially. Its higher historical return is worth tracking, but the extra filters were all chosen retrospectively and the confidence-2 underdog contribution is concentrated in one bet.</p>
<p>The narrower confidence-1 fade improves retrospective growth relative to fading all favorites, but it also removes a losing fade just above the 70¢ boundary. The broader favorite rule is the more interpretable hypothesis to test with new data. Blanket confidence-1 fading produces less growth and more drawdown than favorite-only fading in this comparison.</p>
<h2>My judgment</h2>
<p><strong>For the question “what would have made the most money in this workbook?”</strong>, the narrow undefeated pair with category-specific probability estimates is the answer among the compared portfolios. Its sequential quarter-Kelly version grows 18.615× and full Kelly reaches 6,595.711×. It deserves to be shown, with the explicit explanation that selected winning streaks and increasingly confident local estimates drive the result.</p>
<p><strong>For an aggressive strategy worth testing on subsequent picks</strong>, the high-confidence backing core plus confidence-1 favorite fades remains an interpretable framework. Quarter, half and full Kelly are fundable under the requested timing. The pooled full-history result favors full Kelly, but its drawdown exceeds 50%, and observed-only full Kelly ends below half Kelly with a 79.1% drawdown. Half Kelly is an aggressive comparison point, not a demonstrated optimal future fraction. The simpler ≥35¢ core is close to the optimized 35–81¢ version.</p>
<p>I would treat confidence-2 underdogs and the two narrow winning categories as separately tracked hypotheses before allowing their small local samples to dominate stake estimates. Use the pooled probability estimate as the main sizing comparison, update only after settlement, price the actual side including fees, and do not stack overlapping categories. Search-adjusted evidence from the earlier reports still does not establish a profitable strategy. This new portfolio comparison adds hindsight choices and does not strengthen that statistical confidence.</p>
<details><summary>Secondary sensitivity: simultaneous pre-card funding</summary>
<p>This convention sizes and reserves all selected positions from the bankroll at the start of a card, without recycling that card's payouts. It is a different financing assumption from the requested sequential model. For the bounded high-confidence core plus confidence-1 favorite fades, it gives:</p>
{timing_table}
<p>The pre-card common-fraction ceiling is 64.58% of Kelly, whereas full Kelly is fundable sequentially. For the narrow undefeated pair with category-offset estimates, the pre-card ceiling is 40.90% of Kelly, versus fundable full Kelly sequentially. These ceilings explain the earlier cash warning; they place no constraint on the primary sequential simulations.</p>
</details>
<h2>Verification and interpretation of the stress numbers</h2>
<p>Every combination is checked for duplicate fight exposure. The new single-rule results reconcile with earlier outputs, and independent cash/contract accounting reproduces all feasible trajectories. The continuous fraction optimum is checked using the concavity and first-order conditions of log terminal wealth over the cash-feasible interval. Single-win reversals change one executed winning payout to a loss while holding every forecast stake fraction fixed; they do not refit subsequent probabilities, change interval selection, or estimate a probability of loss. The worst terminal and worst drawdown can come from different reversed fights.</p>
<details><summary>All {meta['scenario_count']} scenarios</summary>{all_table}</details>
<p>Files: <a href="../results/aggressive_growth.csv">scenario summaries</a>, <a href="../results/aggressive_growth_paths.csv.gz">complete fight trajectories</a>, <a href="../results/aggressive_growth_summary.json">portfolio definitions and assumptions</a>, and <a href="research_log.md">research record</a>. Proprietary source rows and intermediate per-fight forecasts stay local.</p></body></html>'''
    (ROOT/'reports/aggressive_growth.html').write_text(text)
    print('Wrote reports/aggressive_growth.html',flush=True)


if __name__=='__main__':run()
