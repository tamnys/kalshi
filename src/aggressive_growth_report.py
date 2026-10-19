"""Explain retrospective growth choices with explicit financing and model risk."""
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
    broad=d[d['sample'].eq('all')&d.cadence.eq('card')&d.model.eq('odds_confidence')&d.choice.eq('half')]
    broad_table=table([[LABELS[r.portfolio],f'{r.final_wealth:.3f}×' if r.feasible else 'Insufficient cash',pct(r.max_drawdown),r.bets,pct(r.max_stake)] for r in broad.itertuples()],
                      ['Portfolio','Half-Kelly terminal bankroll','Max drawdown','Positions','Largest individual allocation'])
    risk=d[d.portfolio.eq('bounded_core_plus_fade1')&d['sample'].eq('all')&d.cadence.eq('card')&d.model.eq('odds_confidence')&d.feasible]
    risk_table=table([[r.choice.replace('_',' '),pct(r.fraction,2),f'{r.final_wealth:.3f}×',pct(r.final_wealth-1),pct(r.max_drawdown),pct(r.max_stake),f'{r.worst_single_win_reversal_wealth:.3f}×',pct(r.worst_single_win_reversal_drawdown)] for r in risk.itertuples()],
                     ['Sizing','Kelly fraction','Terminal bankroll','Growth','Max drawdown','Largest allocation','Worst single-win reversal: terminal bankroll','Worst single-win reversal: drawdown'])
    sensitivity=d[d.portfolio.eq('bounded_core_plus_fade1')&d.cadence.eq('card')&d.choice.eq('half')]
    model_table=table([[r['sample'],MODELS[r.model],f'{r.final_wealth:.3f}×',pct(r.max_drawdown),int(r.bets),pct(r.max_stake)] for _,r in sensitivity.iterrows()],
                      ['Price sample','Probability estimator','Half-Kelly terminal bankroll','Max drawdown','Positions','Largest allocation'])
    chosen=[('odds_confidence','quarter'),('odds_confidence','half'),('odds_confidence','full'),('category_offset','quarter'),('category_offset','half'),('category_offset','hindsight_best_feasible')]
    rows=[]
    for model,choice in chosen:
        r=d[d.portfolio.eq('two_unbeaten_intervals')&d['sample'].eq('all')&d.cadence.eq('card')&d.model.eq(model)&d.choice.eq(choice)].iloc[0]
        rows.append([MODELS[model],choice.replace('_',' '),pct(r.fraction,2),f'{r.final_wealth:.3f}×' if r.feasible else 'Insufficient cash',
                     int(r.bets),pct(r.max_stake),pct(r.maximum_card_allocation),pct(r.max_drawdown)])
    narrow_table=table(rows,['Probability estimator','Sizing','Kelly fraction','Terminal bankroll','Positions','Largest allocation','Largest total card allocation','Historical drawdown'])
    all_table=table([[LABELS[r.portfolio],r.sample,r.cadence,MODELS[r.model],r.choice.replace('_',' '),pct(r.fraction,2),
                     f'{r.final_wealth:.3f}×' if r.feasible else 'Insufficient cash',pct(r.max_drawdown),int(r.bets),pct(r.max_stake)] for r in d.itertuples()],
                    ['Portfolio','Price sample','Timing','Probability model','Sizing','Fraction','Terminal bankroll','Max drawdown','Positions','Largest allocation'])
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for name,color in [('core_bounded','#879390'),('bounded_core_plus_fade1','#14766b'),('core_fade1_plus_back2','#426caa')]:
        s=paths[paths.portfolio.eq(name)&paths['sample'].eq('all')&paths.cadence.eq('card')&paths.model.eq('odds_confidence')&paths.choice.eq('half')]
        assert len(s)==188
        axes[0].plot(s.fight_index,s.wealth,label=LABELS[name],color=color)
    for model,choice,label,color in [('odds_confidence','full','Pooled model, full Kelly','#426caa'),('category_offset','quarter','Category model, quarter Kelly','#14766b'),('category_offset','hindsight_best_feasible','Category model, cash-feasible hindsight optimum','#ad6521')]:
        s=paths[paths.portfolio.eq('two_unbeaten_intervals')&paths['sample'].eq('all')&paths.cadence.eq('card')&paths.model.eq(model)&paths.choice.eq(choice)]
        assert len(s)==188
        axes[1].plot(s.fight_index,s.wealth,label=label,color=color)
    axes[0].set_title('Broader portfolios at half Kelly');axes[1].set_title('Two undefeated intervals: logarithmic scale');axes[1].set_yscale('log')
    for ax in axes:
        ax.axhline(1,color='#777',lw=.6);ax.set(xlabel='Recorded fight index',ylabel='Bankroll / initial bankroll');ax.grid(alpha=.15);ax.legend(fontsize=7,loc='upper left')
    fig.suptitle('Pre-card cash allocations; equity settles fight by fight')
    fig.tight_layout();fig.savefig(ROOT/'reports/figures/aggressive_growth.png',dpi=170);plt.close(fig)
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Retrospective aggressive bankroll growth</title><style>
body{{font:16px/1.6 system-ui,sans-serif;max-width:1200px;margin:auto;padding:28px;color:#203b35;background:#f7faf8}}h1,h2{{line-height:1.2}}a{{color:#176c60}}.note{{padding:18px;background:#e7f0eb;border-left:4px solid #247766}}.table-wrap{{overflow:auto;margin:20px 0}}table{{border-collapse:collapse;background:white;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dce6df;text-align:left;vertical-align:top}}th{{background:#e6eee8}}img{{max-width:100%;height:auto}}figure{{margin:24px 0}}figcaption{{font-size:13px;color:#536b62}}details{{margin-top:24px}}code{{background:#e4ebe5;padding:2px 4px}}</style></head><body>
<p><a href="final_report.html">Main report</a> · <a href="interval_trajectories.html">Interval viewer</a> · <a href="lower_confidence.html">Confidence 1–3 and fades</a></p>
<h1>What would have grown the bankroll aggressively?</h1><p>October 19, 2026 · Retrospective comparison of the categories already discovered. No new interval mining.</p>
<div class="note"><strong>For a broader aggressive framework, combine backing confidence 4–5 at 35–81¢ with opposing confidence-1 favorites.</strong> With pooled probabilities and half Kelly, that combination grows the bankroll to <strong>2.225×</strong> with <strong>33.3% maximum drawdown</strong>, reserving cash before each card and settling every fight. <strong>The largest hindsight gains come from the two narrow undefeated categories</strong>: category-based probability estimates produce 15.253× at quarter Kelly or 57.215× at the largest cash-feasible common fraction, 40.90% of Kelly. Those extreme results are heavily conditioned on finding two winning streaks after seeing outcomes.</div>
<h2>What is being compared</h2>
<p>Twenty single-rule and combined portfolios, three probability estimators, two price samples, two execution conventions, and four sizing choices produce {meta['scenario_count']} scenarios. The {meta['feasible_scenarios']} feasible trajectories each include all 187 recorded rows plus the initial bankroll. The candidates come from the earlier analyses. These results identify winners <em>within this comparison</em>, not a mathematical optimum over every conceivable strategy.</p>
<p>Each portfolio uses one bankroll, at most one position per fight, and a common Kelly fraction across its constituent categories. The probability-model family is also common across constituents. All primary tables reserve cash before each card; odds and forecasts are fixed at that time, while equity updates after each fight. The alternative sequential convention refits and reinvests between fights but assumes the pre-card quotes remain obtainable later. Actual quote depth and settlement timing are unavailable.</p>
<p>Prices are Kalshi purchase asks with modeled taker fees; missing prices remain reconstructed. For an opposite-side purchase, NO ask is one minus the YES bid, preserving the spread under <a href="https://docs.kalshi.com/getting_started/orderbook_responses">Kalshi's documented reciprocal pricing</a>. Intervals described for fades refer to the original pick's YES ask. These are scalable bankroll ratios, not evidence that large dollar amounts could have been filled.</p>
<h2>Choice 1: a broader aggressive combination</h2>
<p>The two components address different signals: confidence-4–5 picks perform better after excluding cheap prices, while confidence-1 favorites have been unusually poor picks and therefore profitable to oppose. They never bet the same fight. Combining them improves the historical growth without adding another overlapping stake on an existing position.</p>
<p>The simplest version uses confidence 4–5 at any ask ≥35¢ and fades confidence-1 favorites. At half Kelly it returns <strong>2.103×</strong> with 33.3% drawdown. Using the retrospectively stronger upper boundary of 81¢ improves terminal wealth to <strong>2.225×</strong>, also with 33.3% drawdown. The small gain from that upper cut is less compelling than the gain from adding the confidence-1 fades.</p>
{risk_table}
<p>Half Kelly is already aggressive: the largest historical individual position is <strong>27.5% of the bankroll at card start</strong>. The all-history cash-feasible ceiling is 64.58% of Kelly, because one card's full-Kelly allocations sum to more than the bankroll. That retrospectively optimal fraction grows to 2.666× with 40.7% drawdown. It is an observed financing boundary, not a forward risk recommendation. Full Kelly cannot fund this portfolio at the pre-card prices without borrowing or changing allocations.</p>
<p>The quarter-to-half increase raises terminal growth from +54.6% to +122.5%, while historical drawdown rises from 18.5% to 33.3%. Moving further to the hindsight cash ceiling adds growth and also makes a single changed winner sufficient to finish below the starting bankroll in the frozen-stake sensitivity. Without a user-selected drawdown preference, no one of these is universally optimal. The distinction between growth optimization and drawdown control is formalized in <a href="https://arxiv.org/abs/1603.06183">Busseti, Ryu and Boyd's risk-constrained Kelly study</a>.</p>
<h2>Does that combination depend on one model or reconstructed quotes?</h2>
{model_table}
<p>All six half-Kelly versions remain profitable. Observed-only pooled fitting and trading gives <strong>2.023× over ten positions</strong>, with 36.5% drawdown. This is useful robustness evidence, but the filters themselves were chosen using the complete historical study. It is not an untouched validation sample. The larger gains from fitting within each category come with substantially larger drawdowns in the full history.</p>
<h2>Choice 2: concentrate on the two undefeated categories</h2>
<ol><li>Back confidence 3–5 at YES asks from <strong>35¢ to below 40¢</strong>: eight wins from eight qualifying fights.</li><li>Oppose confidence-1 picks whose original YES ask was <strong>25–70¢</strong>: six wins from six qualifying fights; actual NO asks ranged from 32 to 76¢.</li></ol>
<p>The categories are disjoint. A category-offset model learns each category's excess win probability from earlier category outcomes only. Because every later qualifying result was a win, its estimated advantage and bankroll both grow. This converts a small winning streak into an enormous compounding outcome.</p>
{narrow_table}
<p>The strongest cash-feasible common fraction in this narrow combination is <strong>40.9016% of Kelly</strong>. It gives 57.215× terminal wealth, taking 12 positions after initial category learning, with a largest individual allocation of 36.4%. On the most heavily allocated card it commits the entire bankroll across the selected fights. Half Kelly would require <strong>122.2% of available card cash</strong>, so its pre-card result is infeasible. Quarter Kelly uses up to 61.1% of card cash and produces 15.253×.</p>
<p>With the pooled model, the same categories give <strong>2.727× at half Kelly and 6.327× at full Kelly</strong>, both cash-feasible. Thus much of the extreme result comes from the probability estimator believing the small category's undefeated history, not simply from choosing a larger fraction of the same estimate.</p>
<p>Under the sequential reinvestment assumption, category-based half Kelly yields 187.927× and full Kelly yields 6,595.711×. Those numbers are mathematically consistent with twelve winning bets carrying large bankroll allocations, but should not be interpreted as achievable historical trading results: the simulator reuses earlier pre-card quotes, assumes scalable fills and prompt availability of proceeds, and selects the categories after observing the streaks. The pre-card cash analysis removes one of those assumptions and sharply reduces the headline.</p>
<p>Using only observed quotes for fitting and trading leaves four category-model positions. Quarter Kelly produces 2.029× and full Kelly 9.224×. The much smaller sample cannot establish that the all-history extreme growth rate persists. All eight narrow backing wins and all six narrow fade wins were used in discovering the filters, so forecast fitting on earlier outcomes does not remove the selection bias.</p>
<p>Zero historical drawdown here means no chosen bet lost. It is not a bound on future drawdown. A loss at the largest sequential full-Kelly stake would remove <strong>88.9% of the then-current bankroll</strong>; even the cash-feasible pre-card version can lose 36.4% on one position, or potentially all of a fully allocated card. The mathematical growth figures are genuine backtest results, but the probability error that matters most is not measured by their zero-loss paths.</p>
{image('aggressive_growth.png','Every curve includes all recorded fights. The right panel uses a logarithmic wealth axis. All shown paths reserve cash before each card; categories and model choices are retrospective.')}
<h2>Would adding more categories have helped?</h2>
{broad_table}
<p>Adding confidence-2 underdogs to the bounded core/fade combination improves pooled half-Kelly wealth from 2.225× to 2.650×. However, this increase comes from <strong>one additional position</strong>; observed-only wealth decreases from 2.023× to 1.984×. It is an exploratory satellite, not evidence of a reliable second growth engine.</p>
<p>Adding confidence-2 middle-price fades gives 2.558× in the full history, but essentially no improvement in the observed-only pooled case. Confidence-3 favorites add little pooled growth. Adding all five groups simultaneously makes pooled half Kelly cash-infeasible; the best common feasible fraction is 48.77%, with 3.031× wealth and 32.6% drawdown. More groups increase both opportunities and overlapping cash needs, and the extra filters are all selected retrospectively.</p>
<p>The narrower confidence-1 fade improves retrospective growth relative to fading all favorites, but it also removes a losing fade just above the 70¢ boundary. The broader favorite rule is the more interpretable hypothesis to test with new data. Blanket confidence-1 fading produces less growth and more drawdown than favorite-only fading in this comparison.</p>
<h2>My judgment</h2>
<p><strong>For the question “what would have made the most money in this workbook?”</strong>, the narrow undefeated pair with category-specific probability estimates is the answer among the compared portfolios. Even its quarter-Kelly, pre-card-funded version grows more than fifteenfold. It deserves to be shown rather than dismissed, but it is an extreme retrospective result.</p>
<p><strong>For an aggressive strategy worth testing on subsequent picks</strong>, the high-confidence backing core plus confidence-1 favorite fades is the stronger framework. Half Kelly is an aggressive comparison point supported by its historical doubling, model/source sensitivity, and available cash; it is not a demonstrated optimal fraction. Quarter Kelly offers a lower-drawdown comparison, and the more aggressive cash-bound fraction shows the price of chasing additional growth. The simpler ≥35¢ core is close to the optimized 35–81¢ version.</p>
<p>I would treat confidence-2 underdogs and the two narrow winning categories as separately tracked hypotheses before allowing their small local samples to dominate stake estimates. Use the pooled probability estimate as the main sizing comparison, update only after settlement, price the actual side including fees, and do not stack overlapping categories. Search-adjusted evidence from the earlier reports still does not establish a profitable strategy. This new portfolio comparison adds hindsight choices and does not strengthen that statistical confidence.</p>
<h2>Verification and interpretation of the stress numbers</h2>
<p>Every combination is checked for duplicate fight exposure. The new single-rule results reconcile with earlier outputs, and independent cash/contract accounting reproduces all feasible trajectories. The continuous fraction optimum is checked using the concavity and first-order conditions of log terminal wealth over the cash-feasible interval. Single-win reversals change one executed winning payout to a loss while holding every forecast stake fraction fixed; they do not refit subsequent probabilities, change interval selection, or estimate a probability of loss. The worst terminal and worst drawdown can come from different reversed fights.</p>
<details><summary>All {meta['scenario_count']} scenarios</summary>{all_table}</details>
<p>Files: <a href="../results/aggressive_growth.csv">scenario summaries</a>, <a href="../results/aggressive_growth_paths.csv.gz">complete fight trajectories</a>, <a href="../results/aggressive_growth_summary.json">portfolio definitions and assumptions</a>, and <a href="research_log.md">research record</a>. Proprietary source rows and intermediate per-fight forecasts stay local.</p></body></html>'''
    (ROOT/'reports/aggressive_growth.html').write_text(text)
    print('Wrote reports/aggressive_growth.html',flush=True)


if __name__=='__main__':run()
