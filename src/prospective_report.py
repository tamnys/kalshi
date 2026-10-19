"""Report earlier-data selection, probability uncertainty and quote-time sizing."""
import base64
import html
import json

import numpy as np
import pandas as pd

from audit import ROOT
from analyze import plt
from report import table, pct


LABELS = {'C4-5_35-81': 'Confidence 4–5, 35–81¢', 'C4-5_35-plus': 'Confidence 4–5, 35¢ and above',
          'C4-5_all': 'Confidence 4–5, all prices', 'C3-5_35-under40': 'Confidence 3–5, 35¢ to below 40¢'}
MODELS = {'market_offset': 'Market plus common offset', 'confidence_offset': 'Market plus confidence offset',
          'odds_only': 'Fitted market slope', 'odds_confidence': 'Fitted odds and confidence',
          'odds_confidence_interaction': 'Odds/confidence interaction', 'confidence_categories': 'Separate confidence levels',
          'category_offset': 'Offset fitted within selected category'}
CRITERIA = {'roi': 'Highest training ROI', 'standardized_return': 'Highest standardized return',
            'quarter_kelly_log_growth': 'Highest prior quarter-Kelly growth'}


def embedded(name, caption):
    data = base64.b64encode((ROOT / 'reports/figures' / name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def run():
    model = pd.read_csv(ROOT / 'results/candidate_model_sensitivity.csv')
    execution = pd.read_csv(ROOT / 'results/candidate_precard_execution.csv')
    adaptive = pd.read_csv(ROOT / 'results/prospective_adaptive.csv')
    frozen = pd.read_csv(ROOT / 'results/prospective_frozen.csv')
    sizing = pd.read_csv(ROOT / 'results/calibration_sizing_uncertainty.csv')
    evidence = json.loads((ROOT / 'results/adaptive_wealth_evidence_summary.json').read_text())
    meta = json.loads((ROOT / 'results/prospective_summary.json').read_text())
    quarter = model[model.fraction.eq(.25) & model.initialization.eq('earliest_rank')]
    pooled = quarter[quarter.price_data.eq('reconstructed_and_observed') & quarter.model.eq('odds_confidence')].set_index('rule')
    precard = execution[execution.price_data.eq('reconstructed_and_observed') & execution.fraction.eq(.25)].set_index('rule')
    observed = execution[execution.price_data.eq('observed_only') & execution.fraction.eq(.25)].set_index('rule')
    rows = []
    for rule, label in LABELS.items():
        a, b, c = pooled.loc[rule], precard.loc[rule], observed.loc[rule]
        rows.append([label, pct(a.final_wealth - 1), pct(b.final_wealth - 1), pct(b.max_drawdown),
                     pct(c.final_wealth - 1), int(c.bets)])
    execution_table = table(rows, ['Fixed rule, selected retrospectively', 'Sequential quarter-Kelly growth',
                                    'Pre-card quarter-Kelly growth', 'Pre-card max drawdown',
                                    'Observed-only pre-card growth', 'Observed-only bets'])
    m = quarter[quarter.price_data.eq('reconstructed_and_observed')]
    models_table = table([[MODELS[name], pct(g.set_index('rule').loc['C4-5_35-81', 'final_wealth'] - 1),
                           pct(g.set_index('rule').loc['C4-5_35-81', 'max_drawdown']),
                           pct(g.set_index('rule').loc['C4-5_35-plus', 'final_wealth'] - 1),
                           pct(g.set_index('rule').loc['C4-5_35-plus', 'max_drawdown'])]
                          for name, g in m.groupby('model', sort=False)],
                         ['Probability estimator', '35–81¢ growth', '35–81¢ drawdown', '35¢+ growth', '35¢+ drawdown'])
    # Show all adaptive quarter-Kelly variants, rather than selecting the winner.
    ar = adaptive[adaptive.fraction.eq(.25)]
    adaptive_table = table([[r.price_data.replace('_', ' '), r.cadence,
                             'Pooled odds/confidence' if r.stake_model == 'global_odds_confidence' else 'Selected-category offset',
                             CRITERIA[r.rule], int(r.flat_n), pct(r.flat_roi), int(r.bets),
                             pct(r.final_wealth - 1), pct(r.max_drawdown)] for r in ar.itertuples()],
                           ['Price sample', 'Reselect after', 'Probability fit', 'Selection criterion', 'Flat N',
                            'Flat ROI', 'Kelly bets', 'Quarter-Kelly growth', 'Max drawdown'])
    fr = frozen[frozen.fraction.eq(.25)]
    frozen_table = table([[r.price_data.replace('_', ' '), r.split, CRITERIA[r.rule], int(r.flat_n),
                           pct(r.flat_roi), int(r.bets), pct(r.final_wealth - 1)] for r in fr.itertuples()],
                         ['Price sample', 'Freeze selection before', 'Criterion', 'Flat N', 'Flat ROI', 'Kelly bets', 'Quarter-Kelly growth'])
    future = sizing[sizing.confidence.ge(4)]
    uncertainty_table = table([[int(r.confidence), f'{100*r.ask:.0f}¢', pct(r.point_probability),
                                f'{pct(r.probability_p025)} to {pct(r.probability_p975)}',
                                pct(r.bootstrap_positive_edge_frequency), pct(r.point_quarter_kelly),
                                f'{pct(r.quarter_kelly_p025)} to {pct(r.quarter_kelly_p975)}'] for r in future.itertuples()],
                              ['Confidence', 'Ask', 'Fitted win probability', 'Central 95% refit range',
                               'Refits with positive edge', 'Fitted quarter-Kelly stake', 'Central 95% stake range'])
    e = execution[execution.price_data.eq('reconstructed_and_observed') & execution.rule.isin(['C4-5_35-81', 'C4-5_35-plus'])]
    risk_table = table([[LABELS[r.rule], pct(r.fraction, 0), 'Yes' if r.feasible_without_borrowing else 'No',
                         pct(r.final_wealth - 1), pct(r.max_drawdown)] for r in e.itertuples()],
                       ['Rule', 'Kelly fraction', 'Fits available pre-card bankroll?', 'Bankroll growth', 'Max drawdown'])

    paths = pd.read_csv(ROOT / 'results/candidate_model_paths.csv.gz')
    batch_paths = pd.read_csv(ROOT / 'results/candidate_precard_paths.csv.gz')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for rule, ax in zip(['C4-5_35-81', 'C4-5_35-plus'], axes):
        s = paths[(paths.rule == rule) & (paths.price_data == 'reconstructed_and_observed') &
                  (paths.model == 'odds_confidence') & (paths.initialization == 'earliest_rank') & (paths.fraction == .25)]
        ax.plot(s.fight_index, s.wealth, color='#126b66', label='Sequential, includes reconstructed prices')
        for source, color, label in [('reconstructed_and_observed', '#3266a3', 'Fixed before card, includes reconstructed prices'),
                                     ('observed_only', '#a56119', 'Fixed before card, observed-price training/trading')]:
            s = batch_paths[(batch_paths.rule == rule) & (batch_paths.price_data == source) & (batch_paths.fraction == .25)]
            ax.plot(s.fight_index, s.wealth, color=color, label=label)
        ax.axhline(1, color='gray', linewidth=.7)
        ax.set(title=LABELS[rule], xlabel='Recorded fight index', ylabel='Bankroll / initial bankroll')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=1, fontsize=9)
    fig.tight_layout(rect=(0, .18, 1, 1))
    fig.savefig(ROOT / 'reports/figures/prospective_execution.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for confidence, ax in zip([4, 5], axes):
        s = sizing[sizing.confidence == confidence].sort_values('ask')
        ax.plot(s.ask * 100, s.point_probability * 100, label='Fitted win probability', color='#126b66')
        ax.fill_between(s.ask * 100, s.probability_p025 * 100, s.probability_p975 * 100, alpha=.2, color='#126b66', label='Central 95% refit range')
        ax.plot(s.ask * 100, s.all_in_cost * 100, '--', color='#a43e3e', label='Breakeven after fee')
        ax.set(title=f'Confidence {confidence}', xlabel='Kalshi ask (cents)', ylabel='Win probability (%)')
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / 'reports/figures/calibration_uncertainty.png', dpi=180); plt.close(fig)

    output = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Earlier-data selection and Kelly robustness</title><style>
body{{margin:0;background:#f4f7f9;color:#172b3e;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1180px;margin:30px auto;background:white;padding:36px;border-radius:12px}}h1{{font-size:31px;line-height:1.25}}h2{{color:#174d69;margin-top:34px}}.note,figcaption{{font-size:14px;color:#526274}}.callout{{background:#edf7f5;border-left:4px solid #126b66;padding:18px}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #dce4eb;vertical-align:top}}th{{background:#eef4f8}}img{{width:100%;height:auto}}figure{{margin:24px 0}}a{{color:#07548c}}details{{margin:20px 0}}summary{{cursor:pointer;font-weight:600}}pre{{white-space:pre-wrap;background:#f0f4f7;padding:15px}}@media(max-width:650px){{main{{margin:0;padding:18px}}}}@media print{{main{{margin:0;padding:0}}}}</style></head><body><main>
<h1>Earlier-data selection and Kelly robustness</h1><p class="note">October 19, 2026 · Extension of the <a href="interval_trajectories.html">interval search and full-trajectory viewer</a> and <a href="final_report.html">source audit</a>.</p>
<p><strong>The fixed high-confidence price filters remain plausible opportunities, but their impressive historical returns are not yet a validated adaptive betting system.</strong> Confidence 4–5 at 35¢ and above earns +22.5% at quarter Kelly when forecasts and allocations are both fixed before each card. Restricting fitting and trading to recovered Kalshi quotes gives +9.5%, from only six bets. Those rules were still selected using the complete history.</p>
<div class="callout"><strong>Updated recommendation:</strong> retain confidence 4–5 at 35¢ and above as the simplest candidate for a fixed, prospectively recorded trial. The 35–81¢ version is a useful comparison, not an established optimal cutoff. Use a pooled probability model and one position per fight; inspect the market-anchored confidence model as a sizing sensitivity. Do not select a new winning bucket after every result or size a bet from a tiny bucket’s eventual win percentage. The data supports investigating this filter, while leaving substantial uncertainty about future profitability and the appropriate Kelly fraction.</div>
<h2>Aligning decisions with when prices were observed</h2>{execution_table}
<p>The sequential version refits and reinvests after each fight, but uses asks recorded before the card. The pre-card version learns only from earlier cards and allocates cash to individual fights at that snapshot time. Every fight still resolves separately in the ledger; allocations use starting-card capital, and pending cash is not reinvested. Drawdowns are measured after each fight rather than only at card boundaries. Quotes and estimates do not prove available depth or actual fills.</p>
<p>The observed-only version removes both imputed-price training rows and imputed-price bets. It therefore tests a different, much smaller history—not merely a fee or price substitution. For the 35¢+ rule, the underlying observed-price sample has 14 decisive fights and +24.1% flat-stake ROI, but the pre-card Kelly procedure places only six positive-estimated-edge bets after training.</p>
{embedded('prospective_execution.png', 'Quarter-Kelly wealth over all 187 recorded rows. The fixed categories were discovered retrospectively.')}
{risk_table}<p>Full-Kelly simultaneous allocations would exceed available cash for these two filters on at least one card. The largest historically feasible common fraction is approximately 0.646, derived from one divided by the largest sum of full-Kelly allocations. This is an accounting limit for that simulation, not a recommended risk budget. Quarter and half remain illustrations; no drawdown tolerance has been chosen on your behalf.</p>
<h2>Does the result depend on the probability model?</h2>{models_table}
<p>These are quarter-Kelly results with each model starting when its design first becomes identifiable. The six original pooled models require 1, 2, 2, 3, 4 and 24 earlier decisive fights respectively. A separate common-24-fight comparison is saved to remove that initialization difference. All six pooled models are profitable for both filtered rules in the reconstructed history. Their different stake sizes and drawdowns show why selecting the model with the highest backtest growth would be another form of optimization on this sample.</p>
<p>The selected-category offset model learns the category’s excess win probability from earlier category results using the same Firth bias reduction. It produces +59.8% quarter-Kelly growth for the 35¢+ rule, but <strong>−10.5%</strong> when restricted to observed-price training and trading. The pooled odds/confidence model yields +7.9% in that observed-only sequential comparison. The 8/8 narrow band grows 6.59-fold with category-specific quarter Kelly; its category was chosen because it contained winners, and that striking result is not independent evidence.</p>
<h2>Selecting the rule using earlier results only</h2>
<p>The adaptive experiments choose a category after each fight or before each card using only earlier decisive results. They compare the highest training ROI, highest return divided by its null standard deviation, and highest prior quarter-Kelly log growth. The last objective uses the already-defined pooled odds/confidence stakes. Each selected rule is then sized either by that pooled model or by a common offset fitted to earlier results inside the selected category. The rule-selection objective is held fixed when comparing these sizing methods.</p>
<p>Price boundaries are midpoints between adjacent distinct training prices, with outer boundaries 0 and 1. They do not use future prices to define unseen-price membership. Score ties within floating-point accumulation error favor more training observations, then broader coverage, then lexical bounds. There is no minimum sample-size gate. A nonpositive best score produces no rule.</p>
<p><strong>All twelve quarter-Kelly adaptive variants using pooled odds/confidence sizing lose money.</strong> With selected-category sizing, the highest-training-ROI rule produces +145.2% growth and 31.0% maximum drawdown when reselected after every fight. That same method loses 49.1% on observed-price-only data. Its card-reselected version grows 69.0% in the reconstructed history and loses 33.1% on observed-only data. The other category-specific selection criteria perform poorly. Positive flat-stake selection returns and negative Kelly returns can coexist because the model sizes or skips different bets.</p>
<details><summary>All 24 adaptive quarter-Kelly combinations</summary>{adaptive_table}</details>
<p>Natural time-block checks give a further useful qualification. Selecting the quarter-Kelly-growth winner before October 2026 already identifies confidence 4–5 at roughly 35.3–81.3¢ from 37 earlier fights. On the subsequent two cards it has five qualifying bets, approximately +0.3% flat-stake ROI and −1.6% quarter-Kelly bankroll growth. The narrow confidence 3–5 band also emerges from 2025 data, but its October 2026 result is only one additional win. These are historical earlier-data tests specified during this investigation, not untouched prospective validation.</p>
<details><summary>Rules frozen at the existing natural calendar splits</summary>{frozen_table}</details>
<h2>How uncertain are current Kelly sizes?</h2>
<p>10,000 resamples each draw 15 complete cards with replacement and refit the odds-plus-confidence model. This includes parameter re-estimation, unlike the earlier bootstrap that merely reordered already-fitted forecast/return paths. It preserves within-card dependence while conditioning on the available prices, model specification and recorded cards. The frequencies below are empirical bootstrap frequencies—not Bayesian probabilities, discovery-adjusted confidence, or guarantees about future events.</p>
{uncertainty_table}
<p>At confidence 4 and a 40¢ ask, the point estimate is a 48.7% win probability and a 3.0% quarter-Kelly bankroll stake. The central refit range is 33.4–63.8% probability and 0–9.5% stake, with a positive estimated edge in 81.2% of fits. At an 85¢ ask, only 56.7% of refits show a positive edge for confidence 4. Higher confidence increases the fitted probability, but confidence 5 has only seven observations in the original sample and wider parameter uncertainty. Every displayed central 95% stake range includes zero.</p>
{embedded('calibration_uncertainty.png', 'Final-fit probability uncertainty from 10,000 whole-card refits. The band is conditional on the model and source sample.')}
<p>The confidence coefficient’s central bootstrap range is approximately −0.022 to +0.639 in log-odds units per confidence level. That is consistent with useful confidence information, but does not settle its magnitude or stability. Full Kelly treats a point estimate as sufficiently accurate for log-growth sizing; these refits show why that assumption is consequential. They do not identify a unique optimal fractional multiplier.</p>
<h2>Evidence after adaptive strategy selection</h2>
<p>A separate conditional diagnostic uses the bankroll itself as an evidence process. If conditional expected settlement never exceeds all-in cost, and an algorithm chooses its stake from past information without borrowing, expected bankroll cannot increase under that null. The reciprocal of peak bankroll then gives a conservative bound valid across stopping times. Averaging prespecified wealth processes also preserves the bound. <a href="https://arxiv.org/abs/1808.03204">Howard et al., time-uniform bounds</a>; <a href="https://arxiv.org/abs/1912.06116">Vovk and Wang, e-value combination</a>.</p>
<p>The profitable adaptive quarter-Kelly ROI strategy ends at 2.45× bankroll and peaks at 3.55×, giving an individual inverse-peak bound of approximately 0.281. Across all {evidence['stated_strategy_family']} adaptive fraction/method variants, Bonferroni bounds are 1.000. The equally weighted wealth mixture peaks at {evidence['equal_mixture_peak']:.2f}×, giving a bound of {evidence['equal_mixture_anytime_bound']:.3f}. This diagnostic offers no strong support for selecting the impressive winner. It covers this explicitly enumerated family under its price and timing assumptions; it does not retroactively preregister the wider research process.</p>
<h2>What to carry forward</h2>
<p>The evidence favors keeping a simple high-confidence price filter stable while collecting new picks, rather than repeatedly optimizing categories. For each eligible fight, obtain the actual ask and fee, update the pooled probability model using settled earlier fights, and calculate the cash fraction from that estimate. Compare fractional stakes against the drawdown tradeoffs, and avoid interpreting the current 3% example as a universal stake. Record all signals and skipped bets, not just fills. Evaluate new data against the fixed rule before deciding whether to revise it.</p>
<p>The absence of an established edge is not evidence that every filtered opportunity is worthless: the high-confidence filters remain positive across models, influence checks, price stress and the quote-time allocation test. The unresolved issues are genuine future replication, signal completeness, timestamp integrity, model drift, and executable prices. Further rearrangement of this same small history cannot resolve those issues.</p>
<h2>Saved outputs and checks</h2>
<p>Saved outputs include {meta['adaptive_scenarios']} adaptive scenarios, {meta['frozen_scenarios']} frozen-rule scenarios, {meta['model_scenarios']} model/initialization scenarios, {meta['execution_scenarios']} quote-time allocation scenarios, complete fight-level paths, and all 10,000 refitted coefficient vectors. Aggregate CSVs and code are in <a href="https://github.com/tamnys/kalshi">tamnys/kalshi</a>; private row-level forecasts, rule histories and bootstrap arrays remain local.</p>
<p><a href="../results/prospective_adaptive.csv">Adaptive results</a> · <a href="../results/candidate_model_sensitivity.csv">Model comparisons</a> · <a href="../results/candidate_precard_execution.csv">Pre-card allocations</a> · <a href="../results/calibration_sizing_uncertainty.csv">Sizing uncertainty</a> · <a href="../results/adaptive_wealth_evidence.csv">Conditional wealth evidence</a>.</p>
<p class="note">Verification directly enumerates a source-card search independently, reconciles every saved rule’s earlier-data membership and profit, changes future outcomes to test pre-card forecast invariance, reconstructs pre-card paths from cash paid and contract payouts, and checks all bootstrap summaries against the saved coefficient draws. Source corrections and Kalshi fee assumptions remain those documented in the original report.</p>
</main></body></html>'''
    path = ROOT / 'reports/prospective_validation.html'
    path.write_text(output)
    print(path, flush=True)


if __name__ == '__main__':
    run()
