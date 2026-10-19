# Research record

## Requested outcome and evidence contract

Analyze the user-specified `MMA betting - UFC Edmonton Oct 17 2026 - results updated.xlsx` thoroughly for actionable confidence/odds combinations. Correct demonstrable errors with an audit trail; preserve unresolved observations without inventing values. Save code and intermediate analyses as well as the final report in `tamnys/kalshi`.

The completed work must provide:

- A source inventory, reconciled fight-level dataset, documented corrections, and explicit effects of missing/ambiguous entries.
- Category returns and reliability estimates that account for odds, sample size, chronology, dependence, and the search across categories.
- Comparisons of confidence information against market odds, and evidence of whether apparent category edges persist outside the observations used to choose them.
- A fight-level fractional Kelly construction, with calibration uncertainty, sequential updating, realistic execution costs, and bankroll/drawdown tradeoffs; no invented user risk tolerance.
- A reproducible final report that distinguishes observed profit, plausible opportunity, and statistically supported inference.

## 2026-10-18 — source discovery

The user identified `/Users/j/Code/Kalshi/MMA betting - UFC Edmonton Oct 17 2026 - results updated.xlsx` as the primary workbook. Related Downloads workbooks were located before this clarification; they will not be pooled as independent observations.

The destination GitHub repository is public. Original workbooks, corrected workbooks, and detailed row-level data remain local. Code, aggregate analyses, and the report are committed.

Picks' pre-fight recording, odds executability, confidence semantics, and completeness of the record are not yet independently established. The analysis will state the resulting limitations.

## Planned analysis, conditional on available fields

Audit workbook values and formulas; validate outcomes when discrepancies warrant it; normalize dates, confidence, odds and settlements. Compute flat-stake results by confidence and odds, uncertainty, and chronological stability. Compare confidence-plus-odds probability models with odds-only baselines using prior-data-only fits. Evaluate the complete declared category search rather than reporting only winning cells. Stress prices and influential fights/cards. Build sequential Kelly examples with probabilities estimated using only information available before each fight, and compare growth against drawdown without selecting a risk budget on the user's behalf.

Category boundaries and computational accuracy will be justified from the data or mathematical/reporting requirements, not arbitrary sample-size filters or betting caps.

## Audit checkpoint

187 recorded picks span 15 cards, March 2025 through October 2026. The apparent 2024 cards are date errors. Every completed fight was reconciled against ESPN's card results. UFC primary sources confirm the canceled UFC 313 Gutierrez–Castaneda entry and the incorrectly marked Rodriguez win against Borjas. The original file's hash is in `results/source_inventory.json`.

The priced binary-outcome sample has 183 fights: 103 wins and 80 losses. Two opponent-quoted prices cannot establish the selected fighter's odds and are excluded; one canceled fight and one draw are accounted separately. The draw's Kalshi position settles at 50 cents rather than an assumed refund. Exact Kalshi cents take priority over rounded equivalent American prices. Preliminary gross profit is 0.153 units on 183 units staked, before fees.

The full correction log and corrected workbook are generated locally under `data/cleaned/`. Reproduction code and aggregate audit counts are committed. Raw scoreboard snapshots are cached locally.

## Kalshi execution and inference checkpoint

The user clarified that all execution should be on Kalshi, with missing prices reconstructed mathematically. The primary analysis now uses Kalshi asks and its standard taker fee, including counterfactual Kalshi prices for periods before the recovered market archive begins. The initial sportsbook summary remains an audit reference, not the final recommendation.

Retrieved 1,570 live/archive KXUFCFIGHT markets and 69 selected-fighter pre-card ask observations. The oldest indexed market opened May 8, 2025. All recovered candles end at card start. Pre-card asks are used to avoid leaking live results; settled last-trade prices are never used. These quotes do not establish available depth or the price obtainable after earlier fights settle.

44 independent sportsbook/ask pairs across five cards support an affine logit conversion. Leave-one-card-out evaluation gives 1.13-cent mean absolute error and 1.45-cent RMSE; identity conversion is less accurate. The 25 recent workbook American prices are conversions of Kalshi cents and are excluded from fitting. Two missing selected-fighter quotes are estimated from the opponent using a separately fitted bid relationship.

The binary sample now has 185 fights, including the two explicitly imputed opponent-price rows. Aggregate net ROI is approximately -0.9%. The exhaustive search covers 52,156 confidence/price intervals with no minimum sample-size filter. All intervals, including losing and tiny cells, are retained in `results/all_interval_search.csv`. The maximum-statistic simulation adjusts for this search. Card-cluster intervals and model-based posterior probabilities describe different uncertainty assumptions and are reported separately.

The user requested growth/drawdown tradeoffs without a preferred risk tolerance. Sequential probability calibration and Kelly staking are being evaluated with earlier outcomes only. All work remains exploratory; no dataset has been declared an untouched holdout.

## Completed analysis and verification

The final report is `reports/final_report.html`. It contains the confidence/price shortlist, frequentist and model-based uncertainty, price reconstruction, temporal and influence checks, and fight-level fractional Kelly construction. The backing and fading searches contain 104,312 intervals in total. The strongest interval's search-adjusted p value is 0.178 within backing and 0.357 after accounting for both directions. No category establishes a robust edge after discovery adjustment.

The broad confidence-4–5 pool returns 9.7% net on 56 flat-stake bets, with an approximate card-cluster 95% interval from -15.2% to +34.6%. Its independent common-shift posterior gives an 88.9% positive-edge probability, which is not discovery-adjusted. Profit remains positive when any one card is omitted. Observed-price and imputed-price subsets are both positive. The narrower 35–40-cent category is an 8/8 exploratory finding, not a probability estimate.

Six calibration models were evaluated with earlier-fight outcomes only, following an initial 24-fight prefix required for common model identifiability. None clearly improves market predictive accuracy. The high-confidence pool with odds-plus-confidence calibration has +10.2% historical bankroll growth and 19.1% maximum drawdown at the quarter-Kelly comparison point. Full Kelly loses 13.6% with 64.9% maximum drawdown. These figures condition on pre-card quotes still being obtainable during sequential betting. The separate benchmark freezes probabilities and allocations before the card and loses 7.7% at quarter Kelly for the high-confidence pool. It does not use later same-card outcomes in advance.

The draw is settled at 50 cents per contract and included in the bankroll ledger. All backing models skip it; the categorical fading model's position incurs its modeled draw loss. No user bankroll size, drawdown budget, fee account type, or optimal betting fraction was invented.

The complete cached pipeline was rerun successfully. Checks cover source preservation, source-hash binding for cell-specific corrections, corrected outcomes/dates, fighter-side matching, no post-card quote use, settlement reconciliation, independent Decimal fee arithmetic, analytic Kelly versus direct log-utility maximization, sequential compounding, finite separated fits, and aggregate totals. The standalone HTML parses with embedded charts and numerical tables; the README passes the documentation boundary check.

Original source workbooks, API caches, normalized fight data, and the corrected workbook remain local. Code, aggregate intermediate analyses, figures, report, and this research record are saved in GitHub checkpoints. There were no trades or external messages.

Rejected as unsupported: exact historical prices where no market/quote exists, missing proprietary picks, pre-fight timestamp verification from post-hoc file metadata, and claims of a unique optimal fraction without a risk preference. These are stated as limitations rather than filled with invented values.

## Completion audit after the continuation request

The preceding turn made progress: the final analysis and report were committed and the remote main branch was verified. This continuation inspected the current files and underlying code against the original request, rather than treating that prior completion statement as evidence.

One material limitation was investigated further. All 15 retrieved ESPN cards have only two or three distinct competition timestamps. Multiple bouts share a timestamp, and every bout has identical `date`, `startDate`, and `endDate` values. These are segment scheduling fields, not individually measured fight starts or finishes. They cannot establish a quote immediately before each fight. The reproducible check is `src/timing_audit.py`, with card-level evidence in `results/quote_timing_audit.csv`; the final report now explains it. No exact fight timestamps were invented.

A stale note in the corrected workbook said the two opponent-price rows were excluded from priced inference. It now distinguishes their missing source prices from their explicit estimates in the final Kalshi analysis. The workbook and report were regenerated. A before/after SHA-256 comparison of the normalized Kalshi ledger confirmed that this clarification changed no prices, outcomes, or calculated returns.

| Original requirement | Current authoritative evidence |
| --- | --- |
| Audit and correct the supplied workbook | Immutable source hash, 187-row inventory, all completed outcomes reconciled in `src/clean.py`, 32 logged source-cell edits, local corrected Excel workbook and correction log. `src/verify.py` passed against the regenerated workbook. |
| Use Kalshi prices and payouts, estimate unavailable prices mathematically | Cached API asks with pre-card cutoffs and settlement-side reconciliation; 69 recovered prices; 44 independent sportsbook/ask pairs with leave-one-card-out comparison; 117 explicit estimates and one cancellation. Coefficients, errors, sources, and counterfactual-price limits are saved. |
| Identify confidence/odds combinations and reliability | `candidate_details.csv`, complete backing/fading interval searches, card-cluster uncertainty, model-based posterior probabilities, and search correction. The interval enumeration and null-return arithmetic were inspected in the current source. |
| Analyze stability and incremental confidence information | Observed/estimated splits, reconstruction sensitivity, influential-fight/card removal, calendar blocks, market-relative model scores, and forecasts trained only on earlier outcomes. The report explicitly distinguishes these exploratory checks from an untouched holdout. |
| Construct fight-level fractional Kelly and show risk tradeoffs | Prior-outcome Firth calibration, analytic cash-fraction formula independently checked against log-utility optimization, sequential bankroll compounding, quarter/half/full and continuous hindsight comparisons, whole-card resampling, and the adverse frozen-before-card execution scenario. User risk tolerance remains unspecified. |
| Save final and intermediate work | Checkpointed code, aggregate CSV/JSON results, figures, standalone HTML report, and this research record in the requested repository. Proprietary source and row-level artifacts remain local because the repository is public. |

The remaining unknowns are absent signal timestamps, complete-system pick coverage, executable depth and fills, and exact individual-fight start/settlement times. They limit what can be inferred; further transformations of the same observations cannot supply them. The supported conclusion remains a plausible confidence-4–5 candidate, not an established profitable strategy.

## 2026-10-19 — expanded ROI ranking and full fight trajectories

The follow-up request explicitly asks for confidence/odds intervals with higher ROI and fractional Kelly simulation over the whole fight sequence. Acceptance evidence will comprise the complete positive-ROI interval set, simulations of every such interval, inspectable full-sequence bankroll paths for the ROI-versus-sample-size frontier and strongest interval in each confidence range, and explicit comparison of return, drawdown, sample size, and search-adjusted uncertainty.

The existing exhaustive backing search contains 52,156 intervals, including 15,292 with positive historical flat-stake ROI. No arbitrary minimum sample size or top-N cutoff will be imposed. Displayed candidates will be selected by the nondominated ROI/sample-size frontier and the maximum variance-standardized return within each confidence interval; all positive intervals will remain available in machine-readable results.

Every path will include all 187 source rows in chronological fight order, with no bet during initial estimation and on the canceled row. The odds-plus-confidence model is identifiable after three prior decisive fights, compared with the previous common 24-fight prefix required to compare all six model families. Both prefixes will be simulated. Probability fitting will use only earlier decisive outcomes, draw settlement will use the actual 50-cent payout, and fractions 0.25, 0.5, and 1 will remain comparison scenarios rather than a user risk budget. Category selection still uses the full historical sample and must not be described as independent validation. Prices retain the documented pre-card/reconstruction assumptions.

### Expanded results and verification

All 15,292 positive backing intervals were simulated under six combinations of training prefix and Kelly fraction: 91,752 scenarios, each with 188 wealth points including the initial bankroll. Compressed complete paths remain local; aggregate simulation summaries and selected paths are committed. The follow-up report is `reports/interval_trajectories.html`. It provides 111 selectable candidates without a top-N or minimum-N filter. Candidates comprise the 58-interval ROI/sample-size frontier, the strongest standardized and raw-ROI interval in every confidence range, the Kelly return/drawdown frontiers and confidence-range leaders, and broad reference categories.

The highest historical quarter-Kelly bankroll among all positive backing intervals is confidence 4–5 at 35–81 cents: 32/43 flat-stake wins, 29.47% flat-stake ROI, and 47.58% bankroll growth from 32 positive-model-edge positions, with 11.42% maximum drawdown. Half Kelly returns 104.12% with 22.18% drawdown; full Kelly returns 218.27% with 42.55% drawdown. The simpler 35-cents-and-above rule has 37/49 wins, 25.39% flat-stake ROI, and 43.41% quarter-Kelly growth. These candidates were selected retrospectively, including by their observed Kelly performance.

For the 35–81-cent rule, removing its largest winner leaves 26.04% ROI, omitting any one card leaves at least 23.52%, and its 13 observed-price fights return 24.56%. Applying the adverse empirical reconstruction error and reselecting gives 42 fights at 22.11% ROI. The 35-cents-and-above rule similarly retains 18.60% stressed ROI. Nominal cluster intervals exclude zero for these two rules, but neither survives the full category-search correction; no adjusted Kelly-performance significance is asserted.

The broad all-price confidence-4–5 quarter-Kelly outcome changes materially with initialization: -7.20% from the earliest identifiable three-fight prefix versus +10.24% after the prior 24-fight comparison prefix. One additional early losing position accounts for the difference and stakes 15.82% of bankroll at quarter Kelly. The main report was revised to surface this result rather than retaining an unqualified positive broad-pool recommendation.

`src/verify_intervals.py` passed. It changes the first forecasted outcome and confirms that the current/earlier forecasts remain unchanged while later forecasts update; checks draw payoff and skipped-row accounting independently; reconciles all 91,752 stored paths and drawdowns; and reproduces the previous broad-pool results under their original training convention. All interval memberships, wins and flat profits reconcile to the complete scan. The new standard plot was visually inspected.

The browser-profile container could not reach the host's local HTTP server, so a live browser interaction check was unavailable. The actual generated report script was instead executed in an isolated Node VM with a minimal DOM harness: all 222 candidate/training-control combinations render three finite 188-point paths and correct summary tables. This verifies control and data behavior, not browser layout or accessibility. The standalone report includes an embedded static plot as well. The README passed the documentation boundary check.

Code, intermediate results and draft report were checkpointed in `e86d0be` while the final checks continued. No new source picks, exact timestamps, fills or independent validation observations were invented.
