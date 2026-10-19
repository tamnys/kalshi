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
