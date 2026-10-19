# MMA pick analysis

Reproducible analysis of confidence levels, Kalshi prices, realized returns, and fight-by-fight bankroll sizing for the supplied UFC picks.

Source workbooks and row-level corrected data are stored locally and excluded from this public repository. Aggregate analyses, code, and reports are versioned here.

Open [the standalone report](reports/final_report.html) in a browser. The report contains the candidate categories, uncertainty, price reconstruction, and fractional Kelly tradeoffs. Its charts are embedded, so the HTML can be shared as one file.

The [confidence/odds interval follow-up](reports/interval_trajectories.html) adds a selectable full fight-by-fight bankroll viewer. It simulates all 15,292 historically positive backing intervals at quarter, half, and full Kelly, using both the earliest identifiable three-fight training prefix and the original 24-fight comparison. Every path includes all 187 recorded rows. The updated practical candidates and the effect of early calibration are explained there.

The [earlier-data selection and Kelly robustness report](reports/prospective_validation.html) tests whether rules chosen from past results perform on subsequent fights. It compares pooled and category-specific probability estimates, observed-only training/trading, allocations fixed at the pre-card quote time, and sizing uncertainty from 10,000 whole-card model refits. The fixed 35¢-and-above high-confidence filter remains a plausible candidate; adaptive selection and stake estimates are much less stable.

The [complete interval simulations](results/interval_kelly_all_positive.csv.gz) and [displayed bankroll paths](results/interval_bankroll_paths.csv.gz) are gzip-compressed CSV files. The [ROI/sample-size frontier](results/interval_roi_frontier.csv), [Kelly return/drawdown frontier](results/interval_kelly_return_drawdown_frontier.csv), and [selected-category statistics](results/interval_selected_details.csv) are ordinary CSVs. Complete paths for every positive interval are generated locally as compressed NumPy arrays in `artifacts/private/`. Category selection uses the full historical sample; prior-outcome probability fitting does not remove that selection bias.

The main data outputs are [category statistics](results/categories.csv), [candidate details](results/candidate_details.csv), [portfolio tradeoffs](results/portfolio_tradeoffs.csv), [Kalshi reconstruction](results/kalshi_price_reconstruction.json), and the complete [backing](results/all_interval_search.csv) and [fading](results/fade_interval_search.csv) interval searches.

To reproduce with Python 3.9 and the exact source workbook:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.lock
.venv/bin/python run_analysis.py --source "/path/to/MMA betting - UFC Edmonton Oct 17 2026 - results updated.xlsx"
```

The source checksum is enforced because the reviewed corrections refer to specific cells in this workbook. A different workbook needs a fresh audit. Public ESPN and Kalshi API access is required on the first run; raw responses are cached locally afterward.

The corrected Excel file is written to `data/cleaned/MMA betting - audited.xlsx`; the normalized fight ledger and cell-level correction log are in the same directory. The original is preserved. Local caches and row-level proprietary data are ignored by Git.

The model uses standard Kalshi taker fees. Historical asks take precedence over workbook snapshots; missing asks are explicitly estimated. Reconstructed prices and pre-card quote snapshots do not establish executable fight-time returns. Quarter/half/full Kelly results are comparison scenarios, not a guaranteed strategy or a chosen risk budget.

See [the research record](reports/research_log.md) for intermediate findings and provenance.
