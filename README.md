# MMA pick analysis

Reproducible analysis of confidence levels, Kalshi prices, realized returns, and fight-by-fight bankroll sizing for the supplied UFC picks.

Source workbooks and row-level corrected data are stored locally and excluded from this public repository. Aggregate analyses, code, and reports are versioned here.

Open [the standalone report](reports/final_report.html) in a browser. The report contains the candidate categories, uncertainty, price reconstruction, and fractional Kelly tradeoffs. Its charts are embedded, so the HTML can be shared as one file.

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
