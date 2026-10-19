"""Refit complete-card resamples to quantify uncertainty in current Kelly sizes."""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from audit import ROOT
from analyze import N_SIM, cost
from kelly import designs, fit_firth, kelly_fraction


def run():
    full = pd.read_csv(ROOT / 'data/cleaned/kalshi_fights.csv')
    d = full[full.y.notna() & full.kalshi_cost.notna()].reset_index(drop=True)
    x, offset = designs(d)['odds_confidence']
    y = d.y.to_numpy()
    fitted = fit_firth(x, y, offset)
    dates = sorted(d.date.unique())
    cards = [np.flatnonzero(d.date.to_numpy() == date) for date in dates]
    seed = int(hashlib.sha256(b'calibration-card-bootstrap').hexdigest()[:8], 16)
    source = json.loads((ROOT / 'results/source_inventory.json').read_text())
    seed ^= int(source['sha256'][:8], 16)
    rng = np.random.default_rng(seed)
    coefficients = []
    for selected in rng.integers(len(cards), size=(N_SIM, len(cards))):
        indices = np.concatenate([cards[i] for i in selected])
        assert np.linalg.matrix_rank(x[indices]) == x.shape[1]
        coefficients.append(fit_firth(x[indices], y[indices], offset[indices], fitted))
    coefficients = np.array(coefficients)
    np.savez_compressed(ROOT / 'artifacts/private/calibration_bootstrap_coefficients.npz',
                        coefficients=coefficients, fitted=fitted, seed=seed)
    # Reuse the existing published price/confidence examples, not a new tuned grid.
    examples = pd.read_csv(ROOT / 'results/future_staking_examples.csv')
    rows = []
    for r in examples.itertuples():
        design = np.array([1., logit(r.price), r.confidence - 3])
        probabilities = expit(coefficients @ design)
        estimate = float(expit(fitted @ design))
        c = cost(r.price)
        sizes = kelly_fraction(probabilities, c)
        rows.append(dict(confidence=r.confidence, ask=r.price, all_in_cost=c,
                         point_probability=estimate, probability_p025=np.quantile(probabilities, .025),
                         probability_median=np.median(probabilities), probability_p975=np.quantile(probabilities, .975),
                         bootstrap_positive_edge_frequency=float(np.mean(probabilities > c)),
                         point_quarter_kelly=float(kelly_fraction(estimate, c) / 4),
                         quarter_kelly_p025=float(np.quantile(sizes, .025) / 4),
                         quarter_kelly_median=float(np.median(sizes) / 4),
                         quarter_kelly_p975=float(np.quantile(sizes, .975) / 4)))
    pd.DataFrame(rows).to_csv(ROOT / 'results/calibration_sizing_uncertainty.csv', index=False)
    influence = []
    for date in dates:
        keep = d.date.ne(date).to_numpy()
        beta = fit_firth(x[keep], y[keep], offset[keep], fitted)
        influence.append(dict(omitted_card=date, intercept=beta[0], logit_price=beta[1], confidence=beta[2]))
    pd.DataFrame(influence).to_csv(ROOT / 'results/calibration_card_influence.csv', index=False)
    covariance = np.cov(coefficients.T)
    metadata = dict(resamples=N_SIM, cards=len(cards), decisive_fights=len(d), seed=seed,
                    coefficient_names=['intercept', 'logit_price', 'confidence_minus_3'],
                    fitted_coefficients=fitted.tolist(), coefficient_p025=np.quantile(coefficients, .025, axis=0).tolist(),
                    coefficient_p975=np.quantile(coefficients, .975, axis=0).tolist(),
                    coefficient_covariance=covariance.tolist(),
                    notes=['Every resample draws 15 complete cards with replacement and refits the probability model.',
                           '10,000 fits give at most 0.005 Monte Carlo standard error for reported frequencies.',
                           'Bootstrap frequencies are not Bayesian probabilities or discovery-adjusted confidence.',
                           'These are final-fit sizing uncertainty summaries, not additional historical strategy returns.',
                           'Conditional on this model, reconstructed prices and the available cards; missing signals and future regime changes are not modeled.',
                           'No confidence cutoff or bankroll-risk preference is imposed.'])
    (ROOT / 'results/calibration_uncertainty_summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)
    print(pd.DataFrame(rows).query('confidence >= 4').to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
