"""Reproduce the largest sequential pooled stake and inspect its probability input."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from audit import ROOT
from kelly import designs, fit_firth


def run():
    full = pd.read_csv(ROOT/'data/cleaned/kalshi_fights.csv').sort_values('sequence').reset_index(drop=True)
    key = 'bounded_core_plus_fade1|all|fight|odds_confidence'
    stakes = pd.read_csv(ROOT/'artifacts/private/aggressive_growth_full_stakes.csv')[key]
    saved = pd.read_csv(ROOT/'artifacts/private/aggressive_growth_probabilities.csv')[key]
    i = int(stakes.idxmax())
    prior = (np.arange(len(full)) < i) & full.y.notna().to_numpy() & full.payout.notna().to_numpy()
    x, offset = designs(full)['odds_confidence']
    # Start afresh and use only earlier outcomes, independently of cached forecasts.
    beta = fit_firth(x[prior], full.y.to_numpy()[prior], offset[prior])
    refit_p = float(expit(sum(float(a)*float(b) for a, b in zip(x[i], beta)) + offset[i]))
    p, c = float(saved.iloc[i]), float(full.kalshi_cost.iloc[i])
    full_fraction = (p-c)/(1-c)
    net_profit_per_dollar = (1-c)/c
    # Optimize expected log bankroll directly, without the Kelly formula.
    optimum = minimize_scalar(
        lambda f: -(p*np.log1p(f*net_profit_per_dollar) + (1-p)*np.log1p(-f)),
        bounds=(0, 1), method='bounded')
    assert optimum.success
    assert np.isclose(full_fraction, float(stakes.iloc[i]))
    # SciPy bounded minimization defaults to xatol=1e-5; the underlying Firth
    # BFGS fit also uses its documented default gradient tolerance of 1e-5.
    assert abs(float(optimum.x)-full_fraction) <= 1e-5
    assert abs(refit_p-p) <= 1e-5
    confidence5 = prior & full.confidence.eq(5).to_numpy()
    result = {
        'portfolio': 'bounded_core_plus_fade1',
        'timing': 'settle each fight before sizing the next bet',
        'probability_model': 'pooled odds/confidence logistic fit',
        'prior_binary_fights': int(prior.sum()),
        'prior_confidence5_fights': int(confidence5.sum()),
        'prior_confidence5_wins': int(full.loc[confidence5, 'y'].sum()),
        'confidence': int(full.confidence.iloc[i]),
        'contract_ask': float(full.kalshi_price.iloc[i]),
        'price_source': str(full.price_source.iloc[i]),
        'all_in_cost': c,
        'saved_probability': p,
        'independently_refit_probability': refit_p,
        'saved_full_kelly_fraction': float(stakes.iloc[i]),
        'algebraic_full_kelly_fraction': full_fraction,
        'direct_expected_log_optimum': float(optimum.x),
        'hypothetical_probability_sensitivity': [
            {'assumed_win_probability': q, 'full_kelly': max(0, (q-c)/(1-c)),
             'half_kelly': .5*max(0, (q-c)/(1-c))}
            for q in (c, .8, .85, p)],
        'interpretation': [
            'The forecast combines all earlier confidence levels and prices; it is not the empirical win rate of four confidence-5 fights.',
            'Full Kelly is conditional on the point forecast being correct; these calculations do not establish that probability.',
            'Sensitivity probabilities are illustrative assumptions, not calibrated estimates or recommended risk limits.',
            'Full and half Kelly use the same forecast but their own current bankrolls, so dollar stakes need not have a two-to-one ratio.'
        ]
    }
    destination = ROOT/'results/largest_stake_diagnostic.json'
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
