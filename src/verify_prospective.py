"""Independent checks of training-only selection and pre-card cash accounting."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from audit import ROOT
from prospective import choose_rules, probabilities, fixed_rules, FRACTIONS, CRITERIA
from kelly import kelly_fraction


def run():
    full = pd.read_csv(ROOT / 'data/cleaned/kalshi_fights.csv').sort_values('sequence').reset_index(drop=True)
    allowed = full.y.notna().to_numpy() & full.kalshi_cost.notna().to_numpy()
    first_card = full[full.date.eq(full.date.iloc[0]) & full.y.notna()].copy()
    # Independent direct-membership sums, rather than cumulative sorted sums.
    logs = np.zeros(len(first_card))
    best = choose_rules(first_card, logs)
    raw_best = {name: -np.inf for name in CRITERIA}
    for lo in range(1, 6):
        for hi in range(lo, 6):
            confidence = first_card.confidence.between(lo, hi)
            unique = np.sort(first_card.loc[confidence, 'kalshi_price'].unique())
            if not len(unique):
                continue
            boundaries = np.r_[0., (unique[1:] + unique[:-1]) / 2, 1.]
            for a in range(len(boundaries) - 1):
                for b in range(a + 1, len(boundaries)):
                    member = confidence & first_card.kalshi_price.ge(boundaries[a]) & first_card.kalshi_price.lt(boundaries[b])
                    g = first_card[member]
                    profit = g.kalshi_return.sum()
                    raw_best['roi'] = max(raw_best['roi'], profit / len(g))
                    variance = ((1 - g.kalshi_cost) / g.kalshi_cost).sum()
                    raw_best['standardized_return'] = max(raw_best['standardized_return'], profit / np.sqrt(variance))
    for name in ('roi', 'standardized_return'):
        assert np.isclose(best[name]['score'], raw_best[name])
    assert best['quarter_kelly_log_growth'] is None

    # Last-card outcomes cannot enter forecasts at any point on that card.
    priced = full.payout.notna().to_numpy() & full.kalshi_price.notna().to_numpy()
    original, _ = probabilities(full, priced, ['odds_confidence'], cadence='card')
    corrupted = full.copy()
    last = full.date.eq(full.date.iloc[-1]) & full.y.notna()
    corrupted.loc[last, 'y'] = 1 - corrupted.loc[last, 'y']
    alternate, _ = probabilities(corrupted, priced, ['odds_confidence'], cadence='card')
    assert np.allclose(original['odds_confidence'], alternate['odds_confidence'], equal_nan=True)

    history = pd.read_csv(ROOT / 'artifacts/private/prospective_rule_history.csv')
    for row in history.itertuples():
        i = int(row.fight_index) - 1
        expected = allowed.copy()
        if row.price_data == 'observed_only':
            expected &= full.price_source.eq('api_precard_ask').to_numpy()
        expected &= np.arange(len(full)) < i if row.cadence == 'fight' else full.date.to_numpy() < full.date.iloc[i]
        assert expected.sum() == row.prior_training_n
        if pd.notna(row.training_n):
            member = expected & full.confidence.between(row.confidence_low, row.confidence_high).to_numpy()
            member &= full.kalshi_price.ge(row.price_low).to_numpy() & full.kalshi_price.lt(row.price_high).to_numpy()
            assert member.sum() == row.training_n
            assert np.isclose(full.loc[member, 'kalshi_return'].sum(), row.training_profit)

    # Reconcile execution via cash paid for contracts and cash received at payout,
    # instead of the production implementation's incremental profit identity.
    forecast = pd.read_csv(ROOT / 'artifacts/private/prospective_forecasts.csv')
    saved = pd.read_csv(ROOT / 'results/candidate_precard_execution.csv')
    paths = pd.read_csv(ROOT / 'results/candidate_precard_paths.csv.gz')
    for provenance in ('reconstructed_and_observed', 'observed_only'):
        tradeable = priced.copy()
        if provenance == 'observed_only':
            tradeable &= full.price_source.eq('api_precard_ask').to_numpy()
        f = np.nan_to_num(kelly_fraction(forecast[provenance + '_before_card'].to_numpy(), full.kalshi_cost.to_numpy()), nan=0.)
        for name, member in fixed_rules(full).items():
            for fraction in FRACTIONS:
                wealth, cash_path, feasible = 1., [1.], True
                for date in full.date.unique():
                    indices = np.flatnonzero(full.date.to_numpy() == date)
                    outlays = wealth * fraction * f[indices] * member[indices] * tradeable[indices]
                    if outlays.sum() > wealth:
                        feasible = False
                        break
                    cash = wealth - outlays.sum()
                    remaining = outlays.sum()
                    for j, i in enumerate(indices):
                        if outlays[j] > 0:
                            cash += outlays[j] / full.kalshi_cost.iloc[i] * full.payout.iloc[i]
                        remaining -= outlays[j]
                        cash_path.append(cash + remaining)
                    wealth = cash
                row = saved[(saved.price_data == provenance) & (saved.rule == name) & (saved.fraction == fraction)].iloc[0]
                assert row.feasible_without_borrowing == feasible
                if feasible:
                    actual = paths[(paths.price_data == provenance) & (paths.rule == name) & (paths.fraction == fraction)].sort_values('fight_index').wealth
                    assert len(actual) == len(full) + 1
                    assert np.allclose(cash_path, actual)
                    assert np.isclose(wealth, row.final_wealth)

    bootstrap = np.load(ROOT / 'artifacts/private/calibration_bootstrap_coefficients.npz')
    meta = json.loads((ROOT / 'results/calibration_uncertainty_summary.json').read_text())
    assert bootstrap['coefficients'].shape == (meta['resamples'], 3)
    assert np.isfinite(bootstrap['coefficients']).all()
    for row in pd.read_csv(ROOT / 'results/calibration_sizing_uncertainty.csv').itertuples():
        # Independently expand the affine expression. Matrix products on this host
        # emitted floating-status warnings despite finite operands/matching values.
        beta = bootstrap['coefficients']
        p = expit(beta[:, 0] + beta[:, 1] * logit(row.ask) + beta[:, 2] * (row.confidence - 3))
        assert np.isclose(np.quantile(p, .025), row.probability_p025)
        assert np.isclose(np.quantile(p, .975), row.probability_p975)
        assert np.isclose(np.mean(p > row.all_in_cost), row.bootstrap_positive_edge_frequency)
    print('PASS: direct exhaustive selection, prior-only rule histories, pre-card future-outcome invariance, independent cash/contract execution, and all bootstrap summaries')


if __name__ == '__main__':
    run()
