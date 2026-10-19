"""Check chronological forecasts, full paths and independent payoff accounting."""
import json

import numpy as np
import pandas as pd

from audit import ROOT
from interval_trajectories import forecast, trajectories


def run():
    full = pd.read_csv(ROOT / 'data/cleaned/kalshi_fights.csv').sort_values('sequence').reset_index(drop=True)
    p = pd.read_csv(ROOT / 'artifacts/private/interval_forecasts.csv').forecast.to_numpy()
    metadata = json.loads((ROOT / 'results/interval_trajectory_summary.json').read_text())
    # Changing an outcome must not alter that fight's forecast or any earlier one.
    changed = full.copy()
    first_bettable = np.flatnonzero(np.isfinite(p))[0]
    changed.loc[first_bettable, 'y'] = 1 - changed.loc[first_bettable, 'y']
    alternate, _, _ = forecast(changed)
    assert np.allclose(alternate[:first_bettable + 1], p[:first_bettable + 1], equal_nan=True)
    assert not np.allclose(alternate[first_bettable + 1:], p[first_bettable + 1:], equal_nan=True)
    # A real draw pays half a contract; skipped/canceled rows preserve wealth.
    price = .4
    cost = price + .07 * price * (1 - price)
    returns = np.array([.5 / cost - 1, 0., -1.])
    wealth, metrics = trajectories(np.array([[True, False, True]]), np.array([.2, .7, .4]), returns, .5)
    expected_draw_wealth = 1 + .1 * (.5 / cost - 1)
    assert np.allclose(wealth[0], [1., expected_draw_wealth, expected_draw_wealth, expected_draw_wealth * .8])
    assert metrics['bets'][0] == 2
    summary = pd.read_csv(ROOT / 'results/interval_kelly_all_positive.csv.gz')
    for prefix in (metadata['earliest_identifiable_prefix'], metadata['comparison_prefix']):
        archive = np.load(ROOT / f'artifacts/private/all_positive_interval_paths_prefix_{prefix}.npz')
        for fraction in metadata['fractions']:
            values = archive['wealth_' + str(fraction)]
            s = summary[(summary.training_prefix == prefix) & (summary.kelly_fraction == fraction)]
            assert np.array_equal(s.interval_id, archive['interval_ids'])
            assert values.shape == (metadata['positive_intervals'], len(full) + 1)
            assert np.all(values[:, 0] == 1.)
            assert np.allclose(values[:, -1], s.final_wealth)
            drawdowns = 1 - values / np.maximum.accumulate(values, axis=1)
            assert np.allclose(drawdowns.max(axis=1), s.max_drawdown)
            for i in np.flatnonzero(full.status.eq('cancelled').to_numpy()):
                assert np.array_equal(values[:, i], values[:, i + 1])
    # Preserve the prior broad result when using its original training convention.
    current = pd.read_csv(ROOT / 'results/interval_kelly_selected.csv')
    previous = pd.read_csv(ROOT / 'results/portfolio_tradeoffs.csv')
    for fraction in metadata['fractions']:
        a = current[(current.interval_id == 'core_all') & (current.training_prefix == 24) &
                    (current.kelly_fraction == fraction)].iloc[0]
        b = previous[(previous.portfolio == 'confidence_4_5') & (previous.fraction == fraction)].iloc[0]
        # Same Firth optimum, potentially different BFGS warm starts; its default
        # gradient convergence tolerance is 1e-5, rather than bitwise identity.
        assert np.isclose(a.final_wealth, b.final_wealth, rtol=1e-5)
        assert np.isclose(a.max_drawdown, b.max_drawdown, rtol=1e-5)
    print('PASS: forecasts exclude current/future outcomes; draw payout and cancellation accounting; all 91,752 full trajectories; prior-result reconciliation')


if __name__ == '__main__':
    run()
