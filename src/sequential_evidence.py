"""Conditional wealth-based evidence for the fully specified adaptive family.

If E[payout_t | past and current inputs] <= cost_t and 0 <= stake_t <= 1,
then E[1 + stake_t * (payout_t/cost_t - 1) | past] <= 1. Wealth is a
nonnegative supermartingale. Ville's bound gives 1 / peak wealth. This
does not retroactively preregister this investigation or validate its prices.
"""
import json

import numpy as np
import pandas as pd

from audit import ROOT


def run():
    paths = pd.read_csv(ROOT / 'results/prospective_paths.csv.gz')
    keys = ['price_data', 'cadence', 'stake_model', 'rule', 'fraction']
    rows = []
    family = paths.groupby(keys).ngroups
    for key, g in paths.groupby(keys):
        wealth = g.sort_values('fight_index').wealth.to_numpy()
        assert wealth[0] == 1. and np.all(wealth >= 0)
        peak = float(wealth.max())
        rows.append(dict(zip(keys, key), terminal_wealth=wealth[-1], peak_wealth=peak,
                         terminal_inverse_wealth_bound=min(1., 1 / wealth[-1]) if wealth[-1] else 1.,
                         anytime_inverse_peak_bound=min(1., 1 / peak),
                         stated_family_bonferroni_bound=min(1., family / peak)))
    pd.DataFrame(rows).to_csv(ROOT / 'results/adaptive_wealth_evidence.csv', index=False)
    mixture = paths.groupby('fight_index', as_index=False).wealth.mean()
    mixture.to_csv(ROOT / 'results/adaptive_equal_mixture_path.csv', index=False)
    summary = dict(stated_strategy_family=family, equal_mixture_terminal=float(mixture.wealth.iloc[-1]),
                   equal_mixture_peak=float(mixture.wealth.max()),
                   equal_mixture_anytime_bound=float(min(1., 1 / mixture.wealth.max())),
                   minimum_individual_anytime_bound=min(r['anytime_inverse_peak_bound'] for r in rows),
                   assumptions=['Conditional expected settlement is no greater than all-in entry cost.',
                                'The algorithm and its stake are chosen without future-outcome information, with no borrowing.',
                                'Quoted/reconstructed costs are treated as the fixed design; actual executability is unverified.',
                                'This covers the stated 72 adaptive variants, not all researcher choices across the wider investigation.'],
                   sources=['https://arxiv.org/abs/1808.03204', 'https://arxiv.org/abs/1912.06116'])
    (ROOT / 'results/adaptive_wealth_evidence_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    run()
