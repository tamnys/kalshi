"""Independent cash accounting, past-only sizing and resampling reconciliation."""
import json
import numpy as np
import pandas as pd
from audit import ROOT
from sequential_kelly import context_inputs, FIXED, SPLITS


def verify_optimum(z, value):
    z = np.asarray(z)
    if not np.any(z):
        assert value == 0
        return
    slope = np.sum(z/(1+value*z))
    # Brent's documented default root tolerance, mapped to gradient error by
    # the objective curvature, plus accumulated floating-point roundoff.
    delta = 2e-12 + 4*np.finfo(float).eps
    points = np.clip([value-delta, value+delta], 0, 1)
    curvature = max(np.sum(z*z/(1+p*z)**2) for p in points)
    error = delta*curvature + np.finfo(float).eps*len(z)*(1+np.sum(np.abs(z/(1+value*z))))
    if value == 0:
        assert slope <= error
    elif value == 1:
        assert slope >= -error
    else:
        assert abs(slope) <= error


def run():
    full, contexts = context_inputs()
    scenarios = pd.read_csv(ROOT/'results/sequential_kelly_scenarios.csv')
    paths = pd.read_csv(ROOT/'results/sequential_kelly_paths.csv.gz')
    frozen = pd.read_csv(ROOT/'results/sequential_kelly_frozen.csv')
    resamples = pd.read_csv(ROOT/'results/sequential_kelly_bootstrap.csv')
    old = pd.read_csv(ROOT/'results/aggressive_growth.csv')
    old = old[old.cadence.eq('fight')].set_index(['portfolio', 'sample', 'model', 'choice'])
    draws = np.load(ROOT/'artifacts/private/sequential_bootstrap_card_indices.npy')
    indices = [np.flatnonzero(full.date.eq(date)) for date in full.date.unique()]
    grouped_paths = {key: g.sort_values('fight_index') for key, g in paths.groupby(['portfolio', 'sample', 'model', 'policy'], sort=False)}
    checked_paths = prior_conditions = reconciled = frozen_conditions = bootstrap_conditions = 0
    for context, full_fraction, returns, costs, payouts in contexts:
        key = tuple(context.values())
        match = np.logical_and.reduce([scenarios[k].eq(v).to_numpy() for k, v in context.items()])
        increments = full_fraction*returns
        for row in scenarios[match].itertuples():
            path = grouped_paths[(*key, row.policy)]
            assert len(path) == len(full)+1
            assert path.fight_index.tolist() == list(range(len(full)+1))
            fractions = path.stake.to_numpy()[1:]
            multipliers = path.multiplier.to_numpy()[1:]
            assert np.allclose(fractions, full_fraction*multipliers)
            cash = 1.; ledger = [cash]
            for i, fraction in enumerate(fractions):
                outlay = cash*fraction
                if outlay:
                    assert costs[i] > 0 and outlay <= cash
                    cash -= outlay
                    cash += outlay/costs[i]*payouts[i]
                ledger.append(cash)
                if row.policy == 'learned_from_prior':
                    # Each saved decision must optimize the STRICT earlier prefix.
                    verify_optimum(increments[:i], multipliers[i])
                    prior_conditions += 1
            assert np.allclose(ledger, path.wealth)
            ledger = np.asarray(ledger)
            dd = 1-ledger/np.maximum.accumulate(ledger)
            assert np.allclose(dd, path.drawdown)
            assert np.isclose(cash, row.final_wealth)
            assert np.isclose(dd.max(), row.max_drawdown)
            if row.policy != 'learned_from_prior':
                prior = old.loc[(*key, row.policy)]
                for column in ('final_wealth', 'max_drawdown', 'max_stake', 'bets'):
                    assert np.isclose(getattr(row, column), prior[column])
                reconciled += 1
            checked_paths += 1
        fm = np.logical_and.reduce([frozen[k].eq(v).to_numpy() for k, v in context.items()])
        for split in SPLITS:
            training = full.date.to_numpy() < split
            rows = frozen[fm & frozen.split.eq(split)]
            for row in rows.itertuples():
                if row.policy == 'prior_fitted':
                    verify_optimum(increments[training], row.fraction)
                    frozen_conditions += 1
                elif row.policy == 'test_hindsight_optimum':
                    verify_optimum(increments[~training], row.fraction)
                    frozen_conditions += 1
                else:
                    assert row.fraction == FIXED[row.policy]
                # Use explicit multiplication rather than trajectory().
                factors = 1+row.fraction*increments[~training]
                wealth = np.r_[1, np.cumprod(factors)]
                assert np.isclose(wealth[-1], row.final_wealth)
                assert np.isclose(np.max(1-wealth/np.maximum.accumulate(wealth)), row.max_drawdown)
        bm = np.logical_and.reduce([resamples[k].eq(v).to_numpy() for k, v in context.items()])
        for row in resamples[bm].itertuples():
            # Terminal wealth can independently be obtained by multiplying card
            # factors without the padded within-card path used by the producer.
            card_factors = np.array([np.prod(1+row.fraction*increments[ix]) for ix in indices])
            terminal = np.prod(card_factors[draws], axis=1)
            assert np.allclose(np.quantile(terminal, [.025, .5, .975]),
                               [row.wealth_p025, row.wealth_median, row.wealth_p975])
            # Treat machine-roundoff around exactly one as a boundary. Agreement
            # of frequencies must hold away from that ambiguous numerical point.
            roundoff = np.finfo(float).eps*len(full)*max(map(len, indices))
            certain_losses = np.mean(terminal < 1-roundoff)
            possible_losses = np.mean(terminal < 1+roundoff)
            assert certain_losses <= row.empirical_loss_frequency <= possible_losses
            worst_indices = np.concatenate([indices[k] for k in draws[row.worst_drawdown_replica]])
            wealth = np.r_[1., np.cumprod(1+row.fraction*increments[worst_indices])]
            assert np.isclose(np.max(1-wealth/np.maximum.accumulate(wealth)), row.max_resampled_drawdown)
            if not row.has_recorded_loss:
                assert row.drawdown_p95 == 0
                assert row.empirical_loss_frequency == 0
            bootstrap_conditions += 1
    result = dict(status='PASS', independent_cash_paths=checked_paths,
                  prior_fixed_scenarios_reconciled=reconciled,
                  strictly_prior_fraction_optimality_conditions=prior_conditions,
                  frozen_fraction_optimality_conditions=frozen_conditions,
                  independent_bootstrap_terminal_and_tail_checks=bootstrap_conditions)
    (ROOT/'results/sequential_kelly_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
