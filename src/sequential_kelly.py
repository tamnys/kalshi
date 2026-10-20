"""Sequential Kelly comparisons, earlier-data sizing and conditional card resampling."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from audit import ROOT
from analyze import N_SIM, SEED
from aggressive_growth import legs, portfolios
from lower_confidence import side_frame


FIXED = {'quarter': .25, 'half': .5, 'full': 1.}
# Reuse the natural chronological splits from the existing validation analysis.
SPLITS = ('2025-07-19', '2026-10-10')


def optimal_fraction(increments):
    """Maximize past log growth over fractional Kelly [0,1]; flat ties use zero."""
    z = np.asarray(increments)
    z = z[z != 0]
    if len(z) == 0 or z.sum() <= 0:
        return 0.
    if np.sum(z/(1+z)) >= 0:
        return 1.
    return float(brentq(lambda value: np.sum(z/(1+value*z)), 0., 1.))


def learned_fractions(increments):
    # Outcomes of skipped but observable opportunities are also available to learn
    # from; otherwise the zero-stake start would prevent the strategy ever learning.
    return np.array([optimal_fraction(increments[:i]) for i in range(len(increments))])


def context_inputs():
    full = pd.read_csv(ROOT/'data/cleaned/kalshi_fights.csv').sort_values('sequence').reset_index(drop=True)
    configs = legs(full)
    sides = {side: side_frame(full, side) for side in ('back', 'fade')}
    saved = pd.read_csv(ROOT/'artifacts/private/aggressive_growth_full_stakes.csv')
    contexts = []
    for sample in ('all', 'observed'):
        allowed = full.payout.notna().to_numpy()
        if sample == 'observed':
            allowed &= full.price_source.eq('api_precard_ask').to_numpy()
        for portfolio, constituents in portfolios().items():
            costs, payouts, used = np.zeros(len(full)), np.zeros(len(full)), np.zeros(len(full), bool)
            for constituent in constituents:
                side, membership, _ = configs[constituent]
                membership = membership & allowed
                assert not (used & membership).any()
                used |= membership
                costs[membership] = sides[side].kalshi_cost.to_numpy()[membership]
                payouts[membership] = sides[side].payout.to_numpy()[membership]
            returns = np.divide(payouts, costs, out=np.ones(len(full)), where=costs > 0)-1
            for model in ('odds_confidence', 'confidence_categories', 'category_offset'):
                key = '|'.join((portfolio, sample, 'fight', model))
                f = saved[key].to_numpy()
                assert np.all((f >= 0) & (f < 1))
                assert np.all(f[~used] == 0)
                contexts.append((dict(portfolio=portfolio, sample=sample, model=model),
                                 f, returns.copy(), costs.copy(), payouts.copy()))
    return full, contexts


def trajectory(f, returns, multipliers):
    multipliers = np.broadcast_to(multipliers, f.shape)
    stake = f*multipliers
    factors = 1+stake*returns
    assert np.all(factors > 0)
    wealth = np.r_[1., np.cumprod(factors)]
    drawdown = 1-wealth/np.maximum.accumulate(wealth)
    placed = stake > 0
    metrics = dict(final_wealth=float(wealth[-1]), max_drawdown=float(drawdown.max()),
                   max_stake=float(stake.max()), bets=int(placed.sum()),
                   winning_positions=int(((returns > 0) & placed).sum()),
                   losing_positions=int(((returns < 0) & placed).sum()),
                   mean_stake=float(stake[placed].mean()) if placed.any() else 0.,
                   mean_multiplier=float(multipliers[placed].mean()) if placed.any() else 0.,
                   minimum_wealth=float(wealth.min()))
    return metrics, wealth, drawdown, stake, multipliers


def bootstrap_card_paths(full, increments, fraction, draws):
    indices = [np.flatnonzero(full.date.eq(date)) for date in full.date.unique()]
    # Padding has zero log return and therefore cannot change wealth or drawdown.
    block_logs = np.zeros((len(indices), max(map(len, indices))))
    for j, ix in enumerate(indices):
        block_logs[j, :len(ix)] = np.log1p(fraction*increments[ix])
    log_paths = np.cumsum(block_logs[draws].reshape(len(draws), -1), axis=1)
    peak_logs = np.maximum(0., np.maximum.accumulate(log_paths, axis=1))
    drawdowns = np.max(-np.expm1(log_paths-peak_logs), axis=1)
    terminal = np.exp(log_paths[:, -1])
    return dict(wealth_p025=float(np.quantile(terminal, .025)),
                wealth_median=float(np.median(terminal)), wealth_p975=float(np.quantile(terminal, .975)),
                empirical_loss_frequency=float(np.mean(terminal < 1)),
                drawdown_median=float(np.median(drawdowns)), drawdown_p95=float(np.quantile(drawdowns, .95)),
                max_resampled_drawdown=float(drawdowns.max()),
                worst_drawdown_replica=int(np.argmax(drawdowns)),
                has_recorded_loss=bool(np.any(increments < 0)),
                recorded_opportunities=int(np.count_nonzero(increments)),
                recorded_cards_with_positions=int(sum(np.any(increments[ix] != 0) for ix in indices)))


def run():
    full, contexts = context_inputs()
    old = pd.read_csv(ROOT/'results/aggressive_growth.csv')
    old = old[old.cadence.eq('fight') & old.choice.eq('hindsight_best_feasible')]
    lookup = old.set_index(['portfolio', 'sample', 'model']).fraction
    rng = np.random.default_rng(SEED)
    dates = full.date.unique()
    draws = rng.integers(len(dates), size=(N_SIM, len(dates)))
    np.save(ROOT/'artifacts/private/sequential_bootstrap_card_indices.npy', draws)
    scenarios, paths, frontiers, frozen, resamples = [], [], [], [], []
    for context, f, returns, _, _ in contexts:
        increments = f*returns
        retrospective = float(lookup.loc[tuple(context.values())])
        methods = {**FIXED, 'hindsight_best_feasible': retrospective,
                   'learned_from_prior': learned_fractions(increments)}
        for policy, multiplier in methods.items():
            metrics, wealth, drawdown, stake, multipliers = trajectory(f, returns, multiplier)
            scenarios.append(dict(**context, policy=policy,
                                  fraction=float(multiplier) if np.isscalar(multiplier) else np.nan,
                                  **metrics))
            paths.extend(dict(**context, policy=policy, fight_index=i, wealth=float(value),
                              drawdown=float(drawdown[i]), stake=float(stake[i-1]) if i else 0.,
                              multiplier=float(multipliers[i-1]) if i else 0.)
                         for i, value in enumerate(wealth))
        # Existing project convention: 1%-Kelly display resolution. Exact
        # continuous optima are computed separately, not constrained to this grid.
        for multiplier in np.linspace(0, 1, 101):
            metrics, *_ = trajectory(f, returns, multiplier)
            frontiers.append(dict(**context, fraction=float(multiplier), **metrics))
        for split in SPLITS:
            training = full.date.to_numpy() < split
            testing = ~training
            train_multiplier = optimal_fraction(increments[training])
            test_oracle = optimal_fraction(increments[testing])
            policies = {**FIXED, 'prior_fitted': train_multiplier, 'test_hindsight_optimum': test_oracle}
            for policy, multiplier in policies.items():
                metrics, *_ = trajectory(f[testing], returns[testing], multiplier)
                frozen.append(dict(**context, split=split, policy=policy, fraction=multiplier,
                                   training_opportunities=int(np.count_nonzero(increments[training])),
                                   test_rows=int(testing.sum()), **metrics))
        for policy, multiplier in FIXED.items():
            resamples.append(dict(**context, policy=policy, fraction=multiplier,
                                  **bootstrap_card_paths(full, increments, multiplier, draws)))
        if context['portfolio'] == 'two_unbeaten_intervals':
            print('Processed', context, flush=True)
    outputs = {'scenarios': scenarios, 'paths': paths, 'frontiers': frontiers,
               'frozen': frozen, 'bootstrap': resamples}
    for name, rows in outputs.items():
        suffix = '.csv.gz' if name == 'paths' else '.csv'
        compression = {'method': 'gzip', 'mtime': 0} if name == 'paths' else None
        pd.DataFrame(rows).to_csv(ROOT/('results/sequential_kelly_'+name+suffix), index=False, compression=compression)
    metadata = dict(contexts=len(contexts), scenarios=len(scenarios), path_points=len(paths),
                    frontier_points=len(frontiers), frozen_comparisons=len(frozen),
                    bootstrap_comparisons=len(resamples), bootstrap_replicates=N_SIM,
                    bootstrap_cards_per_replica=len(dates), recorded_rows=len(full),
                    decisive_fights=int(full.y.notna().sum()), splits=list(SPLITS),
                    notes=[
                        'Every bet settles before the next is sized, using one bankroll and no simultaneous card constraint.',
                        'Chronology follows the audited ESPN fight order within each card.',
                        'Probability models and side-specific Kalshi costs are the existing verified sequential inputs.',
                        'Quarter/half/full are existing comparison fractions, not chosen exposure limits.',
                        'Hindsight optimum uses the full evaluation period and is not a deployable selection rule.',
                        'Learned-from-prior fits a common fraction on earlier realized full-Kelly opportunity returns only; no-history/flat ties use zero.',
                        'Earlier observable opportunities are used even if the multiplier made the executed stake zero.',
                        'Frozen tests fit the fraction before the existing natural date split, then hold that fraction fixed; probability forecasts continue updating after settlements.',
                        'Frozen test wealth starts at one at the split and must be compared with its same-window fixed-fraction controls.',
                        'Bootstrap resamples 15 whole cards, preserving within-card order, cached probabilities and fixed multipliers. It does not refit models or correct retrospective category selection.',
                        'Bootstrap replicate count retains the existing worst-case frequency Monte Carlo SE <= .005 convention.',
                        'A no-loss selected sample cannot generate losing bets by empirical resampling; zero empirical loss frequency is not evidence of zero future risk.',
                        'Prices include modeled fees and retain the assumption of obtainable quotes and settlement before the next fight.'
                    ])
    (ROOT/'results/sequential_kelly_summary.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({k: v for k, v in metadata.items() if k != 'notes'}, indent=2), flush=True)


if __name__ == '__main__':
    run()
