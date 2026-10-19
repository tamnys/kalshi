"""Rank all positive intervals and simulate them over every recorded fight.

Categories are discovered retrospectively. Forecasts use prior outcomes only.
The complete positive interval set is simulated without a minimum-N filter.
"""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from audit import ROOT
from analyze import cost
from kelly import fit_firth, kelly_fraction, path_metrics
from robustness import stats


FRACTIONS = (.25, .5, 1.)  # Comparison points already used in the main report.


def forecast(full):
    """Fit only earlier decisive results; wait for this model's full column rank."""
    valid = full.kalshi_price.notna().to_numpy() & full.status.ne('cancelled').to_numpy()
    binary = full.y.notna().to_numpy() & valid
    x = np.column_stack([np.ones(len(full)), logit(full.kalshi_price), full.confidence - 3])
    binary_x = x[binary]
    warm = next(i for i in range(1, len(binary_x))
                if np.linalg.matrix_rank(binary_x[:i]) == binary_x.shape[1])
    p = np.full(len(full), np.nan)
    prior_n = np.cumsum(binary) - binary
    beta = None
    for i in range(len(full)):
        if not valid[i] or prior_n[i] < warm:
            continue
        earlier = binary & (np.arange(len(full)) < i)
        beta = fit_firth(x[earlier], full.y.to_numpy()[earlier], start=beta)
        p[i] = expit(x[i] @ beta)
    return p, prior_n, warm


def membership(intervals, full):
    c = full.confidence.to_numpy()[None, :]
    p = full.kalshi_price.to_numpy()[None, :]
    return ((c >= intervals.confidence_low.to_numpy()[:, None]) &
            (c <= intervals.confidence_high.to_numpy()[:, None]) &
            (p >= intervals.price_low.to_numpy()[:, None]) &
            (p <= intervals.price_high.to_numpy()[:, None]) &
            full.status.ne('cancelled').to_numpy()[None, :])


def trajectories(mask, full_stake, returns, fraction):
    stakes = mask * (fraction * full_stake)[None, :]
    wealth = np.column_stack([np.ones(len(mask)), np.cumprod(1 + stakes * returns, axis=1)])
    drawdown = 1 - wealth / np.maximum.accumulate(wealth, axis=1)
    count = np.sum(stakes > 0, axis=1)
    metrics = dict(
        bets=count, final_wealth=wealth[:, -1], bankroll_return=wealth[:, -1] - 1,
        max_drawdown=drawdown.max(axis=1), max_stake=stakes.max(axis=1),
        mean_stake=np.divide(stakes.sum(axis=1), count, out=np.zeros(len(mask)), where=count > 0),
    )
    return wealth, metrics


def run():
    full = pd.read_csv(ROOT / 'data/cleaned/kalshi_fights.csv').sort_values('sequence').reset_index(drop=True)
    binary = full.y.notna().to_numpy() & full.kalshi_price.notna().to_numpy()
    # Round-trip restores the exact endpoints written by the exhaustive scan.
    scan = pd.read_csv(ROOT / 'results/all_interval_search.csv', float_precision='round_trip')
    scan = scan.sort_values(['confidence_low', 'confidence_high', 'price_low', 'price_high']).reset_index(drop=True)
    scan['interval_id'] = ['interval_' + str(i) for i in range(len(scan))]
    positive = scan[scan.roi > 0].copy().reset_index(drop=True)
    mask = membership(positive, full)
    assert np.array_equal(mask[:, binary].sum(axis=1), positive.n)
    returns = full.kalshi_return.fillna(0).to_numpy()
    assert np.allclose(mask[:, binary] @ returns[binary], positive.profit)
    assert np.array_equal(mask[:, binary] @ full.loc[binary, 'y'].to_numpy(), positive.wins)

    # Nondominated in ROI and sample size: no chosen top-N or sample-size gate.
    frontier = positive.sort_values(['n', 'roi'], ascending=[False, False]).drop_duplicates('n')
    frontier = frontier[frontier.roi > frontier.roi.cummax().shift(fill_value=-np.inf)]
    frontier.to_csv(ROOT / 'results/interval_roi_frontier.csv', index=False)
    leaders = positive.sort_values(['null_z', 'n'], ascending=[False, False]).groupby(
        ['confidence_low', 'confidence_high'], sort=True).head(1).sort_values(['confidence_low', 'confidence_high'])
    leaders.to_csv(ROOT / 'results/interval_confidence_leaders.csv', index=False)
    roi_leaders = positive.sort_values(['roi', 'n'], ascending=[False, False]).groupby(
        ['confidence_low', 'confidence_high'], sort=True).head(1).sort_values(['confidence_low', 'confidence_high'])
    roi_leaders.to_csv(ROOT / 'results/interval_maximum_roi_by_confidence.csv', index=False)

    p, prior_n, earliest = forecast(full)
    previous = json.loads((ROOT / 'results/kelly_summary.json').read_text())['warmup_fights']
    predictions = full[['row', 'sequence', 'date', 'confidence', 'kalshi_price', 'payout']].copy()
    predictions['prior_decisive_fights'] = prior_n
    predictions['forecast'] = p
    predictions.to_csv(ROOT / 'artifacts/private/interval_forecasts.csv', index=False)
    full_stake = np.nan_to_num(kelly_fraction(p, full.kalshi_cost.to_numpy()), nan=0.)
    private = ROOT / 'artifacts/private'
    all_summaries = []
    for prefix in sorted(set([earliest, previous])):
        stake = np.where(prior_n >= prefix, full_stake, 0.)
        matrices = {}
        for fraction in FRACTIONS:
            wealth, metrics = trajectories(mask, stake, returns, fraction)
            frame = positive.copy()
            frame['training_prefix'] = prefix
            frame['kelly_fraction'] = fraction
            for key, values in metrics.items():
                frame[key] = values
            all_summaries.append(frame)
            matrices['wealth_' + str(fraction)] = wealth
        np.savez_compressed(private / f'all_positive_interval_paths_prefix_{prefix}.npz',
                            interval_ids=positive.interval_id.to_numpy(dtype=str), **matrices)
    all_summary = pd.concat(all_summaries, ignore_index=True)
    all_summary.to_csv(ROOT / 'results/interval_kelly_all_positive.csv.gz', index=False,
                       compression={'method': 'gzip', 'mtime': 0})
    risk_frontiers = []
    for _, g in all_summary.groupby(['training_prefix', 'kelly_fraction']):
        f = g.sort_values(['max_drawdown', 'final_wealth'], ascending=[True, False]).drop_duplicates('max_drawdown')
        risk_frontiers.append(f[f.final_wealth > f.final_wealth.cummax().shift(fill_value=-np.inf)])
    risk_frontier = pd.concat(risk_frontiers, ignore_index=True)
    risk_frontier.to_csv(ROOT / 'results/interval_kelly_return_drawdown_frontier.csv', index=False)
    kelly_leaders = all_summary.sort_values(['final_wealth', 'n'], ascending=[False, False]).groupby(
        ['training_prefix', 'kelly_fraction', 'confidence_low', 'confidence_high']).head(1)
    kelly_leaders.to_csv(ROOT / 'results/interval_kelly_leaders.csv', index=False)

    chosen = (set(frontier.interval_id) | set(leaders.interval_id) | set(roi_leaders.interval_id) |
              set(risk_frontier.interval_id) | set(kelly_leaders.interval_id))
    selected = positive[positive.interval_id.isin(chosen)].copy()
    selected['selection'] = selected.interval_id.map(lambda k: ', '.join(
        label for label, ids in [('ROI/sample frontier', set(frontier.interval_id)),
                                ('strongest signal in confidence range', set(leaders.interval_id)),
                                ('maximum ROI in confidence range', set(roi_leaders.interval_id)),
                                ('Kelly return/drawdown frontier', set(risk_frontier.interval_id)),
                                ('Kelly leader in confidence range', set(kelly_leaders.interval_id))] if k in ids))
    # Retain the prior broad recommendation and readable mined band as references.
    baseline = pd.DataFrame([
        dict(interval_id='core_all', confidence_low=4, confidence_high=5, price_low=0., price_high=1., selection='broad reference'),
        dict(interval_id='core_favorites', confidence_low=4, confidence_high=5, price_low=np.nextafter(.5, 1.), price_high=1., selection='broad reference'),
        dict(interval_id='core_underdogs', confidence_low=4, confidence_high=5, price_low=0., price_high=np.nextafter(.5, 0.), selection='broad reference'),
        dict(interval_id='readable_band', confidence_low=3, confidence_high=5, price_low=.35, price_high=np.nextafter(.4, 0.), selection='35 to below 40 cents'),
        dict(interval_id='core_above_35', confidence_low=4, confidence_high=5, price_low=.35, price_high=1., selection='remove the mined upper boundary; 35 cents and above'),
    ])
    selected = pd.concat([selected, baseline], ignore_index=True)
    selected_mask = membership(selected, full)
    details = []
    for row, m in zip(selected.itertuples(index=False), selected_mask):
        g = full.loc[m & binary].copy()
        record = stats(row.interval_id, g)
        record.update(interval_id=row.interval_id, confidence_low=int(row.confidence_low),
                      confidence_high=int(row.confidence_high), price_low=row.price_low,
                      price_high=row.price_high, selection=row.selection,
                      search_adjusted_p=row.search_adjusted_p,
                      both_directions_adjusted_p=row.both_directions_adjusted_p,
                      remove_best_roi=g.drop(g.kalshi_return.idxmax()).kalshi_return.mean(),
                      observed_roi=g.loc[g.price_source.eq('api_precard_ask'), 'kalshi_return'].mean(),
                      draw_count=int(np.sum(m & full.result.eq('draw').to_numpy())))
        leave_card = [g.loc[g.date.ne(date), 'kalshi_return'].mean() for date in g.date.unique()]
        record['leave_card_out_min_roi'] = min(leave_card)
        record['leave_card_out_max_roi'] = max(leave_card)
        details.append(record)
    details = pd.DataFrame(details)
    # Broad reference p-values come from the same complete search.
    broad = pd.read_csv(ROOT / 'results/categories.csv').set_index('category')
    for iid, name in [('core_all', 'C4-5 / all_prices'), ('core_favorites', 'C4-5 / favorite'),
                      ('core_underdogs', 'C4-5 / underdog')]:
        pv = float(broad.loc[name, 'search_adjusted_p'])
        details.loc[details.interval_id.eq(iid), ['search_adjusted_p', 'both_directions_adjusted_p']] = [pv, min(1., 2 * pv)]
    exact = scan.iloc[scan.null_z.argmax()]
    assert np.array_equal(selected_mask[selected.interval_id.eq('readable_band').to_numpy()][0],
                          membership(pd.DataFrame([exact]), full)[0])
    details.loc[details.interval_id.eq('readable_band'), ['search_adjusted_p', 'both_directions_adjusted_p']] = [exact.search_adjusted_p, exact.both_directions_adjusted_p]
    same_mask = np.all(mask == selected_mask[selected.interval_id.eq('core_above_35').to_numpy()][0], axis=1)
    ref = positive[same_mask].iloc[0]
    details.loc[details.interval_id.eq('core_above_35'), ['search_adjusted_p', 'both_directions_adjusted_p']] = [ref.search_adjusted_p, ref.both_directions_adjusted_p]
    details.to_csv(ROOT / 'results/interval_selected_details.csv', index=False)
    reconstruction = json.loads((ROOT / 'results/kalshi_price_reconstruction.json').read_text())
    price_sensitivity = []
    imputed = full.price_source.str.startswith('imputed').to_numpy()
    for quantile in ('0.025', '0.975'):
        shifted = full.copy()
        shift = reconstruction['residual_quantiles'][quantile]
        shifted['kalshi_price'] = np.clip(full.kalshi_price + imputed * shift, np.nextafter(0., 1.), 1.)
        changed_masks = membership(selected, shifted)
        shifted_returns = shifted.y / cost(shifted.kalshi_price) - 1
        for row, m in zip(selected.itertuples(index=False), changed_masks):
            eligible = m & binary
            price_sensitivity.append(dict(interval_id=row.interval_id, residual_quantile=quantile,
                                          shift_cents=100 * shift, n=int(eligible.sum()),
                                          roi=shifted_returns[eligible].mean()))
    pd.DataFrame(price_sensitivity).to_csv(ROOT / 'results/interval_price_sensitivity.csv', index=False)

    chosen_summaries = []
    paths = []
    for prefix in sorted(set([earliest, previous])):
        stake = np.where(prior_n >= prefix, full_stake, 0.)
        for fraction in FRACTIONS:
            chosen_wealth, chosen_metrics = trajectories(selected_mask, stake, returns, fraction)
            for j, iid in enumerate(selected.interval_id):
                chosen_summaries.append(dict(interval_id=iid, training_prefix=prefix, kelly_fraction=fraction,
                                             **{key: values[j] for key, values in chosen_metrics.items()}))
                for i, value in enumerate(chosen_wealth[j]):
                    paths.append(dict(interval_id=iid, training_prefix=prefix, kelly_fraction=fraction,
                                      fight_index=i, wealth=value))
    pd.DataFrame(chosen_summaries).to_csv(ROOT / 'results/interval_kelly_selected.csv', index=False)
    pd.DataFrame(paths).to_csv(ROOT / 'results/interval_bankroll_paths.csv.gz', index=False,
                            compression={'method': 'gzip', 'mtime': 0})

    # Independent scalar path accounting for every displayed candidate/scenario.
    for row in chosen_summaries:
        j = selected.index[selected.interval_id.eq(row['interval_id'])][0]
        f = np.where(prior_n >= row['training_prefix'], full_stake, 0.) * selected_mask[j] * row['kelly_fraction']
        independently = path_metrics(f, returns)
        assert np.isclose(independently['final_wealth'], row['final_wealth'])
        assert np.isclose(independently['max_drawdown'], row['max_drawdown'])
        assert independently['bets'] == row['bets']
    # Canceled and pre-estimation rows cannot change any bankroll path.
    assert np.all(full_stake[full.status.eq('cancelled')] == 0)
    assert np.all(full_stake[prior_n < earliest] == 0)
    metadata = dict(total_intervals=len(scan), positive_intervals=len(positive),
                    roi_sample_frontier_intervals=len(frontier), displayed_candidates=len(selected),
                    recorded_rows=len(full), decisive_fights=int(binary.sum()),
                    draw_rows=int(full.result.eq('draw').sum()), canceled_rows=int(full.status.eq('cancelled').sum()),
                    earliest_identifiable_prefix=earliest, comparison_prefix=previous,
                    fractions=list(FRACTIONS), simulation_scenarios=len(all_summary),
                    trajectory_points_per_scenario=len(full) + 1,
                    notes=['All positive backing intervals are simulated; no minimum-N or top-N filter.',
                           'Categories are selected using full-sample outcomes; probability fits use only previous decisive fights.',
                           'The first three decisive fights establish full rank; no probability-based stake is placed before then.',
                           'Each path begins at bankroll 1, includes skipped/canceled rows, and settles the draw at 50 cents.',
                           'Pre-card asks, reconstructed prices and scalable taker fees retain the limitations in the main report.'])
    (ROOT / 'results/interval_trajectory_summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)
    print('PASS: all interval memberships, returns, displayed scalar paths, canceled rows and training exclusions', flush=True)


if __name__ == '__main__':
    run()
