"""Earlier-outcome-only category selection, model choice and quote-time execution."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from audit import ROOT
from kelly import designs, fit_firth, kelly_fraction, path_metrics


FRACTIONS = (.25, .5, 1.)
CRITERIA = ('roi', 'standardized_return', 'quarter_kelly_log_growth')


def fixed_rules(full):
    c, p = full.confidence.to_numpy(), full.kalshi_price.to_numpy()
    return {
        'C4-5_35-81': (c >= 4) & (p >= .35) & (p <= .81),
        'C4-5_35-plus': (c >= 4) & (p >= .35),
        'C4-5_all': c >= 4,
        'C3-5_35-under40': (c >= 3) & (p >= .35) & (p < .4),
    }


def probabilities(full, eligible, model_names=None, cadence='fight'):
    """Current row inputs are allowed; only strictly earlier outcomes are fitted."""
    designs_all = designs(full)
    names = list(designs_all) if model_names is None else model_names
    binary = eligible & full.y.notna().to_numpy()
    sequence = np.arange(len(full))
    result, warmups = {}, {}
    for name in names:
        x, offset = designs_all[name]
        pred = np.full(len(full), np.nan)
        beta, last_cutoff = None, None
        warm = None
        for i in range(len(full)):
            if not eligible[i]:
                continue
            if cadence == 'fight':
                tr = binary & (sequence < i)
                cutoff = i
            else:
                tr = binary & (full.date.to_numpy() < full.date.iloc[i])
                cutoff = full.date.iloc[i]
            if np.sum(tr) == 0 or np.linalg.matrix_rank(x[tr]) < x.shape[1]:
                continue
            if warm is None:
                warm = int(np.sum(tr))
            if cutoff != last_cutoff:
                beta = fit_firth(x[tr], full.y.to_numpy()[tr], offset[tr], beta)
                last_cutoff = cutoff
            pred[i] = expit(x[i] @ beta + offset[i])
        result[name] = pred
        warmups[name] = warm
    return result, warmups


def choose_rules(training, prior_quarter_logs):
    """Exhaust all confidence/price intervals using training observations only."""
    if len(training) == 0:
        return {criterion: None for criterion in CRITERIA}
    contenders = {criterion: [] for criterion in CRITERIA}
    confidence = training.confidence.to_numpy()
    prices = training.kalshi_price.to_numpy()
    returns = training.kalshi_return.to_numpy()
    variance = (1 - training.kalshi_cost.to_numpy()) / training.kalshi_cost.to_numpy()
    for lo in range(1, 6):
        for hi in range(lo, 6):
            indices = np.flatnonzero((confidence >= lo) & (confidence <= hi))
            if len(indices) == 0:
                continue
            indices = indices[np.argsort(prices[indices], kind='stable')]
            unique, first = np.unique(prices[indices], return_index=True)
            edges = np.r_[first, len(indices)]
            boundary = np.r_[0., (unique[:-1] + unique[1:]) / 2, 1.]
            start, end = np.triu_indices(len(edges), k=1)
            a, b = edges[start], edges[end]
            n = b - a
            profit = np.r_[0., np.cumsum(returns[indices])]
            var = np.r_[0., np.cumsum(variance[indices])]
            logs = np.r_[0., np.cumsum(prior_quarter_logs[indices])]
            values = {
                'roi': (profit[b] - profit[a]) / n,
                'standardized_return': (profit[b] - profit[a]) / np.sqrt(var[b] - var[a]),
                'quarter_kelly_log_growth': logs[b] - logs[a],
            }
            for criterion, score in values.items():
                # Accumulation-roundoff tie handling, not a statistical threshold.
                tolerance = np.finfo(float).eps * len(training) * (1 + np.max(np.abs(score)))
                tied = np.flatnonzero(score >= np.max(score) - tolerance)
                j = min(tied, key=lambda k: (-n[k], -(boundary[end[k]] - boundary[start[k]]), boundary[start[k]], boundary[end[k]]))
                contenders[criterion].append(dict(confidence_low=lo, confidence_high=hi,
                                                   price_low=float(boundary[start[j]]), price_high=float(boundary[end[j]]),
                                                   training_n=int(n[j]), training_profit=float(profit[b[j]] - profit[a[j]]),
                                                   score=float(score[j])))
    chosen = {}
    for criterion, candidates in contenders.items():
        maximum = max(r['score'] for r in candidates)
        if maximum <= 0:
            chosen[criterion] = None
            continue
        tolerance = np.finfo(float).eps * len(training) * (1 + abs(maximum))
        tied = [r for r in candidates if r['score'] >= maximum - tolerance]
        chosen[criterion] = min(tied, key=lambda r: (-r['training_n'], -(r['confidence_high'] - r['confidence_low']),
                                                    -(r['price_high'] - r['price_low']), r['confidence_low'], r['price_low']))
    return chosen


def applies(rule, row):
    return rule is not None and rule['confidence_low'] <= row.confidence <= rule['confidence_high'] and rule['price_low'] <= row.kalshi_price < rule['price_high']


def local_probability(full, training_mask, index):
    """Existing Firth common-offset model fitted to the selected category only."""
    n = int(training_mask.sum())
    if n == 0:
        return np.nan
    beta = fit_firth(np.ones((n, 1)), full.y.to_numpy()[training_mask],
                     logit(full.kalshi_price.to_numpy()[training_mask]))
    return float(expit(logit(full.kalshi_price.iloc[index]) + beta[0]))


def paths_and_summary(full, eligible, masks, p, context):
    returns = full.kalshi_return.fillna(0).to_numpy()
    full_stake = np.nan_to_num(kelly_fraction(p, full.kalshi_cost.to_numpy()), nan=0.)
    rows, paths = [], []
    for name, mask in masks.items():
        selected = mask & eligible
        binary = selected & full.y.notna().to_numpy()
        flat = returns[selected]
        for fraction in FRACTIONS:
            stake = fraction * full_stake * selected
            metrics = path_metrics(stake, returns)
            rows.append(dict(**context, rule=name, fraction=fraction, flat_n=int(selected.sum()),
                             decisive_n=int(binary.sum()), wins=int(full.y.to_numpy()[binary].sum()),
                             flat_roi=float(flat.mean()) if len(flat) else None, flat_profit=float(flat.sum()), **metrics))
            wealth = np.r_[1., np.cumprod(1 + stake * returns)]
            paths.extend(dict(**context, rule=name, fraction=fraction, fight_index=i, wealth=value)
                         for i, value in enumerate(wealth))
    return rows, paths


def run():
    full = pd.read_csv(ROOT / 'data/cleaned/kalshi_fights.csv').sort_values('sequence').reset_index(drop=True)
    eligible = full.payout.notna().to_numpy() & full.kalshi_price.notna().to_numpy()
    observed = eligible & full.price_source.eq('api_precard_ask').to_numpy()
    returns = full.kalshi_return.fillna(0).to_numpy()
    sequence = np.arange(len(full))
    model_results, model_paths, adaptive, adaptive_paths, frozen, frozen_rules = [], [], [], [], [], []
    execution, execution_paths, histories = [], [], []
    forecasts = full[['row', 'sequence', 'date']].copy()
    warmup_records = {}
    for provenance, allowed in [('reconstructed_and_observed', eligible), ('observed_only', observed)]:
        names = None if provenance == 'reconstructed_and_observed' else ['odds_confidence']
        pred, warmups = probabilities(full, allowed, names)
        warmup_records[provenance] = warmups
        before_card, _ = probabilities(full, allowed, ['odds_confidence'], cadence='card')
        for name, p in pred.items():
            forecasts[provenance + '_' + name] = p
        forecasts[provenance + '_before_card'] = before_card['odds_confidence']
        p = pred['odds_confidence']
        stake = np.nan_to_num(kelly_fraction(p, full.kalshi_cost.to_numpy()), nan=0.)
        prior_logs = np.log1p(.25 * stake * returns)
        binary = allowed & full.y.notna().to_numpy()
        prior_count = np.cumsum(binary) - binary
        common_prefix = json.loads((ROOT / 'results/kelly_summary.json').read_text())['warmup_fights']
        for name, probabilities_for_model in pred.items():
            for convention, available in [('earliest_rank', allowed), ('common_24', allowed & (prior_count >= common_prefix))]:
                rows, paths = paths_and_summary(full, available, fixed_rules(full), probabilities_for_model,
                                                dict(price_data=provenance, model=name, initialization=convention))
                model_results.extend(rows); model_paths.extend(paths)
        # Compare global calibration with learning each fixed category's excess
        # win probability. A single common offset is already used in the study;
        # it requires one earlier category result, with no invented pseudo-count.
        for name, mask in fixed_rules(full).items():
            local = np.full(len(full), np.nan)
            for i in np.flatnonzero(mask & allowed):
                tr = binary & mask & (sequence < i)
                local[i] = local_probability(full, tr, i)
            forecasts[provenance + '_category_offset_' + name] = local
            for convention, available in [('earliest_rank', allowed), ('common_24', allowed & (prior_count >= common_prefix))]:
                rows, paths = paths_and_summary(full, available, {name: mask}, local,
                                                dict(price_data=provenance, model='category_offset', initialization=convention))
                model_results.extend(rows); model_paths.extend(paths)
        for cadence in ('fight', 'card'):
            masks = {criterion: np.zeros(len(full), dtype=bool) for criterion in CRITERIA}
            local_predictions = {criterion: np.full(len(full), np.nan) for criterion in CRITERIA}
            choices, last_cutoff = None, None
            for i, row in enumerate(full.itertuples(index=False)):
                if not allowed[i]:
                    continue
                tr = binary & ((sequence < i) if cadence == 'fight' else (full.date.to_numpy() < row.date))
                cutoff = i if cadence == 'fight' else row.date
                if cutoff != last_cutoff:
                    choices = choose_rules(full.loc[tr], prior_logs[tr])
                    last_cutoff = cutoff
                for criterion, rule in choices.items():
                    masks[criterion][i] = applies(rule, row)
                    if masks[criterion][i]:
                        historical_members = np.array([applies(rule, historical) for historical in full.itertuples(index=False)])
                        local_training = binary & historical_members & (sequence < i)
                        local_predictions[criterion][i] = local_probability(full, local_training, i)
                    histories.append(dict(price_data=provenance, cadence=cadence, criterion=criterion, row=row.row,
                                          fight_index=i + 1, prior_training_n=int(tr.sum()), selected=bool(masks[criterion][i]),
                                          **(rule or {})))
            rows, paths = paths_and_summary(full, allowed, masks, p, dict(price_data=provenance, cadence=cadence, stake_model='global_odds_confidence'))
            adaptive.extend(rows); adaptive_paths.extend(paths)
            for criterion in CRITERIA:
                rows, paths = paths_and_summary(full, allowed, {criterion: masks[criterion]}, local_predictions[criterion],
                                                dict(price_data=provenance, cadence=cadence, stake_model='category_offset'))
                adaptive.extend(rows); adaptive_paths.extend(paths)
        for split in ('2025-07-19', '2026-10-10'):
            tr = binary & (full.date.to_numpy() < split)
            choices = choose_rules(full.loc[tr], prior_logs[tr])
            test = allowed & (full.date.to_numpy() >= split)
            masks = {}
            for criterion, rule in choices.items():
                masks[criterion] = np.array([applies(rule, row) for row in full.itertuples(index=False)])
                frozen_rules.append(dict(price_data=provenance, split=split, criterion=criterion, **(rule or {})))
            rows, _ = paths_and_summary(full, test, masks, p, dict(price_data=provenance, split=split))
            frozen.extend(rows)

        # Forecast and individual-fight allocations are both fixed at card start.
        # Equity is updated at every recorded fight; unspent cash is not borrowed.
        pc = before_card['odds_confidence']
        fc = np.nan_to_num(kelly_fraction(pc, full.kalshi_cost.to_numpy()), nan=0.)
        for name, mask in fixed_rules(full).items():
            card_exposures = [float(np.sum(fc[(full.date.to_numpy() == date) & mask & allowed])) for date in full.date.unique()]
            feasible_fraction = min(1., 1 / max(card_exposures)) if max(card_exposures) > 0 else 1.
            for fraction in FRACTIONS:
                wealth = peak = 1.; max_dd = 0.; bets = 0
                path = [1.]; feasible = True
                for date in full.date.unique():
                    indices = np.flatnonzero(full.date.to_numpy() == date)
                    f = fraction * fc[indices] * mask[indices] * allowed[indices]
                    if f.sum() > 1:
                        feasible = False
                        break
                    starting_equity = wealth
                    for index, allocation in zip(indices, f):
                        wealth += starting_equity * allocation * returns[index]
                        bets += allocation > 0
                        peak = max(peak, wealth); max_dd = max(max_dd, 1 - wealth / peak)
                        path.append(wealth)
                execution.append(dict(price_data=provenance, rule=name, fraction=fraction,
                                      feasible_without_borrowing=feasible, maximum_feasible_fraction=feasible_fraction,
                                      final_wealth=wealth if feasible else None, max_drawdown=max_dd if feasible else None,
                                      bets=int(bets) if feasible else None))
                if feasible:
                    execution_paths.extend(dict(price_data=provenance, rule=name, fraction=fraction, fight_index=i, wealth=value)
                                           for i, value in enumerate(path))
    outputs = {'prospective_adaptive.csv': adaptive, 'prospective_frozen.csv': frozen,
               'prospective_frozen_rules.csv': frozen_rules, 'candidate_model_sensitivity.csv': model_results,
               'candidate_precard_execution.csv': execution}
    for filename, rows in outputs.items():
        pd.DataFrame(rows).to_csv(ROOT / 'results' / filename, index=False)
    for filename, rows in [('prospective_paths.csv.gz', adaptive_paths), ('candidate_model_paths.csv.gz', model_paths),
                           ('candidate_precard_paths.csv.gz', execution_paths)]:
        pd.DataFrame(rows).to_csv(ROOT / 'results' / filename, index=False, compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(histories).to_csv(ROOT / 'artifacts/private/prospective_rule_history.csv', index=False)
    forecasts.to_csv(ROOT / 'artifacts/private/prospective_forecasts.csv', index=False)
    summary = dict(model_initialization_counts=warmup_records,
                   adaptive_scenarios=len(adaptive), frozen_scenarios=len(frozen), model_scenarios=len(model_results),
                   execution_scenarios=len(execution),
                   notes=['Selections use only earlier outcomes and training-price midpoint boundaries.',
                          'Observed-only scenarios fit and trade only recovered asks; reconstructed scenarios retain later price-pair calibration.',
                          'Fixed-rule results condition on retrospective discovery; adaptive algorithms were specified during this investigation.',
                          'Before-card execution uses only prior cards and no within-card reinvestment; its equity ledger still resolves every fight.',
                          'Draws settle at 50 cents and canceled rows cannot create positions.'])
    (ROOT / 'results/prospective_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    print(pd.DataFrame(adaptive).query('fraction == 0.25').to_string(index=False), flush=True)
    print(pd.DataFrame(execution).query('fraction == 0.25').to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
