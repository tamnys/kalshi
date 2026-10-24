"""Reconcile old paths, independently account cash, and audit prior-only tuning."""
import csv
import gzip
import json
import math
from collections import defaultdict
from shrinkage_backtest import (ROOT, MODELS, PORTFOLIOS, FRACTIONS, CAPS,
                               FIRTH_SCORE_TOL, load_inputs, read_csv, position,
                               expit, logit, firth_state, shrink_forecasts)


def run():
    rows, saved, lower, legacy = load_inputs()
    learning_start = next(i for i, r in enumerate(rows) if r['row'] in legacy)
    scenarios = read_csv(ROOT/'results/shrinkage_scenarios.csv')
    # A conservative floating-operation error bound: <=10 arithmetic operations
    # per fight in each cash recurrence, each rounded to machine precision.
    error_bound = 10*len(rows)*math.ulp(1.)
    def close(a, b):
        assert abs(a-b) <= error_bound*max(1., abs(a), abs(b)), (a, b)
    old = {(r['portfolio'], r['sample'], r['policy'], r['cap_name']): r
           for r in read_csv(ROOT/'results/capped_kelly_scenarios.csv') if r['model'] == 'odds_confidence'}
    baselines = 0
    for r in scenarios:
        if r['model'] == 'original':
            base = old[(r['portfolio'], r['sample'], r['policy'], r['cap_name'])]
            for k in ('final_wealth', 'max_drawdown', 'max_stake', 'bets', 'capped_bets'):
                close(float(r[k]), float(base[k]))
            baselines += 1
    forecasts = read_csv(ROOT/'artifacts/private/shrinkage_forecasts.csv')
    grouped_forecasts = {s: [r for r in forecasts if r['sample'] == s] for s in ('all', 'observed')}
    keys = ('portfolio', 'sample', 'model', 'policy', 'cap_name')
    paths = defaultdict(list)
    with gzip.open(ROOT/'results/shrinkage_paths.csv.gz', 'rt', newline='') as f:
        for r in csv.DictReader(f):
            paths[tuple(r[k] for k in keys)].append(r)
    cash_checks = 0
    for r in scenarios:
        points = paths[tuple(r[k] for k in keys)]
        assert len(points) == len(rows)+1
        cash = peak = 1.
        mdd = 0.
        bets = 0
        for i, row in enumerate(rows):
            point = points[i+1]
            allowed = row['payout'] is not None and row['q'] is not None and (r['sample'] == 'all' or row['observed'])
            side = position(row, PORTFOLIOS[r['portfolio']], lower)
            ptext = grouped_forecasts[r['sample']][i][r['model']]
            outlay = 0.
            before = cash
            if allowed and side is not None and ptext:
                probability = float(ptext) if side == 'yes' else 1-float(ptext)
                cost = row['cost'] if side == 'yes' else row['no_cost']
                payout = row['payout'] if side == 'yes' else 1-row['payout']
                # Equivalent odds-based Kelly expression, independent of the
                # production probability-minus-cost expression.
                odds = (1-cost)/cost
                fraction = max(0., probability-(1-probability)/odds)*FRACTIONS[r['policy']]
                cap = CAPS[r['cap_name']]
                if cap is not None:
                    fraction = min(fraction, cap)
                outlay = cash*fraction
                contracts = outlay/cost
                cash = cash-outlay+contracts*payout
                bets += outlay > 0
            close(float(point['stake']), outlay/before)
            close(float(point['wealth']), cash)
            peak = max(peak, cash)
            mdd = max(mdd, 1-cash/peak)
            close(float(point['drawdown']), 1-cash/peak)
            cash_checks += 1
        close(float(r['final_wealth']), cash)
        close(float(r['max_drawdown']), mdd)
        assert bets == int(r['bets'])
    tuning_checks = prefix_checks = 0
    for sample, fs in grouped_forecasts.items():
        allowed = [r['payout'] is not None and r['q'] is not None and (sample == 'all' or r['observed']) for r in rows]
        for base, model, weightcol in [('original', 'shrink_original', 'retention_original'),
                                      ('confidence_offset', 'shrink_confidence', 'retention_confidence'),
                                      ('original', 'shrink_original_warm', 'retention_original_warm'),
                                      ('confidence_offset', 'shrink_confidence_warm', 'retention_confidence_warm')]:
            start = learning_start if model.endswith('_warm') else 0
            history = []
            for i, r in enumerate(fs):
                if not r[model]:
                    continue
                w, p = float(r[weightcol]), float(r[base])
                assert 0 <= w <= 1
                # Convex objective: boundary signs or zero derivative certify
                # the global optimum over the complete contraction interval.
                derivative = sum(delta*(expit(q+w*delta)-y) for q, delta, y in history)
                scale = math.ulp(1.)*max(1., sum(abs(delta) for _, delta, _ in history))*max(1, len(history))
                if w == 0:
                    assert derivative >= -scale
                elif w == 1:
                    assert derivative <= scale
                else:
                    assert abs(derivative) <= scale
                close(float(r[model]), expit(logit(rows[i]['q'])+w*(logit(p)-logit(rows[i]['q']))))
                assert int(r['prior_forecasts_warm' if start else 'prior_forecasts']) == len(history)
                if rows[i]['y'] is not None and i >= start:
                    history.append((logit(rows[i]['q']), logit(p)-logit(rows[i]['q']), rows[i]['y']))
                tuning_checks += 1
            # Mutate current and later outcomes at the two user-discussed cuts.
            # Forecasts through each current fight must be unchanged.
            base_probs = [float(r[base]) if r[base] else None for r in fs]
            for cut in (146, 161):
                changed = [dict(r, y=1-r['y']) if j >= cut and r['y'] is not None else r.copy()
                           for j, r in enumerate(rows)]
                pred, _ = shrink_forecasts(changed, base_probs, allowed, start)
                for j in range(cut+1):
                    if fs[j][model]:
                        close(pred[j], float(fs[j][model]))
                prefix_checks += 1
    fits = read_csv(ROOT/'artifacts/private/shrinkage_offset_fits.csv')
    score_max = 0.
    for f in fits:
        i = int(f['fight_index'])-1
        tr = [(r['q'], r['c']-3, r['y']) for r in rows[:i]
              if r['y'] is not None and r['payout'] is not None and r['q'] is not None
              and (f['sample'] == 'all' or r['observed'])]
        assert len(tr) == int(f['prior_n'])
        _, score, _ = firth_state((float(f['intercept']), float(f['confidence_slope'])), tr)
        score_max = max(score_max, *map(abs, score))
        assert max(map(abs, score)) <= FIRTH_SCORE_TOL
    result = dict(status='PASS', reconciled_existing_scenarios=baselines,
                  independent_cash_paths=len(scenarios), cash_steps=cash_checks,
                  convex_optimality_checks=tuning_checks, future_outcome_mutation_checks=prefix_checks,
                  offset_fit_score_checks=len(fits), maximum_firth_score=score_max,
                  arithmetic_relative_error_bound=error_bound,
                  note='No new category search or return-optimized shrinkage. Prefix mutation checks certify tuning inputs; base fits separately use strict earlier-row records.')
    (ROOT/'results/shrinkage_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
