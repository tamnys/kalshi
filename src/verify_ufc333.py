"""Independent cash reconciliation and earlier-outcome checks for UFC 333."""
from collections import defaultdict
import hashlib
import json
import math

from ufc333_review import (ROOT, PRIVATE, WORKBOOK, MODELS, PORTFOLIOS, FRACTIONS,
    CAPS, FIRTH_SCORE_TOL, load_inputs, read_csv, workbook_rows, original_state,
    fit_original, extend_forecasts, logit, expit, position)


def run():
    old, _, lower, legacy = load_inputs()
    learning_start = next(i for i, r in enumerate(old) if r['row'] in legacy)
    card = read_csv(PRIVATE/'fight_ledger.csv')
    for r in card:
        for key in ('c', 'row', 'sequence'):
            r[key] = int(r[key])
        for key in ('q', 'bid', 'no_ask', 'y', 'payout', 'cost', 'no_cost'):
            r[key] = float(r[key])
    summary = json.loads((ROOT/'results/ufc333_summary.json').read_text())
    assert hashlib.sha256(WORKBOOK.read_bytes()).hexdigest() == summary['workbook_sha256']
    forecasts = read_csv(PRIVATE/'forecasts.csv')
    fs = {(r['sample'], int(r['fight_index'])): r for r in forecasts}
    keys = ('sample', 'portfolio', 'model', 'policy', 'cap_name')
    paths = defaultdict(list)
    for r in read_csv(ROOT/'results/ufc333_paths.csv'):
        paths[tuple(r[k] for k in keys)].append(r)
    # Same error bound as the existing accounting audit: ten arithmetic
    # operations per fight, each rounded to double precision.
    bound = 10*len(card)*math.ulp(1.)
    def close(a, b):
        assert abs(a-b) <= bound*max(1., abs(a), abs(b)), (a, b)
    steps = 0
    for scenario in read_csv(ROOT/'results/ufc333_scenarios.csv'):
        path = paths[tuple(scenario[k] for k in keys)]
        assert len(path) == len(card)+1
        cash = peak = 1.
        mdd = maximum_stake = 0.
        bets = capped = 0
        for i, r in enumerate(card):
            side = position(r, PORTFOLIOS[scenario['portfolio']], lower)
            probability = float(fs[scenario['sample'], i+1][scenario['model']])
            stake = 0.
            if side:
                cost = r['cost'] if side == 'yes' else r['no_cost']
                win = r['y'] if side == 'yes' else 1-r['y']
                probability = probability if side == 'yes' else 1-probability
                odds = (1-cost)/cost
                nominal = FRACTIONS[scenario['policy']]*max(0., probability-(1-probability)/odds)
                cap = CAPS[scenario['cap_name']]
                stake = nominal if cap is None else min(nominal, cap)
                capped += stake < nominal
                outlay = cash*stake
                cash = cash-outlay+outlay/cost*win
            maximum_stake = max(maximum_stake, stake)
            bets += stake > 0
            peak = max(peak, cash)
            dd = 1-cash/peak
            mdd = max(mdd, dd)
            close(float(path[i+1]['wealth']), cash)
            close(float(path[i+1]['stake']), stake)
            close(float(path[i+1]['drawdown']), dd)
            steps += 1
        close(float(scenario['final_wealth']), cash)
        close(float(scenario['max_drawdown']), mdd)
        close(float(scenario['max_stake']), maximum_stake)
        assert int(scenario['bets']) == bets
        assert int(scenario['capped_bets']) == capped
    final_reference = json.loads((ROOT/'results/calibration_uncertainty_summary.json').read_text())['fitted_coefficients']
    training = [(r['q'], r['c']-3, r['y']) for r in old if r['q'] is not None and r['y'] is not None]
    ref_score = max(map(abs, original_state(final_reference, training)[1]))
    beta, score = fit_original(training)
    assert ref_score <= FIRTH_SCORE_TOL and score <= FIRTH_SCORE_TOL
    reference_difference = max(abs(expit(final_reference[0]+final_reference[1]*logit(r['q'])+final_reference[2]*(r['c']-3))-
                                  expit(beta[0]+beta[1]*logit(r['q'])+beta[2]*(r['c']-3))) for r in card)
    fits = json.loads((PRIVATE/'fits.json').read_text())
    assert max(max(r['original_score'], r['confidence_score']) for r in fits) <= FIRTH_SCORE_TOL
    past = read_csv(ROOT/'artifacts/private/shrinkage_forecasts.csv')
    mutation_checks = 0
    for sample in ('all', 'observed'):
        saved = [r for r in past if r['sample'] == sample]
        previous_n = sum(r['y'] is not None and r['q'] is not None and (sample == 'all' or r['observed']) for r in old)
        for i in range(len(card)):
            # Remove every later fight and flip the current result. Every
            # probability and shrinkage weight through this fight must agree.
            prefix = [dict(r) for r in card[:i+1]]
            prefix[-1]['y'] = 1-prefix[-1]['y']
            altered, detail = extend_forecasts(old, prefix, saved, sample, learning_start)
            assert detail[-1]['prior_n'] == previous_n+i
            for j, forecast in enumerate(altered):
                reference = fs[sample, j+1]
                for model in MODELS:
                    close(forecast[model], float(reference[model]))
                    if model.startswith('shrink_'):
                        close(forecast['retention_'+model], float(reference['retention_'+model]))
            mutation_checks += 1
    flat = read_csv(ROOT/'results/ufc333_flat_stakes.csv')[0]
    cash_profit = sum(r['y']/r['cost'] for r in card)-len(card)
    close(cash_profit, float(flat['profit_per_unit_each']))
    result = dict(scenarios=len(paths), settlements_reconciled=steps,
        current_and_future_outcome_checks=mutation_checks,
        fits_within_existing_score_tolerance=len(fits)*2,
        saved_full_history_reference_score=ref_score,
        maximum_probability_difference_from_saved_full_history_fit=reference_difference,
        workbook_preserved=True, passed=True)
    (ROOT/'results/ufc333_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
