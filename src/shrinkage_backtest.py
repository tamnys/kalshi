"""Earlier-forecast log-score shrinkage, with sequential Kalshi accounting.

Standard-library-only so the saved comparison needs no additional packages.
The two base forecasts are the existing Firth odds+confidence model and the
existing market-offset+confidence model. A scalar retains 0 to 100% of each
model's log-odds departure from the quoted market baseline. Its value maximizes
the log score of STRICTLY EARLIER out-of-fit forecasts, not fitted probabilities
or betting returns. This is predictive shrinkage, not a Bayesian posterior.
"""
import csv
import gzip
import hashlib
import html
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('original', 'confidence_offset', 'shrink_original', 'shrink_confidence',
          'shrink_original_warm', 'shrink_confidence_warm')
LABELS = {'original': 'Original odds + confidence',
          'confidence_offset': 'Unshrunk confidence adjustment',
          'shrink_original': 'Shrunk original model',
          'shrink_confidence': 'Shrunk confidence adjustment',
          'shrink_original_warm': 'Shrunk original — established warmup',
          'shrink_confidence_warm': 'Shrunk confidence — established warmup',
          'market': 'Market quote benchmark'}
PORTFOLIOS = {
    'core_bounded': ['core'],
    'bounded_core_plus_fade1': ['core', 'fade1'],
    'bounded_core_plus_narrow_fade1': ['core', 'fade1_narrow'],
    'core_fade1_plus_fade2': ['core', 'fade1', 'fade2'],
    'all_disjoint_candidates': ['core', 'fade1', 'back2', 'back3', 'fade2'],
    'two_unbeaten_intervals': ['back35', 'fade1_narrow'],
}
FRACTIONS = {'quarter': .25, 'half': .5, 'full': 1.}
# Existing user-requested exposure caps, not new tuned parameters.
CAPS = {'uncapped': None, 'cap_05': .05, 'cap_10': .10, 'cap_15': .15, 'cap_20': .20}
SPLITS = ('2025-07-19', '2026-10-10')  # Existing natural-period comparisons.
FIRTH_SCORE_TOL = 1e-5  # Existing scipy BFGS default in src/kelly.py.
N_SIM = 10000  # Existing project MC precision convention: worst-case SE <= .005.


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def logit(p):
    return math.log(p) - math.log1p(-p)


def expit(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def softplus(z):
    return max(z, 0) + math.log1p(math.exp(-abs(z)))


def firth_state(beta, training):
    """Two-parameter market-offset Firth likelihood, score and information."""
    vals = []
    a = b = c = ll = 0.
    for q, z, y in training:
        eta = logit(q) + beta[0] + beta[1]*z
        p = expit(eta)
        w = p*(1-p)
        vals.append((z, y, p, w))
        a += w
        b += w*z
        c += w*z*z
        ll += y*eta-softplus(eta)
    det = a*c-b*b
    if det <= 0:
        return -math.inf, None, None
    s0 = s1 = 0.
    for z, y, p, w in vals:
        leverage = w*(c-2*b*z+a*z*z)/det
        residual = y-p+leverage*(.5-p)
        s0 += residual
        s1 += z*residual
    return ll+.5*math.log(det), (s0, s1), (a, b, c, det)


def firth_offset(training, start):
    """Fisher scoring with objective-monotone backtracking; no fit-count cap."""
    beta = start or (0., 0.)
    while True:
        value, score, info = firth_state(beta, training)
        if score is None:
            raise ArithmeticError('Singular Firth information')
        if max(map(abs, score)) <= FIRTH_SCORE_TOL:
            return beta, max(map(abs, score))
        a, b, c, det = info
        step = ((c*score[0]-b*score[1])/det,
                (a*score[1]-b*score[0])/det)
        scale = 1.
        while True:
            trial = tuple(beta[j]+scale*step[j] for j in (0, 1))
            if trial == beta:
                raise ArithmeticError('Firth iteration stalled before score tolerance')
            candidate, _, _ = firth_state(trial, training)
            if candidate >= value:
                beta = trial
                break
            scale /= 2


def fit_offset_forecasts(rows, allowed):
    training, beta, predictions, diagnostics = [], None, [], []
    for i, r in enumerate(rows):
        p = None
        if allowed[i] and len({v[1] for v in training}) >= 2:
            beta, score = firth_offset(training, beta)
            p = expit(logit(r['q'])+beta[0]+beta[1]*(r['c']-3))
            diagnostics.append(dict(fight_index=i+1, prior_n=len(training),
                                    intercept=beta[0], confidence_slope=beta[1],
                                    max_score=score))
        predictions.append(p)
        if allowed[i] and r['y'] is not None:
            training.append((r['q'], r['c']-3, r['y']))
    return predictions, diagnostics


def retention(history):
    """Minimize prior forecast log loss over the definition of shrinkage [0,1].

    A flat objective uses its market-centered endpoint. Bisection terminates
    when the floating-point midpoint cannot refine the bracket, not an
    arbitrary statistical or optimizer tuning threshold.
    """
    def derivative(weight):
        return math.fsum(delta*(expit(base+weight*delta)-y)
                         for base, delta, y in history)
    if not history or derivative(0.) >= 0:
        return 0.
    if derivative(1.) <= 0:
        return 1.
    lo, hi = 0., 1.
    while True:
        mid = (lo+hi)/2
        if mid == lo or mid == hi:
            return mid
        if derivative(mid) > 0:
            hi = mid
        else:
            lo = mid


def shrink_forecasts(rows, base_predictions, allowed, learning_start=0):
    history, predictions, records = [], [], []
    for i, (r, p) in enumerate(zip(rows, base_predictions)):
        if not allowed[i] or p is None:
            predictions.append(None)
            records.append(None)
            continue
        weight = retention(history)
        delta = logit(p)-logit(r['q'])
        predictions.append(expit(logit(r['q'])+weight*delta))
        records.append(dict(fight_index=i+1, prior_forecasts=len(history),
                            retention=weight, base_probability=p,
                            probability=predictions[-1]))
        if r['y'] is not None and i >= learning_start:
            history.append((logit(r['q']), delta, r['y']))
    return predictions, records


def position(row, legs, fade2_lower):
    c, p = row['c'], row['q']
    if p is None:
        return None
    options = {
        'core': ('yes', c >= 4 and .35 <= p <= .81),
        'fade1': ('no', c == 1 and p >= .5),
        'fade1_narrow': ('no', c == 1 and .25 <= p <= .7),
        'fade2': ('no', c == 2 and fade2_lower <= p <= .62),
        'back2': ('yes', c == 2 and .16 <= p <= .36),
        'back3': ('yes', c == 3 and .77 <= p < .86),
        'back35': ('yes', c >= 3 and .35 <= p < .4),
    }
    selected = [options[name][0] for name in legs if options[name][1]]
    assert len(selected) <= 1
    return selected[0] if selected else None


def simulate(rows, predictions, allowed, legs, lower, fraction, cap):
    wealth, peak, mdd, peak_i, trough_i = 1., 1., 0., 0, 0
    bets = capped = 0
    max_stake = 0.
    path = [dict(fight_index=0, wealth=1., drawdown=0., stake=0.)]
    for i, (r, p) in enumerate(zip(rows, predictions)):
        side = position(r, legs, lower)
        stake = 0.
        if allowed[i] and p is not None and side is not None:
            prob = p if side == 'yes' else 1-p
            cost = r['cost'] if side == 'yes' else r['no_cost']
            payout = r['payout'] if side == 'yes' else 1-r['payout']
            nominal = fraction*max(0., (prob-cost)/(1-cost))
            stake = nominal if cap is None else min(nominal, cap)
            capped += stake < nominal
            bets += stake > 0
            max_stake = max(max_stake, stake)
            wealth *= 1+stake*(payout/cost-1)
        if wealth > peak:
            peak, peak_i = wealth, i+1
        dd = 1-wealth/peak
        if dd > mdd:
            mdd, trough_i, mdd_peak = dd, i+1, peak_i
        path.append(dict(fight_index=i+1, wealth=wealth, drawdown=dd, stake=stake))
    return dict(final_wealth=wealth, max_drawdown=mdd, max_stake=max_stake,
                bets=bets, capped_bets=capped,
                peak_fight_index=mdd_peak if mdd else 0,
                trough_fight_index=trough_i), path


def loss(p, y):
    return -math.log(p if y else 1-p)


def quantile(values, p):
    v = sorted(values)
    x = p*(len(v)-1)
    lo = int(x)
    return v[lo]+(x-lo)*(v[min(lo+1, len(v)-1)]-v[lo])


def paired_interval(rows, indices, p, q, seed):
    """Existing 95% convention; whole-card paired bootstrap of saved losses."""
    dates = sorted({rows[i]['date'] for i in indices})
    blocks = []
    for day in dates:
        ix = [i for i in indices if rows[i]['date'] == day]
        blocks.append((sum(loss(q[i], rows[i]['y'])-loss(p[i], rows[i]['y']) for i in ix), len(ix)))
    rng = random.Random(seed)
    means = []
    for _ in range(N_SIM):
        sampled = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        means.append(sum(v[0] for v in sampled)/sum(v[1] for v in sampled))
    observed = sum(v[0] for v in blocks)/len(indices)
    return dict(improvement=observed, improvement_p025=quantile(means, .025),
                improvement_p975=quantile(means, .975), cards=len(dates))


def load_inputs():
    raw = sorted(read_csv(ROOT/'data/cleaned/kalshi_fights.csv'), key=lambda r: int(r['sequence']))
    rows = []
    for r in raw:
        f = lambda k: float(r[k]) if r[k] else None
        rows.append(dict(sequence=int(r['sequence']), row=int(r['row']), date=r['date'],
                         c=int(r['confidence']), q=f('kalshi_price'), y=f('y'),
                         payout=f('payout'), cost=f('kalshi_cost'), no_cost=f('kalshi_no_cost'),
                         observed=r['price_source'] == 'api_precard_ask'))
    saved = read_csv(ROOT/'artifacts/private/prospective_forecasts.csv')
    assert [int(r['row']) for r in saved] == [r['row'] for r in rows]
    lower = next(float(r['price_low']) for r in read_csv(ROOT/'results/lower_confidence_candidates.csv')
                 if r['id'] == 'fade_C2_pick_411')
    legacy = {int(r['row']) for r in read_csv(ROOT/'artifacts/private/prequential_predictions.csv')
              if r['odds_confidence']}
    return rows, saved, lower, legacy


def run():
    rows, saved, lower, legacy = load_inputs()
    # Existing 24-decisive-fight identifiability prefix from the original model
    # comparison, not a new tuned burn-in. Earlier rows still train base models.
    learning_start = next(i for i, r in enumerate(rows) if r['row'] in legacy)
    seed = int(json.loads((ROOT/'results/source_inventory.json').read_text())['sha256'][:8], 16)
    results = ROOT/'results'
    private = ROOT/'artifacts/private'
    results.mkdir(exist_ok=True, parents=True)
    private.mkdir(exist_ok=True, parents=True)
    scenarios, all_paths, scores, paired, diagnostics, forecast_records, focal, weights, periods = ([] for _ in range(9))
    all_contexts = {}
    crosscheck_errors = []
    for sample, prefix in [('all', 'reconstructed_and_observed'), ('observed', 'observed_only')]:
        allowed = [r['payout'] is not None and r['q'] is not None and (sample == 'all' or r['observed']) for r in rows]
        original = [float(r[prefix+'_odds_confidence']) if r[prefix+'_odds_confidence'] else None for r in saved]
        fitted, fit_details = fit_offset_forecasts(rows, allowed)
        for v in fit_details:
            diagnostics.append(dict(sample=sample, **v))
        if sample == 'all':
            offset = [float(r[prefix+'_confidence_offset']) if r[prefix+'_confidence_offset'] else None for r in saved]
            crosscheck_errors.extend(abs(a-b) for a, b in zip(offset, fitted) if a is not None and b is not None)
        else:
            offset = fitted
        # Identical prediction/trading start to original model; no advantage from
        # a shorter rank-identifiability warmup in the simpler model.
        offset = [p if original[i] is not None else None for i, p in enumerate(offset)]
        sp, wp = shrink_forecasts(rows, original, allowed)
        sc, wc = shrink_forecasts(rows, offset, allowed)
        spw, wpw = shrink_forecasts(rows, original, allowed, learning_start)
        scw, wcw = shrink_forecasts(rows, offset, allowed, learning_start)
        probs = dict(original=original, confidence_offset=offset,
                     shrink_original=sp, shrink_confidence=sc,
                     shrink_original_warm=spw, shrink_confidence_warm=scw,
                     market=[r['q'] for r in rows])
        all_contexts[sample] = (allowed, probs, wp, wc)
        for i, r in enumerate(rows):
            forecast_records.append(dict(sample=sample, fight_index=i+1, row=r['row'], date=r['date'],
                original=original[i], confidence_offset=offset[i], shrink_original=sp[i], shrink_confidence=sc[i],
                shrink_original_warm=spw[i], shrink_confidence_warm=scw[i],
                retention_original=wp[i]['retention'] if wp[i] else None,
                retention_confidence=wc[i]['retention'] if wc[i] else None,
                retention_original_warm=wpw[i]['retention'] if wpw[i] else None,
                retention_confidence_warm=wcw[i]['retention'] if wcw[i] else None,
                prior_forecasts=wc[i]['prior_forecasts'] if wc[i] else None,
                prior_forecasts_warm=wcw[i]['prior_forecasts'] if wcw[i] else None))
        for name, records in [('shrink_original', wp), ('shrink_confidence', wc),
                              ('shrink_original_warm', wpw), ('shrink_confidence_warm', wcw)]:
            valid_records = [v for v in records if v]
            weights.append(dict(sample=sample, model=name, forecasts=len(valid_records),
                mean_retention=sum(v['retention'] for v in valid_records)/len(valid_records),
                zero_retention=sum(v['retention'] == 0 for v in valid_records),
                full_retention=sum(v['retention'] == 1 for v in valid_records),
                last_retention=valid_records[-1]['retention']))
        common = [i for i, r in enumerate(rows) if allowed[i] and r['y'] is not None and all(probs[m][i] is not None for m in MODELS)]
        scopes = {'common_available': common}
        if sample == 'all':
            scopes['legacy_161'] = [i for i in common if rows[i]['row'] in legacy]
        for scope, indices in scopes.items():
            for model in (*MODELS, 'market'):
                p = probs[model]
                scores.append(dict(sample=sample, scope=scope, model=model, n=len(indices),
                    log_loss=sum(loss(p[i], rows[i]['y']) for i in indices)/len(indices),
                    brier=sum((p[i]-rows[i]['y'])**2 for i in indices)/len(indices)))
            for model, baseline in [('shrink_original', 'original'), ('shrink_confidence', 'confidence_offset'),
                                    ('shrink_confidence', 'original'), ('shrink_original', 'market'), ('shrink_confidence', 'market')]:
                paired.append(dict(sample=sample, scope=scope, model=model, baseline=baseline, n=len(indices),
                    **paired_interval(rows, indices, probs[model], probs[baseline], seed)))
            for model, baseline in [('shrink_original_warm', 'original'), ('shrink_confidence_warm', 'confidence_offset'),
                                    ('shrink_original_warm', 'market'), ('shrink_confidence_warm', 'market')]:
                paired.append(dict(sample=sample, scope=scope, model=model, baseline=baseline, n=len(indices),
                    **paired_interval(rows, indices, probs[model], probs[baseline], seed)))
        for portfolio, legs in PORTFOLIOS.items():
            for model in MODELS:
                for policy, fraction in FRACTIONS.items():
                    for cap_name, cap in CAPS.items():
                        metrics, path = simulate(rows, probs[model], allowed, legs, lower, fraction, cap)
                        context = dict(portfolio=portfolio, sample=sample, model=model, policy=policy, cap_name=cap_name)
                        scenarios.append(dict(**context, **metrics))
                        all_paths.extend(dict(**context, **p) for p in path)
                    # Existing later-period splits: probabilities and shrinkage
                    # continue learning sequentially; bankroll resets to one.
                    for split in SPLITS:
                        ix = [i for i, r in enumerate(rows) if r['date'] >= split]
                        metrics, _ = simulate([rows[i] for i in ix], [probs[model][i] for i in ix],
                                              [allowed[i] for i in ix], legs, lower, fraction, None)
                        periods.append(dict(portfolio=portfolio, sample=sample, model=model, policy=policy,
                                            start_date=split, **metrics))
        for i, side in [(146, 'no'), (161, 'yes')]:
            r = rows[i]
            for model in MODELS:
                p = probs[model][i]
                if p is None:
                    continue
                cost = r['cost'] if side == 'yes' else r['no_cost']
                side_p = p if side == 'yes' else 1-p
                records = {'shrink_original': wp, 'shrink_confidence': wc,
                           'shrink_original_warm': wpw, 'shrink_confidence_warm': wcw}.get(model)
                focal.append(dict(sample=sample, fight_index=i+1, confidence=r['c'], side=side, model=model,
                    side_probability=side_p, full_kelly=max(0., (side_p-cost)/(1-cost)),
                    retention=records[i]['retention'] if records else 1.,
                    prior_forecasts=records[i]['prior_forecasts'] if records else None))
        print('Computed', sample, 'forecasts and sequential paths', flush=True)
    # Paired comparisons hold categories, Kelly fraction, caps and inputs fixed.
    lookup = {(r['portfolio'], r['sample'], r['model'], r['policy'], r['cap_name']): r for r in scenarios}
    for r in scenarios:
        original = lookup[(r['portfolio'], r['sample'], 'original', r['policy'], r['cap_name'])]
        r['wealth_change_vs_original'] = r['final_wealth']-original['final_wealth']
        r['wealth_ratio_vs_original'] = r['final_wealth']/original['final_wealth']
        r['drawdown_reduction_vs_original'] = original['max_drawdown']-r['max_drawdown']
    for name, values in [('scenarios', scenarios), ('scores', scores), ('paired_scores', paired),
                         ('focal_stakes', focal), ('weights', weights), ('later_periods', periods)]:
        write_csv(results/('shrinkage_'+name+'.csv'), values)
    write_csv(private/'shrinkage_forecasts.csv', forecast_records)
    write_csv(private/'shrinkage_offset_fits.csv', diagnostics)
    # Deterministic compressed path output for reproducible repository history.
    import io
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(all_paths[0]))
    writer.writeheader()
    writer.writerows(all_paths)
    (results/'shrinkage_paths.csv.gz').write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))
    summary = dict(rows=len(rows), scenarios=len(scenarios), path_points=len(all_paths),
        models=list(MODELS), portfolios=list(PORTFOLIOS), fractions=FRACTIONS, caps=CAPS,
        method='p = logistic(logit(market ask) + retention * (logit(base forecast) - logit(market ask)))',
        selection='Retention maximizes log score of strictly earlier out-of-fit forecasts across all confidence levels; never current/future outcomes or bankroll returns.',
        domain='Retention [0,1] is the definition of contraction toward the market. No-history/flat objective selects its market-centered endpoint.',
        warmup='All model paths use the original odds+confidence identifiable forecast start in the corresponding price sample.',
        warmup_sensitivity_first_fight_index=learning_start+1,
        offset_crosscheck_max_probability_difference=max(crosscheck_errors),
        uncertainty='95% paired whole-card bootstrap of fixed prequential score differences, 10,000 replicates. Does not refit or account for strategy/model selection.',
        notes=['This is learned predictive shrinkage, not a fully Bayesian prior/posterior model.',
               'Each fight settles before the next stake; current bankroll includes all previous gains and losses.',
               'The base confidence model shrinks both its intercept and ordinal confidence adjustment. Shrinking the original model also shrinks its fitted price-slope deviation.',
               'A retained weight can increase again as earlier prediction performance changes; shrinkage is not a hard stake cap.',
               'Established-warmup sensitivity starts shrinkage learning after the original common 24-decisive-fight prefix; earlier rows remain in base-model training. Before learning begins, the market-centered endpoint is used.',
               'All six portfolio definitions were selected retrospectively in earlier research and are held fixed here.',
               'Quotes, fees, draw settlement and price reconstruction are unchanged. Historical pre-card asks are not verified executable sequential entry prices.',
               'The market ask is a probability benchmark, not known fair value. Pure market forecasts have no positive edge after modeled purchase costs.',
               'Earlier reconstructed prices retain retrospective price-map information; observed-only fitting/trading is a sensitivity, not a matched trade universe.',
               'No later-probability model or prior strength was chosen to avoid the two focal losses.',
               'Later-period tables reset wealth, but forecasts and retention keep learning after each settled fight; they are not new untouched holdouts.'])
    (results/'shrinkage_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    return rows, all_contexts, scenarios, all_paths


if __name__ == '__main__':
    run()
