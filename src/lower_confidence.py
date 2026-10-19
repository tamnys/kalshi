"""Individual low-confidence intervals, opposing picks and fight-level sizing.

Original-pick price scans reconcile to the earlier exhaustive scan. Opposing
purchase-price scans are additional searches and enter the family correction.
"""
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta
from audit import ROOT
from analyze import cost, N_SIM, SEED
from robustness import stats
from kalshi import fit_mapping
from kelly import fit_firth, kelly_fraction, path_metrics
from prospective import probabilities

FRACTIONS = (.25, .5, 1.)
FAMILIES = [(c, 'back', 'pick') for c in (1, 2, 3)] + [(c, 'fade', axis) for c in (1, 2) for axis in ('pick', 'purchase')]


def side_frame(full, side):
    d = full.copy()
    d['pick_price'] = full.kalshi_price
    if side == 'fade':
        d['y'] = 1 - full.y
        d['payout'] = 1 - full.payout
        d['kalshi_price'] = full.kalshi_no_price
        d['kalshi_cost'] = full.kalshi_no_cost
        d['kalshi_return'] = full.kalshi_no_return
    return d


def interval_scan(d, axis, simulations=None):
    """All nonempty intervals, keeping equal prices inseparable."""
    column = 'pick_price' if axis == 'pick' else 'kalshi_price'
    order = np.argsort(d[column].to_numpy(), kind='stable')
    p, r, c, y = [d[k].to_numpy()[order] for k in (column, 'kalshi_return', 'kalshi_cost', 'y')]
    unique, first = np.unique(p, return_index=True)
    edges = np.r_[first, len(p)]
    midpoint = np.r_[0., (unique[:-1] + unique[1:]) / 2, 1.]
    a, b = np.triu_indices(len(edges), 1)
    start, end = edges[a], edges[b]
    profit = np.r_[0., np.cumsum(r)]
    variance = np.r_[0., np.cumsum((1-c)/c)]
    wins = np.r_[0., np.cumsum(y)]
    gain = profit[end] - profit[start]
    denom = np.sqrt(variance[end] - variance[start])
    scan = pd.DataFrame(dict(price_low=p[start], price_high=p[end-1],
                             deploy_low=midpoint[a], deploy_high=midpoint[b],
                             n=end-start, wins=(wins[end]-wins[start]).astype(int),
                             profit=gain, roi=gain/(end-start), null_z=gain/denom))
    nullmax = None
    if simulations is not None:
        cum = np.column_stack([np.zeros(len(simulations)), np.cumsum(simulations[:, order], axis=1)])
        nullmax = np.full(len(simulations), -np.inf)
        nominal = []
        # One interval at a time avoids materializing simulations x all intervals.
        for lo, hi, z, den in zip(start, end, scan.null_z, denom):
            null_z = (cum[:, hi]-cum[:, lo])/den
            nullmax = np.maximum(nullmax, null_z)
            nominal.append((1 + np.sum(null_z >= z - 1e-12))/(len(simulations)+1))
        scan['nominal_p'] = nominal
        scan['within_family_p'] = adjusted(scan.null_z, nullmax)
    return scan, nullmax


def adjusted(z, maxima):
    return (1 + len(maxima) - np.searchsorted(np.sort(maxima), np.asarray(z)-1e-12))/(len(maxima)+1)


def member(rule, full):
    p = full.kalshi_price.to_numpy() if rule['axis'] == 'pick' or rule['side'] == 'back' else full.kalshi_no_price.to_numpy()
    upper = p <= rule['price_high'] if rule.get('high_inclusive', True) else p < rule['price_high']
    return ((full.confidence.to_numpy() == rule['confidence']) & (p >= rule['price_low']) & upper & full.payout.notna().to_numpy())


def choose(training, axis, criterion):
    if training.empty:
        return None
    scan, _ = interval_scan(training, axis)
    key = 'roi' if criterion == 'roi' else 'null_z'
    best = scan[key].max()
    if best <= 0:
        return None
    tolerance = np.finfo(float).eps * len(training) * (1 + abs(best))
    ties = scan[scan[key] >= best-tolerance].copy()
    ties['width'] = ties.deploy_high-ties.deploy_low
    row = ties.sort_values(['n', 'width', 'deploy_low'], ascending=[False, False, True]).iloc[0]
    return dict(price_low=float(row.deploy_low), price_high=float(row.deploy_high), high_inclusive=False,
                training_n=int(row.n), training_roi=float(row.roi), training_statistic=float(row.null_z))


def simulate(full, d, selected, probability, fraction, cadence):
    f = np.nan_to_num(kelly_fraction(probability, d.kalshi_cost.to_numpy()), nan=0.) * selected * fraction
    returns = d.kalshi_return.fillna(0).to_numpy()
    exposures = [f[full.date.eq(date)].sum() for date in full.date.unique()]
    max_exposure = max(exposures)
    feasible = cadence == 'fight' or max_exposure <= 1 + np.finfo(float).eps * len(full)
    if not feasible:
        return dict(feasible=False, final_wealth=np.nan, max_drawdown=np.nan, bets=int((f>0).sum()), max_stake=float(f.max()), maximum_card_allocation=float(max_exposure)), None, f
    if cadence == 'fight':
        path = np.r_[1., np.cumprod(1+f*returns)]
    else:
        path = [1.]
        wealth = 1.
        for date in full.date.unique():
            starting = wealth
            for i in np.flatnonzero(full.date.eq(date).to_numpy()):
                wealth += starting*f[i]*returns[i]
                path.append(wealth)
        path = np.asarray(path)
    dd = 1-path/np.maximum.accumulate(path)
    return dict(feasible=True, final_wealth=float(path[-1]), max_drawdown=float(dd.max()),
                bets=int((f>0).sum()), max_stake=float(f.max()), maximum_card_allocation=float(max_exposure)), path, f


def local_predictions(full, d, selected, allowed, cadence):
    out = np.full(len(full), np.nan)
    for i in np.flatnonzero(selected & allowed):
        earlier = np.arange(len(full)) < i if cadence == 'fight' else full.date.to_numpy() < full.date.iloc[i]
        tr = selected & allowed & d.y.notna().to_numpy() & earlier
        if tr.any():
            coef = fit_firth(np.ones((tr.sum(), 1)), d.y.to_numpy()[tr], logit(d.kalshi_price.to_numpy()[tr]))
            out[i] = expit(logit(d.kalshi_price.iloc[i]) + coef[0])
    return out


def price_errors():
    pairs = pd.read_csv(ROOT/'artifacts/private/price_mapping_overlap.csv')
    mapping = json.loads((ROOT/'results/kalshi_price_reconstruction.json').read_text())
    predicted = np.full(len(pairs), np.nan)
    for date in pairs.date.unique():
        te = pairs.date.eq(date).to_numpy()
        coef = fit_mapping(pairs.price.to_numpy()[~te], pairs.bid.to_numpy()[~te], mapping['chosen_model'])
        predicted[te] = expit(coef[0]+coef[1]*logit(pairs.price.to_numpy()[te]))
    no_error = predicted-pairs.bid.to_numpy()  # true NO ask minus reconstructed NO ask
    out = dict(ask_error=mapping['residual_quantiles'],
               no_error={str(q): float(np.quantile(no_error, q)) for q in (.025, .5, .975)},
               no_loco_mae=float(np.mean(abs(no_error))), no_loco_rmse=float(np.sqrt(np.mean(no_error**2))),
               independent_pairs=len(pairs), cards=int(pairs.date.nunique()),
               note='NO errors use a separately cross-validated bid map. Ask and NO errors are stressed separately; these are sensitivity cases, not confidence bounds.')
    return out


def run():
    full = pd.read_csv(ROOT/'data/cleaned/kalshi_fights.csv').sort_values('sequence').reset_index(drop=True)
    valid = full.payout.notna().to_numpy()
    observed = valid & full.price_source.eq('api_precard_ask').to_numpy()
    binary = full.y.notna().to_numpy() & valid
    sides = {side: side_frame(full, side) for side in ('back', 'fade')}
    old = {side: pd.read_csv(ROOT/'results'/('all_interval_search.csv' if side == 'back' else 'fade_interval_search.csv'), float_precision='round_trip') for side in sides}
    all_scans, maxima, simulations = [], {}, {}
    candidates = []
    for c, side, axis in FAMILIES:
        family = f'{side}_C{c}_{axis}'
        ix = binary & full.confidence.eq(c).to_numpy()
        d = sides[side].loc[ix]
        rng = np.random.default_rng(SEED + 300 + 10*c + (side == 'fade'))
        null = (rng.random((N_SIM, len(d))) < d.kalshi_cost.to_numpy())/d.kalshi_cost.to_numpy()-1
        simulations[family] = null
        scan, nullmax = interval_scan(d, axis, null)
        scan['confidence'], scan['side'], scan['axis'], scan['family'] = c, side, axis, family
        scan['id'] = [family+'_'+str(i) for i in range(len(scan))]
        if axis == 'pick':
            previous = old[side].query('confidence_low == @c and confidence_high == @c').sort_values(['price_low', 'price_high']).reset_index(drop=True)
            current = scan.sort_values(['price_low', 'price_high']).reset_index(drop=True)
            for column in ('price_low', 'price_high', 'n', 'wins', 'roi', 'profit', 'null_z'):
                assert np.allclose(previous[column], current[column]), column
            lookup = dict(zip(zip(previous.price_low, previous.price_high), previous.search_adjusted_p))
            scan['legacy_side_search_p'] = [lookup[(r.price_low, r.price_high)] for r in scan.itertuples()]
            scan['study_adjusted_p'] = np.minimum(1, 3*scan.legacy_side_search_p)
        maxima[family] = nullmax
        all_scans.append(scan)
        positive = scan[scan.roi > 0]
        frontier = positive.sort_values(['n', 'roi'], ascending=[False, False]).drop_duplicates('n')
        frontier = frontier[frontier.roi > frontier.roi.cummax().shift(fill_value=-np.inf)]
        selected = set(frontier.id)
        selected.add(scan.sort_values(['null_z', 'n'], ascending=False).iloc[0].id)
        unbeaten = scan[scan.wins.eq(scan.n)]
        unbeaten_ids = set()
        if not unbeaten.empty:
            # The largest all-win sample can be dominated on ROI by a volatile
            # underdog interval of equal size, yet answers a different reliability
            # question. Retain all ties in longest undefeated interval length.
            unbeaten_ids = set(unbeaten[unbeaten.n.eq(unbeaten.n.max())].id)
            selected.update(unbeaten_ids)
        for record in scan[scan.id.isin(selected)].to_dict('records'):
            record['reason'] = ('longest all-win interval' if record['id'] in unbeaten_ids else
                                'ROI/sample-size frontier' if record['id'] in set(frontier.id) else 'highest standardized return')
            record['high_inclusive'] = True
            candidates.append(record)
        print(family, len(scan), 'intervals', len(selected), 'candidates', flush=True)
    scans = pd.concat(all_scans, ignore_index=True)
    purchase_max = np.maximum(maxima['fade_C1_purchase'], maxima['fade_C2_purchase'])
    is_purchase = scans.axis.eq('purchase')
    scans.loc[is_purchase, 'study_adjusted_p'] = np.minimum(1, 3*adjusted(scans.loc[is_purchase, 'null_z'], purchase_max))
    for record in candidates:
        record['study_adjusted_p'] = float(scans.loc[scans.id.eq(record['id']), 'study_adjusted_p'].iloc[0])
    # Broad comparators and interpretable whole-cent bands, including deliberately
    # wider versions of retrospectively selected cuts, expose boundary sensitivity.
    reference = [(c, side, 0., 1., 'all') for c, side in [(1,'back'), (2,'back'), (3,'back'), (1,'fade'), (2,'fade')]]
    reference += [(c, side, lo, hi, label) for c, side in [(1,'fade'), (2,'back'), (2,'fade'), (3,'back')]
                  for lo, hi, label in [(0., .5, 'under50'), (.5, 1., '50plus')]]
    reference += [(3,'back',.35,.4,'35-under40'), (3,'back',.76,.86,'76-under86'),
                  (3,'back',.77,.86,'77-under86'), (2,'back',.15,.37,'15-under37'),
                  (2,'back',.16,.37,'16-under37'), (1,'fade',.25,.71,'25-under71'),
                  (1,'fade',.5,.71,'50-under71'), (2,'fade',.36,.41,'36-under41')]
    for c, side, lo, hi, name in reference:
        candidates.append(dict(id=f'{side}_C{c}_{name}', confidence=c, side=side, axis='pick', family=f'{side}_C{c}_pick',
                               price_low=lo, price_high=hi, high_inclusive=False, reason='reference or boundary sensitivity'))
    # Distinct rules may select identical historical fights; keep their stated
    # bounds, so round-number examples do not masquerade as precise discoveries.
    errors = price_errors()
    details, sensitivity = [], []
    for rule in candidates:
        selected = member(rule, full) & binary
        d = sides[rule['side']].loc[selected]
        detail = {k:v for k,v in rule.items() if k not in ('n','wins','profit','roi')}
        detail.update(stats(rule['id'], d))
        n, w = len(d), int(d.y.sum())
        detail['win_rate'] = w/n
        detail['win_rate_exact_low'] = 0. if w == 0 else float(beta.ppf(.025, w, n-w+1))
        detail['win_rate_exact_high'] = 1. if w == n else float(beta.ppf(.975, w+1, n-w))
        detail['actual_price_low'], detail['actual_price_high'] = d.kalshi_price.min(), d.kalshi_price.max()
        detail['mean_cost'] = d.kalshi_cost.mean()
        family = rule['family']; eligible = binary & full.confidence.eq(rule['confidence']).to_numpy()
        localmask = selected[eligible]
        null = simulations[family]
        z = d.kalshi_return.sum()/np.sqrt(np.sum((1-d.kalshi_cost)/d.kalshi_cost))
        detail['null_z'] = z
        detail['nominal_p'] = float(np.prod(d.kalshi_cost)) if w == n else (1 + np.sum(null[:,localmask].sum(axis=1) >= d.kalshi_return.sum()-1e-12))/(N_SIM+1)
        detail['within_family_p'] = float(adjusted([z], maxima[family])[0])
        if rule['axis'] == 'pick':
            search = old[rule['side']]
            # Full-range and rounded rules have exactly a member-equivalent scan.
            match = search[(search.confidence_low==rule['confidence']) & (search.confidence_high==rule['confidence']) &
                           (search.n==n) & np.isclose(search.profit, d.kalshi_return.sum(), atol=1e-12)]
            assert not match.empty
            detail['study_adjusted_p'] = min(1., 3*match.search_adjusted_p.iloc[0])
        else:
            detail['study_adjusted_p'] = float(min(1., 3*adjusted([z], purchase_max)[0]))
        for label, mask in [('observed', observed), ('imputed', valid & ~observed),
                            ('before_July19_2025', full.date.lt('2025-07-19').to_numpy()),
                            ('Jul_to_Oct2025', full.date.ge('2025-07-19').to_numpy() & full.date.lt('2026').to_numpy()),
                            ('October2026', full.date.ge('2026').to_numpy()),
                            ('without_opponent_imputation', full.price_source.ne('imputed_from_opponent').to_numpy())]:
            h = sides[rule['side']].loc[selected & mask]
            sensitivity.append(dict(id=rule['id'], scenario=label, n=len(h), wins=h.y.sum(), roi=h.kalshi_return.mean()))
        for date in d.date.unique():
            h = d[d.date.ne(date)]
            sensitivity.append(dict(id=rule['id'], scenario='omit_card_'+date, n=len(h), wins=h.y.sum(), roi=h.kalshi_return.mean()))
        h = d.drop(d.kalshi_return.idxmax())
        sensitivity.append(dict(id=rule['id'], scenario='omit_largest_winner', n=len(h), wins=h.y.sum(), roi=h.kalshi_return.mean()))
        # Pricing uncertainty in the purchased side, without reselecting members.
        for q in ('0.025', '0.975'):
            error = errors['ask_error' if rule['side']=='back' else 'no_error'][q]
            p = d.kalshi_price.to_numpy() + d.price_source.ne('api_precard_ask').to_numpy()*error
            p = np.clip(p, np.nextafter(0.,1.), 1.)
            sensitivity.append(dict(id=rule['id'], scenario='fixed_members_price_error_'+q, n=len(d), wins=w, roi=np.mean(d.y.to_numpy()/cost(p)-1)))
        # Coherent YES-price movement changes membership; NO movement uses its own
        # adverse cost error. A scenario, not a joint-probability statement.
        for q in ('0.025', '0.975'):
            shifted = full.copy(); imputed = valid & ~observed
            shifted.loc[imputed,'kalshi_price'] = np.clip(shifted.loc[imputed,'kalshi_price'] + errors['ask_error'][q], np.nextafter(0.,1.), 1.)
            shifted.loc[imputed,'kalshi_no_price'] = np.clip(shifted.loc[imputed,'kalshi_no_price'] + errors['no_error']['0.975'], np.nextafter(0.,1.), 1.)
            m = member(rule, shifted) & binary
            price = shifted.kalshi_price.to_numpy() if rule['side']=='back' else shifted.kalshi_no_price.to_numpy()
            payout = sides[rule['side']].y.to_numpy()
            sensitivity.append(dict(id=rule['id'], scenario='reselect_pick_error_'+q+'_purchase_adverse', n=int(m.sum()), wins=float(payout[m].sum()), roi=float(np.mean(payout[m]/cost(price[m])-1)) if m.any() else np.nan))
        details.append(detail)
    pd.DataFrame(details).to_csv(ROOT/'results/lower_confidence_candidates.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(ROOT/'results/lower_confidence_sensitivity.csv',index=False)
    scans.to_csv(ROOT/'results/lower_confidence_intervals.csv', index=False)
    (ROOT/'results/lower_confidence_price_errors.json').write_text(json.dumps(errors, indent=2)+'\n')
    # Cached earlier-outcome forecasts were independently checked in prior stages.
    cached = pd.read_csv(ROOT/'artifacts/private/prospective_forecasts.csv')
    assert np.array_equal(cached.row, full.row)
    forecasts = {}; prediction_rows = full[['row','sequence','date']].to_dict('list')
    for sample, allowed in [('all',valid), ('observed',observed)]:
        oldprefix = 'reconstructed_and_observed' if sample=='all' else 'observed_only'
        for cadence in ('fight','card'):
            pooled = cached[oldprefix+'_odds_confidence' if cadence=='fight' else oldprefix+'_before_card'].to_numpy()
            forecasts[(sample,cadence,'odds_confidence')] = pooled
            if sample=='all' and cadence=='fight':
                cat = cached[oldprefix+'_confidence_categories'].to_numpy()
            else:
                preds,_ = probabilities(full,allowed,['confidence_categories'],cadence)
                cat = preds['confidence_categories']
            forecasts[(sample,cadence,'confidence_categories')] = cat
    summary_rows, path_rows, stake_rows = [], [], []
    for rule in candidates:
        selected = member(rule,full)
        d = sides[rule['side']]
        for sample, allowed in [('all',valid),('observed',observed)]:
            for cadence in ('fight','card'):
                for model in ('odds_confidence','confidence_categories','category_offset'):
                    if model=='category_offset':
                        probability = local_predictions(full,d,selected,allowed,cadence)
                    else:
                        probability = forecasts[(sample,cadence,model)]
                        if rule['side']=='fade':probability = 1-probability
                    predname = '|'.join([rule['id'],sample,cadence,model])
                    prediction_rows[predname] = probability
                    for fraction in FRACTIONS:
                        context = dict(id=rule['id'],sample=sample,cadence=cadence,model=model,fraction=fraction)
                        metrics,path,stake = simulate(full,d,selected&allowed,probability,fraction,cadence)
                        summary_rows.append(dict(**context,**metrics))
                        if path is not None:
                            path_rows.extend(dict(**context,fight_index=i,wealth=value) for i,value in enumerate(path))
                            # Raw per-fight stake data stay local; aggregate paths
                            # without fighter identities are already public outputs.
                            if fraction==.25:
                                stake_rows.extend(dict(**context,row=int(full.row.iloc[i]),stake=float(value)) for i,value in enumerate(stake))
    pd.DataFrame(summary_rows).to_csv(ROOT/'results/lower_confidence_kelly.csv',index=False)
    pd.DataFrame(path_rows).to_csv(ROOT/'results/lower_confidence_paths.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(prediction_rows).to_csv(ROOT/'artifacts/private/lower_confidence_forecasts.csv',index=False)
    pd.DataFrame(stake_rows).to_csv(ROOT/'artifacts/private/lower_confidence_stakes.csv',index=False)
    print('Kelly scenarios',len(summary_rows),flush=True)
    # Freeze selection using earlier cards at the existing natural time splits.
    frozen, frozen_rules = [], []
    for sample, allowed in [('all',valid),('observed',observed)]:
        for c,side,axis in FAMILIES:
            d = sides[side]
            for split in ('2025-07-19','2026-10-10'):
                train = allowed & binary & full.confidence.eq(c).to_numpy() & full.date.lt(split).to_numpy()
                test = allowed & full.date.ge(split).to_numpy()
                for criterion in ('roi','standardized'):
                    chosen = choose(d.loc[train],axis,criterion)
                    context = dict(sample=sample,confidence=c,side=side,axis=axis,split=split,criterion=criterion)
                    if chosen is None:
                        frozen_rules.append(dict(**context,training_n=0))
                        continue
                    rule = dict(confidence=c,side=side,axis=axis,**chosen)
                    frozen_rules.append(dict(**context,**chosen))
                    selected = member(rule,full)&test
                    p = forecasts[(sample,'card','odds_confidence')]
                    if side=='fade':p=1-p
                    for fraction in FRACTIONS:
                        metrics,_,_ = simulate(full,d,selected,p,fraction,'card')
                        settled = selected & binary
                        frozen.append(dict(**context,fraction=fraction,flat_n=int(settled.sum()),
                                           flat_wins=int(d.y.to_numpy()[settled].sum()),
                                           flat_roi=float(d.kalshi_return.to_numpy()[settled].mean()) if settled.any() else np.nan,**metrics))
    pd.DataFrame(frozen).to_csv(ROOT/'results/lower_confidence_frozen.csv',index=False)
    pd.DataFrame(frozen_rules).to_csv(ROOT/'results/lower_confidence_frozen_rules.csv',index=False)
    meta = dict(intervals=len(scans),positive_intervals=int((scans.roi>0).sum()),candidate_rules=len(candidates),
                kelly_scenarios=len(summary_rows),frozen_scenarios=len(frozen),source_rows=len(full),
                simulations=N_SIM, search_families=len(FAMILIES),
                notes=['Exact intervals are selected on the complete sample, with no minimum sample size.',
                       'Within-family correction covers all intervals at one confidence, direction and axis.',
                       'Study correction is Bonferroni across the complete prior backing search, complete prior fading search, and new NO-price search for confidence 1 and 2.',
                       'Independent net-breakeven null simulations are conditional on the quoted/reconstructed costs.',
                       'Jeffreys common-offset posteriors are conditional on model and category; they do not account for category discovery.',
                       'All Kelly paths include every workbook row. Card mode fixes probabilities and cash allocations before each card, resolving equity after each fight.',
                       'Observed-only scenarios fit and trade only observed prices. Cached pooled forecasts follow that restriction.',
                       'Scalable fees omit quantity rounding; current published formula and series multiplier checked. Actual fills and within-card settlement availability are unknown.'])
    (ROOT/'results/lower_confidence_summary.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':run()
