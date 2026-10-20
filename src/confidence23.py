"""Focused C2/C3 edge diagnostics and strictly earlier-fight learning.

No minimum sample or tuned probability cutoff is imposed. Two local model forms
are the one- and two-parameter Firth calibrations already used by this project.
Interval discovery is separate from subsequent sizing and forecast validation.
"""
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import t
from audit import ROOT
from analyze import N_SIM, SEED, cost
from robustness import stats
from kelly import fit_firth, kelly_fraction, path_metrics
from lower_confidence import side_frame, interval_scan, adjusted, member, local_predictions

OUT = ROOT / 'results'
PRIVATE = ROOT / 'artifacts/private'
FRACTIONS = (.25, .5, 1.)  # Existing requested comparisons.
CAPS = (1., .05, .10, .15, .20)  # 1 means uncapped; other caps requested by user.
SPLITS = ('2025-07-19', '2026-10-10')  # Existing natural study splits.


def read_full():
    return pd.read_csv(ROOT/'data/cleaned/kalshi_fights.csv', float_precision='round_trip').sort_values('sequence').reset_index(drop=True)


def interval_mean(values, dates):
    values = np.asarray(values)
    if not len(values):
        return np.nan, np.nan, np.nan
    mean = values.mean()
    sums = pd.DataFrame(dict(date=np.asarray(dates), v=values-mean)).groupby('date').v.sum()
    if len(sums) < 2:
        return mean, np.nan, np.nan
    se = np.sqrt(len(sums)/(len(sums)-1)*np.sum(sums**2))/len(values)
    radius = t.ppf(.975, len(sums)-1)*se
    return mean, mean-radius, mean+radius


def select_rule(training, shape):
    """Select both side and interval from prior outcomes by standardized profit.

    One-cut rules are lower/upper tails, including all prices. Full interval
    rules have two endpoints. New prices use training-price midpoint bounds.
    Ties prefer more observations, then wider price coverage, then backing.
    """
    choices = []
    for side in ('back', 'fade'):
        d = side_frame(training, side)
        scan, _ = interval_scan(d, 'pick')
        if shape == 'one_cut':
            scan = scan[(scan.deploy_low == 0) | (scan.deploy_high == 1)]
        if len(scan):
            maximum = scan.null_z.max()
            tol = np.finfo(float).eps * len(training) * (1+abs(maximum))
            tied = scan[scan.null_z >= maximum-tol].copy()
            tied['width'] = tied.deploy_high-tied.deploy_low
            best = tied.sort_values(['n','width','deploy_low'], ascending=[False,False,True]).iloc[0]
            choices.append(dict(side=side, price_low=float(best.deploy_low), price_high=float(best.deploy_high),
                                high_inclusive=False, training_n=int(best.n), training_statistic=float(best.null_z),
                                training_profit=float(best.profit)))
    if not choices or max(r['training_statistic'] for r in choices) <= 0:
        return None
    maximum = max(r['training_statistic'] for r in choices)
    tol = np.finfo(float).eps * len(training) * (1+abs(maximum))
    tied = [r for r in choices if r['training_statistic'] >= maximum-tol]
    return min(tied, key=lambda r: (-r['training_n'], -(r['price_high']-r['price_low']), r['side'] != 'back', r['price_low']))


def fixed_rules():
    rules = []
    def add(c, side, lo, hi, label, inclusive=False, origin='existing comparison'):
        rules.append(dict(id=f'C{c}_{side}_{label}', confidence=c, side=side, axis='pick',
                          price_low=lo, price_high=hi, high_inclusive=inclusive, origin=origin))
    for c in (2,3):
        for side in ('back','fade'):
            for lo,hi,label in [(0,1,'all'),(0,.5,'under50'),(.5,1,'50plus')]:
                add(c,side,lo,hi,label)
    add(2,'back',.16,.36,'16_to_36',True)
    add(2,'back',.15,.37,'15_under37')
    add(2,'back',0,.37,'under37',origin='remove lower cutoff from existing cheap-pick hypothesis')
    old = pd.read_csv(OUT/'lower_confidence_candidates.csv', float_precision='round_trip')
    m = old[(old.confidence==2)&(old.side=='fade')&(old.axis=='pick')&(old.n==18)&(old.wins==13)].iloc[0]
    add(2,'fade',float(m.price_low),float(m.price_high),'middle',True)
    for lo,hi,label in [(.35,.4,'35_under40'),(.77,.86,'77_under86'),(.76,.86,'76_under86')]:
        add(3,'back',lo,hi,label)
    add(3,'fade',.4,.5,'40_under50',origin='whole-cent expansion of existing seven-win 40.214–48.9021 cent interval')
    add(3,'fade',.68,.7,'68_under70',origin='whole-cent expansion of existing strongest C3 fade interval')
    add(3,'fade',.65,.7,'65_under70',origin='whole-cent expansion of existing 65.931–69.7042 cent fade interval')
    # Best single-cut rules in the original exhaustive searches, without new
    # arbitrary price grids or sample-size thresholds.
    full = read_full()
    for c in (2,3):
        d = full[(full.confidence==c)&full.y.notna()]
        for side in ('back','fade'):
            scan,_ = interval_scan(side_frame(d,side),'pick')
            tail = scan[(scan.deploy_low==0)|(scan.deploy_high==1)].sort_values(['null_z','n'],ascending=False).iloc[0]
            add(c,side,float(tail.deploy_low),float(tail.deploy_high),'best_one_cut',origin='full-history single-cut maximum; exploratory')
    return rules


def diagnostics(full, rules):
    binary = full.y.notna().to_numpy()
    legacy = {s:pd.read_csv(OUT/('all_interval_search.csv' if s=='back' else 'fade_interval_search.csv')) for s in ('back','fade')}
    simulations, maxima = {}, {}
    for c in (2,3):
        for side in ('back','fade'):
            d=side_frame(full[(full.confidence==c)&full.y.notna()],side)
            rng=np.random.default_rng(SEED+10*c+(side=='fade'))
            null=(rng.random((N_SIM,len(d)))<d.kalshi_cost.to_numpy())/d.kalshi_cost.to_numpy()-1
            _,mx=interval_scan(d,'pick',null)
            simulations[c,side],maxima[c,side]=null,mx
    rows, sensitivity, expansions = [], [], []
    errors=json.loads((OUT/'lower_confidence_price_errors.json').read_text())
    for rule in rules:
        selected=member(rule,full)&binary
        side=rule['side']; c=rule['confidence']; d=side_frame(full,side).loc[selected]
        r=dict(rule,**stats(rule['id'],d))
        z=d.kalshi_return.sum()/np.sqrt(np.sum((1-d.kalshi_cost)/d.kalshi_cost))
        family=full.confidence.eq(c).to_numpy()&binary
        null=simulations[c,side][:,selected[family]].sum(axis=1)
        r['nominal_p']=float(np.prod(d.kalshi_cost)) if d.y.eq(1).all() else float((1+np.sum(null>=d.kalshi_return.sum()-1e-12))/(N_SIM+1))
        r['within_level_side_p']=float(adjusted([z],maxima[c,side])[0])
        scan=legacy[side]
        match=scan[(scan.confidence_low==c)&(scan.confidence_high==c)&(scan.n==len(d))&np.isclose(scan.profit,d.kalshi_return.sum())]
        assert not match.empty
        # Keep the previous three-family study adjustment; do not reset scope
        # to C2/C3 just because this is a focused follow-up.
        r['study_adjusted_p']=min(1.,3*float(match.search_adjusted_p.iloc[0]))
        r['drop_best_roi']=float(d.drop(d.kalshi_return.idxmax()).kalshi_return.mean()) if len(d)>1 else np.nan
        omissions=[d[d.date!=date].kalshi_return.mean() for date in d.date.unique()]
        r['leave_card_min_roi']=float(np.nanmin(omissions)) if len(omissions)>1 else np.nan
        obs=d[d.price_source.eq('api_precard_ask')]
        r['observed_roi']=float(obs.kalshi_return.mean())
        r['observed_wins']=int(obs.y.sum())
        r['all_win_null_probability']=float(np.prod(d.kalshi_cost)) if d.y.eq(1).all() else np.nan
        rows.append(r)
        for name,m in [('spring_2025',full.date<SPLITS[0]),('later_2025',(full.date>=SPLITS[0])&(full.date<SPLITS[1])),('october_2026',full.date>=SPLITS[1])]:
            g=side_frame(full,side).loc[selected&m.to_numpy()]
            sensitivity.append(dict(id=rule['id'],scenario='period',detail=name,n=len(g),wins=g.y.sum(),roi=g.kalshi_return.mean()))
        for date in d.date.unique():
            g=d[d.date!=date]
            sensitivity.append(dict(id=rule['id'],scenario='omit_card',detail=date,n=len(g),wins=g.y.sum(),roi=g.kalshi_return.mean()))
        for sign in ('0.025','0.975'):
            shifted=full.copy(); imp=~full.price_source.eq('api_precard_ask')
            shifted['kalshi_price']=np.clip(full.kalshi_price+imp*errors['ask_error'][sign],np.nextafter(0.,1.),1.)
            shifted['kalshi_cost']=cost(shifted.kalshi_price)
            shifted['kalshi_no_price']=np.clip(full.kalshi_no_price+imp*errors['no_error']['0.975'],np.nextafter(0.,1.),1.)
            shifted['kalshi_no_cost']=cost(shifted.kalshi_no_price)
            m=member(rule,shifted)&binary
            cc=shifted.kalshi_cost if side=='back' else shifted.kalshi_no_cost
            yy=shifted.y if side=='back' else 1-shifted.y
            rr=(yy/cc-1)[m]
            sensitivity.append(dict(id=rule['id'],scenario='price_error_reselect',detail=sign,n=len(rr),wins=yy[m].sum(),roi=rr.mean()))
        # Change one boundary at a time to every outside observed price. These
        # are all nested expansions, not a manually chosen neighborhood width.
        eligible=full.loc[family]; prices=np.sort(eligible.kalshi_price.unique())
        for bound in ('lower','upper'):
            choices=prices[prices<rule['price_low']] if bound=='lower' else prices[prices>rule['price_high']]
            for p in choices:
                rr=dict(rule)
                if bound=='lower':rr['price_low']=float(p)
                else:rr.update(price_high=float(p),high_inclusive=True)
                g=side_frame(full,side).loc[member(rr,full)&binary]
                expansions.append(dict(id=rule['id'],changed_bound=bound,new_bound=p,n=len(g),wins=g.y.sum(),roi=g.kalshi_return.mean()))
    pd.DataFrame(rows).to_csv(OUT/'confidence23_candidates.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'confidence23_sensitivity.csv',index=False)
    pd.DataFrame(expansions).to_csv(OUT/'confidence23_boundary_expansions.csv',index=False)
    return pd.DataFrame(rows)


def model_predictions(full, c, allowed, model):
    p=full.kalshi_price.to_numpy(); binary=full.y.notna().to_numpy()
    target=allowed&full.confidence.eq(c).to_numpy()
    x=np.ones((len(full),1)) if model=='level_offset' else np.column_stack([np.ones(len(full)),logit(p)])
    offset=logit(p) if model=='level_offset' else np.zeros(len(full))
    pred=np.full(len(full),np.nan); records=[]
    for i in np.flatnonzero(target):
        tr=target&binary&(np.arange(len(full))<i)
        if not tr.any() or np.linalg.matrix_rank(x[tr])<x.shape[1]:continue
        beta=fit_firth(x[tr],full.y.to_numpy()[tr],offset[tr])
        pred[i]=expit(x[i]@beta+offset[i])
        records.append(dict(fight_index=int(i),confidence=c,model=model,training_n=int(tr.sum()),
                            last_training_index=int(np.flatnonzero(tr)[-1]),beta0=beta[0],beta1=beta[1] if len(beta)>1 else np.nan))
    tr=target&binary
    beta=fit_firth(x[tr],full.y.to_numpy()[tr],offset[tr])
    return pred,records,dict(confidence=c,model=model,n=int(tr.sum()),beta0=beta[0],beta1=beta[1] if len(beta)>1 else np.nan)


def choose_adaptive(full,c,allowed,shape):
    target=allowed&full.confidence.eq(c).to_numpy(); binary=full.y.notna().to_numpy()
    side=np.zeros(len(full),int); prob=np.full(len(full),np.nan); records=[]
    for i in np.flatnonzero(target):
        tr=target&binary&(np.arange(len(full))<i)
        if not tr.any():continue
        chosen=select_rule(full.loc[tr],shape)
        if chosen is None:continue
        rr=dict(chosen,confidence=c,axis='pick')
        selected=member(rr,full)
        record=dict(fight_index=int(i),confidence=c,shape=shape,last_training_index=int(np.flatnonzero(tr)[-1]),
                    eligible_now=bool(selected[i]),**chosen)
        records.append(record)
        if not selected[i]:continue
        d=side_frame(full,chosen['side']); train=tr&selected
        beta=fit_firth(np.ones((train.sum(),1)),d.y.to_numpy()[train],logit(d.kalshi_price.to_numpy()[train]))
        prob[i]=expit(logit(d.kalshi_price.iloc[i])+beta[0])
        side[i]=1 if chosen['side']=='back' else -1
    return side,prob,records


def summarize_policy(full, context, side, probability, save_paths, summaries, flat):
    c=np.where(side==-1,full.kalshi_no_cost,full.kalshi_cost)
    r=np.where(side==-1,full.kalshi_no_return,full.kalshi_return)
    eligible=(side!=0)&np.isfinite(probability)
    f=np.zeros(len(full)); f[eligible]=kelly_fraction(probability[eligible],c[eligible])
    active=f>0
    r=np.nan_to_num(r,nan=0.)
    mean,low,high=interval_mean(r[active],full.date[active])
    settled=np.where(side==-1,1-full.payout,full.payout)[active]
    decisive=settled[np.isin(settled,[0.,1.])]
    if len(np.unique(decisive))<2:low=high=np.nan
    flat.append(dict(**context,bets=int(active.sum()),yes_bets=int(np.sum(active&(side==1))),no_bets=int(np.sum(active&(side==-1))),
                     roi=mean,cluster_ci_low=low,cluster_ci_high=high,profit=float(r[active].sum())))
    for fraction in FRACTIONS:
        for cap in CAPS:
            stake=np.minimum(f*fraction,cap)
            wealth=np.r_[1.,np.cumprod(1+stake*r)]
            metrics=path_metrics(stake,r)
            key=dict(**context,fraction=fraction,cap=cap)
            summaries.append(dict(**key,**metrics))
            save_paths.extend(dict(**key,fight_index=i,wealth=float(w)) for i,w in enumerate(wealth))
    return dict(**context,side=side.tolist(),probability=[None if not np.isfinite(p) else float(p) for p in probability],
                full_kelly=f.tolist(),returns=r.tolist())


def run():
    full=read_full(); valid=full.payout.notna().to_numpy(); binary=full.y.notna().to_numpy()
    rules=fixed_rules()
    detail=diagnostics(full,rules)
    print('Fixed C2/C3 rules',len(rules),flush=True)
    cached=pd.read_csv(PRIVATE/'prospective_forecasts.csv');assert np.array_equal(cached.row,full.row)
    prediction_rows=[]; fit_rows=[]; decisions=[]; scores=[]; summary=[]; paths=[]; flat=[]; inputs=[]; adaptive_flat=[]
    for sample,allowed in [('all',valid),('observed',valid&full.price_source.eq('api_precard_ask').to_numpy())]:
        for c in (2,3):
            pred={}
            for model in ('level_offset','level_calibration'):
                pred[model],records,fit=model_predictions(full,c,allowed,model)
                prediction_rows.extend(dict(sample=sample,**r) for r in records);fit_rows.append(dict(sample=sample,**fit))
            prefix='reconstructed_and_observed' if sample=='all' else 'observed_only'
            pred['pooled']=cached[prefix+'_odds_confidence'].to_numpy()
            target=allowed&full.confidence.eq(c).to_numpy()
            common=target&binary
            for p in pred.values():common &= np.isfinite(p)
            y=full.y.to_numpy()[common]; q=full.kalshi_price.to_numpy()[common]
            baseline=-y*np.log(q)-(1-y)*np.log1p(-q)
            for model,p in pred.items():
                pp=p[common]; loss=-y*np.log(pp)-(1-y)*np.log1p(-pp)
                diff,lo,hi=interval_mean(loss-baseline,full.date[common])
                scores.append(dict(confidence=c,sample=sample,model=model,n=int(common.sum()),brier=np.mean((pp-y)**2),
                                   log_loss=loss.mean(),market_brier=np.mean((q-y)**2),market_log_loss=baseline.mean(),
                                   log_loss_difference=diff,difference_ci_low=lo,difference_ci_high=hi))
                # Trade either side only if the forecast clears its fee-inclusive
                # ask. A normal spread makes simultaneous positive edges impossible.
                back=target&np.isfinite(p)&(p>full.kalshi_cost.to_numpy())
                fade=target&np.isfinite(p)&((1-p)>full.kalshi_no_cost.to_numpy())
                assert not np.any(back&fade)
                side=np.where(back,1,np.where(fade,-1,0))
                inputs.append(summarize_policy(full,dict(id=f'C{c}_model_{model}',sample=sample,model=model),side,np.where(side==-1,1-p,p),paths,summary,flat))
            for rule in [r for r in rules if r['confidence']==c]:
                selected=member(rule,full)&allowed
                side=np.where(selected,1 if rule['side']=='back' else -1,0)
                for model in ('pooled','level_offset','level_calibration','category_offset'):
                    if model=='category_offset':
                        p=local_predictions(full,side_frame(full,rule['side']),selected,allowed,'fight')
                    else:
                        p=pred[model] if rule['side']=='back' else 1-pred[model]
                    inputs.append(summarize_policy(full,dict(id=rule['id'],sample=sample,model=model),side,p,paths,summary,flat))
            for shape in ('one_cut','interval'):
                side,p,records=choose_adaptive(full,c,allowed,shape)
                decisions.extend(dict(sample=sample,**r) for r in records)
                context=dict(id=f'C{c}_adaptive_{shape}',sample=sample,model='category_offset')
                inputs.append(summarize_policy(full,context,side,p,paths,summary,flat))
                # Direct test of category discovery: stake one unit whenever the
                # earlier-selected interval applies, independently of Kelly model.
                active=(side!=0)&binary
                ret=np.where(side==-1,full.kalshi_no_return,full.kalshi_return)
                roi,lo,hi=interval_mean(ret[active],full.date[active])
                outcomes=np.where(side==-1,1-full.y,full.y)[active]
                if len(np.unique(outcomes))<2:lo=hi=np.nan
                adaptive_flat.append(dict(**context,bets=int(active.sum()),wins=float(np.where(side==-1,1-full.y,full.y)[active].sum()),
                                          roi=roi,profit=float(ret[active].sum()),cluster_ci_low=lo,cluster_ci_high=hi))
            print('Sequential fits and paths',sample,c,flush=True)
    for name,records in [('model_fits',fit_rows),('scores',scores),('kelly',summary),('kelly_selected_flat',flat),('adaptive_flat',adaptive_flat)]:
        pd.DataFrame(records).to_csv(OUT/f'confidence23_{name}.csv',index=False)
    pd.DataFrame(paths).to_csv(OUT/'confidence23_paths.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(prediction_rows).to_csv(PRIVATE/'confidence23_fit_audit.csv',index=False)
    pd.DataFrame(decisions).to_csv(PRIVATE/'confidence23_adaptive_decisions.csv',index=False)
    (PRIVATE/'confidence23_policy_inputs.json').write_text(json.dumps(inputs,allow_nan=False)+'\n')
    meta=dict(rules=len(rules),scenarios=len(summary),path_points=len(paths),adaptive_decisions=len(decisions),
              local_forecasts=len(prediction_rows),simulations=N_SIM,source_fights=len(full),
              samples={f'C{c}':int(np.sum(binary&full.confidence.eq(c).to_numpy())) for c in (2,3)},
              notes=['Intervals remain retrospective. Historical broader search correction is retained.',
                     'Predictive model comparison uses a common identifiable prefix within each confidence/source context.',
                     'Adaptive selections use strictly earlier fights, but the learning procedure was designed during this investigation.',
                     'Each bet settles before the next. Historical pre-card quotes are reused; execution is not established.',
                     'Price stress magnitudes and 10,000 simulations follow prior measured reconstruction errors and study precision.',
                     'No new minimum N, tuned probability cutoff, stake limit, or user risk preference was introduced.'])
    (OUT/'confidence23_summary.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':run()
