"""Investigate discovered categories, fading picks, and reconstruction uncertainty."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
from audit import ROOT
from analyze import N_SIM,SEED,cost,cluster_interval,posterior_offset,groups,price_tolerance
from kalshi import fit_mapping
from kelly import fit_firth,kelly_fraction,path_metrics,designs


def stats(name,g):
    low,high,se=cluster_interval(g)
    if g.y.nunique()<2:
        # An all-win/all-loss subset cannot estimate outcome variance empirically.
        # Do not present its narrow price-only sampling interval as reliability.
        low=high=np.nan
    out=dict(category=name,n=len(g),wins=int(g.y.sum()),cards=g.date.nunique(),
             profit=g.kalshi_return.sum(),roi=g.kalshi_return.mean(),
             cluster_ci_low=low,cluster_ci_high=high,api_n=int(g.price_source.eq('api_precard_ask').sum()),
             price_tolerance=price_tolerance(g))
    out.update(posterior_offset(g))
    return out


def run():
    full=pd.read_csv(ROOT/'data/cleaned/kalshi_fights.csv')
    d=full[full.y.notna()&full.kalshi_cost.notna()&full.status.ne('cancelled')].copy().reset_index(drop=True)
    no=d.copy();no['y']=1-no.y;no['kalshi_price']=no.kalshi_no_price;no['kalshi_cost']=no.kalshi_no_cost;no['kalshi_return']=no.kalshi_no_return
    # Prices defining confidence categories refer to the original pick, including
    # when the action is to buy No against it.
    fades=[]
    for name,mask in groups(d):
        fades.append(stats('fade '+name,no[mask]))
    pd.DataFrame(fades).to_csv(ROOT/'results/fade_categories.csv',index=False)

    # Same exhaustive category search for the opposite side, grouped on original
    # pick price. Family-wise correction also accounts for both searched sides.
    p=d.kalshi_price.to_numpy();conf=d.confidence.to_numpy();c=no.kalshi_cost.to_numpy();r=no.kalshi_return.to_numpy()
    rng=np.random.default_rng(SEED+2)
    null=(rng.random((N_SIM,len(d)))<c)/c-1
    nullmax=np.full(N_SIM,-np.inf);scan=[]
    for cl in sorted(set(conf)):
        for ch in sorted(set(conf)):
            if ch<cl:continue
            ix=np.flatnonzero((conf>=cl)&(conf<=ch));ix=ix[np.argsort(p[ix],kind='stable')]
            prices=p[ix];edges=np.r_[0,np.flatnonzero(np.diff(prices)>0)+1,len(ix)]
            obs=np.r_[0,np.cumsum(r[ix])];var=np.r_[0,np.cumsum((1-c[ix])/c[ix])];wins=np.r_[0,np.cumsum(no.y.to_numpy()[ix])]
            sims=np.column_stack([np.zeros(N_SIM),np.cumsum(null[:,ix],axis=1)])
            for hi in range(1,len(edges)):
                end=edges[hi];starts=edges[:hi];den=np.sqrt(var[end]-var[starts])
                zs=(sims[:,end,None]-sims[:,starts])/den[None,:]
                nullmax=np.maximum(nullmax,zs.max(axis=1))
                for start,z in zip(starts,(obs[end]-obs[starts])/den):
                    scan.append(dict(confidence_low=int(cl),confidence_high=int(ch),price_low=prices[start],price_high=prices[end-1],
                                     n=int(end-start),wins=int(wins[end]-wins[start]),profit=obs[end]-obs[start],roi=(obs[end]-obs[start])/(end-start),null_z=z))
    scan=pd.DataFrame(scan);mx=np.sort(nullmax)
    scan['search_adjusted_p']=(1+N_SIM-np.searchsorted(mx,scan.null_z,side='left'))/(N_SIM+1)
    scan['both_directions_adjusted_p']=np.minimum(1,2*scan.search_adjusted_p)
    scan.sort_values('null_z',ascending=False).to_csv(ROOT/'results/fade_interval_search.csv',index=False)
    yes=pd.read_csv(ROOT/'results/all_interval_search.csv');yes['both_directions_adjusted_p']=np.minimum(1,2*yes.search_adjusted_p)
    yes.to_csv(ROOT/'results/all_interval_search.csv',index=False)
    fades=pd.DataFrame(fades)
    fades['search_adjusted_p']=[(1+np.sum(nullmax>=np.sum(r[m])/np.sqrt(np.sum((1-c[m])/c[m]))))/(N_SIM+1) for _,m in groups(d)]
    fades['both_directions_adjusted_p']=np.minimum(1,2*fades.search_adjusted_p)
    fades.to_csv(ROOT/'results/fade_categories.csv',index=False)

    # Nominally strongest band, translated to whole-cent boundaries for reporting.
    # This translation is checked against the original selected membership.
    best=yes.iloc[0]
    low=np.floor(best.price_low*100)/100;high=np.ceil(best.price_high*100)/100
    exact=(d.confidence.between(best.confidence_low,best.confidence_high)&d.kalshi_price.between(best.price_low,best.price_high))
    roundmask=(d.confidence.between(best.confidence_low,best.confidence_high)&d.kalshi_price.between(low,high,inclusive='left'))
    assert np.array_equal(exact,roundmask)
    candidates={
        'C3-5 / 35-40 cents':roundmask,
        'C4-5 / all prices':d.confidence>=4,
        'C4-5 / underdog':(d.confidence>=4)&(d.kalshi_price<.5),
        'C4-5 / favorite':(d.confidence>=4)&(d.kalshi_price>.5),
        'C2 / underdog':(d.confidence==2)&(d.kalshi_price<.5),
    }
    details=[];temporal=[];sensitivity=[];seq=[]
    reconstruction=json.loads((ROOT/'results/kalshi_price_reconstruction.json').read_text())
    for name,mask in candidates.items():
        g=d[mask];details.append(stats(name,g))
        # Exact nominal all-win probability for the discovered 8/8 cell under
        # independent net-breakeven outcomes; not a selection-adjusted p value.
        if g.y.eq(1).all():details[-1]['all_win_null_probability']=float(np.prod(g.kalshi_cost))
        for label,h in [('spring_2025',g[g.date<'2025-07-19']),('summer_fall_2025',g[(g.date>='2025-07-19')&(g.date<'2026')]),('october_2026',g[g.date>='2026']),('observed_prices',g[g.price_source.eq('api_precard_ask')]),('imputed_prices',g[g.price_source.str.startswith('imputed')])]:
            temporal.append(dict(category=name,period=label,n=len(h),wins=h.y.sum(),profit=h.kalshi_return.sum(),roi=h.kalshi_return.mean()))
        for label,ret in [('remove_best_fight',g.drop(g.kalshi_return.idxmax()).kalshi_return),('exclude_opponent_imputations',g[g.price_source.ne('imputed_from_opponent')].kalshi_return)]:
            sensitivity.append(dict(category=name,scenario=label,n=len(ret),roi=ret.mean()))
        for date in g.date.unique():
            h=g[g.date!=date];sensitivity.append(dict(category=name,scenario='omit '+date,n=len(h),roi=h.kalshi_return.mean()))
        for q in ['0.025','0.975']:
            shift=reconstruction['residual_quantiles'][q]
            pp=d.kalshi_price.to_numpy().copy();imp=d.price_source.str.startswith('imputed').to_numpy();pp[imp]+=shift
            pp=np.clip(pp,np.nextafter(0.,1.),1.)
            if name.startswith('C3-5 / 35'):
                newmask=(d.confidence>=3)&(pp>=low)&(pp<high)
            elif 'underdog' in name:newmask=(d.confidence>=4 if name.startswith('C4') else d.confidence==2)&(pp<.5)
            elif 'favorite' in name:newmask=(d.confidence>=4)&(pp>.5)
            else:newmask=(d.confidence>=4)
            sensitivity.append(dict(category=name,scenario='reselect_after_price_error_'+q,n=int(newmask.sum()),roi=float(np.mean((d.y.to_numpy()/cost(pp)-1)[newmask]))))
        # Evaluate the practical C4-5 filter with the calibrated probability model
        # whose predictions were already generated using prior outcomes only.
        preds=pd.read_csv(ROOT/'artifacts/private/prequential_predictions.csv').set_index('row')
        gh=g[g.row.isin(preds.index[preds.odds_confidence.notna()])]
        f=kelly_fraction(preds.loc[gh.row,'odds_confidence'].to_numpy(),gh.kalshi_cost.to_numpy());r=gh.kalshi_return.to_numpy()
        objective=lambda scale:-np.log1p(scale*f*r).sum()
        op=minimize_scalar(objective,bounds=(0,1),method='bounded');opt=min([0.,op.x,1.],key=objective)
        for fraction in sorted(set([0.,.25,.5,1.,float(opt)])):
            seq.append(dict(category=name,fraction=fraction,hindsight_optimal_fraction=opt,**path_metrics(fraction*f,r)))
    pd.DataFrame(details).to_csv(ROOT/'results/candidate_details.csv',index=False)
    pd.DataFrame(temporal).to_csv(ROOT/'results/candidate_temporal.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(ROOT/'results/candidate_sensitivity.csv',index=False)
    pd.DataFrame(seq).to_csv(ROOT/'results/candidate_kelly.csv',index=False)

    # LOCO coefficient uncertainty and an identity-mapping sensitivity, with
    # observed asks held fixed. Predictions do not use fight outcomes.
    overlap=pd.read_csv(ROOT/'artifacts/private/price_mapping_overlap.csv')
    mapping=[]
    for label,training in [('all_pairs',overlap)]+[('omit '+date,overlap[overlap.date!=date]) for date in overlap.date.unique()]:
        a,b=fit_mapping(training.price.to_numpy(),training.ask.to_numpy(),'affine_logit')
        pp=d.kalshi_price.to_numpy().copy();mask=d.price_source.eq('imputed_from_sportsbook').to_numpy()
        pp[mask]=expit(a+b*logit(d.price.to_numpy()[mask]))
        ret=d.y.to_numpy()/cost(pp)-1
        mapping.append(dict(mapping=label,intercept=a,slope=b,overall_roi=ret.mean(),high_confidence_roi=ret[d.confidence>=4].mean()))
    pp=d.kalshi_price.to_numpy().copy();m=d.price_source.eq('imputed_from_sportsbook').to_numpy();pp[m]=d.price.to_numpy()[m]
    mapping.append(dict(mapping='identity',intercept=0,slope=1,overall_roi=np.mean(d.y.to_numpy()/cost(pp)-1),high_confidence_roi=np.mean((d.y.to_numpy()/cost(pp)-1)[d.confidence>=4])))
    pd.DataFrame(mapping).to_csv(ROOT/'results/price_mapping_sensitivity.csv',index=False)

    # How much bankroll performance is conditional on sequential reinvestment?
    # Compare fixed per-card bankroll sizing with identical prior-outcome forecasts.
    pred=pd.read_csv(ROOT/'artifacts/private/prequential_predictions.csv').dropna(subset=['odds_confidence'])
    # Freeze probabilities before the card begins for a truly executable
    # simultaneous-allocation sensitivity; earlier same-card results are unknown.
    design,offset=designs(d)['odds_confidence']
    frozen_before_card={}
    for date in pred.date.unique():
        tr=d.date.to_numpy()<date;te=d.date.to_numpy()==date
        coef=fit_firth(design[tr],d.y.to_numpy()[tr],offset[tr])
        frozen_before_card.update(zip(d.loc[te,'row'],expit(design[te]@coef+offset[te])))
    pred['odds_confidence']=pred.row.map(frozen_before_card)
    scenarios=[]
    for restrict in [False,True]:
        pp=pred[pred.confidence>=4] if restrict else pred
        for fraction in [.25,.5,1.]:
            wealth=peak=1.;dd=0.;insolvent=False
            for date,g in pp.groupby('date',sort=True):
                f=fraction*kelly_fraction(g.odds_confidence.to_numpy(),g.kalshi_cost.to_numpy())
                if np.sum(f)>1:insolvent=True;break
                # Starting-card bankroll allocated independently across its fights.
                wealth*=1+np.sum(f*g.kalshi_return.to_numpy());peak=max(peak,wealth);dd=max(dd,1-wealth/peak)
            scenarios.append(dict(high_confidence_only=restrict,fraction=fraction,feasible_without_borrowing=not insolvent,
                                  final_wealth=wealth if not insolvent else None,max_drawdown_at_card_boundaries=dd if not insolvent else None))
    pd.DataFrame(scenarios).to_csv(ROOT/'results/precard_allocation_sensitivity.csv',index=False)
    summary=dict(fade_search_intervals=len(scan),total_search_intervals=len(yes)+len(scan),
                 both_sides_best_adjusted_p=min(scan.both_directions_adjusted_p.min(),yes.both_directions_adjusted_p.min()),
                 best_back_band=dict(confidence_low=int(best.confidence_low),confidence_high=int(best.confidence_high),price_low=low,price_high_exclusive=high),
                 notes=['Both-side adjusted p uses Bonferroni across the back and fade search families.',
                        'The narrow category was discovered in the complete sample and has no independent holdout.',
                        'Fixed-category Kelly results condition on category selection after seeing the full sample.'])
    (ROOT/'results/robustness_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(pd.DataFrame(details).to_string(index=False),flush=True)
    print(fades[['category','n','roi','posterior_prob_positive','both_directions_adjusted_p']].to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
