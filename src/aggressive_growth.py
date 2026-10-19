"""Retrospective combinations of discovered rules and aggressive Kelly sizing."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from audit import ROOT
from kelly import kelly_fraction
from lower_confidence import local_predictions, side_frame, simulate


def legs(full):
    c,p=full.confidence.to_numpy(),full.kalshi_price.to_numpy()
    # Reuse the exact discovered endpoint, rather than a rounded price label.
    lower=pd.read_csv(ROOT/'results/lower_confidence_candidates.csv').set_index('id').loc['fade_C2_pick_411','price_low']
    return {
        'core_broad':('back',(c>=4)&(p>=.35),'Back C4–5, ask ≥35¢'),
        'core_bounded':('back',(c>=4)&(p>=.35)&(p<=.81),'Back C4–5, 35–81¢'),
        'c4_bounded':('back',(c==4)&(p>=.35)&(p<=.81),'Back C4, 35–81¢'),
        'fade1_favorites':('fade',(c==1)&(p>=.5),'Fade C1 favorites'),
        'fade1_all':('fade',c==1,'Fade all C1'),
        'fade1_narrow':('fade',(c==1)&(p>=.25)&(p<=.7),'Fade C1, pick ask 25–70¢'),
        'back35_narrow':('back',(c>=3)&(p>=.35)&(p<.4),'Back C3–5, 35 to <40¢'),
        'back2_underdogs':('back',(c==2)&(p>=.16)&(p<=.36),'Back C2, 16–36¢'),
        'back3_favorites':('back',(c==3)&(p>=.77)&(p<.86),'Back C3, 77 to <86¢'),
        'fade2_middle':('fade',(c==2)&(p>=lower)&(p<=.62),'Fade C2 middle-price picks'),
    }


def portfolios():
    singles={key:[key] for key in ('core_broad','core_bounded','c4_bounded','fade1_favorites','fade1_all',
                                  'fade1_narrow','back35_narrow','back2_underdogs','back3_favorites','fade2_middle')}
    return {**singles,
        'broad_core_plus_fade1':['core_broad','fade1_favorites'],
        'bounded_core_plus_fade1':['core_bounded','fade1_favorites'],
        'bounded_core_plus_all_fade1':['core_bounded','fade1_all'],
        'bounded_core_plus_narrow_fade1':['core_bounded','fade1_narrow'],
        'core_fade1_plus_back2':['core_bounded','fade1_favorites','back2_underdogs'],
        'core_fade1_plus_back3':['core_bounded','fade1_favorites','back3_favorites'],
        'core_fade1_plus_fade2':['core_bounded','fade1_favorites','fade2_middle'],
        'all_disjoint_candidates':['core_bounded','fade1_favorites','back2_underdogs','back3_favorites','fade2_middle'],
        'two_unbeaten_intervals':['back35_narrow','fade1_narrow'],
        'narrow_back_plus_broad_fade':['back35_narrow','fade1_favorites'],
    }


def history(full,returns,stake,cadence):
    if cadence=='fight':return np.r_[1.,np.cumprod(1+stake*returns)]
    wealth=1.;path=[1.]
    for date in full.date.unique():
        initial=wealth
        for i in np.flatnonzero(full.date.eq(date).to_numpy()):
            wealth+=initial*stake[i]*returns[i]
            path.append(wealth)
    return np.array(path)


def run():
    full=pd.read_csv(ROOT/'data/cleaned/kalshi_fights.csv').sort_values('sequence').reset_index(drop=True)
    configs=legs(full)
    sideframes={s:side_frame(full,s) for s in ('back','fade')}
    forecasts=pd.read_csv(ROOT/'artifacts/private/prospective_forecasts.csv')
    lowforecasts=pd.read_csv(ROOT/'artifacts/private/lower_confidence_forecasts.csv')
    assert np.array_equal(forecasts.row,full.row) and np.array_equal(lowforecasts.row,full.row)
    valid=full.payout.notna().to_numpy();observed=valid&full.price_source.eq('api_precard_ask').to_numpy()
    summaries=[];paths=[];stakes={};predictions={}
    for sample,allowed in [('all',valid),('observed',observed)]:
        prefix='reconstructed_and_observed' if sample=='all' else 'observed_only'
        for cadence in ('fight','card'):
            p=forecasts[prefix+'_odds_confidence' if cadence=='fight' else prefix+'_before_card'].to_numpy()
            cat=lowforecasts['|'.join(['back_C1_all',sample,cadence,'confidence_categories'])].to_numpy()
            probs={}
            for key,(side,mask,_) in configs.items():
                for model in ('odds_confidence','confidence_categories','category_offset'):
                    if model=='category_offset':prob=local_predictions(full,sideframes[side],mask,allowed,cadence)
                    else:
                        prob=p if model=='odds_confidence' else cat
                        if side=='fade':prob=1-prob
                    probs[(key,model)]=prob
            for portfolio,keys in portfolios().items():
                union=np.zeros(len(full),bool);d=full.copy();d['kalshi_return']=0.;d['kalshi_cost']=np.nan;d['payout']=np.nan
                for key in keys:
                    side,mask,_=configs[key];mask=mask&allowed
                    assert not np.any(union&mask),(portfolio,key)
                    union|=mask
                    for col in ['kalshi_return','kalshi_cost','payout']:
                        d.loc[mask,col]=sideframes[side].loc[mask,col]
                for model in ('odds_confidence','confidence_categories','category_offset'):
                    probability=np.full(len(full),np.nan)
                    for key in keys:probability[configs[key][1]&allowed]=probs[(key,model)][configs[key][1]&allowed]
                    f=np.nan_to_num(kelly_fraction(probability,d.kalshi_cost.to_numpy()),nan=0.)*union
                    returns=d.kalshi_return.to_numpy()
                    max_exposure=max(f[full.date.eq(date)].sum() for date in full.date.unique())
                    feasible_max=min(1.,1/max_exposure) if cadence=='card' and max_exposure>0 else 1.
                    increments=f*returns if cadence=='fight' else np.array([np.sum(f[full.date.eq(date)]*returns[full.date.eq(date)]) for date in full.date.unique()])
                    def objective(fraction):
                        factors=1+fraction*increments
                        return -np.log(factors).sum() if np.all(factors>0) else np.inf
                    opt=minimize_scalar(objective,bounds=(0,feasible_max),method='bounded')
                    best=min([0.,float(opt.x),feasible_max],key=objective)
                    context=dict(portfolio=portfolio,sample=sample,cadence=cadence,model=model)
                    private_key='|'.join([portfolio,sample,cadence,model])
                    stakes[private_key]=f;predictions[private_key]=probability
                    for choice,fraction in [('quarter',.25),('half',.5),('full',1.),('hindsight_best_feasible',best)]:
                        metrics,path,stake=simulate(full,d,union,probability,fraction,cadence)
                        metrics['maximum_feasible_fraction']=feasible_max
                        metrics['worst_single_win_reversal_wealth']=np.nan
                        metrics['worst_single_win_reversal_drawdown']=np.nan
                        if path is not None:
                            # One executed winner changes to a loss. All future
                            # planned stake fractions remain fixed; no refitting.
                            alternatives=[]
                            for i in np.flatnonzero((stake>0)&d.payout.eq(1).to_numpy()):
                                changed=returns.copy();changed[i]=-1.
                                alt=history(full,changed,stake,cadence)
                                alternatives.append((float(alt[-1]),float(np.max(1-alt/np.maximum.accumulate(alt)))))
                            if alternatives:
                                metrics['worst_single_win_reversal_wealth']=min(x[0] for x in alternatives)
                                metrics['worst_single_win_reversal_drawdown']=max(x[1] for x in alternatives)
                            paths.extend(dict(**context,choice=choice,fraction=fraction,fight_index=i,wealth=float(value)) for i,value in enumerate(path))
                        summaries.append(dict(**context,choice=choice,fraction=fraction,**metrics))
    pd.DataFrame(summaries).to_csv(ROOT/'results/aggressive_growth.csv',index=False)
    pd.DataFrame(paths).to_csv(ROOT/'results/aggressive_growth_paths.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(stakes).to_csv(ROOT/'artifacts/private/aggressive_growth_full_stakes.csv',index=False)
    pd.DataFrame(predictions).to_csv(ROOT/'artifacts/private/aggressive_growth_probabilities.csv',index=False)
    metadata=dict(portfolios=portfolios(),leg_descriptions={key:text for key,(_,_,text) in configs.items()},
                  scenario_count=len(summaries),feasible_scenarios=sum(r['feasible'] for r in summaries),
                  notes=['Only already-discovered intervals and disjoint combinations are compared. This is not a global optimum over all strategies.',
                         'Every combination uses one bankroll and at most one side per fight. No overlapping positions are added.',
                         'The retrospective fraction optimum is restricted to fractional Kelly [0,1] and observed cash feasibility.',
                         'Card-mode probabilities and allocations are fixed before each card; equity updates after each fight.',
                         'Observed-only means observed-price training and trading, following earlier model stages.',
                         'Single-win reversal sensitivity fixes all planned stake fractions, so it does not simulate later forecast refitting.',
                         'All categories, combinations and retrospective fractions were considered after seeing the sample. No search-adjusted performance significance is asserted.'])
    (ROOT/'results/aggressive_growth_summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)
    out=pd.DataFrame(summaries)
    print(out[(out['sample']=='all')&(out.cadence=='card')&(out.model=='odds_confidence')&(out.choice=='half')].to_string(index=False),flush=True)
    print(out[(out['sample']=='all')&(out.portfolio=='two_unbeaten_intervals')].to_string(index=False),flush=True)


if __name__=='__main__':run()
