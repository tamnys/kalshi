"""Reconcile new combinations to prior singles and independently fund contracts."""
import json
import numpy as np
import pandas as pd
from audit import ROOT
from aggressive_growth import legs,portfolios
from lower_confidence import side_frame


def run():
    full=pd.read_csv(ROOT/'data/cleaned/kalshi_fights.csv').sort_values('sequence').reset_index(drop=True)
    spec=legs(full);groups=portfolios();sides={s:side_frame(full,s) for s in ('back','fade')}
    results=pd.read_csv(ROOT/'results/aggressive_growth.csv')
    paths=pd.read_csv(ROOT/'results/aggressive_growth_paths.csv.gz')
    stakes=pd.read_csv(ROOT/'artifacts/private/aggressive_growth_full_stakes.csv')
    keys=['portfolio','sample','cadence','model','choice']
    saved={key:g.sort_values('fight_index').wealth.to_numpy() for key,g in paths.groupby(keys,sort=False)}
    lower=pd.read_csv(ROOT/'results/lower_confidence_kelly.csv')
    matches={'fade1_favorites':'fade_C1_50plus','fade1_all':'fade_C1_all','fade1_narrow':'fade_C1_pick_35',
             'back2_underdogs':'back_C2_pick_94','back3_favorites':'back_C3_77-under86','fade2_middle':'fade_C2_pick_411'}
    singles=0
    for key,oldkey in matches.items():
        current=results[(results.portfolio==key)&results.choice.ne('hindsight_best_feasible')]
        for row in current.itertuples():
            old=lower[(lower.id==oldkey)&lower['sample'].eq(row.sample)&lower.cadence.eq(row.cadence)&lower.model.eq(row.model)&lower.fraction.eq(row.fraction)].iloc[0]
            assert bool(old.feasible)==row.feasible
            if row.feasible:
                assert np.isclose(old.final_wealth,row.final_wealth)
                assert np.isclose(old.max_drawdown,row.max_drawdown)
            singles+=1
    oldcore=pd.read_csv(ROOT/'results/candidate_precard_execution.csv')
    for key,prior in [('core_broad','C4-5_35-plus'),('core_bounded','C4-5_35-81'),('back35_narrow','C3-5_35-under40')]:
        current=results[(results.portfolio==key)&results.cadence.eq('card')&results.model.eq('odds_confidence')&results.choice.ne('hindsight_best_feasible')]
        for row in current.itertuples():
            source='reconstructed_and_observed' if row.sample=='all' else 'observed_only'
            old=oldcore[(oldcore.rule==prior)&oldcore.price_data.eq(source)&oldcore.fraction.eq(row.fraction)].iloc[0]
            assert bool(old.feasible_without_borrowing)==row.feasible
            if row.feasible:assert np.isclose(old.final_wealth,row.final_wealth)
            singles+=1
    for row in results.itertuples():
        allowed=full.payout.notna().to_numpy()
        if row.sample=='observed':allowed&=full.price_source.eq('api_precard_ask').to_numpy()
        c=np.zeros(len(full));pay=np.zeros(len(full));used=np.zeros(len(full),bool)
        for leg in groups[row.portfolio]:
            side,mask,_=spec[leg];m=mask&allowed
            assert not (used&m).any();used|=m
            c[m]=sides[side].kalshi_cost.to_numpy()[m];pay[m]=sides[side].payout.to_numpy()[m]
        f=row.fraction*stakes['|'.join([row.portfolio,row.sample,row.cadence,row.model])].to_numpy()
        cash=1.;path=[cash]
        if not row.feasible:
            assert row.cadence=='card' and any(f[full.date.eq(date)].sum()>1 for date in full.date.unique());continue
        if row.cadence=='fight':
            for i in range(len(full)):
                outlay=cash*f[i]
                if outlay:cash-=outlay;cash+=outlay/c[i]*pay[i]
                path.append(cash)
        else:
            for date in full.date.unique():
                ix=np.flatnonzero(full.date.eq(date).to_numpy());outlay=cash*f[ix]
                quantity=np.divide(outlay,c[ix],out=np.zeros(len(ix)),where=outlay>0)
                unsettled=outlay.sum();cash-=unsettled
                assert cash>=-1e-11
                for j,i in enumerate(ix):
                    cash+=quantity[j]*pay[i];unsettled-=outlay[j];path.append(cash+unsettled)
        path=np.array(path);key=(row.portfolio,row.sample,row.cadence,row.model,row.choice)
        assert len(path)==188
        assert np.allclose(path,saved[key],rtol=1e-9,atol=1e-9)
        assert np.isclose(path[-1],row.final_wealth)
        assert np.isclose(np.max(1-path/np.maximum.accumulate(path)),row.max_drawdown)
        # Independently replace the affected multiplicative factor. A winning
        # payout changing from 1 to 0 subtracts allocation / cost from that factor.
        returns=np.divide(pay,c,out=np.ones(len(full)),where=c>0)-1
        alternatives=[]
        for i in np.flatnonzero((f>0)&(pay==1)):
            factor=(1+f[i]*returns[i]) if row.cadence=='fight' else 1+np.sum(f[full.date.eq(full.date.iloc[i])]*returns[full.date.eq(full.date.iloc[i])])
            alternatives.append(path[-1]*(factor-f[i]/c[i])/factor)
        if alternatives:
            assert np.isclose(min(alternatives),row.worst_single_win_reversal_wealth)
    # Log terminal wealth is concave in the common fraction on the cash-feasible
    # domain. Verify its endpoint/interior first-order conditions independently.
    opt=results[results.choice.eq('hindsight_best_feasible')]
    for row in opt.itertuples():
        f=stakes['|'.join([row.portfolio,row.sample,row.cadence,row.model])].to_numpy()
        returns=np.zeros(len(full))
        for leg in groups[row.portfolio]:
            side,mask,_=spec[leg];returns[mask]=sides[side].kalshi_return.fillna(0).to_numpy()[mask]
        inc=f*returns
        if row.cadence=='card':inc=np.array([inc[full.date.eq(date)].sum() for date in full.date.unique()])
        slope=np.sum(inc/(1+row.fraction*inc))
        roundoff=np.finfo(float).eps*len(inc)*(1+np.sum(np.abs(inc/(1+row.fraction*inc))))
        if row.fraction==0.:assert slope<=roundoff
        elif row.fraction==row.maximum_feasible_fraction:assert slope>=-roundoff
        else:
            # Convert SciPy bounded minimizer's default xatol=1e-5 into
            # a derivative tolerance using curvature over that interval.
            delta=1e-5
            endpoints=np.clip([row.fraction-delta,row.fraction+delta],0,row.maximum_feasible_fraction)
            curvature=max(np.sum(inc**2/(1+z*inc)**2) for z in endpoints)
            assert abs(slope)<=curvature*delta+roundoff
    print(json.dumps(dict(status='PASS',prior_single_scenarios_reconciled=singles,
                          cash_reconstructed_paths=len(saved),optimality_conditions=len(opt)),indent=2))


if __name__=='__main__':run()
