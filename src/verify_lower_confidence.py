"""Independent membership, side-price, cash ledger and estimation checks."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, log_expit, logsumexp
from audit import ROOT
from lower_confidence import FAMILIES, side_frame, member, choose


def run():
    full = pd.read_csv(ROOT/'data/cleaned/kalshi_fights.csv').sort_values('sequence').reset_index(drop=True)
    valid = full.payout.notna().to_numpy()
    binary = valid & full.y.notna().to_numpy()
    observed = valid & full.price_source.eq('api_precard_ask').to_numpy()
    assert np.allclose(full.loc[observed,'kalshi_no_price'],1-full.loc[observed,'bid'])
    assert np.all(full.loc[valid,'kalshi_no_price']+full.loc[valid,'kalshi_price'] >= 1-1e-12)
    assert np.allclose(full.loc[valid,'kalshi_no_return'],(1-full.loc[valid,'payout'])/full.loc[valid,'kalshi_no_cost']-1)
    scans = pd.read_csv(ROOT/'results/lower_confidence_intervals.csv',float_precision='round_trip')
    details = pd.read_csv(ROOT/'results/lower_confidence_candidates.csv',float_precision='round_trip')
    sides = {s:side_frame(full,s) for s in ('back','fade')}
    # Direct masks, independent of cumulative-sum search implementation.
    for rule in scans.to_dict('records'):
        selected = member(rule,full)&binary
        d = sides[rule['side']].loc[selected]
        assert len(d)==rule['n']
        assert d.y.sum()==rule['wins']
        assert np.isclose(d.kalshi_return.sum(),rule['profit'])
    for c,side,axis in FAMILIES:
        d=sides[side].loc[binary&full.confidence.eq(c).to_numpy()]
        unique=d['pick_price' if axis=='pick' else 'kalshi_price'].nunique()
        assert len(scans[(scans.confidence==c)&(scans.side==side)&(scans.axis==axis)])==unique*(unique+1)//2
    predictions = pd.read_csv(ROOT/'artifacts/private/lower_confidence_forecasts.csv')
    pv = predictions.drop(columns=['row','sequence','date']).to_numpy()
    assert np.all((pv[np.isfinite(pv)]>0)&(pv[np.isfinite(pv)]<1))
    summaries = pd.read_csv(ROOT/'results/lower_confidence_kelly.csv')
    paths = pd.read_csv(ROOT/'results/lower_confidence_paths.csv.gz')
    keys=['id','sample','cadence','model','fraction']
    indexed={key:g.sort_values('fight_index').wealth.to_numpy() for key,g in paths.groupby(keys,sort=False)}
    independently_fitted=0
    for rule in details.to_dict('records'):
        selected=member(rule,full); d=sides[rule['side']]
        for sample,allowed in [('all',valid),('observed',observed)]:
            for cadence in ('fight','card'):
                col='|'.join([rule['id'],sample,cadence,'category_offset'])
                # Independent one-dimensional Jeffreys likelihood optimization:
                # stable log probabilities, no BLAS matrix multiplication.
                for i in np.flatnonzero(selected&allowed):
                    earlier=np.arange(len(full))<i if cadence=='fight' else full.date.to_numpy()<full.date.iloc[i]
                    tr=selected&allowed&binary&earlier
                    if not tr.any():
                        assert np.isnan(predictions[col].iloc[i]);continue
                    x=logit(d.kalshi_price.to_numpy()[tr]); y=d.y.to_numpy()[tr]
                    def objective(delta):
                        eta=x+delta
                        return -(np.sum(y*log_expit(eta)+(1-y)*log_expit(-eta))+.5*logsumexp(log_expit(eta)+log_expit(-eta)))
                    fit=minimize_scalar(objective)
                    assert fit.success
                    expected=expit(logit(d.kalshi_price.iloc[i])+fit.x)
                    assert np.isclose(expected,predictions[col].iloc[i],atol=1e-5), (col,i,expected,predictions[col].iloc[i])
                    independently_fitted+=1
    # Reconstruct equity from cash plus unsettled contracts at purchase cost.
    # This is different from production's direct accumulation of profit factors.
    rules=details.set_index('id').to_dict('index')
    for row in summaries.itertuples(index=False):
        rule=rules[row.id];d=sides[rule['side']]
        allowed=valid if row.sample=='all' else observed
        sel=member(rule,full)&allowed
        p=predictions['|'.join([row.id,row.sample,row.cadence,row.model])].to_numpy()
        c=d.kalshi_cost.to_numpy();ret=d.kalshi_return.fillna(0).to_numpy()
        stake=np.zeros(len(full));available=sel&np.isfinite(p)
        stake[available]=row.fraction*np.maximum(0,(p[available]-c[available])/(1-c[available]))
        if not row.feasible:
            assert row.cadence=='card'
            assert any(stake[full.date.eq(date)].sum()>1 for date in full.date.unique())
            continue
        cash=1.;path=[cash]
        if row.cadence=='fight':
            for i in range(len(full)):
                outlay=cash*stake[i]
                quantity=outlay/c[i] if outlay else 0.
                cash-=outlay
                cash+=quantity*d.payout.iloc[i] if quantity else 0.
                path.append(cash)
        else:
            for date in full.date.unique():
                ix=np.flatnonzero(full.date.eq(date).to_numpy())
                outlay=cash*stake[ix]
                quantity=np.divide(outlay,c[ix],out=np.zeros(len(ix)),where=outlay>0)
                remaining=outlay.sum();cash-=remaining
                assert cash>=-1e-12
                for j,i in enumerate(ix):
                    if quantity[j]:cash+=quantity[j]*d.payout.iloc[i]
                    remaining-=outlay[j]
                    path.append(cash+remaining)
        path=np.asarray(path)
        key=(row.id,row.sample,row.cadence,row.model,row.fraction)
        assert len(indexed[key])==len(full)+1
        assert np.allclose(path,indexed[key],atol=1e-10)
        assert np.isclose(path[-1],row.final_wealth)
        assert np.isclose(np.max(1-path/np.maximum.accumulate(path)),row.max_drawdown)
    # Frozen selection is reconstructed from the stated earlier rows only.
    frozen=pd.read_csv(ROOT/'results/lower_confidence_frozen_rules.csv')
    for row in frozen.itertuples(index=False):
        allowed=valid if row.sample=='all' else observed
        tr=allowed&binary&full.confidence.eq(row.confidence).to_numpy()&full.date.lt(row.split).to_numpy()
        decision=choose(sides[row.side].loc[tr],row.axis,row.criterion)
        if decision is None:assert row.training_n==0;continue
        assert np.isclose(row.price_low,decision['price_low'])
        assert np.isclose(row.price_high,decision['price_high'])
        assert row.training_n==decision['training_n']
    print(json.dumps(dict(status='PASS',intervals=len(scans),candidates=len(details),
                          independently_refitted_local_forecasts=independently_fitted,
                          cash_reconstructed_paths=len(indexed),frozen_rules=len(frozen)),indent=2))


if __name__=='__main__':run()
