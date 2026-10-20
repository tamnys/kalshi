"""Independent category, cash-flow and prior-only fit/selection reconciliation."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, log_expit, logsumexp
from audit import ROOT


def direct_choice(d, shape):
    """Enumerate direct membership masks, independently of cumulative sums."""
    p=d.kalshi_price.to_numpy(); unique=np.unique(p)
    edges=np.r_[0.,(unique[:-1]+unique[1:])/2,1.]
    a,b=np.triu_indices(len(edges),1)
    if shape=='one_cut':
        keep=(a==0)|(b==len(edges)-1);a,b=a[keep],b[keep]
    masks=(p[None,:]>=edges[a,None])&(p[None,:]<edges[b,None])
    counts=masks.sum(axis=1)
    contenders=[]
    for side in ('back','fade'):
        cost=d.kalshi_cost.to_numpy() if side=='back' else d.kalshi_no_cost.to_numpy()
        outcome=d.y.to_numpy() if side=='back' else 1-d.y.to_numpy()
        profits=np.sum(masks*(outcome/cost-1),axis=1)
        denominator=np.sqrt(np.sum(masks*((1-cost)/cost),axis=1))
        scores=profits/denominator
        tol=np.finfo(float).eps*len(d)*(1+abs(scores.max()))
        for j in np.flatnonzero(scores>=scores.max()-tol):
            contenders.append(dict(side=side,price_low=edges[a[j]],price_high=edges[b[j]],training_n=counts[j],training_statistic=scores[j]))
    best=max(r['training_statistic'] for r in contenders)
    if best<=0:return None
    tol=np.finfo(float).eps*len(d)*(1+abs(best))
    tied=[r for r in contenders if r['training_statistic']>=best-tol]
    return min(tied,key=lambda r:(-r['training_n'],-(r['price_high']-r['price_low']),r['side']!='back',r['price_low']))


def run():
    full=pd.read_csv(ROOT/'data/cleaned/kalshi_fights.csv',float_precision='round_trip').sort_values('sequence').reset_index(drop=True)
    candidates=pd.read_csv(ROOT/'results/confidence23_candidates.csv',float_precision='round_trip')
    for rule in candidates.itertuples():
        m=(full.confidence==rule.confidence)&(full.kalshi_price>=rule.price_low)&full.y.notna()
        m &= full.kalshi_price<=rule.price_high if rule.high_inclusive else full.kalshi_price<rule.price_high
        d=full[m]; outcome=d.y if rule.side=='back' else 1-d.y
        price=d.kalshi_price if rule.side=='back' else d.kalshi_no_price
        c=price+.07*price*(1-price)
        assert len(d)==rule.n and outcome.sum()==rule.wins
        assert np.isclose(np.mean(outcome/c-1),rule.roi)
        assert np.isclose(np.sum(outcome/c-1),rule.profit)
    saved=json.loads((ROOT/'artifacts/private/confidence23_policy_inputs.json').read_text())
    inputs={(r['id'],r['sample'],r['model']):r for r in saved}
    summaries=pd.read_csv(ROOT/'results/confidence23_kelly.csv')
    paths=pd.read_csv(ROOT/'results/confidence23_paths.csv.gz')
    keys=['id','sample','model','fraction','cap']
    grouped={k:g.sort_values('fight_index').wealth.to_numpy() for k,g in paths.groupby(keys,sort=False)}
    for r in summaries.itertuples():
        context=inputs[r.id,r.sample,r.model]
        side=np.array(context['side']);p=np.array(context['probability'],float)
        c=np.where(side==-1,full.kalshi_no_cost,full.kalshi_cost)
        payout=np.where(side==-1,1-full.payout,full.payout)
        fractions=np.zeros(len(full));available=(side!=0)&np.isfinite(p)
        fractions[available]=np.minimum(r.cap,r.fraction*np.maximum(0,(p[available]-c[available])/(1-c[available])))
        cash=1.;expected=[cash]
        for i in range(len(full)):
            outlay=cash*fractions[i]
            if outlay:
                quantity=outlay/c[i]
                cash-=outlay
                cash+=quantity*payout[i]
            expected.append(cash)
        expected=np.array(expected)
        assert np.allclose(expected,grouped[r.id,r.sample,r.model,r.fraction,r.cap])
        assert np.isclose(cash,r.final_wealth)
        assert np.isclose(np.max(1-expected/np.maximum.accumulate(expected)),r.max_drawdown)
        assert np.isclose(fractions.max(),r.max_stake)
    audit=pd.read_csv(ROOT/'artifacts/private/confidence23_fit_audit.csv')
    for row in audit.itertuples():
        i=row.fight_index
        tr=(np.arange(len(full))<i)&full.confidence.eq(row.confidence)&full.y.notna()
        if row.sample=='observed':tr &= full.price_source.eq('api_precard_ask')
        assert tr.sum()==row.training_n and np.flatnonzero(tr)[-1]==row.last_training_index<i
        x=logit(full.loc[tr,'kalshi_price'].to_numpy());y=full.loc[tr,'y'].to_numpy()
        design=np.ones((len(x),1)) if row.model=='level_offset' else np.column_stack([np.ones(len(x)),x])
        beta=np.array([row.beta0]) if row.model=='level_offset' else np.array([row.beta0,row.beta1])
        offset=x if row.model=='level_offset' else 0.
        q=expit(design@beta+offset);w=q*(1-q)
        info=design.T@(w[:,None]*design)
        h=w*np.einsum('ij,jk,ik->i',design,np.linalg.inv(info),design)
        score=design.T@(y-q+h*(.5-q))
        assert np.max(abs(score))<=1e-5  # Existing optimizer's documented tolerance.
        actual=inputs[f'C{row.confidence}_model_{row.model}',row.sample,row.model]
        price=full.kalshi_price.iloc[i]
        expected=expit(row.beta0+(logit(price) if row.model=='level_offset' else row.beta1*logit(price)))
        recorded=actual['probability'][i]
        if actual['side'][i]==-1:recorded=1-recorded
        assert np.isclose(expected,recorded)
    decisions=pd.read_csv(ROOT/'artifacts/private/confidence23_adaptive_decisions.csv')
    for r in decisions.itertuples():
        i=r.fight_index
        tr=(np.arange(len(full))<i)&full.confidence.eq(r.confidence)&full.y.notna()
        if r.sample=='observed':tr &= full.price_source.eq('api_precard_ask')
        assert np.flatnonzero(tr)[-1]==r.last_training_index<i
        expected=direct_choice(full[tr],r.shape)
        assert expected is not None and expected['side']==r.side
        for col in ['price_low','price_high','training_n','training_statistic']:assert np.isclose(expected[col],getattr(r,col)),(r,col,expected)
        if not r.eligible_now:continue
        selected=tr&(full.kalshi_price>=r.price_low)&(full.kalshi_price<r.price_high)
        price=full.kalshi_price if r.side=='back' else full.kalshi_no_price
        x=logit(price[selected].to_numpy());y=(full.y if r.side=='back' else 1-full.y)[selected].to_numpy()
        def negative_log_posterior(delta):
            eta=x+delta
            return -(np.sum(y*log_expit(eta)+(1-y)*log_expit(-eta))+.5*logsumexp(log_expit(eta)+log_expit(-eta)))
        fit=minimize_scalar(negative_log_posterior)
        assert fit.success
        actual=inputs[f'C{r.confidence}_adaptive_{r.shape}',r.sample,'category_offset']['probability'][i]
        assert np.isclose(expit(logit(price.iloc[i])+fit.x),actual,atol=1e-5)
    result=dict(status='PASS',candidate_memberships=len(candidates),independent_cash_paths=len(summaries),
                strict_prior_fit_scores=len(audit),independent_adaptive_selections=len(decisions))
    (ROOT/'results/confidence23_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
