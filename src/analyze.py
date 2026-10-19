"""Category inference, exhaustive interval search and execution sensitivity.

Simulation size follows a stated numerical criterion: worst-case Monte Carlo
standard error <= .005 for probabilities reported to two decimals, hence
ceil(.25/.005**2)=10,000. This is not a significance or bet-eligibility threshold.
"""
import json
import math
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/mma-matplotlib')
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit, log_expit, logsumexp
from scipy.stats import norm, t
import statsmodels.api as sm
from audit import ROOT

N_SIM = math.ceil(.25 / .005**2)
SEED = int(json.loads((ROOT/'results/source_inventory.json').read_text())['sha256'][:8], 16)
OUT = ROOT/'results'


def cost(p):
    return p+.07*p*(1-p)


def groups(d):
    cc=d.confidence.to_numpy(); pp=d.kalshi_price.to_numpy()
    conf=[('all',np.ones(len(d),bool))]
    conf += [(f'C{c}',cc==c) for c in sorted(set(cc))]
    conf += [(f'C{c}-5',cc>=c) for c in sorted(set(cc)) if 1<c<5]
    for name,cm in conf:
        for side,pm in [('all_prices',np.ones(len(d),bool)),('underdog',pp<.5),('favorite',pp>.5),('even',pp==.5)]:
            if (cm&pm).any():
                yield name+' / '+side, cm&pm


def cluster_interval(d):
    r=d.kalshi_return.to_numpy(); n=len(r); mu=r.mean()
    sums=d.assign(centered=r-mu).groupby('date').centered.sum().to_numpy()
    g=len(sums)
    if g<2:
        return np.nan,np.nan,np.nan
    se=np.sqrt(g/(g-1)*np.sum(sums*sums))/n
    z=t.ppf(.975,g-1)
    return mu-z*se,mu+z*se,se


def posterior_offset(d):
    """Jeffreys posterior for one common log-odds shift, varying base prices.

    This is a model-based posterior, NOT a search-adjusted confidence level.
    """
    x=logit(d.kalshi_price.to_numpy()); y=d.y.to_numpy(); c=d.kalshi_cost.to_numpy()
    def ld(delta):
        eta=x+delta
        return np.sum(y*log_expit(eta)+(1-y)*log_expit(-eta)) + .5*logsumexp(log_expit(eta)+log_expit(-eta))
    fit=minimize_scalar(lambda z:-ld(z))
    mode=fit.x; scale=ld(mode)
    dens=lambda z: np.exp(ld(z)-scale)
    total=quad(dens,-np.inf,np.inf)[0]
    exp_roi=lambda z:float(np.mean(expit(x+z)/c-1))
    lo,hi=-1.,1.
    while exp_roi(lo)>0:lo*=2
    while exp_roi(hi)<0:hi*=2
    zero=brentq(exp_roi,lo,hi)
    prob=quad(dens,zero,np.inf)[0]/total
    def quantile(q):
        cdf=lambda z:quad(dens,-np.inf,z)[0]/total-q
        lo,hi=-1.,1.
        while cdf(lo)>0:lo*=2
        while cdf(hi)<0:hi*=2
        return brentq(cdf,lo,hi)
    lower,median,upper=[quantile(q) for q in (.025,.5,.975)]
    return dict(posterior_prob_positive=prob,posterior_roi_low=exp_roi(lower),
                posterior_roi_median=exp_roi(median),posterior_roi_high=exp_roi(upper),
                posterior_logodds_median=median)


def price_tolerance(d):
    p=d.kalshi_price.to_numpy(); y=d.y.to_numpy()
    f=lambda shift:np.mean(y/cost(np.minimum(p+shift,1))-1)
    if f(0)<=0:return 0.
    return brentq(f,0,1-p.min())


def run():
    full=pd.read_csv(ROOT/'data/cleaned/kalshi_fights.csv')
    d=full[full.y.notna() & full.kalshi_cost.notna() & full.status.ne('cancelled')].copy().reset_index(drop=True)
    r=d.kalshi_return.to_numpy(); c=d.kalshi_cost.to_numpy(); p=d.kalshi_price.to_numpy()
    rng=np.random.default_rng(SEED)
    null=(rng.random((N_SIM,len(d)))<c)/c-1
    records=[]
    for name,mask in groups(d):
        g=d[mask]; low,high,se=cluster_interval(g); n=len(g)
        nullsum=null[:,mask].sum(axis=1)
        empirical_p=(1+np.sum(nullsum>=g.kalshi_return.sum()-1e-12))/(N_SIM+1)
        record=dict(category=name,n=n,wins=int(g.y.sum()),cards=g.date.nunique(),
                    win_rate=g.y.mean(),mean_price=g.kalshi_price.mean(),
                    mean_breakeven=g.kalshi_cost.mean(),profit=g.kalshi_return.sum(),
                    roi=g.kalshi_return.mean(),gross_roi=g.kalshi_gross_return.mean(),
                    cluster_ci_low=low,cluster_ci_high=high,cluster_se=se,
                    null_p=empirical_p,mc_se=np.sqrt(empirical_p*(1-empirical_p)/N_SIM),
                    api_n=int(g.price_source.eq('api_precard_ask').sum()),
                    price_tolerance=price_tolerance(g))
        record.update(posterior_offset(g))
        records.append(record)
    cats=pd.DataFrame(records)
    cats.to_csv(OUT/'categories.csv',index=False)
    print(cats[['category','n','wins','profit','roi','cluster_ci_low','cluster_ci_high','null_p','posterior_prob_positive']].to_string(index=False),flush=True)

    # Every contiguous confidence interval and every contiguous observed-price
    # interval, with equal-price ties inseparable. There is no minimum-N filter.
    # Keep all results and adjust for the maximum across this entire search.
    scan=[]; nullmax=np.full(N_SIM,-np.inf)
    conf=d.confidence.to_numpy()
    for cl in sorted(set(conf)):
        for ch in sorted(set(conf)):
            if ch<cl:continue
            ix=np.flatnonzero((conf>=cl)&(conf<=ch))
            ix=ix[np.argsort(p[ix],kind='stable')]
            prices=p[ix]
            edges=np.r_[0,np.flatnonzero(np.diff(prices)>0)+1,len(ix)]
            obs=np.r_[0,np.cumsum(r[ix])]
            var=np.r_[0,np.cumsum((1-c[ix])/c[ix])]
            wins=np.r_[0,np.cumsum(d.y.to_numpy()[ix])]
            simulated=np.column_stack([np.zeros(N_SIM),np.cumsum(null[:,ix],axis=1)])
            for hi in range(1,len(edges)):
                ends=edges[hi]; starts=edges[:hi]
                denom=np.sqrt(var[ends]-var[starts])
                z=(obs[ends]-obs[starts])/denom
                zs=(simulated[:,ends,None]-simulated[:,starts])/denom[None,:]
                nullmax=np.maximum(nullmax,zs.max(axis=1))
                for start,stat in zip(starts,z):
                    scan.append(dict(confidence_low=int(cl),confidence_high=int(ch),
                                     price_low=float(prices[start]),price_high=float(prices[ends-1]),
                                     n=int(ends-start),wins=int(wins[ends]-wins[start]),
                                     profit=float(obs[ends]-obs[start]),roi=float((obs[ends]-obs[start])/(ends-start)),
                                     null_z=float(stat)))
            print('SEARCH',int(cl),int(ch),'intervals',len(scan),flush=True)
    scanned=pd.DataFrame(scan)
    sortedmax=np.sort(nullmax)
    scanned['search_adjusted_p']=(1+N_SIM-np.searchsorted(sortedmax,scanned.null_z,side='left'))/(N_SIM+1)
    scanned.sort_values('null_z',ascending=False).to_csv(OUT/'all_interval_search.csv',index=False)
    cats['search_adjusted_p']=[(1+np.sum(nullmax>=np.sum(r[mask])/np.sqrt(np.sum((1-c[mask])/c[mask]))))/(N_SIM+1) for _,mask in groups(d)]
    cats.to_csv(OUT/'categories.csv',index=False)
    print('BEST SEARCH',scanned.sort_values('null_z',ascending=False).head().to_string(index=False),flush=True)

    # Influence and temporal stability for every declared broad category.
    sensitivity=[]
    reconstruction=json.loads((OUT/'kalshi_price_reconstruction.json').read_text())
    uncertainty=reconstruction['residual_quantiles']
    for name,mask in groups(d):
        g=d[mask]; gp=g.kalshi_price.to_numpy(); yy=g.y.to_numpy(); imp=g.price_source.str.startswith('imputed').to_numpy()
        for date,h in g.groupby('date'):
            other=g[g.date!=date]
            sensitivity.append(dict(category=name,scenario='leave_card_out',detail=date,n=len(other),roi=other.kalshi_return.mean()))
        for period,h in g.groupby(g.date.str[:4]):
            sensitivity.append(dict(category=name,scenario='calendar_year',detail=period,n=len(h),roi=h.kalshi_return.mean()))
        for source,h in g.groupby(g.price_source.eq('api_precard_ask')):
            sensitivity.append(dict(category=name,scenario='price_provenance',detail='observed' if source else 'imputed',n=len(h),roi=h.kalshi_return.mean()))
        # A coherent adverse/favorable price shift equal to empirical cross-card
        # error quantiles, applied only to reconstructed prices.
        for quantile in ('0.025','0.975'):
            pp=np.clip(gp+imp*uncertainty[quantile],np.nextafter(0.,1.),1.)
            sensitivity.append(dict(category=name,scenario='imputed_price_error',detail=quantile,n=len(g),roi=np.mean(yy/cost(pp)-1)))
        # The largest observed winner is omitted, not replaced with a loss.
        other=g.drop(g.kalshi_return.idxmax())
        sensitivity.append(dict(category=name,scenario='remove_best_fight',detail='',n=len(other),roi=other.kalshi_return.mean()))
    pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity.csv',index=False)
    d.groupby(['date','event']).agg(n=('y','size'),wins=('y','sum'),profit=('kalshi_return','sum'),roi=('kalshi_return','mean')).reset_index().to_csv(OUT/'card_results.csv',index=False)

    # Model the incremental information in confidence after controlling for price.
    xx=logit(p); cc=conf-conf.mean(); yy=d.y.to_numpy()
    designs={'odds_only':np.column_stack([np.ones(len(d)),xx]),
             'odds_confidence':np.column_stack([np.ones(len(d)),xx,cc]),
             'odds_confidence_interaction':np.column_stack([np.ones(len(d)),xx,cc,xx*cc])}
    models={}
    for name,x in designs.items():
        fit=sm.GLM(yy,x,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':d.date})
        models[name]=dict(parameters=fit.params.tolist(),cluster_standard_errors=fit.bse.tolist(),
                          cluster_pvalues=fit.pvalues.tolist(),log_likelihood=fit.llf,
                          aic=fit.aic,brier=float(np.mean((fit.predict()-yy)**2)),
                          coefficient_columns=['intercept','logit_price','centered_confidence','interaction'][:x.shape[1]])
    (OUT/'probability_models.json').write_text(json.dumps(models,indent=2)+'\n')
    # Raw-versus-corrected result impact is isolated from price reconstruction.
    draw=full[full.result.eq('draw')]
    summary=dict(binary_fights=len(d),cards=d.date.nunique(),wins=int(d.y.sum()),
                 profit=float(r.sum()),roi=float(r.mean()),gross_profit=float(d.kalshi_gross_return.sum()),
                 draw_n=len(draw),draw_net_profit=float(draw.kalshi_return.sum()),
                 draw_inclusive_roi=float(full.kalshi_return.dropna().mean()),
                 exhaustive_intervals=len(scanned),best_search_adjusted_p=float(scanned.search_adjusted_p.min()),
                 simulation_count=N_SIM,monte_carlo_max_standard_error=.005,seed=SEED,
                 all_in_cost_formula='p + 0.07*p*(1-p)',
                 limitations=['API asks establish quoted prices, not fill depth or execution.',
                              'Reconstructed prices are not observed prices; uncertainty stress is not a confidence interval.',
                              'Global search adjustment uses independent Bernoulli outcomes at net breakeven.',
                              'Card-cluster t intervals are approximate with only 15 cards.',
                              'Posterior probabilities use a common log-odds shift with Jeffreys prior and independent fights; they are not selection-adjusted.'])
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    plots=ROOT/'reports/figures';plots.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,5))
    for name,mask in groups(d):
        if name in ['all / all_prices','all / underdog','C4-5 / favorite','C2 / underdog','C3-5 / underdog']:
            series=np.where(mask,r,0)
            ax.plot(np.arange(1,len(d)+1),np.cumsum(series),label=name)
    ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Fight sequence',ylabel='Net profit, one unit at risk per qualifying fight',title='Historical Kalshi results, including estimated prices')
    ax.legend();fig.tight_layout();fig.savefig(plots/'cumulative_profit.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    matrix=np.full((5,2),np.nan)
    for i,confidence in enumerate(range(1,6)):
        for j,side in enumerate(['underdog','favorite']):
            row=cats[cats.category.eq(f'C{confidence} / {side}')]
            if len(row):
                rr=row.iloc[0];matrix[i,j]=rr.roi
                ax.text(j,i,f'{rr.roi:.1%}\n{int(rr.wins)}/{int(rr.n)} wins',ha='center',va='center',color='black')
    im=ax.imshow(matrix,cmap='RdYlGn',vmin=-np.nanmax(np.abs(matrix)),vmax=np.nanmax(np.abs(matrix)))
    ax.set(xticks=[0,1],xticklabels=['Price below 50¢','Price above 50¢'],yticks=range(5),yticklabels=[f'Confidence {i}' for i in range(1,6)],title='Net ROI after Kalshi fees; small cells are exploratory')
    fig.colorbar(im,ax=ax,label='Net return / capital at risk');fig.tight_layout();fig.savefig(plots/'confidence_price_roi.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    run()
