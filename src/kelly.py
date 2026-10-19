"""Prior-data-only probability forecasts and fight-by-fight Kelly accounting."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import t
from audit import ROOT
from analyze import cost, SEED, N_SIM


def fit_firth(x,y,offset=None,start=None):
    """Jeffreys/Firth logistic fit: finite estimates under separation."""
    if offset is None:offset=np.zeros(len(y))
    if start is None:start=np.zeros(x.shape[1])
    def obj(beta):
        eta=x@beta+offset;p=expit(eta);w=p*(1-p)
        info=x.T@(w[:,None]*x)
        sign,ld=np.linalg.slogdet(info)
        if sign<=0:return np.inf,np.zeros(len(beta))
        leverage=w*np.sum(x*(x@np.linalg.inv(info)),axis=1)
        ll=np.sum(y*eta-np.logaddexp(0,eta))+.5*ld
        score=x.T@(y-p+leverage*(.5-p))
        return -ll,-score
    result=minimize(obj,start,jac=True,method='BFGS')
    # scipy BFGS's documented default gradient tolerance is 1e-5.
    if not result.success and np.max(np.abs(result.jac))>1e-5:
        raise RuntimeError(str(result.message)+': '+str(result.jac))
    return result.x


def kelly_fraction(probability,all_in_cost):
    return np.maximum(0,(probability-all_in_cost)/(1-all_in_cost))


def path_metrics(fractions,returns):
    factors=1+fractions*returns
    if np.any(factors<=0):
        return dict(final_wealth=0.,log_growth=-np.inf,max_drawdown=1.,bets=int(np.sum(fractions>0)))
    wealth=np.r_[1.,np.cumprod(factors)]
    dd=1-wealth/np.maximum.accumulate(wealth)
    return dict(final_wealth=float(wealth[-1]),log_growth=float(np.log(factors).sum()),
                max_drawdown=float(dd.max()),bets=int(np.sum(fractions>0)),
                mean_stake=float(np.mean(fractions[fractions>0])) if np.any(fractions>0) else 0.,
                max_stake=float(np.max(fractions)))


def designs(d):
    q=d.kalshi_price.to_numpy();x=logit(q);c=d.confidence.to_numpy()-3
    one=np.ones(len(d));zero=np.zeros(len(d))
    return {
        'market_offset':(one[:,None],x),
        'confidence_offset':(np.column_stack([one,c]),x),
        'odds_only':(np.column_stack([one,x]),zero),
        'odds_confidence':(np.column_stack([one,x,c]),zero),
        'odds_confidence_interaction':(np.column_stack([one,x,c,x*c]),zero),
        'confidence_categories':(np.column_stack([(d.confidence.to_numpy()==k).astype(float) for k in range(1,6)]+[x]),zero),
    }


def run():
    full=pd.read_csv(ROOT/'data/cleaned/kalshi_fights.csv')
    d=full[full.y.notna() & full.kalshi_cost.notna() & full.status.ne('cancelled')].copy().reset_index(drop=True)
    yy=d.y.to_numpy(); cc=d.kalshi_cost.to_numpy();rr=d.kalshi_return.to_numpy()
    des=designs(d)
    # Common initial training prefix is the first with full column rank for all
    # compared models. This is an identifiability requirement, not a chosen split.
    warm=next(i for i in range(1,len(d)) if all(np.linalg.matrix_rank(x[:i])==x.shape[1] for x,_ in des.values()))
    predictions=d[['row','date','sequence','confidence','kalshi_price','kalshi_cost','price_source','y','kalshi_return','kalshi_no_cost','kalshi_no_return']].copy()
    predictions['market_probability']=d.kalshi_price
    scores=[];strategies=[];fits={};draw_records=[]
    for name,(x,offset) in des.items():
        pred=np.full(len(d),np.nan);beta=None
        for i in range(warm,len(d)):
            beta=fit_firth(x[:i],yy[:i],offset[:i],beta)
            pred[i]=expit(x[i]@beta+offset[i])
        predictions[name]=pred
        final=fit_firth(x,yy,offset,beta);fits[name]=final.tolist()
        v=pred[warm:];y=yy[warm:]
        ll=-(y*np.log(v)+(1-y)*np.log1p(-v))
        market=d.kalshi_price.to_numpy()[warm:]
        baseline=-(y*np.log(market)+(1-y)*np.log1p(-market))
        loss_difference=ll-baseline
        gd=pd.DataFrame({'date':d.date.iloc[warm:].to_numpy(),'v':loss_difference})
        n=len(gd);gm=gd.v.mean();sums=gd.assign(centered=gd.v-gm).groupby('date').centered.sum()
        g=len(sums);se=np.sqrt(g/(g-1)*np.sum(sums*sums))/n
        scores.append(dict(model=name,n=len(v),brier=float(np.mean((v-y)**2)),log_loss=float(ll.mean()),
                           difference_from_market=float(gm),difference_ci_low=float(gm-t.ppf(.975,g-1)*se),
                           difference_ci_high=float(gm+t.ppf(.975,g-1)*se)))
        f=kelly_fraction(v,cc[warm:]);r=rr[warm:]
        fn=kelly_fraction(1-v,d.kalshi_no_cost.to_numpy()[warm:])
        rn=d.kalshi_no_return.to_numpy()[warm:]
        predictions[name+'_full_kelly']=np.r_[np.full(warm,np.nan),f]
        predictions[name+'_fade_kelly']=np.r_[np.full(warm,np.nan),fn]
        sequences=d.sequence.to_numpy()[warm:]
        for _,draw in full[full.result.eq('draw')].iterrows():
            tr=d.sequence.to_numpy()<draw.sequence
            if tr.sum()<warm:continue
            coef=fit_firth(x[tr],yy[tr],offset[tr])
            dx,do=designs(full[full.row.eq(draw.row)])[name]
            probability=float(expit(dx[0]@coef+do[0]))
            fd=float(kelly_fraction(probability,draw.kalshi_cost));fnd=float(kelly_fraction(1-probability,draw.kalshi_no_cost))
            pos=np.searchsorted(sequences,draw.sequence)
            sequences=np.insert(sequences,pos,draw.sequence)
            f=np.insert(f,pos,fd);fn=np.insert(fn,pos,fnd)
            r=np.insert(r,pos,draw.kalshi_return);rn=np.insert(rn,pos,draw.kalshi_no_return)
            draw_records.append(dict(model=name,forecast_win_probability=probability,back_full_kelly=fd,fade_full_kelly=fnd,back_return=draw.kalshi_return,fade_return=draw.kalshi_no_return))
        # At ordinary bid/ask spreads both positive edges cannot coexist; this
        # guards against doubling the same fight's bankroll exposure.
        assert not np.any((f>0)&(fn>0))
        for direction,stake,ret in [('back',f,r),('back_or_fade',f+fn,np.where(fn>0,rn,r))]:
            objective=lambda fraction:-np.log1p(fraction*stake*ret).sum()
            opt=minimize_scalar(objective,bounds=(0,1),method='bounded')
            lam=min([0.,float(opt.x),1.],key=objective)
            # Fractions below are conventional comparison points, not exposure
            # caps, defaults, recommendations, or optimized user risk budgets.
            for fraction in sorted(set([0.,.25,.5,1.,lam])):
                metrics=path_metrics(fraction*stake,ret)
                strategies.append(dict(model=name,direction=direction,fraction=fraction,
                                       hindsight_optimal_fraction=lam,**metrics))
        print('MODEL',name,'walk-forward',scores[-1],flush=True)
    market=d.kalshi_price.to_numpy()[warm:];y=yy[warm:]
    scores.append(dict(model='market_unadjusted',n=len(y),brier=float(np.mean((market-y)**2)),
                       log_loss=float(-np.mean(y*np.log(market)+(1-y)*np.log1p(-market)))))
    pd.DataFrame(scores).to_csv(ROOT/'results/prequential_scores.csv',index=False)
    pd.DataFrame(strategies).to_csv(ROOT/'results/kelly_historical.csv',index=False)
    predictions.to_csv(ROOT/'artifacts/private/prequential_predictions.csv',index=False)
    pd.DataFrame(draw_records).to_csv(ROOT/'results/draw_accounting.csv',index=False)

    # Test natural later time blocks with a model frozen on earlier observations.
    frozen=[]
    for split in ['2025-07-19','2026-10-10']:
        train=d.date.to_numpy()<split;test=~train
        for name,(x,offset) in des.items():
            beta=fit_firth(x[train],yy[train],offset[train]);p=expit(x[test]@beta+offset[test])
            f=kelly_fraction(p,cc[test]);r=rr[test]
            for fraction in [.25,.5,1.]:
                frozen.append(dict(split=split,model=name,fraction=fraction,**path_metrics(fraction*f,r)))
    pd.DataFrame(frozen).to_csv(ROOT/'results/kelly_frozen_validation.csv',index=False)

    # Bootstrap *whole cards* of prior-data-only forecasts/outcomes. This is an
    # empirical uncertainty illustration, not a fresh model-refitting experiment
    # or a guarantee about the next chosen number of fights.
    model='odds_confidence';p=predictions[model].to_numpy()[warm:]
    f=kelly_fraction(p,cc[warm:]);r=rr[warm:]
    dates=d.date.to_numpy()[warm:];unique=np.unique(dates);G=len(unique)
    rng=np.random.default_rng(SEED+1)
    sampled=rng.integers(G,size=(N_SIM,G))
    frontier=[]
    # Frontier resolution is one percentage point of the full-Kelly scale for
    # display; not a betting-size grid constraint. Optimization above is continuous.
    for fraction in np.linspace(0,1,101):
        factors=[np.log1p(fraction*f[dates==date]*r[dates==date]) for date in unique]
        terminal=[];drawdowns=[]
        for carddraw in sampled:
            logs=np.concatenate([factors[k] for k in carddraw]);cumulative=np.r_[0.,np.cumsum(logs)]
            terminal.append(np.exp(cumulative[-1]));drawdowns.append(np.max(1-np.exp(cumulative-np.maximum.accumulate(cumulative))))
        terminal=np.array(terminal);drawdowns=np.array(drawdowns)
        frontier.append(dict(fraction=fraction,median_wealth=float(np.median(terminal)),
                             wealth_p025=float(np.quantile(terminal,.025)),wealth_p975=float(np.quantile(terminal,.975)),
                             probability_loss=float(np.mean(terminal<1)),median_max_drawdown=float(np.median(drawdowns)),
                             drawdown_p95=float(np.quantile(drawdowns,.95))))
    pd.DataFrame(frontier).to_csv(ROOT/'results/kelly_frontier.csv',index=False)
    summary=dict(warmup_fights=warm,evaluated_fights=len(d)-warm,final_parameters=fits,
                 final_model='odds_confidence',
                 notes=['Comparative model family was specified during exploratory analysis, not externally preregistered.',
                        'Each forecast and stake uses earlier fight outcomes only; current/recent card outcomes are never read ahead.',
                        'Prices are pre-card snapshots; compounding assumes those prices remain obtainable when each later fight is bet.',
                        'Calibration of reconstructed historical prices uses later paired prices and is not strictly prospective.',
                        'No preferred risk tolerance was supplied; fractions are illustrated, not asserted to be universally optimal.',
                        'Hindsight-optimal fractions are descriptive, not independently validated recommendations.',
                        'Bootstrap risk uses the observed evaluation length in whole cards; it does not represent all future horizons.'])
    (ROOT/'results/kelly_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib.pyplot as plt
    front=pd.DataFrame(frontier)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(front.fraction,front.median_wealth);axes[0].fill_between(front.fraction,front.wealth_p025,front.wealth_p975,alpha=.2)
    axes[0].axhline(1,color='black',lw=.7);axes[0].set(xlabel='Fraction of estimated full Kelly',ylabel='Final bankroll / initial bankroll',title='Card-bootstrap median and 95% range')
    axes[1].plot(front.fraction,front.median_max_drawdown,label='Median');axes[1].plot(front.fraction,front.drawdown_p95,label='95th percentile')
    axes[1].set(xlabel='Fraction of estimated full Kelly',ylabel='Maximum drawdown',title='Same empirical resamples');axes[1].legend()
    fig.tight_layout();fig.savefig(ROOT/'reports/figures/kelly_frontier.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
