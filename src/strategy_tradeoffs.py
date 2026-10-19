"""Compare candidate portfolios at common fractional-Kelly illustration points."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
from audit import ROOT
from analyze import SEED,N_SIM,cost
from kelly import kelly_fraction,path_metrics


def run():
    d=pd.read_csv(ROOT/'artifacts/private/prequential_predictions.csv').dropna(subset=['odds_confidence'])
    dates=d.date.to_numpy();cards=np.unique(dates)
    rng=np.random.default_rng(SEED+3);sample=rng.integers(len(cards),size=(N_SIM,len(cards)))
    p=d.odds_confidence.to_numpy();f=kelly_fraction(p,d.kalshi_cost.to_numpy());r=d.kalshi_return.to_numpy()
    masks={'all_model_edges':np.ones(len(d),bool),'confidence_4_5':d.confidence>=4,
           'confidence_4_5_favorites':(d.confidence>=4)&(d.kalshi_price>.5),
           'confidence_4_5_underdogs':(d.confidence>=4)&(d.kalshi_price<.5)}
    out=[]
    for name,m in masks.items():
        stake=np.where(m,f,0)
        objective=lambda z:-np.log1p(z*stake*r).sum()
        opt=minimize_scalar(objective,bounds=(0,1),method='bounded');lam=min([0.,opt.x,1.],key=objective)
        for fraction in sorted(set([0.,.25,.5,1.,float(lam)])):
            histories=[np.log1p(fraction*stake[dates==date]*r[dates==date]) for date in cards]
            terminals=[];dds=[]
            for draw in sample:
                logs=np.concatenate([histories[i] for i in draw]);cw=np.r_[0.,np.cumsum(logs)]
                terminals.append(np.exp(cw[-1]));dds.append(np.max(1-np.exp(cw-np.maximum.accumulate(cw))))
            terminals=np.array(terminals);dds=np.array(dds)
            out.append(dict(portfolio=name,fraction=fraction,**path_metrics(fraction*stake,r),
                            bootstrap_median_wealth=float(np.median(terminals)),bootstrap_wealth_p025=float(np.quantile(terminals,.025)),
                            bootstrap_wealth_p975=float(np.quantile(terminals,.975)),bootstrap_probability_loss=float(np.mean(terminals<1)),
                            bootstrap_median_drawdown=float(np.median(dds)),bootstrap_drawdown_p95=float(np.quantile(dds,.95))))
    pd.DataFrame(out).to_csv(ROOT/'results/portfolio_tradeoffs.csv',index=False)
    coef=json.loads((ROOT/'results/kelly_summary.json').read_text())['final_parameters']['odds_confidence']
    examples=[]
    # Examples correspond to economically distinct regions already present in
    # the report (the mined band, near-even quotes, and favorite quotes).
    for price in [.35,.40,.50,.70,.85]:
        for confidence in range(1,6):
            prob=float(expit(coef[0]+coef[1]*logit(price)+coef[2]*(confidence-3)))
            c=cost(price);f=float(kelly_fraction(prob,c))
            examples.append(dict(confidence=confidence,price=price,all_in_cost=c,predicted_probability=prob,
                                 expected_roi=prob/c-1,full_kelly=f,quarter_kelly=f/4,half_kelly=f/2))
    pd.DataFrame(examples).to_csv(ROOT/'results/future_staking_examples.csv',index=False)
    print(pd.DataFrame(out).to_string(index=False),flush=True)


if __name__=='__main__':run()
