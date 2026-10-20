"""Verify caps against current cash and reconcile uncapped sequential benchmarks."""
import json
import numpy as np
import pandas as pd
from audit import ROOT
from sequential_kelly import context_inputs, FIXED
from capped_kelly import LEADING, CAPS


def run():
    full, contexts = context_inputs()
    cases = pd.read_csv(ROOT/'results/capped_kelly_scenarios.csv')
    paths = pd.read_csv(ROOT/'results/capped_kelly_paths.csv.gz')
    old = pd.read_csv(ROOT/'results/sequential_kelly_scenarios.csv').set_index(['portfolio','sample','model','policy'])
    grouped = {key:g.sort_values('fight_index') for key,g in paths.groupby(['portfolio','sample','model','policy','cap_name'], sort=False)}
    lookup = cases.set_index(['portfolio','sample','model','policy','cap_name'])
    checked = comparisons = baselines = limit_checks = 0
    for context, f, returns, costs, payouts in contexts:
        if context['portfolio'] not in LEADING:
            continue
        key = tuple(context.values())
        for policy, multiplier in FIXED.items():
            baseline = old.loc[(*key, policy)]
            for cap_name, cap in CAPS.items():
                saved = grouped[(*key, policy, cap_name)]
                case = lookup.loc[(*key, policy, cap_name)]
                assert len(saved) == len(full)+1
                assert saved.fight_index.tolist() == list(range(len(full)+1))
                cash, ledger, stakes = 1., [1.], []
                cap_count = 0
                for i in range(len(full)):
                    proposed_outlay = cash*multiplier*f[i]
                    outlay = proposed_outlay if cap is None else min(proposed_outlay, cap*cash)
                    stake = outlay/cash
                    if cap is not None:
                        assert outlay <= cap*cash
                        limit_checks += 1
                        cap_count += multiplier*f[i] > cap
                    assert np.isclose(stake, saved.stake.iloc[i+1])
                    assert np.isclose(multiplier*f[i], saved.nominal_stake.iloc[i+1])
                    if outlay:
                        cash -= outlay
                        cash += outlay/costs[i]*payouts[i]
                    ledger.append(cash); stakes.append(stake)
                ledger = np.array(ledger)
                dd = 1-ledger/np.maximum.accumulate(ledger)
                assert np.allclose(ledger, saved.wealth)
                assert np.allclose(dd, saved.drawdown)
                assert np.isclose(case.final_wealth, cash)
                assert np.isclose(case.max_drawdown, dd.max())
                assert np.isclose(case.max_stake, max(stakes))
                assert case.capped_bets == cap_count
                assert case.bets == baseline.bets
                assert np.isclose(case.terminal_wealth_sacrificed_fraction, 1-cash/baseline.final_wealth)
                assert np.isclose(case.drawdown_reduction, baseline.max_drawdown-dd.max())
                if baseline.final_wealth > 1:
                    assert np.isclose(case.net_profit_sacrificed_fraction, (baseline.final_wealth-cash)/(baseline.final_wealth-1))
                else:
                    assert pd.isna(case.net_profit_sacrificed_fraction)
                if cap is None:
                    for field in ('final_wealth','max_drawdown','max_stake','bets'):
                        assert np.isclose(case[field], baseline[field])
                    baselines += 1
                else:
                    comparisons += 1
                    if baseline.max_stake <= cap:
                        assert np.isclose(case.final_wealth, baseline.final_wealth)
                        assert np.isclose(case.max_drawdown, baseline.max_drawdown)
                checked += 1
    result = dict(status='PASS', cash_reconstructed_paths=checked,
                  uncapped_benchmarks_reconciled=baselines, cap_comparisons=comparisons,
                  individual_current_bankroll_cap_checks=limit_checks)
    (ROOT/'results/capped_kelly_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
