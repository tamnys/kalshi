"""Requested per-fight bankroll caps, applied after fractional Kelly sizing."""
import json
import numpy as np
import pandas as pd
from audit import ROOT
from sequential_kelly import context_inputs, FIXED


# The six leading comparison portfolios from the preceding sequential report.
LEADING = ('core_bounded', 'bounded_core_plus_fade1', 'bounded_core_plus_narrow_fade1',
           'core_fade1_plus_fade2', 'all_disjoint_candidates', 'two_unbeaten_intervals')
# These exact exposure limits were requested by the user on October 19, 2026.
CAPS = {'uncapped': None, 'cap_05': .05, 'cap_10': .10, 'cap_15': .15, 'cap_20': .20}


def simulate(f, returns, multiplier, cap):
    nominal = multiplier*f
    stake = nominal.copy() if cap is None else np.minimum(nominal, cap)
    wealth = np.r_[1., np.cumprod(1+stake*returns)]
    drawdown = 1-wealth/np.maximum.accumulate(wealth)
    trough = int(drawdown.argmax())
    peak = int(wealth[:trough+1].argmax())
    metrics = dict(final_wealth=float(wealth[-1]), net_growth=float(wealth[-1]-1),
                   max_drawdown=float(drawdown[trough]), max_stake=float(stake.max()),
                   bets=int(np.count_nonzero(stake)), capped_bets=int(np.count_nonzero(stake < nominal)),
                   mean_stake=float(stake[stake > 0].mean()) if np.any(stake > 0) else 0.,
                   peak_fight_index=peak, trough_fight_index=trough,
                   peak_wealth=float(wealth[peak]), trough_wealth=float(wealth[trough]))
    return metrics, wealth, drawdown, stake, nominal


def run():
    full, contexts = context_inputs()
    contexts = [c for c in contexts if c[0]['portfolio'] in LEADING]
    rows, paths = [], []
    for context, f, returns, _, _ in contexts:
        for policy, multiplier in FIXED.items():
            baseline, basepath, *_ = simulate(f, returns, multiplier, None)
            for cap_name, cap in CAPS.items():
                metrics, wealth, drawdown, stake, nominal = simulate(f, returns, multiplier, cap)
                dd_reduction = baseline['max_drawdown']-metrics['max_drawdown']
                metrics.update(
                    uncapped_wealth=baseline['final_wealth'],
                    uncapped_drawdown=baseline['max_drawdown'],
                    terminal_wealth_change=metrics['final_wealth']-baseline['final_wealth'],
                    terminal_wealth_sacrificed_fraction=1-metrics['final_wealth']/baseline['final_wealth'],
                    net_profit_sacrificed_fraction=(baseline['final_wealth']-metrics['final_wealth'])/(baseline['final_wealth']-1)
                        if baseline['final_wealth'] > 1 else np.nan,
                    drawdown_reduction=dd_reduction,
                    drawdown_reduction_fraction=dd_reduction/baseline['max_drawdown'] if baseline['max_drawdown'] > 0 else np.nan,
                    return_over_uncapped_drawdown_window=float(wealth[baseline['trough_fight_index']]/wealth[baseline['peak_fight_index']]-1))
                rows.append(dict(**context, policy=policy, kelly_multiplier=multiplier,
                                 cap_name=cap_name, cap=cap, **metrics))
                paths.extend(dict(**context, policy=policy, cap_name=cap_name, fight_index=i,
                                  wealth=float(value), drawdown=float(drawdown[i]),
                                  stake=float(stake[i-1]) if i else 0.,
                                  nominal_stake=float(nominal[i-1]) if i else 0.)
                             for i, value in enumerate(wealth))
    pd.DataFrame(rows).to_csv(ROOT/'results/capped_kelly_scenarios.csv', index=False)
    pd.DataFrame(paths).to_csv(ROOT/'results/capped_kelly_paths.csv.gz', index=False,
                              compression={'method':'gzip', 'mtime':0})
    summary = dict(portfolios=list(LEADING), contexts=len(contexts), scenarios=len(rows),
                   path_points=len(paths), caps=CAPS, kelly_multipliers=FIXED,
                   notes=[
                       'Per-fight caps of 5%, 10%, 15% and 20% were explicitly requested by the user.',
                       'Stake is current settled bankroll times min(Kelly multiplier times estimated full Kelly, cap).',
                       'The cap includes the modeled fee in the total amount at risk. It does not force smaller Kelly bets up to the cap.',
                       'Each fight settles before the next bet. Forecasts, price inputs, categories and multipliers are unchanged between capped and uncapped comparisons.',
                       'Maximum drawdown is recomputed over the entire trajectory; its peak and trough can change when stakes are capped.',
                       'Terminal wealth sacrificed is 1 minus capped ending wealth divided by uncapped ending wealth; a negative value means capping increased ending wealth.',
                       'Net profit sacrificed uses uncapped wealth minus initial wealth as denominator and is undefined when the uncapped strategy did not earn a profit.',
                       'Drawdown reduction is measured both in percentage points and relative to uncapped drawdown; zero-uncapped-drawdown relative reductions are undefined.',
                       'All prior model, reconstructed-price, timing and retrospective-category-selection limitations remain. No cap is declared optimal for future betting.'
                   ])
    (ROOT/'results/capped_kelly_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'notes'}, indent=2), flush=True)


if __name__ == '__main__':
    run()
