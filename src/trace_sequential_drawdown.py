"""Attribute the shared 50.5% sequential drawdown to settled individual bets."""
import json
import numpy as np
import pandas as pd
from audit import ROOT


def run():
    paths = pd.read_csv(ROOT/'results/sequential_kelly_paths.csv.gz')
    ledger = pd.read_csv(ROOT/'data/cleaned/kalshi_fights.csv').sort_values('sequence').reset_index(drop=True)
    forecasts = pd.read_csv(ROOT/'artifacts/private/aggressive_growth_probabilities.csv')
    summaries, details = [], []
    names = ('bounded_core_plus_fade1', 'core_fade1_plus_fade2', 'all_disjoint_candidates')
    for name in names:
        selected = paths[paths.portfolio.eq(name) & paths['sample'].eq('all') & paths.model.eq('odds_confidence')]
        full = selected[selected.policy.eq('full')].sort_values('fight_index').reset_index(drop=True)
        trough = int(full.drawdown.idxmax())
        peak = int(full.wealth.iloc[:trough+1].idxmax())
        placed = full[(full.index > peak) & (full.index <= trough) & (full.stake > 0)]
        components = []
        for step in placed.itertuples():
            record = ledger.iloc[step.fight_index-1]
            p = float(forecasts[name+'|all|fight|odds_confidence'].iloc[step.fight_index-1])
            before, after = float(full.wealth.iloc[step.fight_index-1]), float(step.wealth)
            details.append(dict(portfolio=name, fight_index=step.fight_index, date=record.date,
                                fight=record.fight, pick=record.pick, confidence=int(record.confidence),
                                full_kelly_stake=float(step.stake), side_probability=p,
                                pick_won=float(record.y), before=before, after=after,
                                price_source=record.price_source))
            components.append(dict(fight_index=step.fight_index, confidence=int(record.confidence),
                                   full_kelly_stake=float(step.stake), side_probability=p,
                                   realized_bankroll_factor=after/before,
                                   price_source=record.price_source))
        # Both executed bets lost their entire outlay; reproduce the decline
        # independently by multiplying the surviving bankroll fractions.
        assert len(components) == 2
        assert all(np.isclose(c['realized_bankroll_factor'], 1-c['full_kelly_stake']) for c in components)
        compounded = float(np.prod([1-c['full_kelly_stake'] for c in components]))
        assert np.isclose(1-compounded, full.drawdown.iloc[trough])
        sizing = []
        for policy in ('quarter', 'half', 'full'):
            q = selected[selected.policy.eq(policy)].sort_values('fight_index').reset_index(drop=True)
            sizing.append(dict(policy=policy, peak_wealth=float(q.wealth.iloc[peak]),
                               trough_wealth=float(q.wealth.iloc[trough]),
                               decline_over_same_window=float(1-q.wealth.iloc[trough]/q.wealth.iloc[peak]),
                               maximum_drawdown=float(q.drawdown.max()), ending_wealth=float(q.wealth.iloc[-1])))
        summaries.append(dict(portfolio=name, sample='all', model='odds_confidence',
                              peak_fight_index=peak, trough_fight_index=trough,
                              loss_components=components, sizing=sizing))
    pd.DataFrame(details).to_csv(ROOT/'artifacts/private/sequential_drawdown_fight_details.csv', index=False)
    (ROOT/'results/sequential_drawdown_attribution.json').write_text(json.dumps(summaries, indent=2)+'\n')
    print('Verified two-loss attribution and saved three portfolio summaries; identifying fight details remain local.')


if __name__ == '__main__':
    run()
