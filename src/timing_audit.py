"""Record whether retrieved scoreboards identify individual fight start times."""
import json

import pandas as pd

from audit import ROOT


def run():
    fights = pd.read_csv(ROOT / 'data/cleaned/fights.csv')
    records = []
    for date in sorted(fights.date.unique()):
        scoreboard = json.loads((ROOT / f'data/raw/scoreboards/{date}.json').read_text())
        for event in scoreboard.get('events', []):
            competitions = event.get('competitions', [])
            timestamps = pd.Series([c.get('date') for c in competitions], dtype='object')
            records.append(dict(
                date=date,
                event=event['name'],
                scoreboard_bouts=len(competitions),
                distinct_competition_dates=timestamps.nunique(),
                largest_shared_timestamp_group=int(timestamps.value_counts().max()),
                all_date_start_end_equal=all(
                    c.get('date') == c.get('startDate') == c.get('endDate')
                    for c in competitions),
                source='https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard?dates=' + date.replace('-', ''),
            ))
    output = pd.DataFrame(records)
    output.to_csv(ROOT / 'results/quote_timing_audit.csv', index=False)
    print(output.to_string(index=False))


if __name__ == '__main__':
    run()
