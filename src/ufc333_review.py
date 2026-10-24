"""Apply the existing strategies to the appended UFC 333 card; no rule search.

Inputs and identifying outputs stay in ignored local directories. The audited
workbook is read, never regenerated. Old training data and old out-of-fit
forecasts are frozen; each new outcome becomes available only after its bet.
Standard library only, sharing the existing shrinkage and bankroll definitions.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

from shrinkage_backtest import (ROOT, MODELS, LABELS, PORTFOLIOS, FRACTIONS, CAPS,
    FIRTH_SCORE_TOL, read_csv, write_csv, load_inputs, logit, expit, softplus,
    firth_offset, firth_state, retention, position, simulate)

PRIVATE = ROOT/'artifacts/private/ufc333'
DATE = '2026-10-24'
WORKBOOK = ROOT/'data/cleaned/MMA betting - audited.xlsx'
ESPN = 'https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard?dates=20261024'
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
PORT_LABELS = {
    'core_bounded': 'C4–5 core, 35–81¢',
    'bounded_core_plus_fade1': 'Core + oppose C1 favorites',
    'bounded_core_plus_narrow_fade1': 'Core + oppose C1, 25–70¢',
    'core_fade1_plus_fade2': 'Core + oppose C1/C2 candidates',
    'all_disjoint_candidates': 'All previously selected candidates',
    'two_unbeaten_intervals': 'Two previously unbeaten intervals',
}


def fetch(url):
    PRIVATE.mkdir(parents=True, exist_ok=True)
    path = PRIVATE/(hashlib.sha256(url.encode()).hexdigest()+'.json')
    if not path.exists():
        with urllib.request.urlopen(url) as response:
            content = response.read()
        json.loads(content)
        path.write_bytes(content)
        path.with_suffix('.receipt.json').write_text(json.dumps(dict(
            url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
            sha256=hashlib.sha256(content).hexdigest()), indent=2)+'\n')
    return json.loads(path.read_text())


def workbook_rows():
    with zipfile.ZipFile(WORKBOOK) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared = [''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        root = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    rows, date, event = [], None, None
    for row in root.findall('.//s:sheetData/s:row', NS):
        cells = {}
        for cell in row.findall('s:c', NS):
            col = ''.join(c for c in cell.attrib['r'] if c.isalpha())
            v = cell.find('s:v', NS)
            if cell.attrib.get('t') == 'inlineStr':
                text = ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
            elif cell.attrib.get('t') == 's':
                text = shared[int(v.text)]
            else:
                text = v.text if v is not None else ''
            cells[col] = text
        if cells.get('B'):
            event = cells['B'].strip().lower()
            if event == 'ufc 333':
                date = (datetime(1899, 12, 30)+timedelta(days=float(cells['A']))).date().isoformat()
        if event != 'ufc 333' or not cells.get('C'):
            continue
        assert date == DATE
        price = float(cells['P'])
        assert math.isclose(price, float(cells['G'].replace('YES', '').replace('¢', '').strip())/100)
        assert cells['I'].lower() in ('y', 'n')
        rows.append(dict(row=int(row.attrib['r']), date=date, fight=cells['C'], pick=cells['D'],
            c=int(cells['E']), q=price, y=float(cells['I'].lower() == 'y'),
            payout=float(cells['I'].lower() == 'y'), winner=cells['L'],
            quote_url=cells['J'], quote_note=cells['K'], observed=True))
    assert rows, 'UFC 333 entries not found'
    return rows


def audit_card(rows):
    event = next(e for e in fetch(ESPN)['events'] if 'UFC 333' in e['name'])
    assert event['status']['type']['completed']
    quotes = list(ThreadPoolExecutor().map(fetch, [r['quote_url'] for r in rows]))
    order = []
    for r, quote in zip(rows, quotes):
        candidates = [(i, fight) for i, fight in enumerate(event['competitions'])
                      if r['pick'].lower() in ' / '.join(c['athlete']['displayName'].lower() for c in fight['competitors'])]
        assert len(candidates) == 1, (r['pick'], candidates)
        index, fight = candidates[0]
        assert fight['status']['type']['completed']
        winner = next(c['athlete']['displayName'] for c in fight['competitors'] if c.get('winner'))
        assert winner == r['winner'], (winner, r['winner'])
        assert float(r['pick'].lower() in winner.lower()) == r['y']
        order.append(index)
        # Exactly the candle cited in the workbook, not a later closing price.
        target = int(r['quote_url'].split('end_ts=')[1].split('&')[0])
        candle = next(c for c in quote['candlesticks'] if c['end_period_ts'] == target)
        assert float(candle['yes_ask']['close_dollars']) == r['q']
        r['bid'] = float(candle['yes_bid']['close_dollars'])
        r['no_ask'] = 1-r['bid']
        assert 0 < r['bid'] <= r['q'] < 1
        r['quote_time'] = datetime.fromtimestamp(target, timezone.utc).isoformat()
        r['cost'] = r['q']+.07*r['q']*(1-r['q'])
        r['no_cost'] = r['no_ask']+.07*r['no_ask']*(1-r['no_ask'])
        r['sequence'] = len(order)
    assert order == sorted(order), 'Workbook order differs from ESPN bout listing'
    return dict(result_matches=len(rows), ask_matches=len(rows), opposite_asks_recovered=len(rows),
                fight_order='Workbook order agrees with ESPN competition listing', espn=ESPN,
                quote_times=sorted({r['quote_time'] for r in rows}))


def original_state(beta, training):
    """Same three-parameter Jeffreys/Firth objective as src/kelly.py."""
    vals, info, ll = [], [[0.]*3 for _ in range(3)], 0.
    for q, z, y in training:
        x = (1., logit(q), z)
        eta = sum(b*v for b, v in zip(beta, x))
        p = expit(eta)
        w = p*(1-p)
        vals.append((x, y, p, w))
        ll += y*eta-softplus(eta)
        for i in range(3):
            for j in range(3):
                info[i][j] += w*x[i]*x[j]
    a, b, c = info[0]
    _, d, e = info[1]
    f = info[2][2]
    cof = [[d*f-e*e, c*e-b*f, b*e-c*d],
           [c*e-b*f, a*f-c*c, b*c-a*e],
           [b*e-c*d, b*c-a*e, a*d-b*b]]
    det = a*cof[0][0]+b*cof[0][1]+c*cof[0][2]
    if det <= 0:
        return -math.inf, None, None
    inv = [[v/det for v in row] for row in cof]
    score = [0.]*3
    for x, y, p, w in vals:
        leverage = w*sum(x[i]*inv[i][j]*x[j] for i in range(3) for j in range(3))
        residual = y-p+leverage*(.5-p)
        for i in range(3):
            score[i] += x[i]*residual
    return ll+.5*math.log(det), score, inv


def fit_original(training, start=None):
    beta = start or (0., 1., 0.)
    while True:
        value, score, inv = original_state(beta, training)
        if score is None:
            raise ArithmeticError('Singular original-model fit')
        if max(map(abs, score)) <= FIRTH_SCORE_TOL:
            return beta, max(map(abs, score))
        step = [sum(inv[i][j]*score[j] for j in range(3)) for i in range(3)]
        scale = 1.
        while True:
            trial = tuple(beta[i]+scale*step[i] for i in range(3))
            if trial == beta:
                raise ArithmeticError('Original model stalled before score tolerance')
            if original_state(trial, training)[0] >= value:
                beta = trial
                break
            scale /= 2


def extend_forecasts(old, card, saved, sample, learning_start):
    training = [(r['q'], r['c']-3, r['y']) for r in old
                if r['q'] is not None and r['y'] is not None and (sample == 'all' or r['observed'])]
    history = {m: [] for m in MODELS if m.startswith('shrink_')}
    for i, (r, forecast) in enumerate(zip(old, saved)):
        for model in history:
            base = 'original' if 'original' in model else 'confidence_offset'
            if r['y'] is not None and forecast[base] and (not model.endswith('_warm') or i >= learning_start):
                history[model].append((logit(r['q']), logit(float(forecast[base]))-logit(r['q']), r['y']))
    beta, offset, forecasts, diagnostics = None, None, [], []
    for i, r in enumerate(card):
        beta, score = fit_original(training, beta)
        offset, offset_score = firth_offset(training, offset)
        ps = dict(original=expit(beta[0]+beta[1]*logit(r['q'])+beta[2]*(r['c']-3)),
                  confidence_offset=expit(logit(r['q'])+offset[0]+offset[1]*(r['c']-3)))
        weights = {}
        for model in history:
            base = 'original' if 'original' in model else 'confidence_offset'
            w = retention(history[model])
            weights[model] = w
            delta = logit(ps[base])-logit(r['q'])
            ps[model] = expit(logit(r['q'])+w*delta)
            # Settlement becomes visible only after the current forecasts.
            history[model].append((logit(r['q']), delta, r['y']))
        forecasts.append(dict(sample=sample, fight_index=i+1, row=r['row'], **ps,
                              **{'retention_'+m: w for m, w in weights.items()}))
        diagnostics.append(dict(sample=sample, fight_index=i+1, prior_n=len(training),
            original_beta=list(beta), confidence_beta=list(offset),
            original_score=score, confidence_score=offset_score))
        training.append((r['q'], r['c']-3, r['y']))
    return forecasts, diagnostics


def run():
    old, _, lower, legacy = load_inputs()
    learning_start = next(i for i, r in enumerate(old) if r['row'] in legacy)
    card = workbook_rows()
    assert max(r['date'] for r in old) < DATE
    audit = audit_card(card)
    write_csv(PRIVATE/'fight_ledger.csv', card)
    saved = read_csv(ROOT/'artifacts/private/shrinkage_forecasts.csv')
    all_forecasts, diagnostics, scenarios, paths, trades = [], [], [], [], []
    for sample in ('all', 'observed'):
        previous = [r for r in saved if r['sample'] == sample]
        assert [int(r['row']) for r in previous] == [r['row'] for r in old]
        fs, fits = extend_forecasts(old, card, previous, sample, learning_start)
        all_forecasts.extend(fs)
        diagnostics.extend(fits)
        for portfolio, legs in PORTFOLIOS.items():
            for model in MODELS:
                predictions = [r[model] for r in fs]
                for policy, fraction in FRACTIONS.items():
                    for cap_name, cap in CAPS.items():
                        metrics, path = simulate(card, predictions, [True]*len(card), legs, lower, fraction, cap)
                        ctx = dict(sample=sample, portfolio=portfolio, model=model, policy=policy, cap_name=cap_name)
                        scenarios.append(dict(**ctx, **metrics))
                        paths.extend(dict(**ctx, **p) for p in path)
                for i, (r, p) in enumerate(zip(card, predictions)):
                    side = position(r, legs, lower)
                    if side is None:
                        continue
                    probability = p if side == 'yes' else 1-p
                    cost = r['cost'] if side == 'yes' else r['no_cost']
                    payout = r['y'] if side == 'yes' else 1-r['y']
                    trades.append(dict(sample=sample, portfolio=portfolio, model=model,
                        fight_index=i+1, pick=r['pick'], side=side, confidence=r['c'],
                        ask=r['q'] if side == 'yes' else r['no_ask'], probability=probability,
                        cost=cost, win=payout, full_kelly=max(0., (probability-cost)/(1-cost))))
    flat = []
    for confidence in (None, *sorted({r['c'] for r in card})):
        selected = [r for r in card if confidence is None or r['c'] == confidence]
        profit = sum(r['y']/r['cost']-1 for r in selected)
        flat.append(dict(confidence='all' if confidence is None else confidence,
            bets=len(selected), wins=int(sum(r['y'] for r in selected)), profit_per_unit_each=profit,
            roi=profit/len(selected), gross_roi=sum(r['y']/r['q']-1 for r in selected)/len(selected)))
    summary = dict(event='UFC 333', date=DATE, fights=len(card), wins=int(sum(r['y'] for r in card)),
        prior_decisive=sum(r['y'] is not None for r in old), scenarios=len(scenarios),
        workbook_sha256=hashlib.sha256(WORKBOOK.read_bytes()).hexdigest(), **audit,
        methodology=['Apply previously defined intervals and six probability models without selecting new rules.',
            'Refit only on earlier settled fights, including earlier fights on this card. Retention scores earlier out-of-fit forecasts.',
            'Use the spreadsheet YES asks. Buy NO at 1 minus same-candle YES bid. All 13 asks and outcomes match retrieved feeds.',
            'Each fight settles before the next stake. Reset card-start bankroll to one.',
            'Same continuous standard-taker fee approximation: ask + 0.07*ask*(1-ask). Fractional contracts; actual contract and fee rounding omitted.',
            'Quotes were recorded before the card, not at each sequential entry. Executability and liquidity are not established.',
            'The workbook was supplied with results; pre-fight existence of its picks is not independently certified. This is a retrospective application of pre-existing rules.',
            'No new statistical edge claim is inferred from one card. No original or audited workbook is modified.'])
    write_csv(PRIVATE/'forecasts.csv', all_forecasts)
    write_csv(PRIVATE/'trades.csv', trades)
    (PRIVATE/'fits.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    write_csv(ROOT/'results/ufc333_scenarios.csv', scenarios)
    write_csv(ROOT/'results/ufc333_flat_stakes.csv', flat)
    write_csv(ROOT/'results/ufc333_paths.csv', paths)
    (ROOT/'results/ufc333_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(summary=summary, flat_stakes=flat,
        main=[r for r in scenarios if r['sample'] == 'all' and r['portfolio'] == 'bounded_core_plus_fade1'
              and r['cap_name'] in ('uncapped', 'cap_10')]), indent=2), flush=True)


if __name__ == '__main__':
    run()
