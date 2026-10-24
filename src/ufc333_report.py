"""Standalone aggregate report plus a local-only identified trade supplement."""
import html
import json
from collections import defaultdict
from ufc333_review import ROOT, PRIVATE, LABELS, PORT_LABELS, read_csv


def pct(v, signed=False):
    return format(float(v)*100, '+.2f' if signed else '.2f')+'%'


def table(headers, rows):
    return '<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def run():
    scenarios = read_csv(ROOT/'results/ufc333_scenarios.csv')
    paths = read_csv(ROOT/'results/ufc333_paths.csv')
    flat = read_csv(ROOT/'results/ufc333_flat_stakes.csv')
    card = read_csv(PRIVATE/'fight_ledger.csv')
    trades = read_csv(PRIVATE/'trades.csv')
    summary = json.loads((ROOT/'results/ufc333_summary.json').read_text())
    def lookup(model='original', portfolio='bounded_core_plus_fade1', policy='full', cap='uncapped', sample='all'):
        return next(r for r in scenarios if (r['model'], r['portfolio'], r['policy'], r['cap_name'], r['sample']) == (model, portfolio, policy, cap, sample))
    def growth(**args):
        return pct(float(lookup(**args)['final_wealth'])-1, True)
    flat_rows = [[('All picks' if r['confidence'] == 'all' else 'Confidence '+r['confidence']),
                  r['bets'], r['wins']+'–'+str(int(r['bets'])-int(r['wins'])),
                  pct(r['roi'], True), pct(r['gross_roi'], True)] for r in flat]
    portfolio_rows = []
    for p, label in PORT_LABELS.items():
        r = lookup(portfolio=p)
        portfolio_rows.append([label, r['bets'], growth(portfolio=p, policy='quarter'),
            growth(portfolio=p, policy='half'), growth(portfolio=p), pct(r['max_drawdown'])])
    model_rows = [[label, lookup(model=m)['bets'], growth(model=m, policy='quarter'),
                   growth(model=m, policy='half'), growth(model=m), pct(lookup(model=m)['max_drawdown'])]
                  for m, label in LABELS.items() if m != 'market']
    cap_rows = []
    for cap, label in [('uncapped', 'No cap'), ('cap_05', '5%'), ('cap_10', '10%'), ('cap_15', '15%'), ('cap_20', '20%')]:
        r = lookup(cap=cap)
        cap_rows.append([label, growth(cap=cap), pct(r['max_drawdown']), pct(r['max_stake']), r['capped_bets']])
    recovered_rows = [[LABELS[m], lookup(model=m, sample='observed')['bets'],
        growth(model=m, sample='observed', policy='half'), growth(model=m, sample='observed')]
        for m in ('original', 'confidence_offset', 'shrink_original', 'shrink_confidence', 'shrink_confidence_warm')]
    largest = max(card, key=lambda r: float(r['y'])/float(r['cost'])-1)
    profit = float(flat[0]['profit_per_unit_each'])
    without = (profit-(float(largest['y'])/float(largest['cost'])-1))/(len(card)-1)
    # Static, self-contained figure: all plotted points come from saved paths.
    curves = [('original', 'full', '#b83b46', 'Original · full Kelly'),
              ('original', 'half', '#d68c16', 'Original · half Kelly'),
              ('shrink_confidence_warm', 'full', '#337bb9', 'Warmup shrinkage · full Kelly'),
              ('shrink_confidence', 'full', '#25906e', 'Primary shrinkage · no bets')]
    points = []
    for model, policy, color, label in curves:
        line = [r for r in paths if r['sample'] == 'all' and r['portfolio'] == 'bounded_core_plus_fade1'
                and r['model'] == model and r['policy'] == policy and r['cap_name'] == 'uncapped']
        points.append((line, color, label))
    low = min(float(r['wealth']) for line, _, _ in points for r in line)
    width, height, left, top, plotw, ploth = 900, 340, 68, 32, 805, 245
    y = lambda wealth: top+(1-wealth)/(1-low)*ploth
    svg = f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Bankroll after each fight, starting at 100 percent">'
    # Guide values are the initial bankroll and the actual lowest plotted value.
    for value, label in [(1., '100%'), (low, pct(low))]:
        svg += f'<line x1="{left}" x2="{left+plotw}" y1="{y(value)}" y2="{y(value)}" stroke="#d8dfeb"/><text x="8" y="{y(value)+5}">{label}</text>'
    for line, color, _ in points:
        xy = ' '.join(f'{left+int(r["fight_index"])/len(card)*plotw},{y(float(r["wealth"]))}' for r in line)
        svg += f'<polyline points="{xy}" fill="none" stroke="{color}" stroke-width="3"/>'
    for i in range(len(card)+1):
        svg += f'<text text-anchor="middle" x="{left+i/len(card)*plotw}" y="302">{i}</text>'
    svg += '<text x="440" y="330">Settled fights</text></svg>'
    legend = '<div class="legend">'+''.join(f'<span style="color:{color}">● {html.escape(label)}</span>' for _, color, label in points)+'</div>'
    css = '''body{font:16px/1.6 system-ui,sans-serif;color:#182538;background:#f3f6fa;margin:0}main{max-width:1130px;margin:auto;background:white;padding:40px}h1{font-size:36px;line-height:1.2}h2{font-size:24px;margin-top:38px}.eyebrow{letter-spacing:.1em;color:#52657d;font-size:13px;text-transform:uppercase}.lead{font-size:21px;max-width:940px}.callout{border-left:4px solid #337bb9;padding:12px 20px;background:#eef5fc}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;text-align:right;border-bottom:1px solid #dce3ed;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#f0f4fa}svg{width:100%;font:13px system-ui}a{color:#205a97}.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:13px}.muted{color:#52657d;font-size:14px}li{margin:8px 0}@media(max-width:650px){main{padding:20px}h1{font-size:29px}}'''
    body = f'''<p class="eyebrow">UFC 333 · October 24, 2026 · sequential settlement</p>
    <h1>A profitable card for the raw picks; a losing card for the earlier filters</h1>
    <p class="lead">The 13 picks went <strong>8–5</strong>. Equal-dollar backing returned <strong>{pct(flat[0]['roi'], True)}</strong> on stakes after modeled fees.
    The primary shrinkage rule placed <strong>no bets</strong>. The original core-plus-C1-fade strategy lost <strong>{pct(1-float(lookup(policy='half')['final_wealth']))}</strong> of bankroll at half Kelly.</p>
    <p>These calculations use the spreadsheet’s Kalshi asks and the strategy definitions from the preceding research.
    All 13 results match the <a href="https://www.espn.com/mma/fightcenter/_/id/600061267/league/ufc">ESPN results</a>, and all 13 asks match their cited Kalshi candles at 14:00 UTC.
    Opposing bets use the same candle’s NO ask, derived from the YES bid. The spreadsheet is preserved.</p>
    <div class="callout">Flat-stake ROI measures profit divided by total money staked. Kelly returns below measure change in the bankroll available at the start of this card.
    They use different sizing and denominators. Each Kelly bet is a percentage of the bankroll remaining after the previous fight.</div>
    <h2>Backing every pick equally</h2>
    {table(['Group', 'Bets', 'Wins–losses', 'ROI after fees', 'ROI before fees'], flat_rows)}
    <p>For example, $100 of total outlay on each fight means $1,300 staked and approximately <strong>${100*profit:,.2f} net profit</strong>.
    The 17¢ winner generated more than the entire card’s net profit; the other 12 picks together returned {pct(without, True)} on stakes.
    No confidence-5 picks appeared on this card.</p>
    <h2>The earlier strategy filters</h2>
    <p>Original odds-plus-confidence probabilities, fitted to all earlier usable fights. Categories are unchanged; a qualifying fight receives a bet only if its estimated advantage remains positive after costs.</p>
    {table(['Strategy', 'Bets', 'Quarter Kelly', 'Half Kelly', 'Full Kelly', 'Full-Kelly max drawdown'], portfolio_rows)}
    <p>The core’s three confidence-4 bets went 1–2. Opposing the confidence-1 favorite lost; adding the confidence-2 opposing candidate added another loss.
    The confidence-3 favorite qualified for its interval but did not have a positive estimated edge after fees, so it received no Kelly stake.
    The winning 17¢ confidence-3 pick was outside the selected intervals. The formerly unbeaten intervals therefore no longer have an unbeaten record when this card is included.</p>
    <h2>What shrinkage changed</h2>
    <p>Core plus opposing confidence-1 favorites; full historical training sample. “Established warmup” is the previously tested convention that starts shrinkage scoring after the original 24-decisive-fight prefix; it is not chosen from this card’s returns.</p>
    {table(['Probability model', 'Bets', 'Quarter Kelly', 'Half Kelly', 'Full Kelly', 'Full-Kelly max drawdown'], model_rows)}
    <p>The primary versions retained zero model adjustment and stayed in cash. The warmup confidence version placed one losing bet:
    2.47% of current bankroll at full Kelly, or 1.23% at half Kelly. All three high-confidence bets were skipped because their shrunk probabilities did not cover purchase costs.
    The warmup version that shrinks the original model also placed no bets. Staying in cash avoided this card’s filtered-strategy losses; it did not capture the raw picks’ profit.</p>
    {legend}{svg}
    <h2>Existing per-fight caps</h2>
    <p>Original model, full Kelly, core plus opposing confidence-1 favorites. Caps are applied after Kelly sizing and include fees in the amount at risk.</p>
    {table(['Cap per fight', 'Bankroll return', 'Maximum drawdown', 'Largest stake', 'Capped bets'], cap_rows)}
    <h2>Training only on recovered Kalshi quotes</h2>
    <p>This sensitivity excludes reconstructed prices from earlier training. New-card prices are observed under both versions.</p>
    {table(['Probability model', 'Bets', 'Half-Kelly return', 'Full-Kelly return'], recovered_rows)}
    <h2>What this card tells us</h2>
    <p>This card does not validate the historical filters: their preferred high-confidence and opposing-pick positions performed poorly, while a large winner lay outside them.
    It illustrates the uncertainty behind the earlier attractive backtests. One card also cannot establish that the underlying confidence scale is useless or that backing all picks is the better long-run rule.
    The shrinkage versions protected capital mainly by reducing or eliminating participation.</p>
    <h2>Calculation and limits</h2><ul>'''+''.join('<li>'+html.escape(s)+'</li>' for s in summary['methodology'])+f'''</ul>
    <p>The first forecast uses {summary['prior_decisive']} earlier decisive results. The model then learns from each settled fight, including fights on which it did not bet.
    Kelly uses f = max(0, (p − all-in cost)/(1 − all-in cost)), multiplied by the selected Kelly fraction and limited by the selected cap.</p>
    <p>These are conditional results at the supplied pre-card prices. Holding those prices fixed while recycling bankroll does not demonstrate that the same quotes would have remained available before every later fight.</p>
    <p class="muted">The continuous fee approximation follows the standard taker expression in the <a href="https://kalshi.com/docs/kalshi-fee-schedule.pdf">Kalshi fee schedule</a>.
    It omits contract granularity, per-order cent rounding and slippage. No new interval search, stake fraction optimization or statistical-significance claim was introduced for this card.</p>
    <p><a href="../results/ufc333_scenarios.csv">All 1,080 sizing comparisons</a> · <a href="../results/ufc333_flat_stakes.csv">Flat-stake results</a> · <a href="../results/ufc333_paths.csv">Every sequential path</a> · <a href="shrinkage_kelly.html">Earlier shrinkage study</a></p>'''
    document = lambda content: '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UFC 333 — strategy results</title><style>'+css+'</style><main>'+content+'</main></html>'
    (ROOT/'reports/ufc333.html').write_text(document(body))
    named = []
    for i, r in enumerate(card):
        named.append([i+1, r['fight'], r['pick'], r['c'], f"{float(r['q'])*100:.0f}¢",
                      'Win' if float(r['y']) else 'Loss', r['winner'],
                      pct(float(r['y'])/float(r['cost'])-1, True)])
    main_trades = [r for r in trades if r['sample'] == 'all' and r['portfolio'] == 'bounded_core_plus_fade1' and r['model'] == 'original']
    detail = [[r['fight_index'], ('Back ' if r['side'] == 'yes' else 'Oppose ')+r['pick'],
               f"{float(r['ask'])*100:.0f}¢", pct(r['probability']), f"{float(r['cost'])*100:.2f}¢",
               pct(r['full_kelly']), pct(float(r['full_kelly'])/2), 'Win' if float(r['win']) else 'Loss'] for r in main_trades]
    supplement = '<p class="eyebrow">Local private supplement</p><h1>UFC 333 picks, prices and bets</h1><p>Proprietary row-level data; this file is excluded from Git.</p>'
    supplement += table(['Fight order', 'Matchup', 'Pick', 'Confidence', 'YES ask', 'Result', 'Winner', 'Flat return after fees'], named)
    supplement += '<h2>Original core-plus-C1 strategy</h2>'+table(['Fight order', 'Position', 'Ask', 'Estimated win probability', 'Cost / contract', 'Full-Kelly stake', 'Half-Kelly stake', 'Result'], detail)
    supplement += '<p>Each stake is a fraction of the current settled bankroll. Estimated probabilities use only earlier outcomes. NO asks include the observed spread. The price column is before fees; stake and return columns include modeled fees.</p><p><a href="../../../reports/ufc333.html">Full comparison and limitations</a> · <a href="trades.csv">All qualifying positions and probabilities</a> · <a href="fight_ledger.csv">Card ledger and source links</a></p>'
    (PRIVATE/'picks_and_bets.html').write_text(document(supplement))
    print('Saved aggregate report and local private pick/price supplement.', flush=True)


if __name__ == '__main__':
    run()
