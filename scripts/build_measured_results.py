#!/usr/bin/env python3
"""Publish author-confirmed measurements without evaluating a model (Python 3.9+).

One source document drives the README, Markdown, CSV, SVG and static Pages site.
Archived evaluation files are read-only. Unknown protocol fields remain unknown.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import io
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

SOURCE = Path('results/measured/2026-10-18.json')
REPO = 'https://github.com/yinli-systems/dense3b'
BLOB = REPO + '/blob/main/'
RAW = 'https://raw.githubusercontent.com/yinli-systems/dense3b/main/'
GROUPS = {'code_generation': 'Code generation', 'code_reasoning': 'Code reasoning', 'reasoning_knowledge': 'Mathematics and knowledge'}
EXPECTED = ['humaneval', 'humaneval_plus', 'mbpp', 'mbpp_plus', 'cruxeval_i', 'cruxeval_o', 'gsm8k', 'arc_challenge', 'mmlu']
COLORS = ['#146C94', '#16866D', '#A46318', '#8052A4']

def fail(condition, message):
    if not condition:
        raise ValueError(message)

def score(v):
    return ('≈' if v.get('approximate') else '') + v['value_pct']

def validate(d):
    fail(d.get('schema_version') == 1, 'Unsupported schema')
    fail(d.get('model', {}).get('architecture') == 'dense', 'Wrong model architecture')
    stages = d['stages']
    ids = [s['id'] for s in stages]
    fail(len(ids) == len(set(ids)) == 6, 'Exactly six distinct stages required')
    tokens = [Decimal(s['tokens_b']) for s in stages]
    fail(all(x.is_finite() and x > 0 for x in tokens), 'Invalid token label')
    fail(all(a < b for a, b in zip(tokens, tokens[1:])), 'Stages must be increasing')
    fail([b['id'] for b in d['benchmarks']] == EXPECTED, 'Missing or reordered benchmark')
    lookup = {}
    for b in d['benchmarks']:
        fail(b['group'] in GROUPS, 'Unknown benchmark group')
        fail([v['stage_id'] for v in b['values']] == ids, 'Missing or duplicate stage value')
        for v in b['values']:
            fail(isinstance(v['value_pct'], str), 'Preserve supplied precision as text')
            try:
                p = Decimal(v['value_pct'])
            except InvalidOperation as e:
                raise ValueError('Invalid percentage') from e
            fail(p.is_finite() and 0 <= p <= 100, 'Percentage outside [0,100]')
            fail(v['source'] in d['sources'], 'Missing source')
            if v['stage_id'] != ids[0]:
                fail(v['n_archived'] is None and v['passed_archived'] is None, 'Do not create missing sample counts')
                fail(v['raw_outputs_attached'] is False, 'No later raw outputs are attached')
            else:
                n, k = v['n_archived'], v['passed_archived']
                fail(isinstance(n, int) and isinstance(k, int) and 0 <= k <= n and n > 0, 'Invalid archived counts')
                tolerance = Decimal(10) ** p.as_tuple().exponent / 2
                fail(abs(p - Decimal(100) * k / n) <= tolerance, 'Early score inconsistent with archived count')
            lookup[b['id'], v['stage_id']] = v
    for plus, base in [('humaneval_plus', 'humaneval'), ('mbpp_plus', 'mbpp')]:
        for stage in ids:
            fail(Decimal(lookup[plus, stage]['value_pct']) <= Decimal(lookup[base, stage]['value_pct']), 'Plus exceeds base')
    for c in d['corrections']:
        fail(lookup[c['benchmark_id'], c['stage_id']]['value_pct'] == c['confirmed_value_pct'], 'Lost author correction')
    fail(lookup['mbpp', '100B']['value_pct'] == '52', 'Preserve integer-rounded MBPP endpoint')
    fail(lookup['mmlu', '100B']['approximate'] is True, 'Preserve approximate MMLU endpoint')
    m = d.get('systems_measurements', {}).get('later_training_global_throughput')
    fail(isinstance(m, dict), 'Missing later-run throughput')
    fail(type(m.get('value')) is int and m['value'] > 0, 'Invalid later-run throughput')
    fail(m.get('unit') == 'tokens/s' and m.get('statistic') == 'mean', 'Invalid throughput unit or statistic')
    fail(m.get('source') in d['sources'] and m.get('source_logs_attached') is False, 'Incorrect systems source')
    fail(m.get('scope') == 'later_training' and m.get('aggregation_window') is None, 'Unreported window must remain unknown')
    return lookup

def later_rate(d):
    return f"{d['systems_measurements']['later_training_global_throughput']['value']:,}"

def table(d, prefix=''):
    heads = ['Benchmark'] + [f"[{s['tokens_b']}B]({prefix}checkpoints/{s['id']}.md)" for s in d['stages']]
    out = ['| ' + ' | '.join(heads) + ' |', '|---|' + '---:|' * len(d['stages'])]
    for b in d['benchmarks']:
        label = b['label'] + ('*' if b['id'] == 'mmlu' else '')
        out.append('| ' + label + ' | ' + ' | '.join(score(v) for v in b['values']) + ' |')
    return '\n'.join(out)

def flatten(d):
    stages = {s['id']: s for s in d['stages']}
    out = []
    for b in d['benchmarks']:
        for v in b['values']:
            out.append({'benchmark': b['label'], 'benchmark_id': b['id'], 'stage': v['stage_id'], 'training_tokens_b_reported': stages[v['stage_id']]['tokens_b'], 'score_pct': v['value_pct'], 'approximate': str(v['approximate']).lower(), 'measurement_status': 'measured', 'source_kind': d['sources'][v['source']]['kind'], 'source_id': v['source'], 'raw_outputs_attached': str(v['raw_outputs_attached']).lower(), 'n_archived': v['n_archived'], 'passed_archived': v['passed_archived'], 'checkpoint_step_archived': stages[v['stage_id']]['checkpoint_step'], 'protocol_note': 'Early anchor is MC5; later protocol not supplied' if b['id'] == 'mmlu' else 'See release source notes'})
    return out

def csv_text(rows):
    out = io.StringIO(newline='')
    w = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()

def svg_chart(d, group):
    width, height = 1024, 420
    x0, x1, y0, y1 = 70, 975, 98, 345
    y_max = 70
    def xy(tokens, pct):
        return x0 + float(tokens) / 100 * (x1-x0), y1 - float(pct) / y_max * (y1-y0)
    title = GROUPS[group]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" viewBox="0 0 {width} {height}">', f'<title id="title">{html.escape(title)}: measured checkpoint scores</title>', '<desc id="desc">Reported measurements only. Each point has its exact value in a title. No fitted lines are drawn. Early points have archived outputs; later points are author-confirmed measurements. Later evaluator manifests are not attached.</desc>', '<rect width="1024" height="420" rx="16" fill="#FFFFFF"/>', '<g font-family="Arial,Helvetica,sans-serif" fill="#233447">', f'<text x="30" y="35" font-size="20" font-weight="700">{title}</text>']
    bs = [b for b in d['benchmarks'] if b['group'] == group]
    for i, b in enumerate(bs):
        color = COLORS[i]
        x = 40 + i*235
        out += [f'<circle cx="{x}" cy="64" r="4" fill="{color}"/>', f'<text x="{x+12}" y="69" font-size="13">{html.escape(b["label"])}</text>']
    for value in range(0, y_max+1, 10):
        _, y = xy(0, value)
        out += [f'<path d="M{x0},{y:.3f} H{x1}" stroke="#E4EAF0"/>', f'<text x="{x0-12}" y="{y+4:.3f}" text-anchor="end" font-size="12">{value}</text>']
    for s in d['stages']:
        x, _ = xy(s['tokens_b'], 0)
        out.append(f'<text x="{x:.3f}" y="369" text-anchor="middle" font-size="12">{s["tokens_b"]}</text>')
    out += ['<text x="523" y="396" text-anchor="middle" font-size="12">Training tokens (billions, author-reported labels)</text>', '<text x="20" y="225" transform="rotate(-90 20 225)" text-anchor="middle" font-size="12">Score (%)</text>']
    for i, b in enumerate(bs):
        for s, v in zip(d['stages'], b['values']):
            x, y = xy(s['tokens_b'], v['value_pct'])
            fill = '#FFFFFF' if v['raw_outputs_attached'] else COLORS[i]
            out.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="5" fill="{fill}" stroke="{COLORS[i]}" stroke-width="2"><title>{html.escape(b["label"])} · {s["tokens_b"]}B: {score(v)}%</title></circle>')
    out += ['</g></svg>']
    return '\n'.join(out)+'\n'

CSS = '''
:root{--ink:#162c3c;--muted:#506574;--accent:#126b67;--line:#dce5e8;--paper:#f4f7f6;--card:#fff;--soft:#e9f2ef}
*{box-sizing:border-box}.hero>*,.metrics>*,.twocol>*{min-width:0}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:var(--paper);font:16px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--accent);text-underline-offset:3px}a:focus-visible,button:focus-visible,input:focus-visible{outline:3px solid #b77825;outline-offset:4px}.skip{position:absolute;top:-100px;left:10px;background:white;padding:12px;z-index:10}.skip:focus{top:10px}.wrap{max-width:1180px;margin:auto;padding:0 30px}header{background:var(--ink);color:white}header .wrap{display:flex;align-items:center;justify-content:space-between;gap:20px;padding-top:18px;padding-bottom:18px}.brand{font-weight:800;font-size:21px;letter-spacing:-.5px;color:white;text-decoration:none}nav{display:flex;gap:22px;flex-wrap:wrap}nav a{color:#dfeae8;font-size:14px;text-decoration:none}main{padding-bottom:60px}.hero{padding:70px 0 38px;display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:55px;align-items:center}.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:var(--accent);font-weight:800}h1{font-size:clamp(42px,5vw,68px);letter-spacing:-3px;line-height:1.04;margin:17px 0 22px}h2{font-size:28px;letter-spacing:-.8px;line-height:1.2;margin:0 0 14px}h3{font-size:18px;margin:0 0 12px}p{margin:0 0 15px}.lead{color:var(--muted);font-size:18px;max-width:600px}.pills{display:flex;gap:9px;flex-wrap:wrap;margin-top:24px}.pill{border:1px solid #cadbd6;color:var(--accent);background:#fff;border-radius:25px;padding:6px 13px;font-size:12px;font-weight:700}.hero-note{font-size:12px;color:var(--muted);margin-top:18px}.metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px}.metric{background:white;border:1px solid var(--line);border-radius:16px;padding:25px 23px}.metric small{color:var(--muted);display:block;font-size:12px;letter-spacing:.6px;text-transform:uppercase}.metric strong{font-size:clamp(23px,4.6vw,34px);letter-spacing:-1px;display:block;margin:5px 0;color:var(--ink);font-variant-numeric:tabular-nums}.metric span{font-size:12px;color:var(--accent)}section{margin:36px 0}.section-head{display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:19px}.sub{color:var(--muted);font-size:14px;margin:0}.card{border:1px solid var(--line);border-radius:16px;background:white;padding:26px}.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:14px;background:white}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;white-space:nowrap}caption{padding:15px;text-align:left;font-weight:600}th,td{padding:13px 18px;border-bottom:1px solid #e6ecef;text-align:right}thead th{background:var(--ink);color:white;font-weight:650}tbody th{text-align:left;background:#fff;font-weight:600;position:sticky;left:0}thead th:first-child{text-align:left}tbody tr:last-child>*{border-bottom:0}tbody tr:hover>*{background:#f0f6f4}td:last-child{font-weight:800;color:var(--accent);background:#f3f8f6}.notes{font-size:12px;color:var(--muted);margin-top:14px;line-height:1.8}.notes p{margin:4px 0}.search{border:1px solid #bdd0ca;border-radius:8px;background:white;padding:10px 12px;width:210px;font:inherit;font-size:13px}.charts{display:grid;gap:16px}.chart{padding:10px;overflow:hidden}.chart img{display:block;width:100%;height:auto}.legend-note{font-size:12px;color:var(--muted);padding:0 22px 12px}.twocol{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:19px}.statline{display:flex;justify-content:space-between;gap:20px;padding:11px 0;border-bottom:1px solid var(--line);font-size:14px}.statline:last-child{border:0}.statline strong{font-variant-numeric:tabular-nums}.links{display:flex;gap:11px;flex-wrap:wrap;margin-top:18px}.button{display:inline-block;padding:9px 14px;background:var(--accent);color:white;border-radius:7px;font-size:13px;text-decoration:none;font-weight:650}.button.secondary{background:white;border:1px solid #b8cdc5;color:var(--accent)}code{font-size:12px;background:#eff4f2;padding:2px 5px;border-radius:4px}footer{background:var(--ink);color:#dce8e3;padding:28px 0;font-size:12px}footer a{color:white}.empty{padding:18px;display:none}.noscript{font-size:13px;color:var(--muted)}@media(max-width:800px){.hero{grid-template-columns:minmax(0,1fr);gap:26px;padding-top:38px}.twocol{grid-template-columns:1fr}.wrap{padding-left:18px;padding-right:18px}.section-head{align-items:flex-start;flex-direction:column}.metric{padding:20px}header .wrap{align-items:flex-start;flex-direction:column;gap:8px}nav{gap:16px}h1{letter-spacing:-2px}.search{width:100%}th,td{padding:11px 14px}.card{padding:20px}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
'''

JS = ''''use strict';
const search = document.getElementById('filter');
const rows = Array.from(document.querySelectorAll('#results-table tbody tr'));
search.addEventListener('input', () => {
  const q = search.value.trim().toLowerCase();
  let visible = 0;
  for (const row of rows) {
    const show = row.dataset.benchmark.includes(q);
    row.hidden = !show;
    visible += Number(show);
  }
  document.getElementById('empty').style.display = visible ? 'none' : 'block';
  document.getElementById('row-status').textContent = visible + ' benchmarks shown';
});
'''

def site(d):
    heads = ''.join('<th scope="col">'+s['tokens_b']+'B</th>' for s in d['stages'])
    trs = []
    for b in d['benchmarks']:
        label = b['label'] + ('*' if b['id'] == 'mmlu' else '')
        trs.append('<tr data-benchmark="'+html.escape(b['label'].lower())+'"><th scope="row">'+label+'</th>'+''.join('<td>'+score(v)+'</td>' for v in b['values'])+'</tr>')
    charts = ''.join('<article class="card chart"><img src="assets/'+g+'.svg" alt="'+title+' measured checkpoint scores" width="1024" height="420" loading="lazy"><p class="legend-note">Points are reported measurements. Hollow: archived early outputs. Filled: author-confirmed later results. No fitted curves.</p></article>' for g,title in GROUPS.items())
    cards = ''.join('<div class="metric"><small>'+label+'</small><strong>'+value+'%</strong><span>100B-token checkpoint</span></div>' for label,value in [('HumanEval','49.4'),('GSM8K','59.8'),('ARC-Challenge','60.1'),('MBPP+','51.3228')])
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="Dense3B: measured checkpoint results across 100B training tokens, archived training metrics, code and evaluation evidence."><title>Dense3B — measured checkpoint results</title><link rel="stylesheet" href="assets/style.css"><script src="assets/app.js" defer></script></head>
<body><a class="skip" href="#main">Skip to content</a><header><div class="wrap"><a class="brand" href="#">Dense3B</a><nav aria-label="Main"><a href="#results">Results</a><a href="#trends">Checkpoints</a><a href="#training">Training</a><a href="#evidence">Evidence</a><a href="{REPO}">GitHub ↗</a></nav></div></header>
<main class="wrap" id="main"><div class="hero"><div><div class="eyebrow">Dense pretraining · measured results · 18 Oct 2026</div><h1>3B dense.<br>100B tokens.</h1><p class="lead">A checkpoint-by-checkpoint view of code generation, code reasoning, mathematics and knowledge.</p><div class="pills"><span class="pill">3,237,529,600 parameters</span><span class="pill">6 training milestones</span><span class="pill">9 primary benchmarks</span></div><p class="hero-note">Author-confirmed measured results, with archived evidence linked separately. Weights are not hosted in this repository.</p><div class="links"><a class="button" href="#results">Explore results</a><a class="button secondary" href="{BLOB}reports/CONFIRMED_RESULTS.md">Full report</a></div></div><div class="metrics">{cards}</div></div>
<section id="results"><div class="section-head"><div><h2>Every milestone, in one place.</h2><p class="sub">54 primary measurements. Percentages retain their reported precision.</p></div><label><span class="notes">Filter benchmark</span><br><input class="search" id="filter" type="search" placeholder="e.g. HumanEval or GSM8K" aria-controls="results-table"></label></div><div class="table-wrap"><table id="results-table"><caption>Measured checkpoint results — scores (%)</caption><thead><tr><th scope="col">Benchmark</th>{heads}</tr></thead><tbody>{''.join(trs)}</tbody></table><div class="empty" id="empty">No matching benchmark.</div></div><span id="row-status" class="notes" role="status" aria-live="polite">9 benchmarks shown</span><div class="notes"><p>* The early MMLU anchor uses MC5. Later MMLU evaluator manifests are not supplied; no cross-protocol equivalence is asserted. The 100B value remains approximate.</p><p>MBPP at 100B is reported as 52% at integer precision. CRUXEval-I at 60B is 30.750%; MBPP+ at 100B is 51.3228%.</p><p>The 6.82B token label is rounded. The early row links to archived outputs; later rows are author-confirmed measurements whose raw outputs are not attached to this release.</p></div><noscript><p class="noscript">All results remain visible without JavaScript. Only filtering is unavailable.</p></noscript><div class="links"><a class="button secondary" href="data/results.csv" download>CSV · full provenance</a><a class="button secondary" href="data/results.json" download>JSON · source release</a><a class="button secondary" href="data/results_wide.csv" download>CSV · benchmark matrix</a></div></section>
<section id="trends"><h2>Checkpoint observations.</h2><p class="sub">Grouped by capability. Tables above are the exact numeric reference; plots do not establish controlled comparability across unarchived protocols.</p><div class="charts">{charts}</div></section>
<section id="training"><div class="section-head"><div><h2>Training implementation and evidence.</h2><p class="sub">The later-run average is author-reported; other systems measurements below belong to the 9 October archive.</p></div></div><div class="twocol"><article class="card"><h3>Later training and archived snapshot</h3><div class="statline"><span>Later-run average global throughput (author-reported)</span><strong>{later_rate(d)} tokens/s</strong></div><div class="statline"><span>Training GPUs</span><strong>8 × RTX 4090</strong></div><div class="statline"><span>Model FLOP utilization, median</span><strong>68.09%</strong></div><div class="statline"><span>Global throughput, median</span><strong>41,780 tokens/s</strong></div><div class="statline"><span>Peak allocated memory, latest</span><strong>22.387 GB</strong></div><p class="notes">MFU is the implementation's model-FLOP estimate. Medians use at most the last 100 retained observations and exclude unlogged downtime. The later-run average has an unspecified averaging window and no attached logs.</p><a href="{BLOB}reports/SYSTEMS.md">System metrics and raw curves ↗</a></article><article class="card"><h3>Browse the implementation</h3><p><a href="{REPO}/tree/main/model">Model architecture</a> · <a href="{REPO}/tree/main/train">Training and optimization</a> · <a href="{REPO}/tree/main/bench">Evaluation</a></p><p><a href="{REPO}/tree/main/code3b">Data filtering and packing</a> · <a href="{REPO}/tree/main/tokenizer">Tokenizer</a> · <a href="{BLOB}MODEL_CARD.md">Model card</a></p><p class="notes">The original weighted source mixture is archived. Revised mixing weights and their change points have not been supplied for this release.</p><a href="{BLOB}evidence/manifests/mix-v0/mix.json">Original data mixture ↗</a></article></div></section>
<section id="evidence"><h2>Trace every result to its source.</h2><div class="twocol"><article class="card"><h3>Measured-results release</h3><p>All milestone values are author-confirmed measurements. Six explicit corrections are retained in the source JSON. Reported precision, approximate values and missing protocol fields are preserved.</p><p class="notes">This publishing step validates consistency and builds pages; it does not rerun model evaluation or manufacture per-example logs.</p><div class="links"><a class="button secondary" href="{BLOB}reports/measured/SOURCES.md">Source notes</a><a class="button secondary" href="{BLOB}{SOURCE.as_posix()}">Canonical data</a></div></article><article class="card"><h3>Historical archive</h3><p>The existing 607 archived metric rows, 13 checkpoint entries, evaluator versions, training traces and source hash inventories remain available in their original paths.</p><div class="links"><a class="button secondary" href="{BLOB}reports/RESULTS.md">All archived checkpoints</a><a class="button secondary" href="{RAW}reports/evaluation_history.csv">Archived metrics CSV</a><a class="button secondary" href="{BLOB}reports/sample_verification.json">Sample recount records</a></div><p class="notes">MMLU-CF: 28.57% at the early milestone; no later CF result is supplied.</p></article></div></section>
</main><footer><div class="wrap">Dense3B · Yin Li · Measured-results release 2026-10-18 · <a href="{REPO}">Source repository</a><br>Static, dependency-free presentation. No analytics or third-party scripts.</div></footer></body></html>\n'''

def generate(root, d):
    validate(d)
    files = {}
    canonical = json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    long_csv = csv_text(flatten(d))
    wide = [{'Benchmark': b['label'], **{s['tokens_b']+'B':score(v) for s,v in zip(d['stages'],b['values'])}} for b in d['benchmarks']]
    files['reports/measured/results.csv'] = long_csv
    files['reports/measured/results_wide.csv'] = csv_text(wide)
    files['docs/data/results.csv'] = long_csv
    files['docs/data/results_wide.csv'] = csv_text(wide)
    files['docs/data/results.json'] = canonical
    notes = '\n'.join('- '+x for x in d['protocol_notes'])
    source_sha = hashlib.sha256((root/SOURCE).read_bytes()).hexdigest()
    files['reports/measured/SOURCES.md'] = f'''# Results provenance and reporting conventions

Release: **{d['release_id']}** · Published **{d['publication_date']}**.

{d['source_statement']}

{d['evidence_statement']}

## Measurement sources

| Source | Coverage | Evidence available in this repository |
|---|---|---|
| Archived step 6499 evaluation | Early 6.82B-labelled row | [Checkpoint report](../checkpoints/dense3b-step_00006499.md), versioned summaries and sample recount records |
| Author-confirmed measurements, 18 October 2026 | 20B, 40B, 60B, 80B and 100B | Published scores and author corrections; raw outputs and later evaluator manifests are not attached |

## Precision and protocol

{notes}

## Later-run systems measurement

The author reported **{later_rate(d)} tokens/s** mean global training throughput during later training. The averaging interval and supporting logs were not supplied. This is **not** the same statistic as the **41,780 tokens/s** median from the 9 October archived snapshot (last at most 100 observations).

## Final author corrections

| Benchmark | Stage | Confirmed score (%) |
|---|---|---:|
'''+ '\n'.join('| '+c['benchmark_id']+' | '+c['stage_id']+' | '+c['confirmed_value_pct']+' |' for c in d['corrections']) + f'''

## Integrity

Canonical source: [`{SOURCE.as_posix()}`](../../{SOURCE.as_posix()}).

Source SHA-256: `{source_sha}`.

The page builder checks schema, ranges, stage coverage, exact corrections, Plus/base ordering and archived-count compatibility. These checks do not independently verify the author's later measurements. No model weights, synthetic per-example records, optimizer states, credentials or datasets are added by this release.

The historical evaluation and training archive remains at its original paths. Original capture receipts and artifact bytes are unchanged. Archive rebuilds validate sanctioned public redactions against the [existing audit](../../provenance/PUBLIC_NETWORK_REDACTION_AUDIT.json) and [two explicitly verified byte-identical aliases](../../provenance/PUBLIC_NETWORK_REDACTION_ALIASES.json). Unlisted changes still fail hash validation.

The [original publication validation record](validation.json) predates this throughput update. The updated reporting code contains metric validation and a regression test; those Python tests have not been rerun for this patch.
'''
    common = 'All percentages are measured results confirmed by the model author. The early row also has archived outputs; later raw outputs and evaluator manifests are not attached. [Source and protocol notes](measured/SOURCES.md).'
    files['reports/CONFIRMED_RESULTS.md'] = f'''# Dense3B — measured results through 100B tokens

[Repository overview](../README.md) · [Project page](https://yinli-systems.github.io/dense3b/) · [CSV](measured/results.csv) · [JSON](../{SOURCE.as_posix()}) · [Historical archive](RESULTS.md)

**3,237,529,600 dense parameters · 6 milestones · 9 primary benchmarks · 54 primary scores.**

{common}

## Complete checkpoint matrix

Units: percent. Numeric precision is retained exactly as reported.

{table(d, 'measured/')}

*Early MMLU uses MC5; later MMLU protocol details are not supplied. The final value is approximate. MBPP at 100B is reported as 52% at integer precision.*

## Additional archived result

MMLU-CF at 6.82B: **28.57%**. It is distinct from MMLU-MC5. No later MMLU-CF scores are supplied.

## Capability plots

'''+ '\n\n'.join(f'![{title}](../docs/assets/{g}.svg)' for g,title in GROUPS.items()) + '''

Points are observations, not fitted curves. Their presence on one plot does not independently establish matching evaluator protocols.

## Later-run training throughput

**{later_rate(d)} tokens/s** global average throughput, author-reported for later training. The averaging window and raw logs are not attached; the separate 9 October snapshot median is **41,780 tokens/s**. [Systems reporting notes](measured/SOURCES.md).

## Complete historical results and system metrics

[All archived checkpoint entries](RESULTS.md) · [Every archived metric/version](evaluation_history.csv) · [All training metrics](SYSTEMS.md) · [Raw training observations](training_all.csv) · [Sample recount records](sample_verification.json) · [Original snapshot homepage](../ARCHIVED_README.md).
'''
    for idx,s in enumerate(d['stages']):
        rows = '\n'.join('| '+b['label']+(' *' if b['id']=='mmlu' else '')+' | '+score(b['values'][idx])+' |' for b in d['benchmarks'])
        evidence = '[Archived evaluation at step 6499](../../checkpoints/dense3b-step_00006499.md).' if idx==0 else 'Source: author-confirmed measurements, 18 October 2026. Later evaluator manifests and raw per-example outputs are not attached.'
        files[f'reports/measured/checkpoints/{s["id"]}.md'] = f'''# Dense3B · {s['tokens_b']}B training tokens

[All measured milestones](../../CONFIRMED_RESULTS.md) · [Source notes](../SOURCES.md) · [Machine-readable source](../../../{SOURCE.as_posix()})

Training-token label: **{s['tokens_b']}B**, reported by the author. Model: **3,237,529,600 dense parameters**.

{evidence}

| Benchmark | Measured score (%) |
|---|---:|
{rows}

*Early MMLU is MC5; later protocol details are not assigned without a manifest. The final MMLU result remains approximate. MBPP 100B retains its integer-percentage precision.*

No checkpoint step, exact sample count or weight hash is assigned to a later milestone without source evidence. Weights are not included in this Git release.
'''
    files['README.md'] = f'''# Dense3B

### 3.238B dense parameters. 100B training tokens. Measured checkpoint results.

An end-to-end dense-model pretraining project with training code, data preparation, checkpoint measurements and archived execution evidence.

**[Project page](https://yinli-systems.github.io/dense3b/) · [Full results](reports/CONFIRMED_RESULTS.md) · [CSV](reports/measured/results.csv) · [JSON]({SOURCE.as_posix()}) · [Model card](MODEL_CARD.md) · [Historical archive](reports/RESULTS.md)**

## 100B-token results

| HumanEval | HumanEval+ | GSM8K | ARC-Challenge | MBPP+ | MMLU* |
|---:|---:|---:|---:|---:|---:|
| **49.4%** | **46.3%** | **59.8%** | **60.1%** | **51.3228%** | **≈52.0%** |

## Measured checkpoint matrix

All values below are author-confirmed measurements. Percentages retain the supplied precision. [Measurement sources and evaluation conventions](reports/measured/SOURCES.md) distinguish archived raw evidence from later author-confirmed reports.

{table(d,'reports/measured/')}

*Early MMLU uses MC5; later MMLU evaluator manifests are not attached. The final MMLU value is approximate. MBPP at 100B is reported at integer precision (52%). The 6.82B token label is rounded.*

The final author corrections include **HumanEval+ 40B: 39.0%**, **MBPP 20B/40B/60B: 38.4% / 42.9% / 46.3%**, **CRUXEval-I 60B: 30.750%**, and **MBPP+ 100B: 51.3228%**.

![Code generation measured checkpoint scores](docs/assets/code_generation.svg)

[Code reasoning and mathematics/knowledge plots](reports/CONFIRMED_RESULTS.md#capability-plots).

## Training implementation and systems

**Later-run average global throughput (author-reported): {later_rate(d)} tokens/s.** The averaging window and supporting logs were not supplied; this is separate from the older snapshot median.

The following measurements belong to the **9 October 2026 archived snapshot**, not the complete 100B run.

| Archived field | Value |
|---|---:|
| Dense parameters, including embeddings | 3,237,529,600 |
| Training GPUs | 8 × NVIDIA GeForce RTX 4090 |
| Model FLOP utilization, median of last ≤100 records | 68.09% |
| Global training throughput, same window | 41,780 tokens/s |
| Peak allocated memory, latest logged observation | 22.387 GB |

MFU is the implementation's model-FLOP estimate, not device utilization. Loop throughput excludes unrecorded stalls and downtime. [All system metrics and training curves](reports/SYSTEMS.md) · [Raw observations](reports/training_all.csv) · [Recorded run metadata](reports/run_index.json).

## Repository map

| Area | Contents |
|---|---|
| [`results/measured/`](results/measured/) | Canonical, precision-preserving author-confirmed measurements |
| [`reports/measured/`](reports/measured/) | CSV exports, source notes and six milestone pages |
| [`docs/`](docs/) | Static project site and three capability plots |
| [`model/`](model/) · [`train/`](train/) | Architecture, optimization, parallelism and training |
| [`code3b/`](code3b/) · [`data_pipeline/`](data_pipeline/) | Filtering, decontamination, deduplication and token packing |
| [`bench/`](bench/) · [`tokenizer/`](tokenizer/) | Evaluation code and tokenizer artifacts |
| [`evidence/`](evidence/) · [`provenance/`](provenance/) | Original source captures, logs, manifests and hash inventories |
| [`reports/RESULTS.md`](reports/RESULTS.md) | All 13 archived checkpoint entries and 607 archived metric rows |

## Rebuild the presentation

The measured-results presentation uses only the Python standard library; it does not execute model evaluation.

```bash
python3 scripts/build_measured_results.py --root .
python3 scripts/build_measured_results.py --root . --check
python3 -m unittest discover -s tests/publication -v
python3 scripts/verify_publication.py
```

To rebuild the original archive and then refresh the current results presentation:

```bash
python3 -m pip install -r requirements-report.txt
python3 scripts/build_all_results.py --root .
```

The archive builder automatically refreshes the current measured-results pages when the canonical release is present. Original raw evidence is not rewritten by the new presentation builder.

## Availability and provenance

The author confirmed all milestone scores as measurements on **18 October 2026**. Early scores have archived evaluation outputs; later per-example logs, evaluator manifests and weight hashes are not attached to this release. No independent rerun of later evaluations is claimed.

Model weights, optimizer states and data shards are not distributed in Git. Updated data-mixing weights and change points are not reported here. The [initial mixture](evidence/manifests/mix-v0/mix.json), [historical homepage](ARCHIVED_README.md), [evaluation protocols](reports/evaluation_versions.json) and [network-redaction notes](docs/PUBLICATION_SECURITY.md) remain available.
'''
    files['MODEL_CARD.md'] = f'''# Dense3B model card

## Identity

Dense3B is a dense pretraining model with **3,237,529,600 parameters** in the archived configuration. The author reports training milestones through **100 billion tokens**. The public repository contains implementation, measured results and archived evidence; model weights are not included.

## Evaluation

[The complete measured-results report](reports/CONFIRMED_RESULTS.md) contains 54 primary scores across six milestones, covering code generation, code reasoning, mathematics and knowledge. All results are author-confirmed measurements. The early milestone also has archived evaluation outputs. Later protocol manifests, per-example results and checkpoint hashes are not attached.

[Source notes](reports/measured/SOURCES.md) preserve original precision, six author corrections, the approximate final MMLU score, integer-rounded final MBPP score and the separation of MMLU-MC5 from MMLU-CF. Scores across different or unarchived protocols should not be treated as a controlled comparison.

## Training and data

The archived training code and run metadata describe RTX 4090-based training. Original source proportions, repetition factors and data processing are documented in the [initial mixture manifest](evidence/manifests/mix-v0/mix.json) and [`code3b/`](code3b/). The later source mixture, its switch points and full-run hardware utilization have not been provided for this release. The author separately reported a later-run average global throughput of **{later_rate(d)} tokens/s**. No averaging window or supporting logs were supplied; it must not be compared as if it were the archived **41,780 tokens/s median**.

## Scope and limitations

These benchmarks describe the evaluated tasks; they do not establish deployment reliability, absence of benchmark overlap, general instruction-following ability, or performance under an unreported decoding setup. No new training, GPU evaluation or external model comparison is performed by the publication scripts.

## Reproduction and access

Pages and tables can be rebuilt from [`{SOURCE.as_posix()}`]({SOURCE.as_posix()}) with `python3 scripts/build_measured_results.py`. Repeating model evaluation additionally requires weights, datasets and the appropriate protocol; no weight availability is asserted. Source datasets retain their own terms; this model card grants no additional license.
'''
    files['reports/README.md'] = '''# Dense3B — report navigation

## Current measured results

[All measured milestones through 100B](CONFIRMED_RESULTS.md) · [Source notes](measured/SOURCES.md) · [Long CSV](measured/results.csv) · [Matrix CSV](measured/results_wide.csv) · [Canonical JSON](../results/measured/2026-10-18.json).

## Historical evidence

[All archived checkpoint entries](RESULTS.md) · [All archived metric values and versions](evaluation_history.csv) · [System metrics](SYSTEMS.md) · [Raw training observations](training_all.csv) · [Sample recounts](sample_verification.json) · [Evaluator manifests](evaluation_versions.json) · [Capture issues](sync_issues.json).

The historical capture is dated 9 October 2026. The measured-results release is dated 18 October 2026. These source classes are retained separately.
'''
    files['docs/index.html'] = site(d)
    files['docs/assets/style.css'] = CSS.strip()+'\n'
    files['docs/assets/app.js'] = JS
    files['docs/.nojekyll'] = ''
    for group in GROUPS:
        files['docs/assets/'+group+'.svg'] = svg_chart(d,group)
    manifest = {'schema_version':1, 'source_path':SOURCE.as_posix(), 'source_sha256':source_sha, 'measurement_count':54, 'auxiliary_measurement_count':1, 'model_evaluation_executed':False, 'generated_files':{p:hashlib.sha256(text.encode()).hexdigest() for p,text in sorted(files.items())}}
    files['reports/measured/publication_manifest.json'] = json.dumps(manifest,indent=2,sort_keys=True)+'\n'
    return files

def build(root, check=False):
    root = Path(root).resolve()
    d = json.loads((root/SOURCE).read_text(encoding='utf-8'))
    files = generate(root,d)
    different=[]
    for rel,text in files.items():
        path=root/rel
        if check:
            if not path.is_file() or path.read_bytes()!=text.encode('utf-8'):
                different.append(rel)
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(text,encoding='utf-8')
    if different:
        raise ValueError('Generated output differs: '+', '.join(different))
    return {'primary_measurements':54, 'auxiliary_measurements':1, 'generated_files':len(files), 'source_sha256':hashlib.sha256((root/SOURCE).read_bytes()).hexdigest(), 'mode':'check' if check else 'write', 'model_evaluation_executed':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    print(json.dumps(build(a.root,a.check),indent=2))
