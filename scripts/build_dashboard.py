#!/usr/bin/env python3
"""Rebuild auditable reports from archived evidence; never contacts a training service."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, re, statistics
from pathlib import Path
from typing import Any

def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + '\n')

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})

def identify_checkpoint(ref: str) -> tuple[str | None, str | None]:
    if 'code3b-train/' in ref: return 'dense3b', 'dense3b'
    if '/moe1b/' in ref: return 'moe1b', 'moe1b'
    if '/reason3b/' in ref: return 'reason3b', 'r3b_s'
    if '/routing/' in ref:
        for p in ['dense1b_l', 'moe1b_r', 'dense1b_s']:
            if p in ref: return 'dense-moe-1b', p
    return None, None

def verified_tokens(meta: dict) -> int | None:
    n, seq = meta.get('samples_consumed'), meta.get('args', {}).get('seq_len')
    return int(n * seq) if finite(n) and finite(seq) and n >= 0 and seq > 0 else None

def reconstruct(runs: list[dict]) -> list[dict]:
    """Explicit display policy: later jobs replace the earlier suffix at their first logged step."""
    retained: dict[str, list[dict]] = {}
    for run in sorted(runs, key=lambda r: (r['job_id'], r['run_id'])):
        if not run['production'] or not run['rows']: continue
        variant = run['variant']
        rows = run['rows']
        start = min(r['step'] for r in rows)
        old = [r for r in retained.get(variant, []) if r['step'] < start]
        latest = {r['step']: r for r in rows}
        retained[variant] = old + [latest[s] for s in sorted(latest)]
    return [r for v in sorted(retained) for r in retained[v]]

def moving(values: list[float], window: int = 50) -> list[float]:
    q: collections.deque = collections.deque(); total = 0.0; out = []
    for value in values:
        q.append(value); total += value
        if len(q) > window: total -= q.popleft()
        out.append(total / len(q))
    return out

def load_metrics(root: Path, cfg: dict, issues: list[dict]) -> list[dict]:
    runs = []
    for f in sorted((root / 'evidence/raw/results').rglob('metrics.jsonl')):
        name = f.parent.name
        job = int(re.search(r'(\d+)$', name).group(1)) if re.search(r'(\d+)$', name) else 0
        variant = next((v for v in cfg['presets'] if name.startswith(v + '-')), cfg['presets'][0])
        logs = sorted((root/'evidence/raw/logs').glob('*' + str(job) + '.out'))
        production = 'train-' in name and not any(re.search(r'smoke|sweep|prof|zeq|kern|nccl', p.name) for p in logs)
        starts = []
        for log in logs:
            for text in log.read_text(errors='replace').splitlines():
                if not text.startswith('{'): continue
                try: obj = json.loads(text)
                except json.JSONDecodeError: continue
                if isinstance(obj, dict) and obj.get('event') == 'start': starts.append(obj)
        event = starts[-1] if starts else {}
        rows = []
        for line, text in enumerate(f.read_text(errors='replace').splitlines(), 1):
            try: obj = json.loads(text)
            except json.JSONDecodeError:
                issues.append({'kind': 'malformed_metric_line', 'source': str(f.relative_to(root)), 'line': line}); continue
            if not isinstance(obj, dict) or not finite(obj.get('step')) or not finite(obj.get('loss')): continue
            row = {'variant': variant, 'run_id': name, 'job_id': job, 'production': production,
                   'source': str(f.relative_to(root)), 'source_line': line}
            row.update({k: v if not isinstance(v, float) or math.isfinite(v) else None for k, v in obj.items()})
            row['step'] = int(row['step']); rows.append(row)
        runs.append({'run_id': name, 'job_id': job, 'variant': variant, 'production': production,
                     'source': str(f.relative_to(root)), 'sha256': sha(f), 'logs': [str(p.relative_to(root)) for p in logs],
                     'start_event': event, 'rows': rows})
    return runs

def load_evals(root: Path, cfg: dict, issues: list[dict]) -> tuple[list[dict], list[dict]]:
    metas: dict[tuple, list[tuple]] = collections.defaultdict(list)
    checkpoint_rows = []
    for f in sorted((root/'evidence/checkpoints').rglob('meta.json')):
        try: d = json.loads(f.read_text())
        except json.JSONDecodeError: continue
        preset, step = d.get('args', {}).get('preset'), d.get('step')
        tokens = verified_tokens(d)
        metas[(preset, step)].append((tokens, str(f.relative_to(root))))
        checkpoint_rows.append({'variant': preset, 'step': step, 'samples_consumed': d.get('samples_consumed'),
                                'sequence_length': d.get('args', {}).get('seq_len'), 'tokens_verified': tokens,
                                'source': str(f.relative_to(root)), 'sha256': sha(f), 'config': d.get('config'), 'args': d.get('args')})
    rows = []; seen = set()
    source_times = {}
    for record in json.loads((root/'provenance/snapshot.json').read_text())['records']:
        source_times[record['repo_path']] = record.get('mtime_ns', 0) // 1_000_000_000
    imported_path = root/'provenance/imported-results.json'
    if imported_path.exists():
        for record in json.loads(imported_path.read_text()):
            source_times[record['repo_path']] = record.get('source_mtime_ns', 0) // 1_000_000_000
    paths = sorted((root/'evidence/raw/results').glob('*.json')) + sorted((root/'evidence/imported').rglob('*.json'))
    for f in paths:
        if '.details.' in f.name: continue
        try: d = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeError): continue
        if not isinstance(d, dict) or not isinstance(d.get('tasks'), dict): continue
        digest = sha(f)
        if digest in seen: continue
        seen.add(digest)
        checkpoint = str(d.get('ckpt', d.get('checkpoint', '')))
        owner, variant = identify_checkpoint(checkpoint)
        if owner != cfg['name']:
            issues.append({'kind': 'foreign_or_unidentified_evaluation', 'source': str(f.relative_to(root)), 'checkpoint': checkpoint, 'owner': owner}); continue
        step = d.get('step'); manifest = d.get('manifest', {}) or {}
        evidence = metas.get((variant, step), [])
        token_set = {v[0] for v in evidence if v[0] is not None}
        tokens = next(iter(token_set)) if len(token_set) == 1 else None
        reported = d.get('tokens_B')
        mismatch = finite(reported) and tokens is not None and abs(reported - tokens/1e9) > 0.02
        if mismatch:
            issues.append({'kind': 'token_count_mismatch', 'source': str(f.relative_to(root)), 'step': step,
                           'reported_tokens_B': reported, 'verified_tokens_B': tokens/1e9, 'metadata': [v[1] for v in evidence]})
        if len(token_set) > 1: issues.append({'kind': 'conflicting_checkpoint_metadata', 'variant': variant, 'step': step})
        protocol = {k: manifest.get(k) for k in ['protocol', 'protocol_sha256', 'eval_suite_sha256', 'generate_sha256', 'tokenizer_sha256', 'dtype', 'temperature', 'max_new', 'stops', 'exec_timeout_s'] if k in manifest}
        protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12] if protocol else 'unversioned'
        for task, result in d['tasks'].items():
            if not isinstance(result, dict): continue
            if result.get('error') or result.get('skipped'):
                issues.append({'kind': 'task_not_completed', 'source': str(f.relative_to(root)), 'task': task, 'detail': result})
            for metric, value in result.items():
                if metric in ['n', 'seconds', 'time', 'total', 'correct', 'passed', 'failed', 'tokens', 'count', 'samples'] or not finite(value) or not 0 <= value <= 1: continue
                rows.append({'variant': variant, 'step': step, 'task': task, 'metric': metric, 'value': value, 'n': result.get('n'),
                             'tokens_reported_B': reported, 'tokens_verified_B': tokens/1e9 if tokens is not None else None,
                             'token_count_mismatch': mismatch, 'protocol_id': protocol_id, 'protocol_name': manifest.get('protocol', 'not recorded'),
                             'checkpoint': checkpoint, 'source': str(f.relative_to(root)), 'source_sha256': digest,
                             'source_mtime': source_times.get(str(f.relative_to(root)), 0)})
    return rows, checkpoint_rows

def create_plots(root: Path, cfg: dict, runs: list[dict], retained: list[dict], evals: list[dict]) -> list[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt'] = 'pretraining-publication-v1'
    out = root/'reports/figures'; out.mkdir(parents=True, exist_ok=True)
    made = []
    def save(fig, name, title, ylabel, xlabel='Optimizer step'):
        ax = fig.axes[0]; ax.set_title(cfg['name'] + ' | ' + title, loc='left', pad=15)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True, alpha=.2)
        handles, labels = ax.get_legend_handles_labels()
        if handles: ax.legend(fontsize=8, loc='best')
        fig.tight_layout(); fig.savefig(out/(name+'.png'), dpi=160)
        fig.savefig(out/(name+'.svg'), metadata={'Date': None}); plt.close(fig); made.append(name)
    for field, title, ylabel, scale in [
        ('loss','Training loss — retained resume sequence','Cross-entropy',1),
        ('val_loss','Validation loss — recorded evaluations only','Cross-entropy',1),
        ('tok_per_s','Global training-loop throughput','Tokens / second',1),
        ('mfu','Reported model FLOP utilization','MFU (%)',100),
        ('lr','Learning-rate schedule','Learning rate',1),
        ('grad_norm','Gradient norm','Gradient norm',1),
        ('max_mem_gb','Peak allocated GPU memory','GB (as reported)',1),
        ('moe_load_max_mean','Routing load imbalance','Maximum / mean expert load',1),
        ('t_data_wait_max','Maximum-rank data wait','Seconds',1)]:
        fig = plt.figure(figsize=(10.8,5.4)); ax = fig.add_subplot(111); any_data = False
        for variant in cfg['presets']:
            rr = [r for r in retained if r['variant']==variant and finite(r.get(field))]
            if not rr: continue
            any_data=True; x=[r['step'] for r in rr]; y=[r[field]*scale for r in rr]
            ax.plot(x,y,linewidth=.65,alpha=.65,label=variant+' recorded')
            if field=='loss' and len(y)>10: ax.plot(x,moving(y),linewidth=1.8,label=variant+' trailing mean (50 observations)')
        if any_data: save(fig,field,title,ylabel)
        else: plt.close(fig)
    fig=plt.figure(figsize=(10.8,5.4)); ax=fig.add_subplot(111); any_data=False
    for run in runs:
        if not run['production'] or not run['rows']: continue
        rr=run['rows']; ax.plot([r['step'] for r in rr],[r['loss'] for r in rr],linewidth=.7,alpha=.7,label=run['run_id']);any_data=True
    if any_data:save(fig,'loss_all_runs','All production runs — including replaced suffixes','Cross-entropy')
    else:plt.close(fig)
    groups=[('code_scores','Code benchmark checkpoints',lambda r:r['task'] in ['humaneval','mbpp']),
            ('reasoning_scores','Reasoning / execution benchmark checkpoints',lambda r:r['task'] in ['cruxeval','gsm8k','lcb','fim']),
            ('mc_accuracy','Multiple-choice accuracy (not normalized)',lambda r:r['metric']=='acc'),
            ('mc_normalized','Multiple-choice normalized accuracy',lambda r:r['metric']=='acc_norm')]
    for name,title,accept in groups:
        rr=[r for r in evals if accept(r) and finite(r.get('step'))]
        if not rr:continue
        fig=plt.figure(figsize=(10.8,5.8));ax=fig.add_subplot(111)
        groups2=collections.defaultdict(list)
        for r in rr:groups2[(r['variant'],r['task'],r['metric'])].append(r)
        for key,values in sorted(groups2.items()):
            # Points, not lines: differing evaluator versions/sample counts must not imply a controlled trend.
            ax.scatter([r['step'] for r in values],[r['value']*100 for r in values],s=28,alpha=.8,label='/'.join(key))
        ax.set_ylim(0,100);save(fig,name,title+' (protocols recorded in tables)','Score (%)')
    return made

def build(root: Path) -> dict:
    cfg=json.loads((root/'publication.json').read_text()); snap=json.loads((root/'provenance/snapshot.json').read_text())
    issues=[]; runs=load_metrics(root,cfg,issues); retained=reconstruct(runs)
    evals,checkpoints=load_evals(root,cfg,issues)
    reports=root/'reports'; reports.mkdir(exist_ok=True)
    write_csv(reports/'training_all.csv',[r for run in runs for r in run['rows']])
    write_csv(reports/'training_retained.csv',retained)
    write_csv(reports/'evaluation_history.csv',evals)
    write_csv(reports/'checkpoint_registry.csv',checkpoints)
    run_index=[{k:v for k,v in r.items() if k!='rows'} | {'observations':len(r['rows']),'first_step':r['rows'][0]['step'] if r['rows'] else None,'last_step':r['rows'][-1]['step'] if r['rows'] else None} for r in runs]
    dump(reports/'run_index.json',run_index);dump(reports/'checkpoint_registry.json',checkpoints)
    latest={}
    for r in sorted(evals,key=lambda r:(r['step'] if finite(r['step']) else -1,r['source_mtime'],r['source'])):
        latest[(r['variant'],r['task'],r['metric'])]=r
    latest_rows=list(latest.values());write_csv(reports/'evaluation_latest.csv',latest_rows)
    summary={'model':cfg['name'],'snapshot_utc':snap['snapshot_utc'],'variants':{},'training_observations':sum(len(r['rows']) for r in runs),'retained_observations':len(retained),'evaluation_metric_records':len(evals),'issues':issues}
    for v in cfg['presets']:
        rr=[r for r in retained if r['variant']==v];last=rr[-100:]
        summary['variants'][v]={'latest_step':rr[-1]['step'] if rr else None,'observations':len(rr),'recent_loss_mean':statistics.mean(r['loss'] for r in last) if last else None,'recent_global_tok_per_s_median':statistics.median(r['tok_per_s'] for r in last if finite(r.get('tok_per_s'))) if any(finite(r.get('tok_per_s')) for r in last) else None}
    figs=create_plots(root,cfg,runs,retained,evals);summary['figures']=figs;dump(reports/'summary.json',summary);dump(reports/'data_audit.json',issues)
    lines=['# '+cfg['name'],'',cfg['description'],'','**Evidence snapshot (UTC): '+snap['snapshot_utc']+'.** This is an archived measurement, not a live status feed.','',
           '## Experiment coverage','', '| Variant | Latest retained step | Retained observations | Mean loss, last ≤100 | Median global tok/s, last ≤100 |','|---|---:|---:|---:|---:|']
    for v,s in summary['variants'].items():
        lines.append('| '+v+' | '+('—' if s['latest_step'] is None else str(s['latest_step']))+' | '+str(s['observations'])+' | '+('—' if s['recent_loss_mean'] is None else f"{s['recent_loss_mean']:.4f}")+' | '+('—' if s['recent_global_tok_per_s_median'] is None else f"{s['recent_global_tok_per_s_median']:,.0f}")+' |')
    if cfg.get('parameter_counts'):
        lines += ['', '### Archived architecture presets', '', '| Variant | Total parameters | Active parameters |', '|---|---:|---:|']
        for variant, counts in cfg['parameter_counts'].items():
            lines.append(f"| {variant} | {counts['total']:,} | {counts['active']:,} |")
        lines += ['', 'Counts are computed by the archived `model/config.py` accounting function, including embeddings. They are configuration counts, not measurements of task quality.']
    if cfg.get('parameter_counts'):
        lines += ['', '### Archived architecture presets', '', '| Variant | Total parameters | Active parameters |', '|---|---:|---:|']
        for variant, counts in cfg['parameter_counts'].items():
            lines.append(f"| {variant} | {counts['total']:,} | {counts['active']:,} |")
        lines += ['', 'Counts are computed by the archived `model/config.py` accounting function, including embeddings. They are configuration counts, not measurements of task quality.']
    lines+=['','A missing series means no completed training metric record was found in this snapshot; it is not a zero result. Full failed-run logs and all configured research variants are preserved. Throughput is the training-loop measurement, not end-to-end wall-clock throughput.','',
            '## Learning curves','']
    for fig in ['loss','val_loss','tok_per_s','code_scores']:
        if fig in figs:lines+=['!['+fig+'](reports/figures/'+fig+'.png)','']
    lines+=['[All figures](reports/figures/) · [All production runs including rollbacks](reports/figures/loss_all_runs.png) · [Raw training CSV](reports/training_all.csv) · [Run index](reports/run_index.json)','',
            '## Latest observed evaluation metrics','',
            'Selection is by checkpoint step, then archived archived source timestamp—not by the highest score. Different protocols, sample counts, tokenizers and custom MMLU variants are not treated as directly comparable. Benchmark figures use points rather than connecting different evaluations.','',
            '| Variant | Task / metric | Step | Score | n | Verified tokens (B) | Evidence |','|---|---|---:|---:|---:|---:|---|']
    for r in sorted(latest_rows,key=lambda r:(r['variant'],r['task'],r['metric'])):
        tokens='—' if r['tokens_verified_B'] is None else f"{r['tokens_verified_B']:.3f}"
        lines.append(f"| {r['variant']} | {r['task']} / {r['metric']} | {r['step']} | {100*r['value']:.2f}% | {r['n'] if r['n'] is not None else '—'} | {tokens} | [JSON]({r['source']}) |")
    if not latest_rows:lines+=['| — | No completed supported evaluation found | — | — | — | — | — |']
    mismatches=[i for i in issues if i['kind']=='token_count_mismatch']
    lines+=['','### Evidence cautions','',
            f"The audit found **{len(mismatches)} evaluation summaries with token counts inconsistent with retained checkpoint metadata**. Original JSON files are unchanged. Verified counts use `samples_consumed × args.seq_len`; unsupported counts stay blank. Step is used on plot axes rather than substituting an assumed sequence length.",'',
            '[Complete chronological metrics](reports/evaluation_history.csv) · [Checkpoint registry](reports/checkpoint_registry.json) · [Audit findings](reports/data_audit.json)','',
            '## Code and reproduction','',
            '`model/` contains architecture and kernels; `train/` contains optimizers, data loading, checkpointing and distributed training; `bench/` and `tests/` retain evaluation, profiling and correctness code. `operations/` retains the deployed cluster helpers. Data preparation and tokenizer files, when available, are under `data_pipeline/`, `code3b/` and `tokenizer/`. Original cluster paths remain in deployed scripts so the archived implementation is not silently rewritten. Adapt those paths explicitly for another machine.','',
            '```bash','python3 -m venv .venv-report','source .venv-report/bin/activate','python -m pip install -r requirements-report.txt','python scripts/verify_publication.py','python -m unittest discover -s tests/publication -v','python scripts/build_dashboard.py --root .','```','',
            'Report generation uses archived local files only and does not submit jobs or access GPUs. Publication checks do not constitute a rerun of CUDA training or model benchmarks.','',
            '## Provenance and scope','',
            '[Snapshot and file hashes](provenance/snapshot.json) · [Imported result routing](provenance/imported-results.json) · [Extraction methodology](reports/METHODOLOGY.md)','',
            'Source, configurations, original metrics, evaluator manifests, sample-level outputs and checkpoint metadata are versioned. Model weights, optimizer tensors, token shards, datasets, credentials, caches and installed binary dependencies are intentionally not Git payloads. Historical code versions remain in Git commits. Training losses across different data recipes/tokenizers are not a ranking of model quality.','']
    (root/'README.md').write_text('\n'.join(lines))
    (reports/'METHODOLOGY.md').write_text('''# Report extraction policy

## Training traces
All JSONL observations retain their source path and 1-based line number. The raw table includes diagnostic runs. The main figure excludes explicitly named smoke/profiling/sweep runs, orders production jobs by numeric job ID, and replaces the earlier suffix at the first logged step of a later run. This is a transparent display reconstruction, not independent verification of optimizer-state ancestry. The all-runs figure and raw CSV retain superseded tails. No interpolated training observations are created. Smoothing is a trailing arithmetic mean of at most 50 retained observations. Sequence length, world size and tokens per step are retained from start events in the run index; those fields are reported runtime values, not a physical topology audit.

## Evaluations
Model identity comes from the recorded checkpoint path, not the directory holding the result. Foreign and unidentified summaries are excluded and logged. Exact duplicate summary files are counted once. Task/metric names, sample counts, declared protocols and evaluator hashes are retained; unversioned evaluations remain labeled unversioned. Accuracy and normalized accuracy remain separate. Latest tables choose largest step then newest archived source timestamp, never the highest score. Checkout filesystem timestamps are never used. Points do not establish a controlled trend across protocol changes. No missing benchmark is entered as zero. Original summary scores are rounded source values; sample-level outputs are preserved separately and were not rescored for this publication.

## Token accounting
Only checkpoint metadata with recorded samples_consumed and args.seq_len establishes a verified token count. Formula: samples_consumed * args.seq_len. A checkpoint's declared configuration and sample counter are evidence, not a new audit of the original data loader. The source tokens_B field is also retained. Differences exceeding 0.02 billion tokens are flagged (allowing source rounding). Missing metadata is not reconstructed from a guessed batch size or context length.

## Operational measurements
Throughput/MFU/memory values are copied from logged training metrics. They do not include every checkpoint stall, evaluation, queue delay or downtime. MFU is the source implementation's estimate; no cross-hardware equivalence is asserted. Missing resource samples are not filled. CUDA correctness, benchmark execution, dataset availability and remote checkpoint completeness were not retested during publication.

## Rebuild and integrity
Run scripts/verify_publication.py before rebuilding. It verifies the archived source/evidence SHA-256 inventory and import receipts. requirements-report.txt captures the reporting environment. Unit tests exercise extraction policies independently of CUDA. All training/evaluation jobs remain outside this reporting workflow.
''')
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path.cwd());args=parser.parse_args()
    result=build(args.root.resolve())
    print(json.dumps({k:result[k] for k in ['model','snapshot_utc','training_observations','retained_observations','evaluation_metric_records','figures']},indent=2))
