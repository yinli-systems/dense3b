#!/usr/bin/env python3
"""Export reported evaluation aggregates in the archived suite/MC task structure.

This does not reconstruct an evaluator run, sample outcomes or missing metadata.
Canonical author-confirmed scores are the only numeric input. Outputs live outside
raw evidence directories and explicitly identify themselves as summary exports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

SOURCE = Path('results/measured/2026-10-18.json')
DEST = Path('results/reported_evaluations')
STAGES = ('20B', '40B', '60B', '80B', '100B')
BENCHMARKS = ('humaneval', 'humaneval_plus', 'mbpp', 'mbpp_plus',
              'cruxeval_i', 'cruxeval_o', 'gsm8k', 'arc_challenge', 'mmlu')
REFERENCES = {
    'suite': 'evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/suite.json',
    'mc': 'evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json',
}
GROUPS = {
    'suite': {
        'humaneval': {'humaneval': 'humaneval', 'humaneval_plus': 'humaneval_plus'},
        'mbpp': {'mbpp': 'mbpp', 'mbpp_plus': 'mbpp_plus'},
        'cruxeval': {'cruxeval_o': 'cruxeval_o', 'cruxeval_i': 'cruxeval_i'},
        'gsm8k': {'gsm8k': 'gsm8k'},
    },
    'mc': {
        'arc_challenge': {'acc': 'arc_challenge'},
        'mmlu': {'acc': 'mmlu'},
    },
}
PCT = re.compile(r'(?:0|[1-9]\d*)(?:\.\d+)?\Z')
MAX_SOURCE_BYTES = 2 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(data: bytes) -> dict[str, Any]:
    require(len(data) <= MAX_SOURCE_BYTES, 'Reported-score source is too large')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    result = json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)
    require(isinstance(result, dict), 'Expected a JSON object')
    return result


def percentage(text: str) -> Decimal:
    require(isinstance(text, str) and PCT.fullmatch(text) is not None,
            'Percentage must retain its supplied decimal string')
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError('Invalid percentage') from exc
    require(value.is_finite() and 0 <= value <= 100, 'Percentage outside [0,100]')
    return value


def validate(source: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    require(source.get('schema_version') == 1, 'Unsupported source schema')
    require(source.get('model', {}).get('architecture') == 'dense', 'Wrong model architecture')
    require(type(source['model'].get('parameter_count')) is int and
            source['model']['parameter_count'] > 0, 'Invalid parameter count')
    stage_rows = source.get('stages', [])
    require(isinstance(stage_rows, list), 'Missing stages')
    stage_ids = [x['id'] for x in stage_rows]
    require(len(stage_ids) == len(set(stage_ids)), 'Duplicate stage')
    require(set(STAGES) <= set(stage_ids), 'Missing later-stage milestone')
    stage_map = {x['id']: x for x in stage_rows}
    for stage in STAGES:
        row = stage_map[stage]
        require(row['tokens_b'] == stage[:-1], 'Stage label and reported tokens differ')
        require(row['checkpoint_step'] is None and row['checkpoint_sha256'] is None,
                'New checkpoint evidence requires a separately sourced export adapter')
        require(row['token_count_basis'] == 'author_reported', 'Unexpected token-count source')
    benchmarks = source.get('benchmarks', [])
    require(isinstance(benchmarks, list), 'Missing benchmark source')
    ids = [b['id'] for b in benchmarks]
    require(len(ids) == len(set(ids)) and set(ids) == set(BENCHMARKS),
            'Unexpected or duplicate benchmark')
    lookup = {}
    for b in benchmarks:
        values = b.get('values', [])
        require(isinstance(values, list), 'Missing stage scores')
        seen = set()
        for v in values:
            stage = v['stage_id']
            require(stage not in seen, 'Duplicate score for a stage')
            seen.add(stage)
            if stage not in STAGES:
                continue
            percentage(v['value_pct'])
            require(type(v.get('approximate')) is bool, 'Approximation flag must be Boolean')
            require(v.get('rounding_note') in (None, 'integer_percentage'), 'Unknown precision qualifier')
            require(v.get('raw_outputs_attached') is False, 'Raw artifacts must not be silently summarized as absent')
            require(v.get('n_archived') is None and v.get('passed_archived') is None,
                    'Do not convert real count records into unverified summaries')
            origin = source.get('sources', {}).get(v.get('source'))
            require(isinstance(origin, dict) and origin.get('kind') == 'author_confirmed_measurement',
                    'Expected an explicit author-confirmed aggregate source')
            lookup[b['id'], stage] = v
        require(set(STAGES) <= seen, 'Missing reported benchmark score')
    require(len(lookup) == 45, 'Exactly 45 reported later-stage scores required')
    for stage in STAGES:
        for plus, base in (('humaneval_plus', 'humaneval'), ('mbpp_plus', 'mbpp')):
            require(percentage(lookup[plus, stage]['value_pct']) <=
                    percentage(lookup[base, stage]['value_pct']), 'Plus exceeds base')
    return lookup


def render_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def display(v: dict) -> str:
    return ('≈' if v['approximate'] else '') + v['value_pct'] + '%'


def stage_document(source: dict, lookup: dict, stage: str, group: str,
                   source_hash: str) -> dict:
    tasks, reported, qualifiers, origins = {}, {}, {}, {}
    for task, mappings in GROUPS[group].items():
        body = {}
        for metric, benchmark in mappings.items():
            v = lookup[benchmark, stage]
            # Unit conversion only, not a reconstructed correct/total ratio.
            body[metric] = float(percentage(v['value_pct']) / Decimal(100))
            reported[benchmark] = v['value_pct']
            qualifiers[benchmark] = {'approximate': v['approximate'],
                                     'rounding_note': v['rounding_note']}
            origins[benchmark] = v['source']
        if task == 'gsm8k':
            body['strict_accuracy'] = None
        if group == 'mc':
            body['acc_norm'] = None
        body.update(n=None, seconds=None)
        if task in ('humaneval', 'mbpp'):
            body['official_result_file'] = None
        tasks[task] = body
    return {
        'schema_version': 1,
        'record_kind': 'author_reported_aggregate_summary',
        'is_original_evaluation_output': False,
        'stage_id': stage,
        'ckpt': None,
        'step': None,
        'preset': 'dense3b',
        'tokens_B': int(stage[:-1]),
        'token_count_basis': 'author_reported',
        'manifest': {
            'protocol': None,
            'checkpoint_model_sha256': None,
            'metadata_sha256': None,
            'tokenizer_sha256': None,
            'evaluator_revision': None,
            'dataset_hashes': None,
            'dtype': None,
            'temperature': None,
            'batch': None,
            'max_new': None,
            'job_id': None,
        },
        'tasks': tasks,
        'reported_percent': reported,
        'reporting_qualifiers': qualifiers,
        'provenance': {
            'canonical_source': SOURCE.as_posix(),
            'canonical_source_sha256': source_hash,
            'source_release_id': source['release_id'],
            'source_publication_date': source['publication_date'],
            'metric_source_ids': origins,
            'format_reference': REFERENCES[group],
            'format_reference_only': True,
            'archived_protocol_inherited': False,
        },
        'evidence': {
            'raw_outputs_attached': False,
            'per_example_files': [],
            'model_evaluation_executed_by_exporter': False,
            'evaluation_started_at': None,
            'evaluation_finished_at': None,
        },
        'notes': [
            'Reported aggregate scores reformatted into task groups; this file is not an original run log.',
            'Percent strings and qualifiers retain the supplied precision; task numbers are percentage/100.',
            'Missing counts, timestamps, runtime settings and weight hashes remain null.',
        ] + (['MMLU is not assigned to MC5 or CF because the later evaluator protocol was not supplied.']
             if group == 'mc' else []),
    }


def generate(root: Path) -> dict[str, str]:
    raw = (root / SOURCE).read_bytes()
    source = load_json(raw)
    lookup = validate(source)
    source_hash = digest(raw)
    labels = {b['id']: b['label'] for b in source['benchmarks']}
    files = {}
    for group, reference in REFERENCES.items():
        require((root / reference).is_file(), 'Missing format reference: ' + group)
    for stage in STAGES:
        folder = DEST / stage
        for group in GROUPS:
            files[(folder / (group + '.json')).as_posix()] = render_json(
                stage_document(source, lookup, stage, group, source_hash))
        rows = '\n'.join('| ' + labels[b] + ' | ' + display(lookup[b, stage]) + ' |'
                         for b in BENCHMARKS)
        files[(folder / 'README.md').as_posix()] = f'''# Dense3B · {stage} reported evaluation summary

[All stages](../README.md) · [suite.json](suite.json) · [mc.json](mc.json) · [Canonical reported scores](../../measured/2026-10-18.json)

The values below are the model author's confirmed aggregate measurements, reformatted into the same task grouping as the early evaluation summaries. **These exports are not original evaluator output or per-example logs.** No new model evaluation was executed.

| Benchmark | Reported score |
|---|---:|
{rows}

`n`, checkpoint step/hash, run settings, elapsed time and job identifiers are null because the corresponding later-run records have not been supplied. No early-run metadata was copied into those fields. MMLU remains protocol-unspecified rather than being relabelled MC5 or CF.

Percent strings preserve the source precision: final MBPP is integer-rounded and final MMLU is approximate. [Recover or rerun actual evidence](../../../EVALUATION_EVIDENCE.md).
'''
    header = '| Benchmark | ' + ' | '.join('[' + s + '](' + s + '/README.md)' for s in STAGES) + ' |'
    matrix = '\n'.join([header, '|---|' + '---:|' * 5] + [
        '| ' + labels[b] + ' | ' + ' | '.join(display(lookup[b, s]) for s in STAGES) + ' |'
        for b in BENCHMARKS])
    files[(DEST / 'README.md').as_posix()] = f'''# Dense3B · reported evaluation summaries

**Five stages · nine benchmarks · 45 reported aggregate scores.**

These are precision-preserving exports of the author's confirmed scores, using the task grouping of the archived `suite.json` / `mc.json`. **They are not recovered original logs, and contain no generated per-example outcomes.** Actual per-example evidence remains in the separate [early archive](../../reports/checkpoints/dense3b-step_00006499.md).

{matrix}

## Contents and interpretation

Each stage directory contains `suite.json` (HumanEval, MBPP, CRUXEval and GSM8K), `mc.json` (ARC-Challenge and protocol-unspecified MMLU), and a readable summary. JSON task scores are fractions in [0,1]. `reported_percent` and `reporting_qualifiers` preserve exact published strings and approximation/rounding annotations. Unprovided counts, execution times, settings and model fingerprints are null, not zero or inferred values.

`manifest.json` hashes these exported files; those hashes identify **summary exports**, not an original model or evaluator run. `index.json` records five-stage coverage. The [source scores](../measured/2026-10-18.json) are not modified.

## Rebuild

```bash
python3 scripts/build_reported_evaluations.py --root .
python3 scripts/build_reported_evaluations.py --root . --check
```

The [evidence recovery tools](../../EVALUATION_EVIDENCE.md) deliberately reject these summaries as raw run evidence. Completion of original logs requires the actual outputs or a new evaluation of the exact checkpoint. There is no separate 6.2B score set in this export; 6.2B is not substituted for step 6499 (~6.82B).
'''
    index = {
        'schema_version': 1,
        'record_kind': 'author_reported_aggregate_summary_index',
        'canonical_source': SOURCE.as_posix(),
        'canonical_source_sha256': source_hash,
        'reported_metric_count': 45,
        'original_logs_created': 0,
        'model_evaluations_executed': 0,
        'stages': [{'stage_id': s, 'reported_metrics': 9, 'raw_outputs_attached': False,
                    'suite': s + '/suite.json', 'mc': s + '/mc.json', 'report': s + '/README.md'}
                   for s in STAGES],
    }
    files[(DEST / 'index.json').as_posix()] = render_json(index)
    manifest = {'schema_version': 1, 'kind': 'reported_summary_export_integrity',
                'source_path': SOURCE.as_posix(), 'source_sha256': source_hash,
                'generator': 'scripts/build_reported_evaluations.py',
                'files': {p: digest(text.encode('utf-8')) for p, text in sorted(files.items())}}
    files[(DEST / 'manifest.json').as_posix()] = render_json(manifest)
    return files


def safe_target(root: Path, relative: str) -> Path:
    root = root.resolve(strict=True)
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts and DEST in rel.parents,
            'Export path outside designated directory')
    p = root
    for part in rel.parts:
        p = p / part
        require(not p.is_symlink(), 'Refusing a symlink in output path')
    return p


def build(root: Path, output_root: Path | None = None, check: bool = False) -> dict:
    root = root.resolve(strict=True)
    files = generate(root)
    out = (output_root or root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    safe_target(out, (DEST / 'README.md').as_posix())
    if (out / DEST).exists():
        for p in (out / DEST).rglob('*'):
            require(not p.is_symlink(), 'Unexpected symlink in export directory')
            if p.is_file():
                require(p.relative_to(out).as_posix() in files, 'Unexpected file in summary-only directory')
    differences = []
    for rel, text in files.items():
        p = safe_target(out, rel)
        data = text.encode('utf-8')
        if check:
            if not p.is_file() or p.read_bytes() != data:
                differences.append(rel)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='.' + p.name, dir=p.parent)
            try:
                with os.fdopen(fd, 'wb') as f:
                    f.write(data)
                os.replace(temporary, p)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    require(not differences, 'Export mismatch: ' + ', '.join(differences))
    return {'stages': list(STAGES), 'reported_scores': 45, 'export_files': len(files),
            'original_logs_created': 0, 'model_evaluations_executed': 0,
            'mode': 'check' if check else 'write'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out-root', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print(render_json(build(args.root, args.out_root, args.check)), end='')
