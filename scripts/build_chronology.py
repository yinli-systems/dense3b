"""Validate and attach a conditional chronology, never overwriting archived evidence."""
from __future__ import annotations
import csv
import hashlib
import io
import json
from datetime import datetime, timedelta
from pathlib import Path

SOURCE = Path('results/chronology/2026-10-18.json')
README_SECTION = Path('reports/chronology/README_SECTION.md')
SITE_SECTION = Path('docs/partials/chronology.html')
REPORT = Path('reports/CHRONOLOGY.md')

def validate(c):
    assert c['schema_version'] == 1
    assert c['status'] == 'theoretical_backcast_not_observed_dates'
    assert c['lineage']['claim'] == '100B_weights_descend_from_step_00006499'
    assert c['lineage']['evidence'] == 'author_confirmation_only'
    assert c['lineage']['optimizer_state_continuity'] is None
    assert c['lineage']['token_counter_continuity'] is None
    assert c['backcast']['reference_is_verified_completion'] is False
    assert c['backcast']['mean_tokens_per_second'] == 45380
    assert c['backcast']['uptime_fraction'] == 1.0
    anchor = datetime.fromisoformat(c['backcast']['reference_report_time'])
    assert anchor.utcoffset() == timedelta(hours=4)
    stages = c['backcast']['stages']
    assert [s['tokens_b'] for s in stages] == [0, 6.2, 6.815744, 20, 40, 60, 80, 100]
    assert len(stages) == 8
    for s in stages:
        date = anchor - timedelta(seconds=(100 - s['tokens_b']) * 1e9 / 45380)
        assert s['date_dubai'] == date.date().isoformat()
        assert s['theoretical_at_dubai'] == date.isoformat(timespec='minutes')
    assert c['conflicts']['requires_reconciliation'] is True
    assert c['conflicts']['archived_step_6499_metadata_modified_utc'] == '2026-10-09T22:52:48Z'
    return c

def insert(files, path, marker, new_text):
    assert files[path].count(marker) == 1, 'Missing or duplicate chronology marker: ' + path
    files[path] = files[path].replace(marker, new_text + marker, 1)

def apply_chronology(files, root):
    root = Path(root)
    canonical = (root / SOURCE).read_text(encoding='utf-8')
    c = validate(json.loads(canonical))
    section = (root / README_SECTION).read_text(encoding='utf-8')
    site = (root / SITE_SECTION).read_text(encoding='utf-8')
    report = (root / REPORT).read_text(encoding='utf-8')
    for s in c['backcast']['stages']:
        assert s['date_dubai'] in section and s['date_dubai'] in site and s['date_dubai'] in report
    assert 'not audited checkpoint creation dates' in section.lower()
    assert 'not audited checkpoint creation dates' in site.lower()
    assert 'theoretical' in report.lower() and 'unresolved' in report.lower()
    files['docs/data/chronology.json'] = canonical
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=['milestone', 'tokens_b', 'theoretical_date_dubai', 'theoretical_timestamp_dubai'], lineterminator='\n')
    writer.writeheader()
    for s in c['backcast']['stages']:
        writer.writerow({'milestone': s['name'], 'tokens_b': s['tokens_b'], 'theoretical_date_dubai': s['date_dubai'], 'theoretical_timestamp_dubai': s['theoretical_at_dubai']})
    files['reports/chronology/theoretical_stages.csv'] = out.getvalue()
    insert(files, 'README.md', '## Training implementation and systems', section)
    insert(files, 'reports/CONFIRMED_RESULTS.md', '## Later-run training throughput',
           '## Theoretical training chronology\n\n[Weight-lineage statement, date back-calculation and unresolved historical conflict](CHRONOLOGY.md). These dates are **not** independently verified checkpoint creation times.\n\n')
    insert(files, 'MODEL_CARD.md', '## Scope and limitations',
           '## Weight lineage and theoretical dates\n\nThe author confirms the 100B weights descend from step 6499. The [theoretical timeline](reports/CHRONOLOGY.md) uses the 18 October report as an endpoint and assumes 45,380 tokens/s without downtime. It dates step 6499 to 24 September, conflicting with the 7–10 October archived training chronology if treated as one uninterrupted token count. Actual completion dates and optimizer/token-counter continuity are unverified.\n\n')
    insert(files, 'reports/measured/SOURCES.md', '## Integrity',
           '## Timing provenance\n\n[Theoretical milestone dates and archive conflict](../CHRONOLOGY.md). These back-calculated dates are **not** original creation timestamps. The step-6499-to-100B weight lineage is author-confirmed only.\n\n')
    insert(files, 'reports/README.md', '## Historical evidence',
           '## Training timeline\n\n[Theoretical dates, assumptions and unresolved log discrepancy](CHRONOLOGY.md) · [Timeline CSV](chronology/theoretical_stages.csv).\n\n')
    insert(files, 'docs/index.html', '<section id="training">', site + '\n')
    nav = '<a href="#trends">Checkpoints</a>'
    assert files['docs/index.html'].count(nav) == 1
    files['docs/index.html'] = files['docs/index.html'].replace(nav, nav + '<a href="#chronology">Chronology</a>', 1)
    return {
        'source_path': SOURCE.as_posix(),
        'source_sha256': hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
        'readme_section_sha256': hashlib.sha256(section.encode('utf-8')).hexdigest(),
        'site_section_sha256': hashlib.sha256(site.encode('utf-8')).hexdigest(),
        'report_sha256': hashlib.sha256(report.encode('utf-8')).hexdigest()
    }
