"""Publish author-confirmed Dense3B checkpoint dates and model weight lineage.

Dates, sources and weight-continuation declarations reside in one canonical JSON.
This script only builds public documentation; original experiment logs remain intact.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
from datetime import date
from pathlib import Path

SOURCE = Path("results/chronology/2026-10-18.json")
README_SECTION = Path("reports/chronology/README_SECTION.md")
SITE_SECTION = Path("docs/partials/chronology.html")
REPORT = Path("reports/CHRONOLOGY.md")
CSV_PATH = "reports/chronology/checkpoint_dates.csv"
STAGES = [
    ("Training start", 0, "2026-09-23"),
    ("6.2B", 6.2, "2026-09-24"),
    ("6.82B (step 6499)", 6.815744, "2026-09-24"),
    ("20B", 20, "2026-09-28"),
    ("40B", 40, "2026-10-03"),
    ("60B", 60, "2026-10-08"),
    ("80B", 80, "2026-10-13"),
    ("100B", 100, "2026-10-18"),
]

def validate(record):
    assert record["schema_version"] == 2
    assert record["title"] == "Dense3B training timeline"
    assert record["timezone"] == "Asia/Dubai"
    assert record["date_precision"] == "day"
    assert record["source"]["type"] == "author_confirmation"
    assert record["source"]["confirmed_on"] == "2026-10-18"
    assert record["lineage"]["source_checkpoint"] == "step_00006499"
    assert record["lineage"]["continued_from_source_weights"] is True
    assert record["lineage"]["destination_training_tokens_b"] == 100
    assert record["lineage"]["source"] == "author_confirmation"
    assert record["training"]["mean_global_throughput_tokens_per_second"] == 45380
    assert record["training"]["throughput_source"] == "author_confirmation"
    actual = [(m["name"], m["tokens_b"], m["date_dubai"]) for m in record["training"]["milestones"]]
    assert actual == STAGES
    previous = None
    for name, tokens, stamp in actual:
        dt = date.fromisoformat(stamp)
        assert dt.isoformat() == stamp, name
        assert previous is None or previous <= dt
        previous = dt
    assert (6500 * 1_048_576) / 1e9 == record["training"]["milestones"][2]["tokens_b"]
    assert record["archived_evidence"]["retained_unchanged"] is True
    return record

def insert(files, path, marker, value):
    assert files[path].count(marker) == 1, "Expected one insertion marker: " + path
    files[path] = files[path].replace(marker, value + marker, 1)

def apply_chronology(files, root):
    root = Path(root)
    canonical = (root / SOURCE).read_text(encoding="utf-8")
    c = validate(json.loads(canonical))
    section = (root / README_SECTION).read_text(encoding="utf-8")
    site = (root / SITE_SECTION).read_text(encoding="utf-8")
    report = (root / REPORT).read_text(encoding="utf-8")
    for milestone in c["training"]["milestones"]:
        stamp = milestone["date_dubai"]
        assert stamp in section and stamp in site and stamp in report
    assert "author confirms" in section.lower()
    assert "<h2>Training timeline.</h2>" in site
    assert "author" in report.lower()
    files["docs/data/chronology.json"] = canonical
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=["milestone", "tokens_b", "completion_date_dubai", "source"], lineterminator="\n")
    writer.writeheader()
    for m in c["training"]["milestones"]:
        writer.writerow({"milestone":m["name"],"tokens_b":m["tokens_b"],"completion_date_dubai":m["date_dubai"],"source":"author_confirmation"})
    files[CSV_PATH] = out.getvalue()
    insert(files, "README.md", "## Training implementation and systems", section)
    insert(files, "reports/CONFIRMED_RESULTS.md", "## Later-run training throughput",
        "## Training timeline\n\n[Training milestone dates and checkpoint weight continuation](CHRONOLOGY.md) · [CSV](chronology/checkpoint_dates.csv).\n\n")
    insert(files, "MODEL_CARD.md", "## Scope and limitations",
        "## Training timeline and checkpoint lineage\n\n"
        "The project author confirms that the 100B checkpoint continued training from the original step 6499 weights. "
        "The [training timeline](reports/CHRONOLOGY.md) lists eight milestone dates from 23 September to 18 October 2026 in Dubai time (UTC+4).\n\n")
    insert(files, "reports/measured/SOURCES.md", "## Integrity",
        "## Training timeline\n\n[Checkpoint dates and weight continuation](../CHRONOLOGY.md) are supplied by the model author. "
        "Earlier archived training records and benchmark evidence remain unchanged.\n\n")
    insert(files, "reports/README.md", "## Historical evidence",
        "## Training timeline\n\n[Milestone dates and weight lineage](CHRONOLOGY.md) · [Checkpoint dates CSV](chronology/checkpoint_dates.csv).\n\n")
    insert(files, "docs/index.html", '<section id="training">', site + "\n")
    nav = '<a href="#trends">Checkpoints</a>'
    assert files["docs/index.html"].count(nav) == 1
    files["docs/index.html"] = files["docs/index.html"].replace(nav, nav+'<a href="#chronology">Timeline</a>', 1)
    return {
        "source_path":SOURCE.as_posix(),
        "source_sha256":hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "readme_section_sha256":hashlib.sha256(section.encode("utf-8")).hexdigest(),
        "site_section_sha256":hashlib.sha256(site.encode("utf-8")).hexdigest(),
        "report_sha256":hashlib.sha256(report.encode("utf-8")).hexdigest(),
    }
