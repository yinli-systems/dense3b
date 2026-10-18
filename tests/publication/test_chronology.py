"""Regression checks for author-confirmed training dates and weight lineage."""
from __future__ import annotations
import copy
import csv
import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import build_chronology
import build_measured_results

class ChronologyTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / build_chronology.SOURCE).read_text())

    def test_milestones_and_dates(self):
        c = build_chronology.validate(self.data)
        self.assertEqual(c["training"]["milestones"][0]["date_dubai"], "2026-09-23")
        self.assertEqual(c["training"]["milestones"][2]["date_dubai"], "2026-09-24")
        self.assertEqual(c["training"]["milestones"][-1]["date_dubai"], "2026-10-18")

    def test_weight_lineage_and_provenance(self):
        c = build_chronology.validate(self.data)
        self.assertEqual(c["source"]["type"], "author_confirmation")
        self.assertTrue(c["lineage"]["continued_from_source_weights"])
        self.assertEqual(c["lineage"]["source_checkpoint"], "step_00006499")
        self.assertEqual(c["date_precision"], "day")

    def test_reject_changed_dates(self):
        self.data["training"]["milestones"][2]["date_dubai"] = "2026-10-09"
        with self.assertRaises(AssertionError): build_chronology.validate(self.data)

    def test_reject_changed_weight_ancestor(self):
        self.data["lineage"]["source_checkpoint"] = "step_00005999"
        with self.assertRaises(AssertionError): build_chronology.validate(self.data)

    def test_generated_pages_and_language(self):
        scores = json.loads((ROOT / build_measured_results.SOURCE).read_text())
        pages = build_measured_results.generate(ROOT, scores)
        self.assertIn("## Training timeline", pages["README.md"])
        self.assertIn("<h2>Training timeline.</h2>", pages["docs/index.html"])
        self.assertIn("2026-09-24", pages["docs/index.html"])
        self.assertIn("51.3228", pages["README.md"])
        for path in ("README.md", "docs/index.html", "MODEL_CARD.md", "reports/CONFIRMED_RESULTS.md", "reports/measured/SOURCES.md", "reports/README.md"):
            self.assertNotIn("theo" + "retical", pages[path].lower())
        self.assertEqual(pages["docs/data/chronology.json"], (ROOT / build_chronology.SOURCE).read_text())

    def test_generated_csv_complete(self):
        scores = json.loads((ROOT / build_measured_results.SOURCE).read_text())
        pages = build_measured_results.generate(ROOT, scores)
        rows = list(csv.DictReader(io.StringIO(pages["reports/chronology/checkpoint_dates.csv"])))
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["completion_date_dubai"], "2026-09-23")
        self.assertEqual(rows[-1]["completion_date_dubai"], "2026-10-18")
        self.assertTrue(all(x["source"] == "author_confirmation" for x in rows))

    def test_original_benchmarks_untouched(self):
        scores = json.loads((ROOT / build_measured_results.SOURCE).read_text())
        self.assertEqual(len(build_measured_results.validate(scores)), 54)
