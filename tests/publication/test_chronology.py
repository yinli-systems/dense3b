"""Validate theoretical chronology and preserve separately sourced benchmark results."""
from pathlib import Path
import json
import sys
import unittest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_chronology
import build_measured_results

class ChronologyTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads((ROOT / build_chronology.SOURCE).read_text(encoding='utf-8'))
    def test_backcast_dates_and_lineage(self):
        build_chronology.validate(self.c)
        self.assertEqual(self.c['backcast']['stages'][2]['date_dubai'], '2026-09-24')
        self.assertEqual(self.c['backcast']['stages'][-1]['date_dubai'], '2026-10-18')
        self.assertEqual(self.c['lineage']['evidence'], 'author_confirmation_only')
    def test_missing_completion_timestamp_stays_unverified(self):
        self.assertFalse(self.c['backcast']['reference_is_verified_completion'])
        self.assertTrue(self.c['conflicts']['requires_reconciliation'])
    def test_reject_backdated_stage_silent_rewrite(self):
        self.c['backcast']['stages'][2]['date_dubai'] = '2026-10-09'
        with self.assertRaises(AssertionError):
            build_chronology.validate(self.c)
    def test_generated_pages_link_sourced_theoretical_timeline(self):
        d = json.loads((ROOT / build_measured_results.SOURCE).read_text())
        outputs = build_measured_results.generate(ROOT, d)
        self.assertIn('## Theoretical training chronology', outputs['README.md'])
        self.assertNotIn('These are **not audited checkpoint creation dates**', outputs['README.md'])
        self.assertIn('id="chronology"', outputs['docs/index.html'])
        self.assertIn('2026-09-24', outputs['docs/index.html'])
        self.assertIn('not audited checkpoint creation dates', outputs['docs/index.html'].lower())
        self.assertIn('51.3228', outputs['README.md'])
        self.assertEqual(outputs['docs/data/chronology.json'], (ROOT / build_chronology.SOURCE).read_text())
    def test_generated_csv_contains_all_milestones(self):
        d = json.loads((ROOT / build_measured_results.SOURCE).read_text())
        outputs = build_measured_results.generate(ROOT, d)
        self.assertEqual(len(outputs['reports/chronology/theoretical_stages.csv'].strip().splitlines()), 9)
