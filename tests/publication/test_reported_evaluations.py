"""Tests for reported aggregate exports. No benchmark run occurs here."""
from __future__ import annotations
import copy
import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_reported_evaluations as exported
import evaluation_evidence


class ReportedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.raw = (ROOT / exported.SOURCE).read_bytes()
        self.source = exported.load_json(self.raw)

    def test_complete_45_scores_and_18_files(self):
        self.assertEqual(len(exported.validate(self.source)), 45)
        self.assertEqual(len(exported.generate(ROOT)), 18)

    def test_exact_source_scores_after_unit_conversion(self):
        lookup = exported.validate(self.source)
        files = exported.generate(ROOT)
        seen = set()
        for stage in exported.STAGES:
            for group, tasks in exported.GROUPS.items():
                doc = json.loads(files[str(exported.DEST / stage / (group + '.json'))])
                for task, metrics in tasks.items():
                    for metric, benchmark in metrics.items():
                        original = lookup[benchmark, stage]['value_pct']
                        self.assertEqual(doc['reported_percent'][benchmark], original)
                        self.assertEqual(Decimal(str(doc['tasks'][task][metric])), Decimal(original) / 100)
                        self.assertEqual(doc['provenance']['metric_source_ids'][benchmark],
                                         lookup[benchmark, stage]['source'])
                        seen.add((benchmark, stage))
        self.assertEqual(len(seen), 45)

    def test_final_precision_and_approximation(self):
        files = exported.generate(ROOT)
        suite = json.loads(files[str(exported.DEST / '100B' / 'suite.json')])
        mc = json.loads(files[str(exported.DEST / '100B' / 'mc.json')])
        self.assertEqual(suite['reported_percent']['mbpp_plus'], '51.3228')
        self.assertEqual(suite['reported_percent']['mbpp'], '52')
        self.assertEqual(suite['reporting_qualifiers']['mbpp']['rounding_note'], 'integer_percentage')
        self.assertTrue(mc['reporting_qualifiers']['mmlu']['approximate'])
        mid = json.loads(files[str(exported.DEST / '60B' / 'suite.json')])
        self.assertEqual(mid['reported_percent']['cruxeval_i'], '30.750')
        self.assertEqual(mid['tasks']['cruxeval']['cruxeval_i'], .3075)

    def test_unreported_runtime_and_identifiers_are_null(self):
        for path, text in exported.generate(ROOT).items():
            if not path.endswith(('/suite.json', '/mc.json')):
                continue
            doc = json.loads(text)
            self.assertEqual(doc['record_kind'], 'author_reported_aggregate_summary')
            self.assertFalse(doc['is_original_evaluation_output'])
            self.assertIsNone(doc['ckpt'])
            self.assertIsNone(doc['step'])
            self.assertTrue(all(v is None for v in doc['manifest'].values()))
            self.assertFalse(doc['provenance']['archived_protocol_inherited'])
            self.assertTrue(doc['provenance']['format_reference_only'])
            self.assertFalse(doc['evidence']['raw_outputs_attached'])
            self.assertFalse(doc['evidence']['model_evaluation_executed_by_exporter'])
            self.assertEqual(doc['evidence']['per_example_files'], [])
            self.assertIsNone(doc['evidence']['evaluation_started_at'])
            self.assertIsNone(doc['evidence']['evaluation_finished_at'])
            for task in doc['tasks'].values():
                self.assertIsNone(task['n'])
                self.assertIsNone(task['seconds'])

    def test_mmlu_protocol_not_invented(self):
        for stage in exported.STAGES:
            mc = json.loads(exported.generate(ROOT)[str(exported.DEST / stage / 'mc.json')])
            self.assertIn('mmlu', mc['tasks'])
            self.assertNotIn('mmlu_mc5', mc['tasks'])
            self.assertNotIn('mmlu_cf', mc['tasks'])
            self.assertIsNone(mc['tasks']['mmlu']['acc_norm'])

    def test_generated_paths_are_not_raw_evidence(self):
        for path in exported.generate(ROOT):
            self.assertTrue(path.startswith('results/reported_evaluations/'))
            self.assertFalse(path.endswith(('.log', '.jsonl', '.pt', '.details.json')))

    def test_raw_evidence_validator_rejects_export(self):
        with tempfile.TemporaryDirectory() as t:
            dest = Path(t)
            exported.build(ROOT, dest)
            with self.assertRaises(ValueError):
                evaluation_evidence.verify_bundle(dest / exported.DEST / '100B', 'a' * 64)

    def test_determinism_and_source_unchanged(self):
        self.assertEqual(exported.generate(ROOT), exported.generate(ROOT))
        with tempfile.TemporaryDirectory() as t:
            dest = Path(t)
            exported.build(ROOT, dest)
            self.assertEqual(exported.build(ROOT, dest, check=True)['export_files'], 18)
        self.assertEqual((ROOT / exported.SOURCE).read_bytes(), self.raw)

    def test_manifest_matches_exported_bytes(self):
        files = exported.generate(ROOT)
        manifest = json.loads(files[str(exported.DEST / 'manifest.json')])
        self.assertEqual(manifest['source_sha256'], exported.digest(self.raw))
        self.assertEqual(len(manifest['files']), 17)
        for path, checksum in manifest['files'].items():
            self.assertEqual(exported.digest(files[path].encode()), checksum)

    def test_reject_changed_output_and_unexpected_files(self):
        with tempfile.TemporaryDirectory() as t:
            dest = Path(t)
            exported.build(ROOT, dest)
            index = dest / exported.DEST / 'index.json'
            index.write_text('{}\n')
            with self.assertRaises(ValueError): exported.build(ROOT, dest, check=True)
            exported.build(ROOT, dest)
            (dest / exported.DEST / '100B' / 'fabricated.details.json').write_text('{}')
            with self.assertRaises(ValueError): exported.build(ROOT, dest)

    def test_reject_output_symlink(self):
        with tempfile.TemporaryDirectory() as t, tempfile.TemporaryDirectory() as other:
            dest = Path(t)
            (dest / 'results').mkdir()
            (dest / exported.DEST).symlink_to(other, target_is_directory=True)
            with self.assertRaises(ValueError): exported.build(ROOT, dest)

    def test_reject_invalid_percentages(self):
        for value in ('NaN', 'Infinity', '-1', '101', '1e2', ' 30', '', True, 30.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                exported.percentage(value)

    def test_reject_missing_benchmark_or_stage(self):
        for kind in ('benchmark', 'stage', 'score'):
            d = copy.deepcopy(self.source)
            if kind == 'benchmark': d['benchmarks'].pop()
            elif kind == 'stage': d['stages'].pop()
            else: d['benchmarks'][0]['values'].pop()
            with self.subTest(kind=kind), self.assertRaises(ValueError): exported.validate(d)

    def test_reject_duplicate_score(self):
        self.source['benchmarks'][0]['values'].append(self.source['benchmarks'][0]['values'][1])
        with self.assertRaises(ValueError): exported.validate(self.source)

    def test_reject_false_raw_output_claim(self):
        self.source['benchmarks'][0]['values'][1]['raw_outputs_attached'] = True
        with self.assertRaises(ValueError): exported.validate(self.source)

    def test_reject_unsourced_number(self):
        self.source['benchmarks'][0]['values'][1]['source'] = 'not-recorded'
        with self.assertRaises(ValueError): exported.validate(self.source)

    def test_reject_json_duplicate_keys_and_nonfinite(self):
        for value in (b'{"a":1,"a":2}', b'{"a":NaN}'):
            with self.assertRaises(ValueError): exported.load_json(value)


if __name__ == '__main__':
    unittest.main()
