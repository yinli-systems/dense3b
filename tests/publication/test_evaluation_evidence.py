"""Synthetic CPU fixtures test validation, never serve as benchmark evidence."""
from pathlib import Path
import gzip
import json
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluation_evidence as e
H = 'a' * 64


def fixture(task='humaneval', n=2):
    flags = e.TASKS[task][1]
    items = [{'index': i, **{flag: i == 0 for flag in flags.values()}} for i in range(n)]
    summary = {'step': 6499, 'manifest': {'protocol': 'synthetic-unit-test',
               'checkpoint_model_sha256': H},
               'tasks': {task: {'n': n, **{metric: 1 / n for metric in flags}}}}
    details = {'task': task, 'n': n, 'checkpoint_sha256': H, 'items': items}
    return summary, details


class EvidenceTests(unittest.TestCase):
    def test_valid_recorded_counts(self):
        a, b = fixture()
        self.assertEqual(e.verify_task(a, b, 'humaneval', 2)['scores']['humaneval']['correct'], 1)

    def test_all_task_flag_schemas(self):
        for task in e.TASKS:
            with self.subTest(task=task):
                a, b = fixture(task)
                self.assertEqual(e.verify_task(a, b, task, 2)['n'], 2)

    def test_expected_full_count(self):
        a, b = fixture()
        with self.assertRaises(ValueError): e.verify_task(a, b, 'humaneval')

    def test_duplicate_item_id(self):
        a, b = fixture(); b['items'][1]['index'] = 0
        with self.assertRaises(ValueError): e.verify_task(a, b, 'humaneval', 2)

    def test_missing_id(self):
        a, b = fixture(); b['items'][1].pop('index')
        with self.assertRaises(ValueError): e.verify_task(a, b, 'humaneval', 2)

    def test_boolean_not_count(self):
        a, b = fixture(); a['tasks']['humaneval']['n'] = True
        with self.assertRaises(ValueError): e.verify_task(a, b, 'humaneval', 2)

    def test_wrong_weight_hash(self):
        a, b = fixture(); b['checkpoint_sha256'] = 'b' * 64
        with self.assertRaises(ValueError): e.verify_task(a, b, 'humaneval', 2)

    def test_missing_model_manifest(self):
        a, b = fixture(); a.pop('manifest')
        with self.assertRaises(ValueError): e.verify_task(a, b, 'humaneval', 2)

    def test_wrong_task(self):
        a, b = fixture(); b['task'] = 'mbpp'
        with self.assertRaises(ValueError): e.verify_task(a, b, 'humaneval', 2)

    def test_missing_flags(self):
        a, b = fixture(); b['items'][0].pop('base')
        with self.assertRaises(ValueError): e.verify_task(a, b, 'humaneval', 2)

    def test_integer_not_boolean_flag(self):
        a, b = fixture(); b['items'][0]['base'] = 1
        with self.assertRaises(ValueError): e.verify_task(a, b, 'humaneval', 2)

    def test_score_tampering(self):
        a, b = fixture(); a['tasks']['humaneval']['humaneval'] = .99
        with self.assertRaises(ValueError): e.verify_task(a, b, 'humaneval', 2)

    def test_plus_not_subset(self):
        a, b = fixture()
        b['items'][0]['plus'] = False; b['items'][1]['plus'] = True
        with self.assertRaises(ValueError): e.verify_task(a, b, 'humaneval', 2)

    def test_zero_correct_is_valid(self):
        a, b = fixture()
        for item in b['items']: item.update(base=False, plus=False)
        a['tasks']['humaneval'].update(humaneval=0, humaneval_plus=0)
        self.assertEqual(e.verify_task(a, b, 'humaneval', 2)['scores']['humaneval']['correct'], 0)

    def test_json_duplicate_keys_and_nonfinite(self):
        for text in (b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError): e.parse(text)

    def test_paths_and_gzip(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            (p / 'd.json.gz').write_bytes(gzip.compress(b'{"n":1}'))
            self.assertEqual(e.read_json(p, 'd.json.gz'), {'n':1})
            for path in ('../a', '/tmp/a', ''):
                with self.assertRaises(ValueError): e.contained(p, path)
            (p / 'link').symlink_to(p / 'd.json.gz')
            with self.assertRaises(ValueError): e.read_bytes(p, 'link')

    def test_inventory_limit_and_missing_roots(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            for i in range(3): (p / str(i)).write_text('')
            self.assertFalse(e.inventory([p], 1)['complete'])
            self.assertFalse(e.inventory([p / 'absent'])['complete'])
            self.assertTrue(e.inventory([p])['complete'])

    def test_bundle_requires_real_details(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t); a, _ = fixture()
            (p / 'suite.json').write_text(json.dumps(a))
            with self.assertRaises(ValueError): e.verify_bundle(p, H)

    def test_valid_bundle_and_byte_preserving_pack(self):
        with tempfile.TemporaryDirectory() as t:
            base = Path(t); src = base / 'source'; src.mkdir()
            summaries = {'suite.json': {'step':1, 'manifest':{'protocol':'synthetic-unit-test',
                        'checkpoint_model_sha256':H}, 'tasks':{}},
                         'mc.json': {'step':1, 'manifest':{'protocol':'synthetic-unit-test',
                        'checkpoint_model_sha256':H}, 'tasks':{}}}
            for task in e.PRIMARY:
                n = e.TASKS[task][0]; a, b = fixture(task, n)
                name = 'mc.json' if task in ('arc_challenge','mmlu_mc5') else 'suite.json'
                summaries[name]['tasks'][task] = a['tasks'][task]
                (src / (task+'.details.json')).write_text(json.dumps(b))
            for name, value in summaries.items(): (src/name).write_text(json.dumps(value))
            receipt = e.pack(src, base/'packed', H, '100B', 'new_rerun')
            self.assertFalse(receipt['model_evaluation_executed'])
            self.assertEqual(len(receipt['tasks']), len(e.PRIMARY))
            for f in receipt['artifacts']:
                self.assertEqual((src/f['path']).read_bytes(), (base/'packed'/f['path']).read_bytes())
            with self.assertRaises(ValueError): e.pack(src, base/'packed', H, '100B', 'new_rerun')
            with self.assertRaises(ValueError): e.pack(src, src/'inside', H, '100B', 'new_rerun')

    @unittest.skipUnless((ROOT/'reports/sample_verification.json').exists(), 'Full repository archive required')
    def test_real_step_6499_archive(self):
        result = e.audit_archive(ROOT)
        self.assertFalse(result['model_evaluation_executed'])
        self.assertGreaterEqual(result['early_task_families'], len(e.PRIMARY))
        self.assertEqual(result['stages'][0]['checkpoint_step'], 6499)
        for item in result['stages'][1:]:
            self.assertEqual(item['tasks'], [])
            self.assertEqual(item['status'], e.BLOCKED)


if __name__ == '__main__': unittest.main()
