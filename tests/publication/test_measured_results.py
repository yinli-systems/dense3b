"""Consistency tests for the author-confirmed measured-results publication.

These tests validate reporting, not model evaluation accuracy.
"""
from __future__ import annotations
import copy
import csv
import io
import json
import re
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import build_measured_results as pub

class MeasuredResultsTests(unittest.TestCase):
    def setUp(self):
        self.d=json.loads((ROOT/pub.SOURCE).read_text())

    def test_54_primary_measurements_and_auxiliary(self):
        self.assertEqual(len(pub.validate(self.d)),54)
        self.assertEqual(len(self.d['auxiliary_results']),1)

    def test_exact_author_corrections(self):
        l=pub.validate(self.d)
        for k,v in {('humaneval_plus','40B'):'39.0',('mbpp','20B'):'38.4',('mbpp','40B'):'42.9',('mbpp','60B'):'46.3',('cruxeval_i','60B'):'30.750',('mbpp_plus','100B'):'51.3228'}.items():
            self.assertEqual(l[k]['value_pct'],v)

    def test_later_run_global_throughput_is_distinct_from_archive(self):
        m=self.d['systems_measurements']['later_training_global_throughput']
        self.assertEqual(m['value'],45380)
        self.assertEqual(m['statistic'],'mean')
        self.assertEqual(m['unit'],'tokens/s')
        self.assertIsNone(m['aggregation_window'])
        self.assertFalse(m['source_logs_attached'])
        self.assertIn('45,380 tokens/s',pub.site(self.d))
        self.assertIn('41,780 tokens/s',pub.site(self.d))

    def test_approximate_and_coarse_endpoints(self):
        l=pub.validate(self.d)
        self.assertEqual(pub.score(l['mmlu','100B']),'≈52.0')
        self.assertEqual(pub.score(l['mbpp','100B']),'52')

    def test_later_missing_counts_are_preserved(self):
        for b in self.d['benchmarks']:
            for v in b['values'][1:]:
                self.assertIsNone(v['n_archived'])
                self.assertIsNone(v['passed_archived'])
                self.assertFalse(v['raw_outputs_attached'])

    def test_reject_duplicate_stage(self):
        self.d['stages'][1]['id']=self.d['stages'][0]['id']
        with self.assertRaises(ValueError):pub.validate(self.d)

    def test_reject_missing_benchmark(self):
        self.d['benchmarks'].pop()
        with self.assertRaises(ValueError):pub.validate(self.d)

    def test_reject_missing_score(self):
        self.d['benchmarks'][0]['values'].pop()
        with self.assertRaises(ValueError):pub.validate(self.d)

    def test_reject_invalid_percentage(self):
        for invalid in ['NaN','Infinity','-1','101','x']:
            d=copy.deepcopy(self.d)
            d['benchmarks'][0]['values'][1]['value_pct']=invalid
            with self.assertRaises(ValueError):pub.validate(d)

    def test_reject_lost_correction(self):
        self.d['benchmarks'][1]['values'][2]['value_pct']='38.9'
        with self.assertRaises(ValueError):pub.validate(self.d)

    def test_reject_manufactured_raw_evidence(self):
        self.d['benchmarks'][0]['values'][1]['raw_outputs_attached']=True
        with self.assertRaises(ValueError):pub.validate(self.d)

    def test_reject_unknown_source(self):
        self.d['benchmarks'][0]['values'][1]['source']='not-a-source'
        with self.assertRaises(ValueError):pub.validate(self.d)

    def test_reject_plus_above_base(self):
        self.d['benchmarks'][1]['values'][1]['value_pct']='50.0'
        with self.assertRaises(ValueError):pub.validate(self.d)

    def test_csv_preserves_decimal_strings(self):
        rows=list(csv.DictReader(io.StringIO(pub.csv_text(pub.flatten(self.d)))))
        self.assertEqual(len(rows),54)
        lookup={(r['benchmark_id'],r['stage']):r for r in rows}
        self.assertEqual(lookup['mbpp_plus','100B']['score_pct'],'51.3228')
        self.assertEqual(lookup['cruxeval_i','60B']['score_pct'],'30.750')
        self.assertEqual(lookup['mbpp','100B']['score_pct'],'52')
        self.assertEqual(lookup['humaneval','20B']['n_archived'],'')

    def test_svg_exact_point_count_and_no_fit(self):
        for group in pub.GROUPS:
            xml=ET.fromstring(pub.svg_chart(self.d,group))
            points=[x for x in xml.iter('{http://www.w3.org/2000/svg}circle') if x.find('{http://www.w3.org/2000/svg}title') is not None]
            self.assertEqual(len(points),6*sum(b['group']==group for b in self.d['benchmarks']))
            self.assertIn('No fitted lines',pub.svg_chart(self.d,group))

    def test_reproducible_build(self):
        self.assertEqual(pub.generate(ROOT,self.d),pub.generate(ROOT,self.d))
        pub.build(ROOT,check=True)

    def test_generated_site_assets_exist(self):
        files=pub.generate(ROOT,self.d)
        for rel in re.findall(r'(?:src|href)="([^"#]+)"',files['docs/index.html']):
            if not rel.startswith(('http:','https:')):
                self.assertIn('docs/'+rel,files)

    def test_markdown_relative_links_resolve(self):
        files=pub.generate(ROOT,self.d)
        for rel,text in files.items():
            if not rel.endswith('.md'):continue
            for link in re.findall(r'\]\(([^)]+)\)',text):
                link=link.split('#')[0]
                if not link or link.startswith(('http:','https:','mailto:')):continue
                candidate=(ROOT/rel).parent/link
                self.assertTrue(candidate.exists(),str(candidate))

    def test_builder_does_not_target_raw_evidence(self):
        for rel in pub.generate(ROOT,self.d):
            self.assertFalse(rel.startswith(('evidence/','provenance/','model/','train/','tokenizer/')))
            self.assertNotIn(rel,['reports/evaluation_all.json','reports/evaluation_history.csv','reports/sample_verification.json'])

if __name__=='__main__':unittest.main()
