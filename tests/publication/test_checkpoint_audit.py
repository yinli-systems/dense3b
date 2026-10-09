"""CPU-only regression tests for checkpoint publication integrity."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('checkpoint_audit',ROOT/'scripts/audit_checkpoints.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
class CheckpointAuditTests(unittest.TestCase):
    def test_identity_uses_checkpoint_not_result_folder(self):
        self.assertEqual(a.identity('/data/run01/scxi253/code3b-train/ckpt-v0/step_00004999'),('dense3b','dense3b'))
    def test_research_variants_stay_separate(self):
        self.assertEqual(a.identity('/data/run01/scxi253/routing/ckpt-moe1b_r/step_00000119'),('dense-moe-1b','moe1b_r'))
    def test_unknown_checkpoint_not_guessed(self):self.assertEqual(a.identity('/tmp/step_001'),(None,None))
    def test_loss_above_one_is_retained(self):
        r=list(a.flatten('heldout',{'web':3.2446,'swallow-code':.7329}));self.assertEqual(len(r),2);self.assertTrue(all(x['unit']=='cross_entropy' for x in r))
    def test_loss_not_percent(self):self.assertEqual(a.display_value({'reported_value':3.2446,'unit':'cross_entropy'}),'3.244600')
    def test_fraction_is_percent(self):self.assertEqual(a.display_value({'reported_value':.25,'unit':'fraction'}),'25.0000%')
    def test_fim_nested_denominators_and_lengths(self):
        r={x['metric']:x for x in a.flatten('fim',{'single':{'n':164,'pass_rate':.25,'mean_overgen_chars':271.6},'multi':{'n':117,'pass_rate':0.0}})}
        self.assertEqual(r['single.pass_rate']['n'],164);self.assertEqual(r['multi.pass_rate']['n'],117);self.assertEqual(r['single.mean_overgen_chars']['unit'],'characters')
    def test_lcb_slice_denominator(self):
        r={x['metric']:x for x in a.flatten('lcb',{'lcb_v6_slice_175_post_cutoff':0.,'lcb_release_v6_cumulative':.1,'n':{'added_in_v6':175,'release_v6_cumulative':1055}})}
        self.assertEqual(r['lcb_v6_slice_175_post_cutoff']['n'],175);self.assertEqual(r['lcb_release_v6_cumulative']['n'],1055)
    def test_smoke_not_full(self):self.assertEqual(a.coverage('humaneval','humaneval',6),'smoke_test')
    def test_subset_not_full(self):self.assertEqual(a.coverage('gsm8k','gsm8k',200),'subset')
    def test_full_count_label(self):self.assertEqual(a.coverage('humaneval','humaneval',164),'full_recorded_count')
    def test_zero_samples_not_full(self):self.assertEqual(a.coverage('humaneval','humaneval',0),'zero_samples')
    def test_zero_score_is_not_missing(self):self.assertEqual(list(a.flatten('gsm8k',{'gsm8k':0.,'n':1319}))[0]['reported_value'],0.)
    def test_nan_and_boolean_not_metrics(self):self.assertEqual(list(a.flatten('heldout',{'nan':float('nan'),'bool':False})),[])
    def test_routing_ratio_above_one(self):
        r=list(a.flatten('routing_layers',{'1':{'load_max_mean':3.5}}))[0];self.assertEqual(r['reported_value'],3.5);self.assertEqual(r['unit'],'raw_statistic')
    def test_generation_revision_changes_protocol(self):
        self.assertNotEqual(a.protocol({'manifest':{'generate_sha256':'a'}})[0],a.protocol({'manifest':{'generate_sha256':'b'}})[0])
    def test_no_manifest_is_explicit(self):self.assertEqual(a.protocol({})[:2],('unversioned','unversioned'))
    def row(self,value=.5):return {'task':'humaneval','metric':'humaneval','n':2,'reported_value':value,'evaluation_id':'fixture'}
    def detail(self,duplicate=False,plus=False):
        return ({'path':'details.json','sha256':'abc'},{'humaneval':[{'task':'a','base':True,'plus':True},{'task':'a' if duplicate else 'b','base':False,'plus':plus}]})
    def test_recount_agrees(self):self.assertEqual(a.recount(self.row(),[self.detail()],[])['verification'],'sample_flags_recounted')
    def test_recount_disagreement_visible(self):
        issues=[];r=a.recount(self.row(.0),[self.detail()],issues);self.assertEqual(r['verification'],'summary_sample_mismatch');self.assertEqual(len(issues),1)
    def test_duplicate_ids_rejected(self):self.assertEqual(a.recount(self.row(),[self.detail(True)],[])['verification'],'conflicting_sample_evidence')
    def test_plus_cannot_pass_when_base_fails(self):
        row=self.row();row['metric']='humaneval_plus';self.assertEqual(a.recount(row,[self.detail(plus=True)],[])['verification'],'conflicting_sample_evidence')
    def test_partial_examples_not_full_evidence(self):
        row=self.row();row['n']=164;self.assertEqual(a.recount(row,[self.detail()],[])['verification'],'incomplete_sample_evidence')
    def test_missing_details_not_claimed_verified(self):self.assertEqual(a.recount(self.row(),[],[])['verification'],'summary_only')
    def test_git_sibling_pairing_requires_common_commit(self):
        x={'origins':[{'origin_group':'git:r','origin':'git_history','logical_path':'results/suite.json','git_occurrences':['a']} ]}
        y={'origins':[{'origin_group':'git:r','origin':'git_history','logical_path':'results/suite.details.json','git_occurrences':['b']} ]}
        self.assertFalse(a.detail_matches(x,y));y['origins'][0]['git_occurrences']=['a'];self.assertTrue(a.detail_matches(x,y))
    def test_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'x.json').write_text('{}');m={'records':[{'sha256':'bad','repo_path':'x.json'}]}
            with self.assertRaises(ValueError):a.load_blobs(p,m)
    def test_wilson_zero_success_still_has_uncertainty(self):
        lo,hi=a.wilson(0,164);self.assertAlmostEqual(lo,0);self.assertGreater(hi,0)
    def test_json_origin_survives_content_dedup_alias(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data=b'{"step": 7}'
            (root/'same.txt').write_bytes(data)
            m={'records':[{'sha256':a.digest(data),'repo_path':'same.txt','logical_path':'result.json'}]}
            self.assertEqual(a.load_blobs(root,m)[a.digest(data)]['obj']['step'],7)
    def test_preflight_banner_is_not_a_broken_benchmark(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data=b'NCCL version test\n{"world": 8}\n'
            (root/'preflight.json').write_bytes(data)
            m={'records':[{'sha256':a.digest(data),'repo_path':'preflight.json','logical_path':'preflight.json'}]}
            blob=a.load_blobs(root,m)[a.digest(data)]
            self.assertEqual(blob['obj']['world'],8);self.assertFalse(blob.get('json_parse_failed',False))
if __name__=='__main__':unittest.main()
