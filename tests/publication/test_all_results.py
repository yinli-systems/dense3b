"""CPU-only tests for complete synchronization reports and sample-level verification."""
import importlib.util,json,gzip,tempfile,unittest,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import build_all_results as a
class AllResultsTests(unittest.TestCase):
    def test_all_numeric_fields_including_new_fields(self):
        self.assertEqual(a.numeric_fields({'new_gpu_counter':4,'timing':{'wait':.2},'missing':None}),{'new_gpu_counter':4,'timing.wait':.2})
    def test_nonfinite_and_booleans_not_numbers(self):
        self.assertEqual(a.numeric_fields({'nan':float('nan'),'bool':True,'zero':0}),{'zero':0})
    def test_mfu_percentage_without_mutating_raw(self):self.assertEqual(a.field_unit('mfu'),('Logged MFU (%)',100.))
    def test_loss_keeps_native_unit(self):self.assertEqual(a.field_unit('loss'),('Cross-entropy',1.))
    def test_per_gpu_throughput_has_token_units(self):self.assertEqual(a.field_unit('tok_per_s_gpu'),('Tokens / second',1.))
    def test_valid_metadata_tokens(self):self.assertEqual(a.token_count({'args':{'seq_len':4096},'samples_consumed':512}),2097152)
    def test_no_fabricated_tokens(self):self.assertIsNone(a.token_count({'args':{'seq_len':4096,'global_batch':512},'step':2499}))
    def test_checkpoint_and_job_do_not_change_protocol(self):
        self.assertEqual(a.fingerprint({'protocol':'v1','job_id':'1','checkpoint_model_sha256':'x'}),a.fingerprint({'protocol':'v1','job_id':'2','checkpoint_model_sha256':'y'}))
    def test_scorer_revision_changes_protocol(self):self.assertNotEqual(a.fingerprint({'official_backend_sha256':'a'}),a.fingerprint({'official_backend_sha256':'b'}))
    def fixture(self):
        return {'humaneval':.5,'humaneval_plus':.5,'n':2},{'checkpoint_sha256':'weights','items':[{'task':'0','base':True,'plus':True},{'task':'1','base':False,'plus':False}]}
    def test_recount_complete(self):
        summary,details=self.fixture();v,n=a.verify_task('humaneval',summary,details,'weights');self.assertEqual(n,2);self.assertEqual(v['humaneval']['passed_recounted'],1)
    def test_recount_rejects_wrong_model(self):
        s,d=self.fixture()
        with self.assertRaises(ValueError):a.verify_task('humaneval',s,d,'other')
    def test_recount_rejects_wrong_denominator(self):
        s,d=self.fixture();s['n']=164
        with self.assertRaises(ValueError):a.verify_task('humaneval',s,d,'weights')
    def test_recount_rejects_duplicate_ids(self):
        s,d=self.fixture();d['items'][1]['task']='0'
        with self.assertRaises(ValueError):a.verify_task('humaneval',s,d,'weights')
    def test_recount_rejects_wrong_score(self):
        s,d=self.fixture();s['humaneval']=1
        with self.assertRaises(ValueError):a.verify_task('humaneval',s,d,'weights')
    def test_plus_must_pass_base(self):
        s,d=self.fixture();d['items'][0]['plus']=False;d['items'][1]['plus']=True
        with self.assertRaises(ValueError):a.verify_task('humaneval',s,d,'weights')
    def test_strict_relaxed_separate(self):
        s={'gsm8k':1.,'strict_accuracy':.5,'n':2};d={'items':[{'index':0,'correct_relaxed':True,'correct_strict':True},{'index':1,'correct_relaxed':True,'correct_strict':False}]}
        v,n=a.verify_task('gsm8k',s,d,None);self.assertEqual(v['gsm8k']['passed_recounted'],2);self.assertEqual(v['strict_accuracy']['passed_recounted'],1)
    def test_compressed_raw_hash_checked(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);b=b'{"score":0}';z=gzip.compress(b,mtime=0);(p/'x.gz').write_bytes(z)
            r={'repo_path':'x.gz','encoding':'gzip','sha256':a.sha(b),'stored_sha256':a.sha(z)}
            self.assertEqual(a.raw(p,r),b);r['sha256']='wrong'
            with self.assertRaises(ValueError):a.raw(p,r)
    def test_zero_score_is_value_not_missing(self):self.assertEqual(a.value_text({'unit':'fraction','reported_value':0}),'0.0000%')
    def test_missing_is_not_zero(self):self.assertEqual(a.fmt(None),'—')
if __name__=='__main__':unittest.main()
