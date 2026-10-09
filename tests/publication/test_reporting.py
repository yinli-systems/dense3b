"""CPU-only checks for report extraction, rollback, and token accounting."""
from pathlib import Path
import sys, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from build_dashboard import finite, identify_checkpoint, moving, reconstruct, verified_tokens

class ReportingTests(unittest.TestCase):
    def test_numeric_not_boolean(self):
        self.assertTrue(finite(0));self.assertFalse(finite(True));self.assertFalse(finite(float('nan')));self.assertFalse(finite(float('inf')))
    def test_owner_follows_checkpoint_not_result_directory(self):
        self.assertEqual(identify_checkpoint('/data/run01/u/code3b-train/ckpt-v0/step_00004999'),('dense3b','dense3b'))
        self.assertEqual(identify_checkpoint('/data/run01/u/moe1b/ckpt/step_00002249'),('moe1b','moe1b'))
    def test_unknown_checkpoint_is_not_guessed(self):
        self.assertEqual(identify_checkpoint('results/suite-step_00004999.json'),(None,None))
    def test_research_variant_identity(self):
        self.assertEqual(identify_checkpoint('/data/routing/ckpt-dense1b_s/step_00000001'),('dense-moe-1b','dense1b_s'))
    def test_tokens_use_recorded_samples_and_sequence(self):
        self.assertEqual(verified_tokens({'samples_consumed':1792000,'args':{'seq_len':4096}}),7340032000)
    def test_tokens_not_inferred_from_step(self):
        self.assertIsNone(verified_tokens({'step':3499,'args':{'seq_len':4096,'global_batch':512}}))
    def test_missing_sequence_length_stays_missing(self):
        self.assertIsNone(verified_tokens({'samples_consumed':1792000}))
    def test_later_resume_replaces_old_tail(self):
        first={'job_id':1,'run_id':'train-1','production':True,'variant':'m','rows':[{'step':s,'loss':1} for s in range(5)]}
        later={'job_id':2,'run_id':'train-2','production':True,'variant':'m','rows':[{'step':2,'loss':2},{'step':3,'loss':3}]}
        result=reconstruct([later,first]);self.assertEqual([r['step'] for r in result],[0,1,2,3]);self.assertEqual(result[2]['loss'],2)
    def test_diagnostic_run_does_not_replace_production(self):
        runs=[{'job_id':1,'run_id':'train-1','production':True,'variant':'m','rows':[{'step':0,'loss':1}]},
              {'job_id':2,'run_id':'smoke-2','production':False,'variant':'m','rows':[{'step':0,'loss':9}]}]
        self.assertEqual(reconstruct(runs)[0]['loss'],1)
    def test_empty_variant_has_no_fabricated_rows(self):
        self.assertEqual(reconstruct([{'job_id':1,'run_id':'empty','production':True,'variant':'m','rows':[]}]),[])
    def test_trailing_mean_never_uses_future_values(self):
        self.assertEqual(moving([1,3,5],2),[1,2,4])
if __name__=='__main__':unittest.main()
