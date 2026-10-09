from pathlib import Path
import json,subprocess,shutil,os,datetime
from score_support import environment,P,C
W=C/'preflight-score-patched'
positive={}
for dataset in ['humaneval','mbpp']:
 p=W/(dataset+'_eval_results.json');d=json.loads(p.read_text());assert len(d['eval'])==3
 assert all(x['base_status']=='pass' and x['plus_status']=='pass' for rr in d['eval'].values() for x in rr)
 positive[dataset]=3
samples=[json.loads(x) for x in (W/'humaneval.jsonl').read_text().splitlines()]
problems={json.loads(x)['task_id']:json.loads(x) for x in (W/'official-humaneval.jsonl').read_text().splitlines()}
for r in samples:
 if r['task_id']=='HumanEval/32':r['solution']=problems[r['task_id']]['prompt']+'\n    return 999999.0\n'
p=W/'incorrect-32.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in samples))
env=environment(W);env['HUMANEVAL_OVERRIDE_PATH']=str(W/'official-humaneval.jsonl')
cmd=['unshare','-Urnfp','--kill-child=SIGKILL','--',P+'/bin/python','-u',str(C/'official_score.py'),'--dataset','humaneval','--samples',str(p),'--workers','2']
r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=180)
(W/'negative-32.log').write_text(r.stdout+'\n'+r.stderr);print(r.stdout[-1600:],r.stderr[-800:],flush=True);r.check_returncode()
d=json.loads((W/'incorrect-32_eval_results.json').read_text());bad=d['eval']['HumanEval/32'][0]
assert bad['base_status']=='fail' and bad['plus_status']=='fail',bad
assert bad['base_fail_tests'], 'Wrong solution must have recorded failed test evidence'
for tid in ['HumanEval/0','HumanEval/1']:assert all(x['base_status']=='pass' and x['plus_status']=='pass' for x in d['eval'][tid])
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'positive_reference_tasks':positive,'HumanEval32_correct_reference':'base and Plus pass','HumanEval32_incorrect_reference':'base and Plus fail with failed-input evidence','isolation_positive_probe':'passed','host_read_and_new_socket_denied':True,'installed_evalplus_modified':False,'status':'passed'}
(C/'scoring-preflight-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)
