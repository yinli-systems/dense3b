from pathlib import Path
import json,subprocess,os,shutil
C=Path('/data/run01/scxi253/checkpoint-eval-backfill-20261009T2000Z');W=C/'preflight-score'
P='/ssd/scxi253/model_deployment/envs/frontier_torch213';E='/data/run01/scxi253/code3b-train/evalplus-venv/lib/python3.10/site-packages';D='/data/run01/scxi253/code3b-dense3b-src/.deps'
# Only these three inspected official reference programs run in this diagnostic control.
samples=W/'canonical-control.jsonl';shutil.copy2(W/'humaneval.jsonl',samples)
env={'PATH':P+'/bin:/usr/bin:/bin','PYTHONPATH':E+':'+D,'HOME':str(W/'home'),'TMPDIR':str(W/'tmp'),'XDG_CACHE_HOME':str(W/'cache'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','HUMANEVAL_OVERRIDE_PATH':str(W/'official-humaneval.jsonl'),'EVALPLUS_MAX_MEMORY_BYTES':'4294967296'}
r=subprocess.run(['unshare','-Urnfp','--',P+'/bin/python','-m','evalplus.evaluate','--dataset','humaneval','--samples',str(samples),'--parallel','2','--test-details'],capture_output=True,text=True,env=env,timeout=120)
print('REFERENCE_ONLY_CONTROL',r.returncode,r.stdout[-2000:],r.stderr[-1500:],flush=True)
r.check_returncode();d=json.loads((W/'canonical-control_eval_results.json').read_text());print({k:[(x['base_status'],x['plus_status']) for x in v] for k,v in d['eval'].items()},flush=True)
