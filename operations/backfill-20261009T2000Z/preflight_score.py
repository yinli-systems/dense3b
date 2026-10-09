from pathlib import Path
import os,subprocess,json,shutil,time
C=Path('/data/run01/scxi253/checkpoint-eval-backfill-20261009T2000Z')
P='/ssd/scxi253/model_deployment/envs/frontier_torch213';E='/data/run01/scxi253/code3b-train/evalplus-venv/lib/python3.10/site-packages';D='/data/run01/scxi253/code3b-dense3b-src/.deps'
W=C/'preflight-score-patched';W.mkdir(exist_ok=True)
for sub in ['tmp','cache','home']:(W/sub).mkdir(exist_ok=True)
sentinel=C/'host-access-denied-sentinel.txt';sentinel.write_text('sandbox test only\n')
for src,dst in [('HumanEvalPlus-v0.1.10.jsonl','official-humaneval.jsonl'),('MbppPlus-v0.2.0.jsonl','official-mbpp.jsonl')]:
 shutil.copy2(Path('/data/home/scxi253/.cache/evalplus')/src,C/'runtime'/dst)
for dataset,ids in [('humaneval',['HumanEval/0','HumanEval/1','HumanEval/32']),('mbpp',['Mbpp/2','Mbpp/3','Mbpp/4'])]:
 allrows=[json.loads(x) for x in (C/'runtime'/f'official-{dataset}.jsonl').read_text().splitlines()]
 rows=[x for x in allrows if x['task_id'] in ids];assert len(rows)==3,(dataset,[r['task_id'] for r in rows])
 (W/f'official-{dataset}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
 (W/f'{dataset}.jsonl').write_text(''.join(json.dumps({'task_id':r['task_id'],'solution':r['prompt']+r['canonical_solution']})+'\n' for r in rows))
 print('OFFICIAL_DATA',dataset,len(allrows),flush=True)
env={'PATH':P+'/bin:/usr/bin:/bin','PYTHONPATH':str(C/'score-runtime')+':'+E+':'+D+':'+str(C),'HOME':str(W/'home'),'TMPDIR':str(W/'tmp'),'XDG_CACHE_HOME':str(W/'cache'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'','SANDBOX_EVALPLUS_SOURCE':str(C/'score-runtime'),'SANDBOX_PYBASE':P,'SANDBOX_EVALDEPS':E,'SANDBOX_DEPS':D,'SANDBOX_TMP_ROOT':str(W/'tmp'),'SANDBOX_DENIED_SENTINEL':str(sentinel),'HUMANEVAL_OVERRIDE_PATH':str(W/'official-humaneval.jsonl'),'MBPP_OVERRIDE_PATH':str(W/'official-mbpp.jsonl'),'EVALPLUS_MAX_MEMORY_BYTES':'4294967296'}
for label,args in [('probe',['--probe']),('humaneval',['--dataset','humaneval','--samples',str(W/'humaneval.jsonl'),'--workers','2']),('mbpp',['--dataset','mbpp','--samples',str(W/'mbpp.jsonl'),'--workers','2'])]:
 cmd=['unshare','-Urnfp','--',P+'/bin/python','-u',str(C/'official_score.py'),*args]
 r=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=240)
 (W/(label+'.log')).write_text(r.stdout+'\n'+r.stderr)
 print(label,'RC',r.returncode,r.stdout[-2200:],r.stderr[-1200:],flush=True)
 r.check_returncode()
 if label!='probe':
  result=json.loads((W/(label+'_eval_results.json')).read_text())
  assert all(x['base_status']=='pass' and x['plus_status']=='pass' for cases in result['eval'].values() for x in cases), result
  print('POSITIVE_CANARIES_PASS',label,len(result['eval']),flush=True)
print('SCORE_PREFLIGHT_DONE',flush=True)
