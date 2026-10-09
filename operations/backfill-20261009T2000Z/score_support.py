from pathlib import Path
import json,os,subprocess,tempfile
C=Path(__file__).resolve().parent
P='/ssd/scxi253/model_deployment/envs/frontier_torch213'
E='/data/run01/scxi253/code3b-train/evalplus-venv/lib/python3.10/site-packages'
D='/data/run01/scxi253/code3b-dense3b-src/.deps'
O=C/'score-runtime'
def environment(work):
 work=Path(work)
 for sub in ['tmp','cache','home']:(work/sub).mkdir(parents=True,exist_ok=True)
 return {'PATH':P+'/bin:/usr/bin:/bin','PYTHONPATH':str(O)+':'+E+':'+D+':'+str(C),'HOME':str(work/'home'),'TMPDIR':str(work/'tmp'),'XDG_CACHE_HOME':str(work/'cache'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'','SANDBOX_EVALPLUS_SOURCE':str(O),'SANDBOX_PYBASE':P,'SANDBOX_EVALDEPS':E,'SANDBOX_DEPS':D,'SANDBOX_TMP_ROOT':str(work/'tmp'),'SANDBOX_DENIED_SENTINEL':str(C/'host-access-denied-sentinel.txt'),'HUMANEVAL_OVERRIDE_PATH':str(C/'runtime/official-humaneval.jsonl'),'MBPP_OVERRIDE_PATH':str(C/'runtime/official-mbpp.jsonl'),'EVALPLUS_MAX_MEMORY_BYTES':'4294967296'}
def probe(work):
 env=environment(work)
 r=subprocess.run(['unshare','-Urnfp','--kill-child=SIGKILL','--',P+'/bin/python',str(C/'official_score.py'),'--probe'],env=env,capture_output=True,text=True,timeout=30)
 if r.returncode:raise RuntimeError('Isolation probe failed: '+r.stderr)
 return json.loads(r.stdout)
def official(dataset,samples,work):
 env=environment(work);log=Path(work)/(dataset+'-scoring.log')
 cmd=['unshare','-Urnfp','--kill-child=SIGKILL','--',P+'/bin/python','-u',str(C/'official_score.py'),'--dataset',dataset,'--samples',str(samples),'--workers','4']
 with log.open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=2400)
 if r.returncode:raise RuntimeError('Official EvalPlus failed; see '+str(log))
 path=Path(str(samples).replace('.jsonl','_eval_results.json'));d=json.loads(path.read_text());items=[]
 for tid,rr in d['eval'].items():
  if len(rr)!=1:raise RuntimeError('Expected exactly one greedy sample per problem')
  x=rr[0];base=x['base_status']=='pass';plus=base and x['plus_status']=='pass'
  items.append({'task':tid,'base':base,'plus':plus,'base_status':x['base_status'],'plus_status':x['plus_status'],'base_fail_tests':x.get('base_fail_tests'),'plus_fail_tests':x.get('plus_fail_tests')})
 expected=164 if dataset=='humaneval' else 378
 if len(items)!=expected:raise RuntimeError(f'{dataset}: expected {expected}, got {len(items)}')
 sampleids={json.loads(x)['task_id'] for x in Path(samples).read_text().splitlines()}
 assert {x['task'] for x in items}==sampleids
 return {dataset:sum(x['base'] for x in items)/len(items),dataset+'_plus':sum(x['plus'] for x in items)/len(items),'n':len(items)},items,path

def execute(code,timeout=10,stdin=None,root=None):
 with tempfile.TemporaryDirectory(prefix='case-',dir=root) as d:
  work=Path(d);env=environment(work);area=work/'tmp';program=area/'candidate.py';program.write_text(code)
  stdout=area/'stdout.txt';stderr=area/'stderr.txt'
  cmd=['unshare','-Urnfp','--kill-child=SIGKILL','--',P+'/bin/python',str(C/'child_exec.py'),str(program)]
  with stdout.open('w') as out,stderr.open('w') as err:
   try:r=subprocess.run(cmd,input=stdin,text=True,env=env,cwd=area,stdout=out,stderr=err,timeout=timeout+3)
   except subprocess.TimeoutExpired:return False,''
  return r.returncode==0,stdout.read_text(errors='replace')[:8<<20]
