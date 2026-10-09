#!/usr/bin/env python3
"""Full HumanEval/MBPP official Plus scoring, CRUXEval, and GSM8K with sample evidence."""
from pathlib import Path
import argparse,ast,datetime,functools,hashlib,json,os,platform,re,time,traceback
from run_mc import C,sha,atomic,load_model
from bench import eval_suite as suite
import score_support as scoring

def examples(task):return suite.load_jsonl({'humaneval':'humaneval.jsonl','mbpp':'mbppplus.jsonl','gsm8k':'gsm8k_test.jsonl'}[task])
def gen_code(g,task,out):
 rows=examples(task)
 if task=='humaneval':prompts=[r['prompt'] for r in rows];stops=suite.CODE_STOP
 else:
  prompts=[]
  for r in rows:
   tl=r['test_list'] if isinstance(r['test_list'],list) else ast.literal_eval(r['test_list'])
   prompts.append(f'"""\n{r["prompt"]}\n{tl[0]}\n"""\n')
  stops=['\nassert','\n"""','\nprint(','\nif __name__','\n\n\n\n']
 path=out/(task+'.samples.jsonl');rawpath=out/(task+'.generation.json')
 if path.exists() and rawpath.exists():
  old=json.loads(rawpath.read_text());assert len(old['items'])==len(rows)
  return path
 comps,eos=g.run(prompts,512,stops)
 raw=[];samples=[]
 for r,p,c,e in zip(rows,prompts,comps,eos):
  if task=='humaneval':tid=r['task_id'];solution=r['prompt']+c
  else:
   tid='Mbpp/'+str(r['task_id']);imports=r['test_imports'] if isinstance(r['test_imports'],list) else ast.literal_eval(r['test_imports'])
   solution='\n'.join(imports)+'\n'+c
  raw.append({'task':tid,'prompt':p,'completion':c,'eos_or_stop_observed':bool(e)})
  samples.append({'task_id':tid,'solution':solution})
 tmp=path.with_suffix('.jsonl.tmp');tmp.write_text(''.join(json.dumps(x)+'\n' for x in samples));tmp.replace(path)
 atomic(rawpath,{'n':len(raw),'items':raw,'max_new_tokens':512,'stops':stops,'mbpp_imports_policy':'prepend benchmark test_imports, as in archived harness; no post-generation repair'})
 return path

def gsm(g):
 rows=examples('gsm8k');assert len(rows)==1319
 shots=''.join(f'Question: {q}\nAnswer: {a}\n\n' for q,a in suite.GSM_SHOTS)
 prompts=[shots+f'Question: {r["question"]}\nAnswer:' for r in rows]
 outs,_=g.run(prompts,320,['\nQuestion:','\n\n']);details=[]
 for i,(r,o) in enumerate(zip(rows,outs)):
  gold=r['answer'].split('####')[-1].strip().replace(',','')
  m=re.search(r'The answer is\s*\$?(-?[\d,]*\.?\d+)',o)
  strict=m.group(1).replace(',','').rstrip('.') if m else None
  ns=[m.group(1)] if m else re.findall(r'-?\d[\d,]*\.?\d*',o)
  relaxed=ns[-1].replace(',','').rstrip('.') if ns else None
  def equal(v):
   try:return v is not None and abs(float(v)-float(gold))<1e-6
   except (ValueError,OverflowError):return False
  details.append({'index':i,'question_sha256':hashlib.sha256(r['question'].encode()).hexdigest(),'gold':gold,'prediction_strict':strict,'prediction_relaxed':relaxed,'correct_strict':equal(strict),'correct_relaxed':equal(relaxed),'completion':o})
 return {'gsm8k':sum(d['correct_relaxed'] for d in details)/len(details),'strict_accuracy':sum(d['correct_strict'] for d in details)/len(details),'n':len(details)},details

def main():
 p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();row=json.loads((C/'plan.json').read_text())['plan'][a.index]
 out=C/'results'/row['repo']/f"{row['preset']}-step_{row['step']:08d}";out.mkdir(parents=True,exist_ok=True)
 tasks=[t for t in ['humaneval','mbpp','cruxeval','gsm8k'] if t in row['missing_tasks']]
 if not tasks:print('NO_GENERATION_GAPS',a.index);return
 status={'job_id':os.environ.get('SLURM_JOB_ID'),'index':a.index,'tasks':tasks,'state':'preflight','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};atomic(out/'generation-status.json',status)
 work=out/'scoring';work.mkdir(exist_ok=True);isolation=scoring.probe(work)
 assert scoring.execute('assert 2+2==4',root=work)[0], 'Sandbox positive execution canary failed'
 assert not scoring.execute('assert False',root=work)[0], 'Sandbox negative execution canary failed'
 prior_protocol=json.loads((C/'runtime/historical-protocol.json').read_text())
 for name,h in prior_protocol['dataset_sha256'].items():assert sha(Path(suite.J)/name)==h, 'Archived dataset changed: '+name
 status['state']='loading';atomic(out/'generation-status.json',status)
 model,tok,meta,tokpath=load_model(row);suite.TOK=str(tokpath);suite.EOS=int(meta['config']['eos_id']);suite.BAN=None if row['preset'].startswith('r3b') else 50280
 g=suite.Gen(model,tok,16);suite.execute=functools.partial(scoring.execute,root=work)
 modelhash=sha(Path(row['pinned_checkpoint'])/'model.pt');n=meta.get('samples_consumed');seq=meta['args'].get('seq_len') or meta['config']['max_seq_len'];tokens=n*seq if n is not None else None
 manifest={'protocol':'checkpoint-backfill-code-v2-official-evalplus','checkpoint_model_sha256':modelhash,'checkpoint_step':row['step'],'metadata_sha256':row['metadata_sha256'],'tokenizer_sha256':sha(tokpath),'runner_sha256':sha(Path(__file__)),'generate_sha256':sha(C/'source/model/generate.py'),'archived_suite_sha256':sha(C/'source/bench/eval_suite.py'),'official_evaluate_sha256':sha(Path(scoring.O)/'evalplus/evaluate.py'),'official_backend_sha256':sha(Path(scoring.O)/'evalplus/eval/__init__.py'),'official_oracle_sha256':sha(Path(scoring.O)/'evalplus/eval/_special_oracle.py'),'official_humaneval_data_sha256':sha(C/'runtime/official-humaneval.jsonl'),'official_mbpp_data_sha256':sha(C/'runtime/official-mbpp.jsonl'),'evalplus_patch_provenance':json.loads((C/'evalplus-patch-provenance.json').read_text()),'isolation_sha256':sha(C/'isolation.py'),'isolation_probe':isolation,'dtype':'bfloat16','temperature':0,'batch':16,'max_new':{'humaneval':512,'mbpp':512,'cruxeval':96,'gsm8k':320},'code_scoring':'official EvalPlus tests/oracles with documented successful find_zero progress-counter fix; no HumanEval/32 bypass','execution_limits':{'cruxeval_case_wall_s':5,'namespace_startup_allowance_s':3,'child_cpu_s':10,'child_address_space_bytes':8<<30,'evalplus_memory_bytes':4<<30},'cruxeval_protocol':'archived 2-shot assertion execution inside restricted child','gsm8k_protocol':'archived custom 4-shot exemplars; strict phrase extraction and relaxed last-number separately','job_id':os.environ.get('SLURM_JOB_ID')}
 res={'ckpt':row['original_checkpoint'],'pinned_checkpoint':row['pinned_checkpoint'],'step':row['step'],'preset':row['preset'],'tokens_B':tokens/1e9 if tokens is not None else None,'token_count_basis':'samples_consumed * seq_len' if tokens is not None else 'not recorded; left null','manifest':manifest,'tasks':{}}
 final=out/'suite.json'
 if final.exists():
  old=json.loads(final.read_text());assert old['manifest']['checkpoint_model_sha256']==modelhash;res['tasks']=old['tasks']
 atomic(out/'generation-manifest.json',manifest)
 for t in tasks:
  if t in res['tasks'] and (out/(t+'.details.json')).exists():continue
  status.update(state='running',current_task=t);atomic(out/'generation-status.json',status);t0=time.time()
  print('TASK_START',t,flush=True)
  if t in ['humaneval','mbpp']:
   samples=gen_code(g,t,out);status['state']='official_cpu_scoring';atomic(out/'generation-status.json',status)
   r,details,official_path=scoring.official(t,samples,work);r['official_result_file']=str(official_path)
  elif t=='cruxeval':
   r=suite.task_cruxeval(g);details=r.pop('details');assert r['n']==800
  else:r,details=gsm(g)
  r['seconds']=time.time()-t0
  atomic(out/(t+'.details.json'),{'task':t,'checkpoint_sha256':modelhash,'n':r['n'],'items':details});res['tasks'][t]=r;atomic(final,res)
  print(json.dumps({t:r}),flush=True)
 status.update(state='completed',completed_tasks=list(res['tasks']),ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());atomic(out/'generation-status.json',status);print('GENERATION_DONE',str(final),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  try:
   index=int(os.sys.argv[os.sys.argv.index('--index')+1]);row=json.loads((C/'plan.json').read_text())['plan'][index]
   target=C/'results'/row['repo']/f"{row['preset']}-step_{row['step']:08d}"/'generation-status.json'
   status=json.loads(target.read_text()) if target.exists() else {'index':index}
   status.update(state='failed',error=repr(exc),failed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());atomic(target,status)
  except Exception:pass
  traceback.print_exc();raise
