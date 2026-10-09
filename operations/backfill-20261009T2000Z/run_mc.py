#!/usr/bin/env python3
"""Full-count multiple-choice backfill with exact checkpoint config and per-item evidence."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,platform,time,traceback
C=Path(__file__).resolve().parent
import sys
sys.path.insert(0,str(C/'source'))
import torch
from tokenizers import Tokenizer
from model.config import MoEConfig
from model.model import Transformer
from bench import mc_eval as mc
MC_TASKS=['arc_easy','arc_challenge','hellaswag','winogrande','boolq','openbookqa','sciq','mmlu_cf','mmlu_mc5']
EXPECTED={'arc_easy':2376,'arc_challenge':1172,'hellaswag':10042,'winogrande':1267,'boolq':3270,'openbookqa':500,'sciq':1000,'mmlu_cf':14042,'mmlu_mc5':14042}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(16<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def load_model(row):
 pin=Path(row['pinned_checkpoint']);meta=json.loads((pin/'meta.json').read_text());assert sha(pin/'meta.json')==row['metadata_sha256']
 cfgd=dict(meta['config']);cfgd['doc_mask']=False
 cfg=MoEConfig(**cfgd)
 # Only known training-owned state dict files are loaded; generated Python is never executed here.
 model=Transformer(cfg)
 state=torch.load(pin/'model.pt',map_location='cpu',weights_only=True,mmap=True)
 missing,extra=model.load_state_dict(state,strict=False)
 assert not [x for x in missing if x not in ['cos','sin']] and not extra,(missing,extra)
 del state
 model=model.bfloat16().cuda().eval();model.cos,model.sin=model.cos.float(),model.sin.float()
 tokpath=C/'runtime'/('tokenizer-reason.json' if row['preset'].startswith('r3b') else 'tokenizer-code.json')
 tok=Tokenizer.from_file(str(tokpath));mc.EOS=int(cfg.eos_id)
 assert max(tok.get_vocab().values())<cfg.vocab_size
 return model,tok,meta,tokpath
@torch.no_grad()
def task(model,tok,name):
 items=mc.build(name);assert len(items)==EXPECTED[name],(name,len(items))
 reqs=[];mapping=[]
 for i,(ctx,choices,gold,kind) in enumerate(items):
  for j,cont in enumerate(choices):
   text=ctx[j] if kind=='wino' else ctx
   reqs.append((tok.encode(text,add_special_tokens=False).ids,tok.encode(cont,add_special_tokens=False).ids))
   mapping.append((i,j,len(cont.encode('utf-8'))))
 scores=mc.score(model,tok,reqs);ll={}
 for (i,j,nb),(lp,n) in zip(mapping,scores):
  assert n>0 and torch.isfinite(torch.tensor(lp)),(name,i,j,n,lp)
  ll.setdefault(i,{})[j]={'logprob':lp,'per_byte':lp/max(1,nb),'scored_tokens':n,'bytes':nb}
 details=[]
 for i,(ctx,choices,gold,kind) in enumerate(items):
  pred=max(ll[i],key=lambda j:ll[i][j]['logprob']);norm=max(ll[i],key=lambda j:ll[i][j]['per_byte'])
  details.append({'index':i,'item_sha256':hashlib.sha256(json.dumps([ctx,choices,gold,kind],ensure_ascii=False).encode()).hexdigest(),'gold':int(gold),'prediction':pred,'prediction_norm':norm,'correct':pred==gold,'correct_norm':norm==gold,'choices':ll[i]})
 return {'acc':sum(x['correct'] for x in details)/len(details),'acc_norm':sum(x['correct_norm'] for x in details)/len(details),'n':len(details)},details

def main():
 p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
 plan=json.loads((C/'plan.json').read_text());row=plan['plan'][a.index]
 out=C/'results'/row['repo']/f"{row['preset']}-step_{row['step']:08d}";out.mkdir(parents=True,exist_ok=True)
 tasks=[t for t in MC_TASKS if t in row['missing_tasks']]
 status={'index':a.index,'job_id':os.environ.get('SLURM_JOB_ID'),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'loading','tasks':tasks}
 atomic(out/'mc-status.json',status);print('MC_START',json.dumps(row),flush=True)
 model,tok,meta,tokpath=load_model(row);pin=Path(row['pinned_checkpoint']);modelhash=sha(pin/'model.pt')
 n=meta.get('samples_consumed');seq=meta['args'].get('seq_len') or meta['config']['max_seq_len']
 tokens=n*seq if n is not None else None
 manifest={'protocol':'checkpoint-backfill-mc-v2','model_sha256':modelhash,'metadata_sha256':row['metadata_sha256'],'tokenizer_sha256':sha(tokpath),'runner_sha256':sha(Path(__file__)),'mc_source_sha256':sha(C/'source/bench/mc_eval.py'),'architecture_sha256':sha(C/'source/model/model.py'),'source_config':'saved checkpoint config; doc_mask=False for inference','dtype':'bfloat16','normalization':'UTF-8 byte length, not token count','max_sequence':2048,'tokenizer_add_special_tokens':False,'torch':torch.__version__,'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),'hostname':platform.node(),'job_id':os.environ.get('SLURM_JOB_ID'),'dataset_files':{str(p):sha(p) for p in sorted(Path(mc.R).rglob('*.jsonl'))}}
 res={'checkpoint':row['original_checkpoint'],'pinned_checkpoint':row['pinned_checkpoint'],'step':row['step'],'preset':row['preset'],'tokens':tokens,'tokens_B':tokens/1e9 if tokens is not None else None,'token_count_basis':'samples_consumed * seq_len' if tokens is not None else 'not in metadata; left null','manifest':manifest,'tasks':{}}
 final=out/'mc.json'
 if final.exists():
  prior=json.loads(final.read_text())
  if prior.get('manifest',{}).get('model_sha256')==modelhash:res['tasks']=prior.get('tasks',{})
 atomic(out/'mc-manifest.json',manifest)
 for t in tasks:
  if t in res['tasks'] and (out/(t+'.details.json')).exists():continue
  status.update(state='running',current_task=t);atomic(out/'mc-status.json',status);t0=time.time()
  r,details=task(model,tok,t);r['seconds']=time.time()-t0;res['tasks'][t]=r
  atomic(out/(t+'.details.json'),{'task':t,'checkpoint_sha256':modelhash,'n':len(details),'items':details});atomic(final,res)
  print(json.dumps({t:r}),flush=True)
 status.update(state='completed',completed_tasks=list(res['tasks']),ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());atomic(out/'mc-status.json',status)
 print('MC_DONE',str(final),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc();raise
