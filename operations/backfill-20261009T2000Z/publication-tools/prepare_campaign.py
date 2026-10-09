"""Snapshot missing evaluations and hardlink immutable model files without editing training."""
from pathlib import Path
import os,json,hashlib,collections,shutil,datetime
B=Path('/data/run01/scxi253');C=B/'checkpoint-eval-backfill-20261009T2000Z'
C.mkdir(exist_ok=True)
if (C/'plan.json').exists():raise SystemExit('Existing campaign plan; refusing to overwrite')
NAMES={'dense3b':'code3b-train','moe1b':'moe1b','reason3b':'reason3b','dense-moe-1b':'routing'}
EXPECTED={'humaneval':164,'mbpp':378,'cruxeval':800,'gsm8k':1319,'arc_easy':2376,'arc_challenge':1172,'hellaswag':10042,'winogrande':1267,'boolq':3270,'openbookqa':500,'sciq':1000,'mmlu_cf':14042,'mmlu_mc5':14042}
results=collections.defaultdict(set)
for run in NAMES.values():
 for p in (B/run/'results').glob('*.json'):
  if '.details.' in p.name:continue
  try:d=json.loads(p.read_text())
  except (ValueError,UnicodeError):continue
  if not isinstance(d,dict):continue
  ref=str(d.get('ckpt',d.get('checkpoint','')));step=d.get('step')
  for repo,run2 in NAMES.items():
   if '/'+run2+'/' in ref:
    variant=next((v for v in ['dense1b_l','dense1b_s','moe1b_r'] if v in ref),repo)
    for t,r in d.get('tasks',{}).items():
     if t in EXPECTED and isinstance(r,dict) and r.get('n')==EXPECTED[t] and not r.get('error'):results[(repo,variant,step)].add(t)
plan=[];all_ck=[]
for repo,run in NAMES.items():
 seen=set()
 for m in sorted((B/run).glob('ckpt*/step_*/meta.json'))+sorted((B/run).glob('milestones/step_*/meta.json')):
  w=m.parent/'model.pt'
  if not w.exists() or w.stat().st_size<1:continue
  d=json.loads(m.read_text());preset=d['args']['preset'];step=d['step'];variant=preset if repo=='dense-moe-1b' else repo
  if (preset,step) in seen:continue
  seen.add((preset,step));found=results[(repo,variant,step)];missing=sorted(set(EXPECTED)-found)
  row={'repo':repo,'preset':preset,'step':step,'original_checkpoint':str(m.parent),'found_tasks':sorted(found),'missing_tasks':missing}
  all_ck.append(row)
  if not missing:continue
  if not ((m.parent/'COMPLETE').exists() or (m.parent/'MILESTONE').exists() or 'milestones' in m.parts):
   row['excluded']='no completion/milestone evidence';continue
  pin=C/'pinned'/repo/f'{preset}-step_{step:08d}';pin.mkdir(parents=True,exist_ok=False)
  before=w.stat();os.link(w,pin/'model.pt');shutil.copy2(m,pin/'meta.json');after=w.stat()
  if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise RuntimeError('checkpoint changed during pin')
  meta_bytes=(pin/'meta.json').read_bytes()
  row.update(pinned_checkpoint=str(pin),metadata_sha256=hashlib.sha256(meta_bytes).hexdigest(),model_bytes=after.st_size,model_inode=after.st_ino,model_device=after.st_dev,hardlink=True)
  (pin/'PIN_PROVENANCE.json').write_text(json.dumps(row,indent=2)+'\n');plan.append(row)
# Round-robin model families, newest first; oldest intermediate checkpoints remain in the queue.
groups={k:sorted([x for x in plan if x['repo']==k],key=lambda x:-x['step']) for k in NAMES}
ordered=[]
while any(groups.values()):
 for k in NAMES:
  if groups[k]:ordered.append(groups[k].pop(0))
for i,r in enumerate(ordered):r['index']=i
for name in ['logs','results','runtime','source']: (C/name).mkdir(exist_ok=True)
src=B/'routing-src'
for sub in ['model','train','bench','fused_scan']:
 if (src/sub).exists():shutil.copytree(src/sub,C/'source'/sub,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'),dirs_exist_ok=True)
for name,srcp in [('tokenizer-code.json',B/'moe7b-quality-stage2-20261002T2327Z/tokenizer.json'),('tokenizer-reason.json',B/'reason3b-data/tokenizer-v1/tokenizer.json'),('historical-protocol.json',B/'code3b-train/eval-protocol-v1.json')]:
 shutil.copy2(srcp,C/'runtime'/name)
obj={'campaign':C.name,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(C),'scope':'existing checkpoints missing primary full-count tasks only','plan':ordered,'all_checkpoints':all_ck,'planned_concurrency':2,'partition':'gpu_5090','nice':10000,'training_modified':False}
(C/'plan.json').write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps({'root':str(C),'checkpoints_to_evaluate':len(ordered),'by_repository':dict(collections.Counter(x['repo'] for x in ordered)),'pinned_logical_bytes':sum(r['model_bytes'] for r in ordered),'copied_tensor_bytes':0,'plan':[{k:r[k] for k in ['index','repo','preset','step','missing_tasks']} for r in ordered]},indent=2))
