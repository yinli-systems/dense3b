"""One bounded final rescan; append indices without changing already submitted work."""
from pathlib import Path
import json,os,shutil,hashlib,datetime
C=Path('/data/run01/scxi253/checkpoint-eval-backfill-20261009T2000Z');B=C.parent
plan=json.loads((C/'plan.json').read_text())
if plan.get('final_rescan_utc'):raise SystemExit('Final rescan already recorded')
shutil.copy2(C/'plan.json',C/'plan-initial-18.json')
seen={(r['repo'],r['preset'],r['step']) for r in plan['plan']};added=[]
expected=['arc_easy','arc_challenge','hellaswag','winogrande','boolq','openbookqa','sciq','mmlu_cf','mmlu_mc5','humaneval','mbpp','cruxeval','gsm8k']
for repo,run in [('dense3b','code3b-train'),('moe1b','moe1b'),('reason3b','reason3b'),('dense-moe-1b','routing')]:
 for m in sorted((B/run).glob('ckpt*/step_*/meta.json')):
  d=json.loads(m.read_text());key=(repo,d['args']['preset'],d['step']);w=m.parent/'model.pt'
  if key in seen or not w.exists() or not (m.parent/'COMPLETE').exists():continue
  # Do not duplicate an already fully evaluated checkpoint omitted from the initial gap list.
  existing=set()
  for p in (B/run/'results').glob('*.json'):
   if '.details.' in p.name:continue
   try:r=json.loads(p.read_text())
   except (ValueError,UnicodeError):continue
   if not isinstance(r,dict) or r.get('step')!=d['step']:continue
   ref=str(r.get('ckpt',r.get('checkpoint','')))
   if run=='routing' and d['args']['preset'] not in ref:continue
   if '/'+run+'/' not in ref:continue
   for task,v in r.get('tasks',{}).items():
    if task in expected and isinstance(v,dict) and isinstance(v.get('n'),int) and v['n']>100 and not v.get('error'):existing.add(task)
  missing=[t for t in expected if t not in existing]
  if not missing:continue
  preset=d['args']['preset'];step=d['step'];pin=C/'pinned'/repo/f'{preset}-step_{step:08d}';pin.mkdir(parents=True,exist_ok=False)
  st=w.stat();os.link(w,pin/'model.pt');shutil.copy2(m,pin/'meta.json')
  row={'index':len(plan['plan']),'repo':repo,'preset':preset,'step':step,'original_checkpoint':str(m.parent),'pinned_checkpoint':str(pin),'found_tasks':sorted(existing),'missing_tasks':missing,'metadata_sha256':hashlib.sha256((pin/'meta.json').read_bytes()).hexdigest(),'model_bytes':st.st_size,'model_inode':st.st_ino,'model_device':st.st_dev,'hardlink':True,'added_in_final_rescan':True}
  (pin/'PIN_PROVENANCE.json').write_text(json.dumps(row,indent=2)+'\n');plan['plan'].append(row);seen.add(key);added.append(row)
plan['final_rescan_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();plan['final_rescan_added']=len(added)
tmp=C/'plan.json.tmp';tmp.write_text(json.dumps(plan,indent=2)+'\n');tmp.replace(C/'plan.json')
(C/'final-rescan-additions.json').write_text(json.dumps(added,indent=2)+'\n')
print(json.dumps({'total_checkpoints':len(plan['plan']),'added':[{k:r[k] for k in ['index','repo','preset','step']} for r in added],'snapshot_utc':plan['final_rescan_utc']},indent=2))
