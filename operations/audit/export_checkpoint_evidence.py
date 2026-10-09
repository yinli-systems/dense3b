"""Read-only checkpoint/evaluation audit exporter. Never reads tensor payloads."""
from pathlib import Path
import os,sys,json,io,tarfile,hashlib,re,datetime,subprocess
ROOT=Path('/data/run01/scxi253')
MODELS={'dense3b':('code3b-train','code3b-dense3b-src'),'moe1b':('moe1b','moe1b-src'),'reason3b':('reason3b','reason3b-src'),'dense-moe-1b':('routing','routing-src')}
SKIP={'.git','__pycache__','evalplus-venv','inductor-cache','triton-cache','web15b','web-v1','abdata','node_modules','.venv','venv','site-packages','tokenizer','v0','shards','data','datasets','hf-cache','hf_cache'}
TEXT={'.json','.jsonl','.log','.out','.err','.txt','.md','.csv','.yaml','.yml','.toml','.py','.sh','.sbatch'}
index={'snapshot_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'roots':{},'records':[],'checkpoints':[],'skipped':[],'errors':[],'weight_payloads_read':False}
tar=tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz')
def put(name,b):
 t=tarfile.TarInfo(name);t.size=len(b);t.mtime=0;tar.addfile(t,io.BytesIO(b))
def capture(p,arc,repo,role):
 try:
  st=p.stat()
  if st.st_size>96*1024*1024:
   index['skipped'].append({'source':str(p),'reason':'text size limit','bytes':st.st_size});return
  b=p.read_bytes();st2=p.stat()
  put(arc,b)
  index['records'].append({'repository':repo,'role':role,'source':str(p),'resolved_source':str(p.resolve()),'archive_path':arc,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'mtime_ns':st.st_mtime_ns,'changed_during_read':(st.st_size,st.st_mtime_ns)!=(st2.st_size,st2.st_mtime_ns)})
 except Exception as e:index['errors'].append({'source':str(p),'error':type(e).__name__+': '+str(e)})
for repo,(run,src) in MODELS.items():
 index['roots'][repo]=[]
 for name,role in [(run,'run'),(src,'source')]:
  base=ROOT/name
  index['roots'][repo].append({'path':str(base),'resolved_path':str(base.resolve()),'exists':base.exists(),'role':role})
  if not base.exists():continue
  for current,dirs,files in os.walk(base,followlinks=False):
   dirs[:]=sorted(d for d in dirs if d not in SKIP and not d.startswith('.'))
   cur=Path(current);rel=cur.relative_to(base)
   if re.search(r'(?:step[_-]?\d+|checkpoint[_-]?\d+)$',cur.name):
    ent={'repository':repo,'source':str(cur),'resolved_source':str(cur.resolve()),'files':[]}
    for f in sorted(files):
     q=cur/f
     try:
      st=q.stat();ent['files'].append({'name':f,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev,'hardlinks':st.st_nlink})
     except OSError as e:index['errors'].append({'source':str(q),'error':str(e)})
    if (cur/'meta.json').exists():
     try:ent['metadata']=json.loads((cur/'meta.json').read_text())
     except Exception as e:ent['metadata_error']=str(e)
    index['checkpoints'].append(ent)
   for f in sorted(files):
    p=cur/f
    if p.suffix in TEXT and not f.startswith('.') and f not in ['tokenizer.json','vocab.json','merges.txt']:
     capture(p,repo+'/'+role+'/'+str(rel/f),repo,role)
for p in sorted((ROOT/'eval-logs').glob('*')):
 if p.is_file() and p.suffix in TEXT:capture(p,'shared/eval-logs/'+p.name,'shared','eval_log')
# Enumerate archived metadata separately, without reading or copying weights.
for base in [ROOT/'publication_snapshots']:
 if not base.exists():continue
 for cur,dirs,files in os.walk(base,followlinks=False):
  dirs[:]=[d for d in dirs if d not in SKIP]
  if 'meta.json' not in files:continue
  p=Path(cur)/'meta.json'
  if any(n in str(p) for n in MODELS):capture(p,'archive/'+str(p.relative_to(base)),'archive','historical_metadata')
try:index['queue_snapshot']=subprocess.check_output(['squeue','-u',os.environ.get('USER','scxi253'),'-o','%.12i %.42j %.10T %.12M %.40R'],text=True)
except Exception as e:index['queue_error']=str(e)
put('index.json',json.dumps(index,indent=2).encode());tar.close()
print('AUDIT_EXPORT',len(index['records']),'files',len(index['checkpoints']),'checkpoint directories','errors',len(index['errors']),file=sys.stderr)
