"""Read-only export of bounded campaign source, current results, and scheduler receipts."""
from pathlib import Path
import sys,io,tarfile,json,hashlib,datetime,subprocess,os
C=Path('/data/run01/scxi253/checkpoint-eval-backfill-20261009T2000Z')
idx={'snapshot_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(C),'records':[],'errors':[],'excluded':'No tensors, optimizer states, cache files, node runtimes, credentials or unrelated files'}
tar=tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz')
def put(name,data):
 ti=tarfile.TarInfo(name);ti.size=len(data);ti.mtime=0;tar.addfile(ti,io.BytesIO(data))
def capture(p):
 try:
  st=p.stat();data=p.read_bytes();after=p.stat();rel=str(p.relative_to(C))
  if p.suffix=='.json':json.loads(data)
  put(rel,data);idx['records'].append({'path':rel,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'mtime_ns':st.st_mtime_ns,'changed_during_read':(st.st_size,st.st_mtime_ns)!=(after.st_size,after.st_mtime_ns)})
 except Exception as e:idx['errors'].append({'path':str(p),'error':repr(e)})
for p in sorted(C.iterdir()):
 if p.is_file() and p.suffix in ['.py','.sh','.sbatch','.json','.patch','.txt']:capture(p)
for folder in ['source','score-runtime','runtime','results','preflight-score-patched','logs']:
 for root,dirs,files in os.walk(C/folder):
  dirs[:]=[d for d in dirs if d not in ['__pycache__','cache','tmp','home','.git']]
  for f in sorted(files):
   p=Path(root)/f
   if f.startswith('._') or f.endswith(('.pyc','.pkl','.tmp')):continue
   if p.suffix in ['.py','.sh','.sbatch','.json','.jsonl','.log','.out','.md','.txt','.patch'] or f in ['LICENSE','NOTICE']:capture(p)
for label,cmd in [('queue',['squeue','-u','scxi253','-o','%.22i %.28j %.10T %.12M %.45R']),('accounting',['sacct','-j','1681614,1681638,1681668,1681670','-n','-P','--format=JobID,State,ExitCode,Elapsed,Start,End'])]:
 try:idx[label]=subprocess.check_output(cmd,text=True,timeout=20)
 except Exception as e:idx[label+'_error']=repr(e)
put('export-index.json',json.dumps(idx,indent=2).encode());tar.close()
print('SNAPSHOT',idx['snapshot_utc'],'FILES',len(idx['records']),'ERRORS',len(idx['errors']),file=sys.stderr)
