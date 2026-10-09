"""Use two native dependency chains, so one spare GPU need not wait for the other."""
from pathlib import Path
import subprocess,json,datetime,shutil
C=Path('/data/run01/scxi253/checkpoint-eval-backfill-20261009T2000Z')
def run(*args):
 p=subprocess.run(args,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(str(args)+'\n'+p.stderr)
 return p.stdout.strip()
def save(path,obj):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)
assert json.loads((C/'scoring-preflight-receipt.json').read_text())['status']=='passed'
if (C/'dependency-graph.json').exists():raise SystemExit('Graph already exists; do not duplicate submission')
plan=json.loads((C/'plan.json').read_text());initial=json.loads((C/'plan-initial-18.json').read_text())
covered={(r['repo'],r['preset'],r['step']) for r in initial['all_checkpoints'] if not r['missing_tasks']}
kept=[];excluded=[]
for row in plan['plan']:
 if row.get('added_in_final_rescan') and (row['repo'],row['preset'],row['step']) in covered:
  excluded.append(row);continue
 row=dict(row);row['index']=len(kept);kept.append(row)
assert kept[:18]==initial['plan'], 'Never change indices or identities already submitted'
shutil.copy2(C/'plan.json',C/'plan-rescan-before-cross-root-dedup.json')
plan['plan']=kept;plan['already_evaluated_exclusions']=excluded;plan['final_rescan_added']=len(kept)-18;save(C/'plan.json',plan)
rest=(C/'mc-rest-job-id.txt').read_text().strip();pilot=(C/'mc-pilot-job-id.txt').read_text().strip()
states=run('squeue','-h','-r','-j',rest,'-o','%i|%T')
if any(line.split('|')[-1]!='PENDING' for line in states.splitlines()):raise RuntimeError('Remainder already active; do not rewrite running task dependencies')
run('scontrol','hold',rest)
created={}
def submit(label,indices,script):
 p=C/(label+'-job-id.txt')
 if p.exists() and p.read_text().strip():raise RuntimeError('Recorded job already exists: '+str(p))
 job=run('sbatch','--parsable','--hold','--array='+','.join(map(str,indices))+'%2',str(C/script))
 p.write_text(job+'\n');created[label]=job;print('HELD_ARRAY',label,job,flush=True);return job
extra_indices=list(range(18,len(kept)))
extra=submit('mc-extra',extra_indices,'mc.sbatch') if extra_indices else None
gen_indices=[r['index'] for r in kept if any(t in r['missing_tasks'] for t in ['humaneval','mbpp','cruxeval','gsm8k'])]
gen=submit('generation',gen_indices,'generation.sbatch')
mc={0:pilot+'_0',2:pilot+'_2'}
for r in kept:
 i=r['index']
 if i in mc:continue
 mc[i]=(rest if i<18 else extra)+'_'+str(i)
gen_jobs={i:gen+'_'+str(i) for i in gen_indices}
def block(i):
 rows=[{'index':i,'phase':'mc','job':mc[i],'repo':kept[i]['repo'],'preset':kept[i]['preset'],'step':kept[i]['step']}]
 if i in gen_jobs:rows.append(dict(rows[0],phase='generation',job=gen_jobs[i]))
 return rows
lanes=[block(0),block(2)];costs=[4,3]
for row in kept:
 i=row['index']
 if i in [0,2]:continue
 lane=0 if costs[0]<=costs[1] else 1
 lanes[lane].extend(block(i));costs[lane]+=4 if row['repo']=='dense3b' else 3 if row['repo'] in ['moe1b','reason3b'] else 2
for lane in lanes:
 for previous,current in zip(lane,lane[1:]):
  run('scontrol','update','JobId='+current['job'],'Dependency=afterany:'+previous['job'])
  current['dependency']='afterany:'+previous['job']
graph={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'max_concurrent_gpu_jobs':2,'partition':'gpu_5090','nice':10000,'job_arrays':{'pilot':pilot,'mc_remainder':rest,'mc_final_rescan':extra,'generation':gen},'checkpoints':len(kept),'mc_tasks':len(kept),'generation_tasks':len(gen_indices),'lanes':lanes,'training_jobs_modified':False,'already_evaluated_duplicate_excluded':len(excluded)}
save(C/'dependency-graph.json',graph)
for job in [rest,extra,gen]:
 if job:run('scontrol','release',job)
graph['released_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(C/'dependency-graph.json',graph)
print(json.dumps(graph,indent=2),flush=True)
print(run('squeue','-u','scxi253','-o','%.22i %.26j %.10T %.12M %.26R'),flush=True)
