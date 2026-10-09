#!/usr/bin/env python3
"""Build complete per-model results and every recorded training metric from immutable captures.

CPU-only: no remote calls, submissions, weight loading or generated-code execution.
"""
from __future__ import annotations
import argparse,collections,csv,gzip,hashlib,json,math,os,re,statistics
from pathlib import Path
from typing import Any
from audit_checkpoints import flatten,coverage,identity,wilson

SCORE_FLAGS={'humaneval':{'humaneval':'base','humaneval_plus':'plus'},'mbpp':{'mbpp':'base','mbpp_plus':'plus'},'cruxeval':{'cruxeval_o':'ok_o','cruxeval_i':'ok_i'},'gsm8k':{'gsm8k':'correct_relaxed','strict_accuracy':'correct_strict'}}
CORE_TASKS=['humaneval','mbpp','cruxeval','gsm8k','arc_easy','arc_challenge','hellaswag','winogrande','boolq','openbookqa','sciq','mmlu_cf','mmlu_mc5']

def finite(v:Any)->bool:return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def dump(p:Path,d:Any)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False,default=str)+'\n')
def clean(x):
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    return x

def csvout(p:Path,rows:list[dict])->None:
    p.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def raw(root:Path,r:dict)->bytes:
    p=root/r['repo_path'];b=p.read_bytes()
    if r.get('stored_sha256') and sha(b)!=r['stored_sha256']:raise ValueError('Stored hash mismatch: '+str(p))
    data=gzip.decompress(b) if r.get('encoding')=='gzip' else b
    if sha(data)!=r['sha256']:raise ValueError('Source hash mismatch: '+str(p))
    return data

def numeric_fields(d:dict,prefix:str='')->dict:
    out={}
    for k,v in d.items():
        name=(prefix+'.'+k).strip('.')
        if finite(v):out[name]=v
        elif isinstance(v,dict):out.update(numeric_fields(v,name))
    return out

def token_count(meta:dict):
    a=meta.get('args',{});seq=a.get('seq_len') or meta.get('config',{}).get('max_seq_len');n=meta.get('samples_consumed')
    return int(seq*n) if finite(seq) and finite(n) and seq>0 and n>=0 else None

def fingerprint(m:dict)->str:
    # Hardware, timings, checkpoint identity and file paths are not scoring protocol changes.
    keys=['protocol','protocol_sha256','eval_suite_sha256','runner_sha256','generate_sha256','tokenizer_sha256','archived_suite_sha256','official_evaluate_sha256','official_backend_sha256','official_oracle_sha256','official_humaneval_data_sha256','official_mbpp_data_sha256','mc_source_sha256','dtype','temperature','batch','max_new','stops','normalization','max_sequence','tokenizer_add_special_tokens','exec_timeout_s','execution_limits','gsm8k_protocol','cruxeval_protocol','dataset_files']
    d={k:m[k] for k in keys if k in m}
    return sha(json.dumps(d,sort_keys=True).encode())[:16] if d else 'unversioned'

def verify_task(task:str,v:dict,detail:dict,expected_hash:str|None)->tuple[dict,int]:
    items=detail.get('items');n=v.get('n')
    if not isinstance(items,list) or not isinstance(n,int) or n<=0 or len(items)!=n:raise ValueError('Wrong per-example count: '+task)
    if expected_hash and detail.get('checkpoint_sha256')!=expected_hash:raise ValueError('Detail belongs to a different checkpoint: '+task)
    ids=[x.get('task',x.get('id',x.get('index'))) for x in items]
    if any(i is None for i in ids) or len(set(map(str,ids)))!=n:raise ValueError('Missing/duplicate sample IDs: '+task)
    pairs=SCORE_FLAGS.get(task,{'acc':'correct','acc_norm':'correct_norm'});out={}
    for metric,flag in pairs.items():
        if metric not in v:continue
        flags=[x.get(flag) for x in items]
        if not all(isinstance(f,bool) for f in flags):raise ValueError('Missing Boolean outcome: '+task+'/'+metric)
        count=sum(flags);value=count/n
        tolerance=.0000501 if task=='cruxeval' else 1e-10
        if abs(value-v[metric])>tolerance:raise ValueError('Summary/recount disagreement: '+task+'/'+metric)
        lo,hi=wilson(count,n)
        out[metric]={'verification':'sample_flags_recounted','passed_recounted':count,'n_recounted':n,'value_recounted':value,'ci95_low':lo,'ci95_high':hi}
    if task in ['humaneval','mbpp'] and any(x['plus'] and not x['base'] for x in items):raise ValueError('Plus passes while Base fails')
    return out,n

def training(root:Path,manifest:dict,cfg:dict,issues:list)->tuple:
    old=json.loads((root/'provenance/checkpoint_audit_sources.json').read_text())['records']
    sources=[]
    for priority,records in [(0,old),(1,manifest['records'])]:
        for r in records:
            logical=r.get('logical_path',r.get('source',''))
            name=Path(logical).name
            if name=='metrics.jsonl' or (name.endswith(('.out','.log')) and r.get('kind') not in ['shared_evaluation_log','evaluation_evidence']):
                sources.append((priority,r.get('mtime_ns',0),r,logical))
    sources.sort(key=lambda x:(x[0],x[1],x[3]));metric_versions=[];starts={};events=[];seen_files=set()
    for priority,mtime,r,logical in sources:
        key=(logical,r['sha256'])
        if key in seen_files:continue
        seen_files.add(key)
        text=raw(root,r).decode(errors='replace');name=Path(logical).name
        jobmatch=re.search(r'(\d+)(?:\.out|\.log)?$',Path(logical).parent.name if name=='metrics.jsonl' else name)
        if not jobmatch:continue
        job=int(jobmatch.group(1));variant=next((v for v in cfg['presets'] if v in logical),cfg['presets'][0] if len(cfg['presets'])==1 else None)
        if variant is None:continue
        production=not re.search(r'smoke|sweep|profile|prof-|kern|nccl|zeq|fp8dbg|node-bench|xnode',logical,re.I) and ('train-' in logical or 'epitope-' in logical or re.search(r'Ig[MEG]-',logical))
        if name=='metrics.jsonl':metric_versions.append((priority,mtime,r,logical,job,variant,production,text));continue
        for line,t in enumerate(text.splitlines(),1):
            if not t.startswith('{'):continue
            try:d=json.loads(t)
            except ValueError:continue
            if not isinstance(d,dict) or not d.get('event'):continue
            e={'variant':variant,'job_id':job,'production':bool(production),'source':r['repo_path'],'source_line':line,'record':clean(d)}
            events.append(e)
            if d['event']=='start':starts[(variant,job)]=e
    unique={};run_sources=collections.defaultdict(set);malformed=[]
    for priority,mtime,r,logical,job,variant,production,text in metric_versions:
        for line,t in enumerate(text.splitlines(),1):
            try:d=json.loads(t)
            except ValueError:
                malformed.append({'source':r['repo_path'],'line':line,'kind':'malformed_training_line','is_last_line':line==len(text.splitlines())});continue
            if not isinstance(d,dict) or not finite(d.get('step')):continue
            sig=sha(json.dumps(d,sort_keys=True).encode());step=int(d['step']);run_id=Path(logical).parent.name
            row={'variant':variant,'run_id':run_id,'job_id':job,'production':bool(production),**clean(d),'step':step,'source':r['repo_path'],'source_sha256':r['sha256'],'source_line':line,'capture_priority':priority,'source_mtime_ns':mtime}
            unique[(variant,job,step,sig)]=row;run_sources[(variant,job)].add(r['repo_path'])
    issues.extend(malformed);allrows=sorted(unique.values(),key=lambda x:(x['variant'],x['job_id'],x['step'],x['capture_priority'],x['source_mtime_ns'],x['source_line']))
    runs=[];byrun=collections.defaultdict(list)
    for row in allrows:byrun[(row['variant'],row['job_id'])].append(row)
    retained={}
    for (v,job),rr in sorted(byrun.items()):
        perstep={}
        for x in rr:perstep[x['step']]=x
        ordered=[perstep[s] for s in sorted(perstep)];production=rr[-1]['production'];first=ordered[0]['step']
        runs.append({'variant':v,'job_id':job,'run_id':rr[-1]['run_id'],'production':production,'observations':len(rr),'first_step':first,'last_step':ordered[-1]['step'],'sources':sorted(run_sources[(v,job)]),'start_event':starts.get((v,job)), 'rows':ordered})
        if production:
            prev=[x for x in retained.get(v,[]) if x['step']<first]
            retained[v]=prev+ordered
    current=[x for v in cfg['presets'] for x in retained.get(v,[])];schema={}
    for x in allrows:
        # Only numeric fields from the logged payload are metrics, not provenance identifiers.
        excluded={'variant','run_id','job_id','production','step','source','source_line','source_sha256','capture_priority','source_mtime_ns'}
        for field,val in numeric_fields({k:v for k,v in x.items() if k not in excluded}).items():
            a=schema.setdefault(field,{'field':field,'observations':0,'variants':set(),'min':val,'max':val});a['observations']+=1;a['variants'].add(x['variant']);a['min']=min(a['min'],val);a['max']=max(a['max'],val)
    catalog=[dict(x,variants=sorted(x['variants'])) for _,x in sorted(schema.items())]
    return allrows,current,runs,events,catalog

def evaluation(root:Path,manifest:dict,issues:list)->tuple:
    old=json.loads((root/'reports/checkpoint_audit/all_metrics.json').read_text());versions=json.loads((root/'reports/checkpoint_audit/evaluation_versions.json').read_text())
    seen_old=set()
    for r in old:
        key=(r['source'],r['source_sha256'])
        if key not in seen_old:
            if sha((root/r['source']).read_bytes())!=r['source_sha256']:raise ValueError('Historical source hash mismatch: '+r['source'])
            seen_old.add(key)
    rows=[dict(r,origin='historical_audit',capture_priority=0,source_mtime_ns=0) for r in old];known={r['source_sha256'] for r in old};rawmap={r['logical_path']:r for r in manifest['records'] if r.get('logical_path')};checked=0;detail_checks=[];states=[]
    for rec in manifest['records']:
        if rec.get('kind') not in ['training_evidence','evaluation_evidence']:continue
        logical=Path(rec['logical_path']);fn=logical.name
        if fn.endswith('-status.json') and rec['kind']=='evaluation_evidence':
            states.append({'checkpoint':logical.parent.name,'phase':fn,'source':rec['repo_path'],**json.loads(raw(root,rec))});continue
        if logical.suffix!='.json' or fn=='meta.json' or '.details.' in fn or rec['sha256'] in known:continue
        # Do not parse large generation/sample files when looking for aggregate summaries.
        if rec['kind']=='evaluation_evidence' and fn not in ['mc.json','suite.json']:continue
        try:d=json.loads(raw(root,rec))
        except (ValueError,UnicodeError):continue
        if not isinstance(d,dict):continue
        ref=d.get('ckpt',d.get('checkpoint'));owner,var=identity(ref)
        if owner!=manifest['repository']:continue
        step=d.get('step')
        if not finite(step):continue
        tasks=d.get('tasks');kind='suite'
        if not isinstance(tasks,dict):
            tasks={}
            for src,dest in [('heldout_loss','heldout'),('humaneval','humaneval'),('gsm8k','gsm8k'),('ce','domain_ce'),('permute_dce','permuted_router_delta_ce'),('layers','routing_layers')]:
                if isinstance(d.get(src),dict):tasks[dest]=d[src]
            kind='legacy_or_probe'
        if not tasks:continue
        known.add(rec['sha256']);m=d.get('manifest') or {};pid=fingerprint(m);eid=rec['sha256'][:16];mh=m.get('checkpoint_model_sha256',m.get('model_sha256'))
        versions.append({'evaluation_id':eid,'variant':var,'step':int(step),'protocol_id':pid,'protocol_name':m.get('protocol','unversioned'),'manifest':m,'kind':kind,'source':rec['repo_path'],'source_sha256':rec['sha256'],'original_path':rec['logical_path'],'task_names':list(tasks),'origin':rec['origin']})
        for task,val in tasks.items():
            if not isinstance(val,dict):continue
            verification={};detail_rec=None
            if val.get('error') or val.get('skipped'):
                issues.append({'kind':'task_not_completed','source':rec['repo_path'],'task':task});continue
            if rec['kind']=='evaluation_evidence':
                detail_rec=rawmap.get(str(logical.parent/(task+'.details.json')))
                if not detail_rec:raise ValueError('New task summary has no complete sample evidence: '+rec['logical_path']+'/'+task)
                detail=json.loads(raw(root,detail_rec));verification,n=verify_task(task,val,detail,mh);checked+=n
                detail_checks.append({'evaluation_id':eid,'variant':var,'step':int(step),'task':task,'n':n,'summary_source':rec['repo_path'],'details_source':detail_rec['repo_path'],'details_sha256':detail_rec['sha256'],'checks':verification})
            for x in flatten(task,val):
                if x['metric']=='strict_accuracy':x['unit']='fraction'
                row={'evaluation_id':eid,'variant':var,'step':int(step),'checkpoint':ref,'declared_checkpoint_sha256':mh,'protocol_id':pid,'protocol_status':'manifest_recorded' if m else 'unversioned','protocol_name':m.get('protocol','unversioned'),'source':rec['repo_path'],'source_sha256':rec['sha256'],'reported_tokens_B':d.get('tokens_B'),'tokens_from_metadata':d.get('tokens') if d.get('token_count_basis','').startswith('samples_consumed') else None,'capture_priority':1,'source_mtime_ns':rec.get('mtime_ns',0),'origin':rec['origin'],**x,'coverage':coverage(task,x['metric'],x['n']),'verification':'summary_only','detail_sources':[detail_rec['repo_path']] if detail_rec else []}
                row.update(verification.get(x['metric'],{}));rows.append(row)
    rows.sort(key=lambda r:(r['variant'],r['step'],r['task'],r['metric'],r['protocol_id'],r['capture_priority'],r['source_mtime_ns']))
    # Latest version of one task under a given protocol; never select by score.
    per_checkpoint={}
    for r in rows:per_checkpoint[(r['variant'],r['step'],r['task'],r['metric'],r['protocol_id'],r['coverage'])]=r
    selected=list(per_checkpoint.values());latest={}
    for r in sorted(selected,key=lambda x:(x['step'],x['capture_priority'],x['source_mtime_ns'])):
        latest[(r['variant'],r['task'],r['metric'],r['protocol_id'],r['coverage'])]=r
    return rows,selected,list(latest.values()),versions,states,detail_checks,checked

def checkpoints(root:Path,ptr:dict,rows:list,current:list)->list:
    previous=json.loads((root/'reports/checkpoint_audit/checkpoints.json').read_text());bank={}
    for c in previous:
        bank[(c['variant'],c['step'])]={'variant':c['variant'],'step':c['step'],'historical_evidence':c['evidence_kinds'],'metadata_sources':c['sources'],'live_locations':[],'metadata_token_candidates':[c['tokens_from_metadata']] if finite(c.get('tokens_from_metadata')) else []}
    def get(v,s):return bank.setdefault((v,int(s)),{'variant':v,'step':int(s),'historical_evidence':[],'metadata_sources':[],'live_locations':[],'metadata_token_candidates':[]})
    inv=json.loads((root/ptr['checkpoint_inventory']).read_text());pins=json.loads((root/ptr['pinned_inventory']).read_text())
    for e in inv['checkpoints']:
        m=e.get('metadata',{});v=m.get('args',{}).get('preset');s=m.get('step')
        if v is None or not finite(s):continue
        c=get(v,s);t=token_count(m)
        if t is not None:c['metadata_token_candidates'].append(t)
        c['metadata_sources'].append(ptr['checkpoint_inventory'])
        fs=e['files'];model=next((f for f in fs if f['name']=='model.pt'),None)
        c['live_locations'].append({'path':e['source'],'kind':'training_directory','model_file_present':bool(model and model['bytes']>0),'model_bytes':model['bytes'] if model else None,'optimizer_shards':sum(bool(re.fullmatch('rank[0-9]+.pt',f['name'])) for f in fs),'complete_marker':any(f['name']=='COMPLETE' for f in fs),'stat_capture_utc':inv['snapshot_utc']})
    for e in pins['records']:
        c=get(e['preset'],e['step']);t=token_count(e['metadata'])
        if t is not None:c['metadata_token_candidates'].append(t)
        c['metadata_sources'].append(ptr['pinned_inventory'])
        c['live_locations'].append({'path':e['path'],'kind':'evaluation_hardlink','model_file_present':e['model_bytes']>0,'model_bytes':e['model_bytes'],'metadata_matches_plan':e['metadata_matches_plan'],'stat_capture_utc':pins['snapshot_utc']})
    for r in rows:get(r['variant'],r['step'])
    bycp=collections.defaultdict(list)
    for r in rows:bycp[(r['variant'],r['step'])].append(r)
    for key,c in bank.items():
        rr=bycp[key];t=set(c.pop('metadata_token_candidates'));c['tokens_verified']=next(iter(t)) if len(t)==1 else None;c['token_conflict']=len(t)>1
        c['weights_present_in_captures']=any(x['model_file_present'] for x in c['live_locations']);c['metadata_sources']=sorted(set(c['metadata_sources']))
        c['metric_rows']=len(rr);c['evaluation_versions']=sorted({x['evaluation_id'] for x in rr});c['full_count_tasks']=sorted({x['task'] for x in rr if x['coverage']=='full_recorded_count' and x['unit']=='fraction'})
        c['missing_primary_tasks']=sorted(set(CORE_TASKS)-set(c['full_count_tasks']))
        c['checkpoint_weight_hashes']=sorted({x['declared_checkpoint_sha256'] for x in rr if x.get('declared_checkpoint_sha256')})
        obs=[x for x in current if x['variant']==c['variant'] and c['step']-10<=x['step']<=c['step']]
        c['nearby_training_observation']=max(obs,key=lambda x:x['step']) if obs else None
    return sorted(bank.values(),key=lambda x:(x['variant'],x['step']))

def field_unit(name:str)->tuple[str,float]:
    if name=='mfu':return 'Logged MFU (%)',100.
    if 'tok_per_s' in name:return 'Tokens / second',1.
    if name in ['loss','val_loss','ce']:return 'Cross-entropy',1.
    if 'gb' in name.lower():return 'GB (as logged)',1.
    if name.endswith('_s') or name.startswith('t_'):return 'Seconds',1.
    if name=='lr':return 'Learning rate',1.
    if 'load_max_mean' in name:return 'Expert maximum / mean load',1.
    return name.replace('_',' '),1.

def plots(root:Path,cfg:dict,current:list,runs:list,catalog:list,evals:list)->list:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=root/'reports/figures/all';dest.mkdir(parents=True,exist_ok=True);made=[]
    def save(fig,name,kind):
        fig.tight_layout();fig.savefig(dest/(name+'.png'),dpi=140);fig.savefig(dest/(name+'.svg'),metadata={'Date':None});plt.close(fig)
        made.append({'name':name,'kind':kind,'png':'reports/figures/all/'+name+'.png','svg':'reports/figures/all/'+name+'.svg'})
    for c in catalog:
        field=c['field'];unit,scale=field_unit(field);fig,ax=plt.subplots(figsize=(10.5,5.3));has=False
        for v in cfg['presets']:
            rr=[(x['step'],numeric_fields(x).get(field)) for x in current if x['variant']==v];rr=[(s,y*scale) for s,y in rr if finite(y)]
            if not rr:continue
            has=True;ax.plot([x[0] for x in rr],[x[1] for x in rr],linewidth=.8,alpha=.85,label=v)
        if not has:plt.close(fig);continue
        ax.set(title=cfg['name']+' | '+field,xlabel='Logged optimizer step',ylabel=unit);ax.grid(alpha=.2);ax.legend(fontsize=8)
        save(fig,'training_'+field.replace('.','_'),'training')
    fig,ax=plt.subplots(figsize=(10.5,5.7))
    for run in runs:
        if not run['production']:continue
        rr=[r for r in run['rows'] if finite(r.get('loss'))]
        if rr:ax.plot([x['step'] for x in rr],[x['loss'] for x in rr],linewidth=.7,alpha=.75,label=run['variant']+' / '+str(run['job_id']))
    ax.set(title=cfg['name']+' | all production runs, including replaced tails',xlabel='Logged optimizer step',ylabel='Training cross-entropy');ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2)
    save(fig,'training_all_runs','training')
    groups=collections.defaultdict(list)
    for r in evals:
        if r['unit']=='fraction' and r['coverage']=='full_recorded_count':groups[(r['task'],r['metric'])].append(r)
    for (task,metric),rr in sorted(groups.items()):
        fig,ax=plt.subplots(figsize=(10.5,5.4));series=collections.defaultdict(list)
        for x in rr:series[(x['variant'],x['protocol_id'])].append(x)
        for (v,pid),xx in sorted(series.items()):
            ax.scatter([x['step'] for x in xx],[100*x['reported_value'] for x in xx],s=24,alpha=.85,label=v+' / '+pid[:8])
        ax.set(title=cfg['name']+' | '+task+' / '+metric,xlabel='Checkpoint step (protocols separated; no interpolation)',ylabel='Score (%)');ax.set_ylim(0,100);ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
        save(fig,'eval_'+task+'_'+metric.replace('.','_'),'evaluation')
    return made

def fmt(x,digits=3):return '—' if not finite(x) else f'{x:,.{digits}f}'
def link(root:Path,target:str,page:Path,label='source'):
    return '['+label+']('+os.path.relpath(root/target,page.parent).replace(os.sep,'/')+')'
def value_text(r):return fmt(100*r['reported_value'],4)+'%' if r['unit']=='fraction' else fmt(r['reported_value'],6)

def reports(root:Path,cfg:dict,man:dict,ptr:dict,alltrain:list,current:list,runs:list,catalog:list,rows:list,selected:list,latest:list,versions:list,states:list,checks:list,checked:int,cps:list,figs:list,issues:list)->dict:
    out=root/'reports';stats=[]
    for v in cfg['presets']:
        rr=[x for x in current if x['variant']==v];recent=rr[-100:];last=rr[-1] if rr else {};entry={'variant':v,'latest_step':last.get('step'),'latest_job':last.get('job_id'),'retained_observations':len(rr),'recent_observations':len(recent)}
        for f in ['loss','mfu','tok_per_s','tok_per_s_gpu','step_s','max_mem_gb','reserved_gb','grad_norm','lr','t_data_wait_max','t_fwd_bwd','t_grad_comm','t_optim_gather','moe_load_max_mean','moe_load_max_mean_avg']:
            vals=[x[f] for x in recent if finite(x.get(f))];entry[f+'_latest']=last.get(f);entry[f+'_median_last100']=statistics.median(vals) if vals else None
        run=next((x for x in runs if x['variant']==v and x['job_id']==last.get('job_id')),None)
        entry['recorded_start_event']=run['start_event'] if run else None;stats.append(entry)
    plan=json.loads((root/ptr['campaign_plan']).read_text());state_lookup={(s['checkpoint'],s['phase']):s for s in states};campaign_status=[]
    for p in plan['rows']:
        name=f"{p['preset']}-step_{p['step']:08d}";needed=['mc-status.json']+(['generation-status.json'] if any(t in p['missing_tasks'] for t in ['humaneval','mbpp','cruxeval','gsm8k']) else [])
        phases={n:state_lookup.get((name,n),{}).get('state','not_started_in_capture') for n in needed}
        st='completed' if all(v=='completed' for v in phases.values()) else 'failed' if 'failed' in phases.values() else 'in_progress' if any(v!='not_started_in_capture' for v in phases.values()) else 'not_started_in_capture'
        campaign_status.append({'variant':p['preset'],'step':p['step'],'state':st,'phases':phases,'missing_tasks_scheduled':p['missing_tasks'],'original_checkpoint':p['original_checkpoint']})
    for file,data in [('training_all.csv',alltrain),('training_retained.csv',current),('evaluation_history.csv',rows),('evaluation_latest.csv',latest),('checkpoint_registry.csv',cps),('systems_summary.csv',stats),('metrics_catalog.csv',catalog),('campaign_status.csv',campaign_status)]:csvout(out/file,data)
    for file,data in [('evaluation_all.json',rows),('evaluation_versions.json',versions),('checkpoint_registry.json',cps),('run_index.json',[{k:v for k,v in x.items() if k!='rows'} for x in runs]),('systems_summary.json',stats),('metrics_catalog.json',catalog),('campaign_status.json',campaign_status),('sample_verification.json',checks),('sync_issues.json',issues),('figures/all/index.json',figs)]:dump(out/file,data)
    summary={'repository':cfg['name'],'training_capture_started_utc':man['training_snapshot_utc'],'evaluation_capture_started_utc':man['evaluation_snapshot_utc'],'source_records':len(man['records']),'training_observations':len(alltrain),'retained_training_observations':len(current),'numeric_training_metrics':len(catalog),'evaluation_metric_rows_all_versions':len(rows),'new_sample_records_recounted':checked,'checkpoint_entries':len(cps),'checkpoints_with_results':sum(c['metric_rows']>0 for c in cps),'training_figures':sum(f['kind']=='training' for f in figs),'evaluation_figures':sum(f['kind']=='evaluation' for f in figs),'campaign_checkpoints':len(campaign_status),'campaign_states':dict(collections.Counter(x['state'] for x in campaign_status)),'systems':stats,'issues':issues,'not_a_live_feed':True,'training_or_gpu_evaluation_changed':False}
    dump(out/'summary.json',summary)
    syspage=out/'SYSTEMS.md';lines=['# '+cfg['name']+' — training and system metrics','',f"Capture started **{man['training_snapshot_utc']}**. Files were captured individually while jobs continued; this is not an atomic live view.",'',
        '**MFU is the training implementation’s logged estimate**, not GPU device utilization. Summary medians use the last at most 100 retained observations. Throughput and step timing are training-loop measurements; queue time, all checkpoint stalls and downtime are not automatically included.','',
        '| Variant | Latest step | Meanings | MFU median | Global tok/s median | Per-GPU tok/s median | Step seconds median | Peak allocated GB, latest |','|---|---:|---|---:|---:|---:|---:|---:|']
    for s in stats:
        lines.append(f"| {s['variant']} | {s['latest_step'] if s['latest_step'] is not None else '—'} | Last {s['recent_observations']} observations | {fmt(s['mfu_median_last100']*100 if s['mfu_median_last100'] is not None else None,2)}% | {fmt(s['tok_per_s_median_last100'],0)} | {fmt(s['tok_per_s_gpu_median_last100'],0)} | {fmt(s['step_s_median_last100'],3)} | {fmt(s['max_mem_gb_latest'],3)} |")
    lines+=['','[All recorded observations](training_all.csv) · [Retained resume sequence](training_retained.csv) · [Run index and startup metadata](run_index.json) · [Full latest/median summary](systems_summary.csv)','',
        '## Every logged numeric metric','',
        'No metric is imputed. The all-observation table preserves diagnostic runs and replaced resume suffixes; main plots use production runs, with a later job replacing earlier observations from its first logged step. This is an explicit display policy, not proof of optimizer-state ancestry.','',
        '| Metric | Recorded observations | Variants | Chart |','|---|---:|---|---|']
    fmap={f['name']:f for f in figs}
    for c in catalog:
        name='training_'+c['field'].replace('.','_');f=fmap.get(name);url=link(root,f['png'],syspage,'PNG')+' / '+link(root,f['svg'],syspage,'SVG') if f else 'Raw data only'
        lines.append(f"| `{c['field']}` | {c['observations']} | {', '.join(c['variants'])} | {url} |")
    for f in ['loss','val_loss','mfu','tok_per_s','max_mem_gb','step_s','t_grad_comm','t_data_wait_max','moe_load_max_mean']:
        key='training_'+f
        if key in fmap:lines+=['','## '+f,'','!['+f+'](figures/all/'+key+'.png)']
    syspage.write_text('\n'.join(lines)+'\n')
    index=out/'RESULTS.md';lines=['# '+cfg['name']+' — all checkpoints and available results','',f"Evaluation capture started **{man['evaluation_snapshot_utc']}**; pinned-weight file stats have their own timestamp in provenance.",'',
        '**Every located checkpoint appears below, including checkpoints with no benchmark result.** Historical/custom and corrected EvalPlus protocols remain distinct. Finished task results from still-running jobs are included; unfinished tasks have no invented score.','',
        '[All result values and versions (CSV)](evaluation_history.csv) · [All result values (JSON)](evaluation_all.json) · [Protocol manifests](evaluation_versions.json) · [Per-example recount checks](sample_verification.json) · [Training / MFU / throughput](SYSTEMS.md) · [Campaign progress](CAMPAIGN.md)','',
        '| Variant | Step / result page | Weight file present in latest captures | Verified metadata tokens (B) | Result rows | Missing full-count primary task families |','|---|---:|---|---:|---:|---|']
    bycp=collections.defaultdict(list)
    for r in rows:bycp[(r['variant'],r['step'])].append(r)
    for c in cps:
        page=out/'checkpoints'/f"{c['variant']}-step_{c['step']:08d}.md";page.parent.mkdir(exist_ok=True);rr=bycp[(c['variant'],c['step'])]
        lines.append(f"| {c['variant']} | [{c['step']}](checkpoints/{page.name}) | {c['weights_present_in_captures']} | {fmt(c['tokens_verified']/1e9 if c['tokens_verified'] is not None else None,6)} | {len(rr)} | {', '.join(c['missing_primary_tasks']) or 'none at recorded sample counts'} |")
        pl=[f"# {c['variant']} — checkpoint {c['step']}",'','[All checkpoints](../RESULTS.md) · [System metrics](../SYSTEMS.md)','',
            f"**Verified metadata tokens:** {fmt(c['tokens_verified']/1e9 if c['tokens_verified'] is not None else None,6)} B. Missing metadata is not reconstructed from an assumed batch size.",'',
            '**Missing full-count task families:** '+(', '.join(c['missing_primary_tasks']) or 'none')+'. Counts alone do not certify evaluation protocol validity.','',
            '## Retained file locations','',
            'File size and markers are filesystem evidence, not a new weight-loading or full-tensor checksum test. Hardlinks retain model bytes outside the trainer’s rolling checkpoint directory.','']
        for loc in c['live_locations']:pl.append('`'+loc['path']+'` — '+loc['kind']+'; '+str(loc.get('model_bytes'))+' bytes; captured '+loc['stat_capture_utc']+'.\n')
        if not c['live_locations']:pl.append('No current file location found; historical metadata/results are preserved.\n')
        obs=c['nearby_training_observation']
        if obs:
            pl+=['## Nearest recorded training observation','',f"Job {obs['job_id']}, **actual logged step {obs['step']}** (not relabeled as checkpoint step): loss {fmt(obs.get('loss'),6)}, MFU {fmt(obs.get('mfu')*100 if finite(obs.get('mfu')) else None,2)}%, global throughput {fmt(obs.get('tok_per_s'),0)} tok/s, peak allocated memory {fmt(obs.get('max_mem_gb'),3)} GB. "+link(root,obs['source'],page,'Raw log/metrics')+'.','']
        pl+=['## All recorded evaluation values','',
             'Each row retains its evaluation/version ID, protocol, sample count and native unit. Repeated historical versions are evidence, not independent new runs.','',
             '| Evaluation | Task / metric | Value | Unit | n | Scope | Recount | Protocol | Source |','|---|---|---:|---|---:|---|---|---|---|']
        for x in rr:
            count=str(x.get('passed_recounted'))+'/'+str(x.get('n_recounted')) if x.get('passed_recounted') is not None else x.get('verification','summary_only')
            pl.append(f"| {x['evaluation_id'][:10]} | {x['task']} / {x['metric']} | {value_text(x)} | {x['unit']} | {x.get('n') if x.get('n') is not None else '—'} | {x['coverage']} | {count} | {x['protocol_id']} | {link(root,x['source'],page)} |")
        if not rr:pl.append('| — | No benchmark result found | — | — | — | — | — | — | — |')
        page.write_text('\n'.join(pl)+'\n')
    lines+=['','## Evaluation figures','',
            'Full-count scores are plotted as separate points by protocol. No curve connects incompatible evaluators; statistical significance is not implied.','']
    for f in figs:
        if f['kind']=='evaluation':lines.append(link(root,f['png'],index,f['name'])+' / '+link(root,f['svg'],index,'SVG')+'  ')
    if (out/'DIAGNOSTICS.md').exists():lines+=['','[Additional diagnostic experiments](DIAGNOSTICS.md)','']
    index.write_text('\n'.join(lines)+'\n')
    cl=['# '+cfg['name']+' — bounded backfill progress','',f"Evidence capture started: **{man['evaluation_snapshot_utc']}**. This is not a live feed.",'',
        '| Variant | Step | Status of scheduled missing tasks | Phase states |','|---|---:|---|---|']
    for c in campaign_status:cl.append(f"| {c['variant']} | {c['step']} | {c['state']} | {'; '.join(k+': '+v for k,v in c['phases'].items())} |")
    cl+=['','A checkpoint with only MC tasks scheduled is complete for this backfill when those tasks complete; its existing code scores retain their historical protocol. A running phase may already have completed individual tasks.','',
         '[Plan and dependencies]('+os.path.relpath(root/ptr['campaign_plan'],out)+') · [Captured scheduler output]('+os.path.relpath(root/ptr['capture'],out)+') · [All result pages](RESULTS.md)','']
    (out/'CAMPAIGN.md').write_text('\n'.join(cl))
    read=['# '+cfg['name'],'',cfg['description'],'',
        '**[All checkpoints and results](reports/RESULTS.md) · [MFU, loss, throughput and every system metric](reports/SYSTEMS.md) · [Backfill progress](reports/CAMPAIGN.md)**','',
        f"Training/evaluation capture started **{man['training_snapshot_utc']} / {man['evaluation_snapshot_utc']}**. Jobs continued while files were copied; these are timestamped snapshots, not a live dashboard.",'',
        '## Training and efficiency','',
        '| Variant | Latest logged step | Latest loss | MFU median, last ≤100 | Global tok/s median, last ≤100 | Peak allocated GB, latest |','|---|---:|---:|---:|---:|---:|']
    for s in stats:read.append(f"| {s['variant']} | {s['latest_step'] if s['latest_step'] is not None else '—'} | {fmt(s['loss_latest'],4)} | {fmt(s['mfu_median_last100']*100 if s['mfu_median_last100'] is not None else None,2)}% | {fmt(s['tok_per_s_median_last100'],0)} | {fmt(s['max_mem_gb_latest'],3)} |")
    read+=['','MFU is the logged model-FLOP estimate, not device utilization. Per-step throughput excludes unrecorded downtime. The systems page includes every numeric logged field, full unsmoothed data, per-run provenance and the explicit resume-display policy.','']
    for f in ['loss','mfu','tok_per_s']:
        k='training_'+f
        if k in fmap:read+=['!['+f+'](reports/figures/all/'+k+'.png)','']
    read+=['## Latest completed task results in the new scoring campaign','',
        'Latest checkpoint per variant/task/metric, not the highest score. The table may combine tasks at different steps, which are shown explicitly. Historical results and all protocols remain in the complete result index.','',
        '| Variant | Step | Task / metric | Score | n | Verified count |','|---|---:|---|---:|---:|---|']
    head={}
    for x in sorted(selected,key=lambda r:(r['step'],r['source_mtime_ns'])):
        if x['origin']=='campaign_snapshot' and x['unit']=='fraction':head[(x['variant'],x['task'],x['metric'])]=x
    for x in sorted(head.values(),key=lambda r:(r['variant'],r['task'],r['metric'])):
        count=f"{x['passed_recounted']}/{x['n_recounted']}" if x.get('passed_recounted') is not None else x['verification']
        read.append(f"| {x['variant']} | [{x['step']}](reports/checkpoints/{x['variant']}-step_{x['step']:08d}.md) | {x['task']} / {x['metric']} | {value_text(x)} | {x['n']} | {count} |")
    if not head:read.append('| No completed new task in this capture | — | — | — | — | — |')
    read+=['','## Archived architecture and accounting','', '| Variant | Total parameters | Active parameters |','|---|---:|---:|']
    for v,x in cfg.get('parameter_counts',{}).items():read.append(f"| {v} | {x['total']:,} | {x['active']:,} |")
    read+=['','These are archived configuration counts, including embeddings; they are not task-quality rankings.','',
        '## Reproduction and provenance','',
        '```bash','python -m pip install -r requirements-report.txt','python scripts/verify_publication.py','python scripts/build_all_results.py --root .','python -m unittest discover -s tests/publication -v','```','',
        '[Current immutable capture manifest]('+ptr['manifest']+') · [Latest capture pointer](provenance/sync_latest.json) · [Original snapshot inventory](provenance/snapshot.json) · [Earlier checkpoint audit](reports/checkpoint_audit/CHECKPOINTS.md)','',
        'Original logs/results and deployed source snapshots are preserved by hash. No original training source was overwritten by this synchronization. Tensor weights, optimizer states, data shards, credentials and installed binary caches are not Git payloads. Live file/pin inventories include exact paths, sizes and capture times. Loss values keep cross-entropy units; unmeasured values stay absent.','',
        'New Plus results use the documented corrected EvalPlus backend; historical custom Plus scores are not silently merged into the same protocol. MC accuracy and byte-normalized accuracy remain separate. GSM8K strict phrase extraction and relaxed answer extraction are separate metrics. A sample-flag recount verifies arithmetic, not all assumptions of the original execution harness.','',
        '[Build verification](reports/sync_verification.json) · [Input/capture issues](reports/sync_issues.json) · [All figures](reports/figures/all/index.json)','']
    if (out/'DIAGNOSTICS.md').exists():read+=['','[Additional diagnostic experiments and raw evidence](reports/DIAGNOSTICS.md)','']
    (root/'README.md').write_text('\n'.join(read))
    return summary

def build(root:Path,make_plots=True)->dict:
    cfg=json.loads((root/'publication.json').read_text());ptr=json.loads((root/'provenance/sync_latest.json').read_text());man=json.loads((root/ptr['manifest']).read_text());issues=[];verified=0
    for r in man['records']:
        raw(root,r);verified+=1
        if r.get('changed_during_read'):issues.append({'kind':'changed_during_capture','source':r['repo_path'],'note':'Captured bytes are hashed; live state may have advanced.'})
    if man['errors']:issues.extend(man['errors'])
    alltrain,current,runs,events,catalog=training(root,man,cfg,issues)
    rows,selected,latest,versions,states,checks,checked=evaluation(root,man,issues)
    cps=checkpoints(root,ptr,rows,current)
    figs=plots(root,cfg,current,runs,catalog,selected) if make_plots else []
    summary=reports(root,cfg,man,ptr,alltrain,current,runs,catalog,rows,selected,latest,versions,states,checks,checked,cps,figs,issues)
    dump(root/'reports/training_events.json',events)
    verification={'manifest_records_verified':verified,'new_per_example_records_recounted':checked,'new_score_checks':sum(len(x['checks']) for x in checks),'arithmetic_mismatches':0,'input_capture_issues':issues,'tensor_payloads_read':0,'new_gpu_tests_run_by_publication':0}
    dump(root/'reports/sync_verification.json',verification)
    return {k:v for k,v in summary.items() if k!='systems'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--no-plots',action='store_true');a=p.parse_args()
    print(json.dumps(build(a.root.resolve(),not a.no_plots),indent=2))
