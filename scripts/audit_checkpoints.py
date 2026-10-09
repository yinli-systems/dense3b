#!/usr/bin/env python3
"""Rebuild checkpoint tables from archived JSON evidence."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, os, re
from pathlib import Path
from typing import Any
VARIANTS={'dense3b':['dense3b'],'moe1b':['moe1b'],'reason3b':['r3b_s'],'dense-moe-1b':['dense1b_l','moe1b_r','dense1b_s']}
# Counts observed in the archived full-run outputs, not newer benchmark releases.
FULL_N={'humaneval':164,'mbpp':378,'cruxeval':800,'gsm8k':1319,'arc_easy':2376,'arc_challenge':1172,'hellaswag':10042,'winogrande':1267,'boolq':3270,'openbookqa':500,'sciq':1000,'mmlu_cf':14042,'mmlu_mc5':14042}
LCB_N={'added_in_v1':400,'added_in_v2':111,'added_in_v3':101,'added_in_v4':101,'added_in_v5':167,'added_in_v6':175,'release_v6_cumulative':1055}
BENCHMARKS=set(FULL_N)|{'heldout','fim','lcb'}
RATE_FIELDS={'acc','acc_norm','accuracy','pass_at_1','pass_rate','exact','terminated','suffix_dup','humaneval','humaneval_plus','mbpp','mbpp_plus','cruxeval_o','cruxeval_i','gsm8k'}
REF_RE=re.compile(r'/data/run01/scxi253/(?:backup/backup/)?(?:code3b-train|moe1b|reason3b|routing)/[^\s\"\'<>]*?step_\d+')
def finite(x:Any)->bool:return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def digest(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def jd(path:Path,obj:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def wc(path:Path,rows:list[dict])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
def identity(ref:Any)->tuple[str|None,str|None]:
    if not isinstance(ref,str):return None,None
    for s,repo,var in [('code3b-train/','dense3b','dense3b'),('/moe1b/','moe1b','moe1b'),('/reason3b/','reason3b','r3b_s')]:
        if s in ref:return repo,var
    if '/routing/' in ref:
        for v in VARIANTS['dense-moe-1b']:
            if v in ref:return 'dense-moe-1b',v
    return None,None
def step_of(ref:str)->int|None:
    m=re.search(r'step_(\d+)',ref);return int(m.group(1)) if m else None
def sample_n(task:str,metric:str,record:dict,inherited:Any=None)->int|None:
    n=record.get('n',inherited)
    if isinstance(n,dict) and task=='lcb':
        key=metric.removeprefix('lcb_')
        if key=='v6_slice_175_post_cutoff':key='added_in_v6'
        n=n.get(key)
    return int(n) if finite(n) and n>=0 and int(n)==n else None
def unit_for(task:str,metric:str)->str:
    field=metric.split('.')[-1]
    if field in ['seconds','time']:return 'seconds'
    if task in ['heldout','heldout_loss','domain_ce']:return 'cross_entropy'
    if task=='permuted_router_delta_ce':return 'delta_cross_entropy'
    if 'chars' in field:return 'characters'
    if 'tokens' in field:return 'tokens'
    if task=='routing_layers':return 'raw_statistic'
    if field in RATE_FIELDS or (task=='lcb' and metric.startswith('lcb_')):return 'fraction'
    return 'raw_statistic'
def flatten(task:str,record:dict,prefix:str='',inherited:Any=None):
    if not isinstance(record,dict):return
    current_n=record.get('n',inherited)
    for k,v in record.items():
        if k in ['n','samples','details','completion','outputs','passed','correct','failed','total','count','error','skipped']:continue
        key=(prefix+'.'+k).strip('.')
        if isinstance(v,dict):yield from flatten(task,v,key,current_n)
        elif finite(v):yield {'task':task,'metric':key,'reported_value':v,'unit':unit_for(task,key),'n':sample_n(task,key,record,current_n)}
def coverage(task:str,metric:str,n:int|None)->str:
    if n is None:return 'not_recorded'
    if n==0:return 'zero_samples'
    expected=FULL_N.get(task)
    if task=='fim':expected=164 if metric.startswith('single.') else 117 if metric.startswith('multi.') else None
    if task=='lcb':
        key=metric.removeprefix('lcb_')
        if key=='v6_slice_175_post_cutoff':key='added_in_v6'
        expected=LCB_N.get(key)
    if n in [4,6] and expected and n<expected:return 'smoke_test'
    if expected is not None:
        if n==expected:return 'full_recorded_count'
        return 'subset' if n<expected else 'nonstandard_count'
    return 'count_recorded'
def wilson(k:int,n:int):
    if n<=0:return None,None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-half),min(1.,center+half)
def protocol(d:dict)->tuple[str,str,dict]:
    m=d.get('manifest') or {}
    keys=['protocol','protocol_sha256','eval_suite_sha256','generate_sha256','tokenizer_sha256','dtype','temperature','batch','max_new','stops','exec_timeout_s']
    fields={k:m[k] for k in keys if k in m}
    if fields:
        sig=digest(json.dumps(fields,sort_keys=True).encode())[:16]
        status='manifest_recorded' if m.get('eval_suite_sha256') and m.get('generate_sha256') and m.get('tokenizer_sha256') else 'partial_manifest'
    else:sig,status='unversioned','unversioned'
    return sig,status,fields
def detail_matches(summary:dict,detail:dict)->bool:
    for a in summary['origins']:
        ap=Path(a['logical_path'])
        for b in detail['origins']:
            bp=Path(b['logical_path'])
            if a['origin_group']!=b['origin_group'] or ap.parent!=bp.parent or bp.name!=ap.stem+'.details.json':continue
            if a['origin']=='git_history' and b['origin']=='git_history' and not set(a.get('git_occurrences',[]))&set(b.get('git_occurrences',[])):continue
            return True
    return False

def recount(row:dict,candidates:list,issues:list)->dict:
    task,metric=row['task'],row['metric'];field=None
    if task in ['humaneval','mbpp']:field='plus' if metric.endswith('_plus') else 'base' if metric in [task,'pass_at_1'] else None
    elif task=='cruxeval':field={'cruxeval_o':'ok_o','cruxeval_i':'ok_i'}.get(metric)
    result={'verification':'summary_only','passed_recounted':None,'n_recounted':None,'value_recounted':None,'detail_sources':[],'ci95_low':None,'ci95_high':None}
    if field is None:return result
    checks=[]
    for blob,obj in candidates:
        samples=obj.get(task)
        if not isinstance(samples,list):continue
        flags=[s.get(field) for s in samples if isinstance(s,dict)]
        if not flags or not all(isinstance(x,bool) for x in flags):continue
        ids=[s.get('task',s.get('id')) for s in samples]
        duplicates=len(ids)-len(set(map(str,ids)))
        bad_plus=sum(bool(s.get('plus')) and not bool(s.get('base')) for s in samples) if field=='plus' else 0
        checks.append({'source':blob['path'],'sha256':blob['sha256'],'passed':sum(flags),'n':len(flags),'duplicates':duplicates,'plus_not_base':bad_plus})
    result['detail_sources']=[c['source'] for c in checks]
    complete=[c for c in checks if c['n']==row['n']]
    if not complete:
        if checks:result['verification']='incomplete_sample_evidence'
        return result
    values={(c['passed'],c['n']) for c in complete}
    if len(values)!=1 or any(c['duplicates'] or c['plus_not_base'] for c in complete):
        result['verification']='conflicting_sample_evidence';issues.append({'kind':'sample_evidence_conflict','evaluation_id':row['evaluation_id'],'task':task,'metric':metric,'checks':complete});return result
    k,n=next(iter(values));v=k/n if n else None
    result.update(passed_recounted=k,n_recounted=n,value_recounted=v)
    if v is not None and abs(v-row['reported_value'])<=0.0000501:
        result['verification']='sample_flags_recounted';result['ci95_low'],result['ci95_high']=wilson(k,n)
    else:
        result['verification']='summary_sample_mismatch';issues.append({'kind':'summary_sample_mismatch','evaluation_id':row['evaluation_id'],'task':task,'metric':metric,'reported':row['reported_value'],'recounted':v,'checks':complete})
    return result

def load_blobs(root:Path,manifest:dict)->dict:
    blobs={}
    for r in manifest['records']:
        h=r['sha256']
        if h not in blobs:
            p=root/r['repo_path'];data=p.read_bytes()
            if digest(data)!=h:raise ValueError('Evidence SHA-256 mismatch: '+str(p))
            blobs[h]={'sha256':h,'path':r['repo_path'],'text':data.decode('utf-8',errors='replace'),'obj':None,'origins':[]}
        blobs[h]['origins'].append(r)
    for b in blobs.values():
        names=[Path(o.get('logical_path',o['repo_path'])) for o in b['origins']]
        if Path(b['path']).suffix!='.json' and not any(p.suffix=='.json' for p in names):continue
        try:b['obj']=json.loads(b['text'])
        except ValueError:
            if any(p.name=='preflight.json' for p in names):
                # Startup diagnostics are mixed stdout, not benchmark result JSON.
                diagnostics=[]
                for line in b['text'].splitlines():
                    try:d=json.loads(line)
                    except ValueError:continue
                    if isinstance(d,dict):diagnostics.append(d)
                b['obj']=diagnostics[0] if diagnostics else None
                b['format_note']='Communication startup stdout with '+str(len(diagnostics))+' parseable JSON diagnostic record(s); raw banners/errors retained unchanged'
            else:b['json_parse_failed']=True
    return blobs

def training_events(blobs:dict,variants:list)->tuple:
    starts=[];observations=[];seen_start=set();seen_obs=set()
    for b in blobs.values():
        names=[Path(o['logical_path']).name for o in b['origins']]
        is_metric='metrics.jsonl' in names;is_log=any(n.endswith(('.out','.log')) for n in names)
        if not (is_metric or is_log):continue
        paths=' '.join(o['logical_path'] for o in b['origins'])
        if re.search(r'smoke|sweep|node-bench|fp8dbg|nccl|xnode|profile',paths,re.I):continue
        if not is_metric and not re.search(r'train-|IgM-|IgG-|IgE-|epitope-|rs-dense1b_|rs-moe1b_',paths):continue
        variant=next((v for v in variants if v in paths),variants[0] if len(variants)==1 else None)
        if variant is None:continue
        jobs=re.findall(r'(?:train-|epitope-[^-]+-|Ig[MEG]-[^-]+-|rs-[^-]+-)(\d+)',paths)
        job=max(map(int,jobs)) if jobs else 0
        for line,text in enumerate(b['text'].splitlines(),1):
            if not text.startswith('{'):continue
            try:d=json.loads(text)
            except ValueError:continue
            if not isinstance(d,dict):continue
            if d.get('event')=='start' and finite(d.get('start')):
                key=(variant,job,json.dumps(d,sort_keys=True))
                if key not in seen_start:
                    seen_start.add(key);starts.append({'variant':variant,'job_id':job,'start':int(d['start']),'seq':d.get('seq'),'tokens_per_step':d.get('tokens_per_step'),'source':b['path'],'source_line':line})
            if finite(d.get('step')) and (finite(d.get('loss')) or finite(d.get('val_loss'))):
                key=(variant,job,json.dumps(d,sort_keys=True))
                if key in seen_obs:continue
                seen_obs.add(key)
                observations.append({'variant':variant,'job_id':job,'step':int(d['step']),**{k:d.get(k) for k in ['loss','val_loss','lr','grad_norm','tok_per_s','mfu']},'source':b['path'],'source_line':line})
    return sorted(starts,key=lambda x:(x['variant'],x['job_id'],x['start'])),sorted(observations,key=lambda x:(x['variant'],x['step'],x['job_id']))

def build(root:Path)->dict:
    cfg=json.loads((root/'publication.json').read_text());name=cfg['name'];variants=VARIANTS[name]
    manifest=json.loads((root/'provenance/checkpoint_audit_sources.json').read_text())
    inventory=json.loads((root/'provenance/checkpoint_inventory.json').read_text())
    blobs=load_blobs(root,manifest);issues=[];registry={};metas=collections.defaultdict(list)
    for b in blobs.values():
        if b.get('json_parse_failed'):issues.append({'kind':'unparsed_json_source','source':b['path']})
        if any(o.get('changed_during_read') for o in b['origins']):issues.append({'kind':'source_changed_during_snapshot_read','source':b['path']})
    def cp(variant,step,kind=None,source=None):
        key=(variant,int(step))
        if key not in registry:registry[key]={'variant':variant,'step':int(step),'evidence_kinds':[],'sources':[],'locations':[]}
        c=registry[key]
        if kind and kind not in c['evidence_kinds']:c['evidence_kinds'].append(kind)
        if source and source not in c['sources']:c['sources'].append(source)
        return c
    for b in blobs.values():
        d=b['obj']
        if not isinstance(d,dict) or not finite(d.get('step')):continue
        v=d.get('args',{}).get('preset')
        if v in variants and 'config' in d:
            cp(v,d['step'],'metadata',b['path']);metas[(v,int(d['step']))].append((d,b))
    for entry in inventory['checkpoints']:
        d=entry.get('metadata',{});v=d.get('args',{}).get('preset');step=d.get('step')
        if v not in variants or not finite(step):continue
        c=cp(v,step,'filesystem_inventory','provenance/checkpoint_inventory.json')
        files=entry['files'];weights=[f for f in files if f['name']=='model.pt' and f['bytes']>0]
        ranks=[f for f in files if re.fullmatch(r'rank\d+\.pt',f['name']) and f['bytes']>0]
        complete=any(f['name']=='COMPLETE' for f in files)
        c['locations'].append({'path':entry['source'],'resolved_path':entry['resolved_source'],'model_file_present':bool(weights),'model_bytes':weights[0]['bytes'] if weights else 0,'optimizer_files':len(ranks),'expected_writers':d.get('writers'),'complete_marker':complete,'milestone_marker':any(f['name']=='MILESTONE' for f in files),'tensor_deserialization_tested':False})
    starts,observations=training_events(blobs,variants)
    for s in starts:
        if s['start']>0:cp(s['variant'],s['start']-1,'resume_start_inference',s['source'])
    details=[b for b in blobs.values() if isinstance(b['obj'],dict) and any(Path(o['logical_path']).name.endswith('.details.json') for o in b['origins'])]
    rows=[];evaluations=[];eval_docs={};references=[]
    for b in blobs.values():
        d=b['obj']
        if not isinstance(d,dict):continue
        ref=d.get('ckpt',d.get('checkpoint'));owner,var=identity(ref)
        if owner!=name:continue
        step=d.get('step',step_of(ref))
        if not finite(step):continue
        tasks=d.get('tasks');kind='suite'
        if not isinstance(tasks,dict):
            tasks={}
            if 'heldout_loss' in d:tasks['heldout']=d['heldout_loss'];kind='legacy'
            for t in ['humaneval','gsm8k']:
                if isinstance(d.get(t),dict):tasks[t]=d[t];kind='legacy'
            if 'ce' in d:tasks['domain_ce']=d['ce'];kind='domain_probe'
            if 'permute_dce' in d:tasks['permuted_router_delta_ce']=d['permute_dce']
            if 'layers' in d:tasks['routing_layers']=d['layers']
        if not tasks:continue
        c=cp(var,step,'evaluation_summary',b['path'])
        pid,pstatus,pfields=protocol(d)
        if pid=='unversioned':pid+='-'+('mc' if any(t in FULL_N for t in tasks) and kind!='legacy' and 'humaneval' not in tasks else kind)
        partners=[(x,x['obj']) for x in details if detail_matches(b,x)]
        eid=b['sha256'][:16];declared_hash=(d.get('manifest') or {}).get('checkpoint_model_sha256')
        e={'evaluation_id':eid,'variant':var,'step':int(step),'kind':kind,'checkpoint':ref,'declared_checkpoint_sha256':declared_hash,'protocol_id':pid,'protocol_status':pstatus,'protocol_fields':pfields,'task_names':list(tasks),'source':b['path'],'source_sha256':b['sha256'],'origin_count':len(b['origins']),'reported_tokens_B':d.get('tokens_B'),'probe_reported_tokens':d.get('tokens'),'sample_detail_candidates':[x['path'] for x,_ in partners]}
        e['seen_in_live_snapshot']=any(o['origin']=='remote_snapshot' for o in b['origins'])
        e['source_origins']=[{k:o.get(k) for k in ['source','origin','logical_path','mtime_ns','git_commit','git_occurrences']} for o in b['origins']]
        evaluations.append(e);eval_docs[eid]=d
        for task,result in tasks.items():
            if not isinstance(result,dict):continue
            if result.get('error') or result.get('skipped'):issues.append({'kind':'task_not_completed','evaluation_id':eid,'task':task,'detail':result})
            for item in flatten(task,result):
                if kind=='legacy' and item['metric'] in ['pass_at_1','accuracy']:
                    item['original_metric']=item['metric'];item['metric']=task
                row={k:e[k] for k in ['evaluation_id','variant','step','checkpoint','declared_checkpoint_sha256','protocol_id','protocol_status','source','source_sha256','reported_tokens_B']}
                row.update(item);row['coverage']=coverage(task,item['metric'],item['n']);row['source_line']=None
                row.update(recount(row,partners,issues))
                if row['unit']=='fraction' and not 0<=row['reported_value']<=1:issues.append({'kind':'out_of_range_fraction','evaluation_id':eid,'task':task,'metric':row['metric']})
                if row['verification']=='summary_only' and task in ['humaneval','gsm8k'] and kind=='legacy':
                    n=row['n'];k=result.get('passed',result.get('correct'))
                    if finite(k) and n and abs(k/n-row['reported_value'])<=.0000501:row['verification']='summary_count_consistent_not_sample_verified'
                rows.append(row)
    # Log lines are corroboration, not extra evaluations, unless a metric is absent from every JSON version.
    existing={(r['variant'],r['step'],r['task'],r['metric'],r['n'],r['reported_value']) for r in rows}
    correlations=[];attempts=[];filename_targets=collections.defaultdict(set)
    for e in evaluations:
        for o in blobs[e['source_sha256']]['origins']:filename_targets[Path(o['logical_path']).name].add((e['variant'],e['step']))
    for b in blobs.values():
        is_log=any(Path(o['logical_path']).suffix in ['.out','.log'] for o in b['origins'])
        refs=sorted(set(REF_RE.findall(b['text'])))
        for ref in refs:
            owner,v=identity(ref)
            if owner==name:references.append({'variant':v,'step':step_of(ref),'reference':ref,'source':b['path'],'evidence':'text_reference_only'})
        if not is_log:continue
        targets={(identity(ref)[1],step_of(ref)) for ref in refs if identity(ref)[0]==name}
        if not targets:
            for line in b['text'].splitlines():
                if line.startswith('DONE '):targets.update(filename_targets.get(Path(line[5:].strip()).name,set()))
        if len(targets)!=1:continue
        var,step=next(iter(targets))
        record={'variant':var,'step':step,'source':b['path'],'eval_start':'EVAL_START' in b['text'] or 'evaluating ' in b['text'],'eval_done':'EVAL_DONE' in b['text'],'has_traceback':'Traceback (most recent call last)' in b['text'],'corroborated_metrics':0,'recovered_metrics':0}
        if record['eval_start'] or record['eval_done']:attempts.append(record)
        for line,text in enumerate(b['text'].splitlines(),1):
            if not text.startswith('{'):continue
            try:d=json.loads(text)
            except ValueError:continue
            if not isinstance(d,dict):continue
            for task,result in d.items():
                if task not in BENCHMARKS or not isinstance(result,dict):continue
                for item in flatten(task,result):
                    key=(var,step,task,item['metric'],item['n'],item['reported_value'])
                    if key in existing:
                        record['corroborated_metrics']+=1
                        correlations.append({'variant':var,'step':step,'task':task,'metric':item['metric'],'value':item['reported_value'],'source':b['path'],'source_line':line})
                        continue
                    existing.add(key);record['recovered_metrics']+=1
                    cp(var,step,'log_result',b['path'])
                    rows.append({'evaluation_id':'log-'+b['sha256'][:12],'variant':var,'step':step,'checkpoint':None,'declared_checkpoint_sha256':None,'protocol_id':'unversioned-log','protocol_status':'unversioned','source':b['path'],'source_sha256':b['sha256'],'reported_tokens_B':None,**item,'coverage':coverage(task,item['metric'],item['n']),'source_line':line,'verification':'log_only','passed_recounted':None,'n_recounted':None,'value_recounted':None,'detail_sources':[],'ci95_low':None,'ci95_high':None})
    by_cp=collections.defaultdict(list)
    for r in rows:by_cp[(r['variant'],r['step'])].append(r)
    for key,c in registry.items():
        v,step=key;ms=metas[key];values=set();configs=set()
        for d,b in ms:
            configs.add(digest(json.dumps(d['config'],sort_keys=True).encode()))
            seq=d.get('args',{}).get('seq_len') or d['config'].get('max_seq_len')
            consumed=d.get('samples_consumed')
            if finite(seq) and finite(consumed):values.add(int(seq*consumed))
        c['metadata_versions']=len(ms);c['config_hashes']=sorted(configs)
        c['tokens_from_metadata']=next(iter(values)) if len(values)==1 else None
        c['tokens_schedule_reconstruction']=None;c['token_status']='not_verified'
        if len(values)>1:issues.append({'kind':'conflicting_checkpoint_token_metadata','variant':v,'step':step,'values':sorted(values)})
        if len(configs)>1:issues.append({'kind':'conflicting_architecture_metadata','variant':v,'step':step,'config_hashes':sorted(configs)})
        if len(values)==1:c['token_status']='samples_consumed_times_sequence_length'
        else:
            schedules={s['tokens_per_step'] for s in starts if s['variant']==v and finite(s['tokens_per_step'])}
            seqs={s['seq'] for s in starts if s['variant']==v and finite(s['seq'])}
            if len(schedules)==1 and len(seqs)==1:
                c['tokens_schedule_reconstruction']=int((step+1)*next(iter(schedules)));c['token_status']='constant_observed_schedule_reconstruction_not_metadata_verified'
        c['tokens_B']=((c['tokens_from_metadata'] or c['tokens_schedule_reconstruction'])/1e9) if (c['tokens_from_metadata'] is not None or c['tokens_schedule_reconstruction'] is not None) else None
        c['weights_present']=any(l['model_file_present'] for l in c['locations'])
        c['storage_status']='retained_model_files' if c['weights_present'] else 'historical_metadata_only' if ms else 'historical_result_only' if by_cp[key] else 'resume_reference_only'
        c['model_hashes_recorded_in_evaluations']=sorted({r['declared_checkpoint_sha256'] for r in by_cp[key] if r.get('declared_checkpoint_sha256')})
        if len(c['model_hashes_recorded_in_evaluations'])>1:issues.append({'kind':'multiple_weight_hashes_at_same_step','variant':v,'step':step,'hashes':c['model_hashes_recorded_in_evaluations']})
        c['evaluation_ids']=sorted({r['evaluation_id'] for r in by_cp[key]})
        c['metric_rows']=len(by_cp[key]);c['full_count_tasks']=sorted({r['task'] for r in by_cp[key] if r['coverage']=='full_recorded_count' and r['unit']=='fraction'})
        c['missing_primary_tasks']=[t for t in FULL_N if t not in c['full_count_tasks']]
        c['evaluation_status']='no_result_found' if not by_cp[key] else 'primary_task_counts_present' if not c['missing_primary_tasks'] else 'partial_task_coverage'
        obs=[x for x in observations if x['variant']==v and step-10<=x['step']<=step]
        nearest={}
        for x in obs:
            j=x['job_id']
            if j not in nearest or x['step']>nearest[j]['step']:nearest[j]=x
        c['nearby_training_observations']=list(nearest.values())
        c['validation_observations']=[x for x in observations if x['variant']==v and x['step']==step and finite(x.get('val_loss'))]
        for r in by_cp[key]:
            r['tokens_from_metadata']=c['tokens_from_metadata'];r['tokens_schedule_reconstruction']=c['tokens_schedule_reconstruction'];r['token_status']=c['token_status']
            reported=r.get('reported_tokens_B')
            if finite(reported) and c['tokens_B'] is not None and abs(reported-c['tokens_B'])>.00501:
                # One issue per file, not one duplicate per metric.
                issue={'kind':'token_count_mismatch' if c['tokens_from_metadata'] is not None else 'token_count_schedule_discrepancy','evaluation_id':r['evaluation_id'],'variant':v,'step':step,'reported_tokens_B':reported,'metadata_or_schedule_tokens_B':c['tokens_B'],'basis':c['token_status']}
                if issue not in issues:issues.append(issue)
    # Locate exact archived evaluator/tokenizer files, including files already tracked outside the raw evidence tree.
    hash_paths={h:b['path'] for h,b in blobs.items()}
    wanted={h for e in evaluations for k,h in e['protocol_fields'].items() if k.endswith('_sha256') and isinstance(h,str)}-set(hash_paths)
    for p in root.rglob('*'):
        if not wanted:break
        if not p.is_file() or any(x in p.relative_to(root).parts for x in ['.git','reports','evidence']):continue
        if p.suffix not in ['.py','.json','.txt','.yaml','.toml']:continue
        h=digest(p.read_bytes())
        if h in wanted:hash_paths[h]=str(p.relative_to(root));wanted.remove(h)
    for e in evaluations:
        e['resolved_manifest_files']={k:hash_paths[h] for k,h in e['protocol_fields'].items() if k.endswith('_sha256') and isinstance(h,str) and h in hash_paths}
        e['unresolved_manifest_hashes']={k:h for k,h in e['protocol_fields'].items() if k.endswith('_sha256') and isinstance(h,str) and h not in hash_paths}
    cps=sorted(registry.values(),key=lambda c:(c['variant'],c['step']))
    rows.sort(key=lambda r:(r['variant'],r['step'],r['task'],r['metric'],r['evaluation_id']))
    references=[dict(r,checkpoint_has_other_evidence=(r['variant'],r['step']) in registry) for r in references]
    summary={'repository':name,'snapshot_utc':inventory['snapshot_utc'],'checkpoint_entries':len(cps),'retained_checkpoint_steps':sum(c['weights_present'] for c in cps),'checkpoint_directories':len(inventory['checkpoints']),'checkpoints_with_results':sum(bool(c['metric_rows']) for c in cps),'checkpoints_without_results':sum(not c['metric_rows'] for c in cps),'retained_without_results':sum(c['weights_present'] and not c['metric_rows'] for c in cps),'unique_json_evaluation_versions':len(evaluations),'metric_rows':len(rows),'sample_recounted_rows':sum(r['verification']=='sample_flags_recounted' for r in rows),'source_origins':len(manifest['records']),'unique_source_blobs_verified':len(blobs),'issues_by_kind':dict(collections.Counter(x['kind'] for x in issues)),'variants_without_checkpoints':[v for v in variants if not any(c['variant']==v for c in cps)],'tensor_files_loaded':0,'gpu_evaluations_launched':0}
    out=root/'reports/checkpoint_audit';out.mkdir(parents=True,exist_ok=True)
    for filename,obj in [('summary.json',summary),('checkpoints.json',cps),('evaluation_versions.json',evaluations),('all_metrics.json',rows),('issues.json',issues),('evaluation_attempts.json',attempts),('log_corroboration.json',correlations),('training_starts.json',starts)]:jd(out/filename,obj)
    for filename,obj in [('checkpoints.csv',cps),('all_metrics.csv',rows),('checkpoint_references.csv',references),('training_observations.csv',observations)]:wc(out/filename,obj)
    write_reports(root,summary,cps,rows,evaluations,issues)
    slots=write_schedule_coverage(root,cps,inventory,observations)
    summary['unconfirmed_schedule_slots']=sum(not x['has_direct_or_resume_evidence'] for x in slots)
    summary['checkpoints_with_exact_validation_loss']=sum(bool(c['validation_observations']) for c in cps)
    notes=[{'source':b['path'],'format':b['format_note']} for b in blobs.values() if b.get('format_note')]
    summary['diagnostic_json_with_nccl_banner']=len(notes)
    jd(out/'diagnostic_format_notes.json',notes);jd(out/'summary.json',summary)
    return summary
def mdlink(root:Path,target:str,page:Path,label:str='source',line:Any=None)->str:
    rel=os.path.relpath(root/target,page.parent).replace(os.sep,'/')
    return '['+label+']('+rel+(f'#L{line}' if line else '')+')'
def display_value(r:dict)->str:
    v=r['reported_value']
    if r['unit']=='fraction':return f'{v*100:.4f}%'
    if r['unit'] in ['cross_entropy','delta_cross_entropy']:return f'{v:.6f}'
    return f'{v:.8g}'
def write_reports(root:Path,summary:dict,cps:list,rows:list,evaluations:list,issues:list)->None:
    out=root/'reports/checkpoint_audit';index=out/'CHECKPOINTS.md';name=summary['repository']
    lines=[f'# {name}: every located checkpoint and result','',f"Snapshot: **{summary['snapshot_utc']}**.",'',
        '**This is an evidence audit, not a new GPU evaluation.** Tensor contents were not read, loaded or re-hashed. A model file present in the snapshot does not prove that it can be deserialized. Missing scores are never zero-filled.','',
        f"Located **{summary['checkpoint_entries']} checkpoint entries**: **{summary['retained_checkpoint_steps']} with model files present**, **{summary['checkpoints_with_results']} with downstream/probe evaluation results**, and **{summary['checkpoints_without_results']} with no downstream/probe evaluation result found. Training and validation observations are listed separately. The inventory contains {summary['checkpoint_directories']} directories; directory aliases do not create extra training steps.",'',
        f"Recovered **{summary['unique_json_evaluation_versions']} distinct JSON evaluation versions** and **{summary['metric_rows']} typed metric rows**, including timing and routing statistics. **{summary['sample_recounted_rows']} score rows were independently re-aggregated from saved Boolean sample flags**. This validates arithmetic and sample counts, not the execution harness itself.",'',
        '[All evaluation values (CSV)](all_metrics.csv) · [All training observations](training_observations.csv) · [Unconfirmed scheduled slots](SCHEDULE_COVERAGE.md) · [All numeric results (JSON)](all_metrics.json) · [Every evaluation version](evaluation_versions.json) · [Integrity and protocol issues](ISSUES.md) · [Missing evaluations](GAPS.md) · [Method](METHOD.md)','']
    evalmap={e['evaluation_id']:e for e in evaluations};gaprows=[]
    for variant in VARIANTS[name]:
        selected=[c for c in cps if c['variant']==variant]
        lines.extend([f'## {variant}',''])
        if not selected:
            lines.extend(['No checkpoint metadata, retained model file or completed benchmark result was found in the searched sources. Do not read this as a score of zero.','']);continue
        lines.extend(['| Step / full result page | Model file in snapshot | Tokens (B) | Token basis | Result rows | Full-count task families | Missing primary task families |','|---|---|---:|---|---:|---:|---:|'])
        for c in selected:
            cpfile=out/'checkpoints'/f"{variant}-step_{c['step']:08d}.md";cpfile.parent.mkdir(exist_ok=True)
            tok='—' if c['tokens_B'] is None else f"{c['tokens_B']:.6f}"
            basis='metadata' if c['tokens_from_metadata'] is not None else 'schedule estimate' if c['tokens_schedule_reconstruction'] is not None else 'unverified'
            lines.append(f"| [{c['step']}](checkpoints/{cpfile.name}) | {'present' if c['weights_present'] else 'not found'} | {tok} | {basis} | {c['metric_rows']} | {len(c['full_count_tasks'])}/{len(FULL_N)} | {len(c['missing_primary_tasks'])} |")
            cr=[r for r in rows if r['variant']==variant and r['step']==c['step']]
            page=[f"# {variant} — checkpoint {c['step']}",'','[All checkpoints](../CHECKPOINTS.md)','',
                f"**Storage:** `{c['storage_status']}`. **Evaluation coverage:** `{c['evaluation_status']}`.",'',
                f"**Tokens:** {tok} B; basis: `{c['token_status']}`. Schedule reconstructions are explicitly not metadata-verified; they describe logical stream progress and exclude replayed work after rollbacks.",'',
                '**Evidence types:** '+', '.join(c['evidence_kinds'])+'.','',
                '**Missing primary task families:** '+(', '.join(c['missing_primary_tasks']) or 'none at the recorded full sample counts')+'.','',
                '**Recorded weight hashes:** '+(', '.join('`'+x+'`' for x in c['model_hashes_recorded_in_evaluations']) or 'not recorded')+'. These are hashes stated in evaluation manifests, not new checksums from this audit.','',
                '## Checkpoint locations','',
                '| Path | Model bytes | Optimizer shard files | COMPLETE marker |','|---|---:|---:|---|']
            if c['locations']:
                for loc in c['locations']:page.append(f"| `{loc['path']}` | {loc['model_bytes']} | {loc['optimizer_files']} | {loc['complete_marker']} |")
            else:page.append('| Not present in the live inventory | — | — | — |')
            page.extend(['','Historical metadata, results and resume traces are retained even when the model weights were pruned. A resume trace only establishes that the trainer reported a resumed start; it is weaker than a saved metadata file.','',
                         '## Training observations near this checkpoint','',
                         'Each row keeps its original job and actual observed step. An observation up to 10 steps earlier is not relabeled as a checkpoint measurement.','',
                         '| Job | Observed step | Train loss | Validation loss | Source |','|---|---:|---:|---:|---|'])
            if c['nearby_training_observations']:
                for x in c['nearby_training_observations']:
                    fmt=lambda z:'—' if not finite(z) else f'{z:.6f}'
                    page.append(f"| {x['job_id']} | {x['step']} | {fmt(x['loss'])} | {fmt(x['val_loss'])} | {mdlink(root,x['source'],cpfile,line=x['source_line'])} |")
            else:page.append('| — | — | No nearby observation found | — | — |')
            page.extend(['','## Every evaluation version','',
                         'A later file is not automatically better. Protocol fingerprints, checkpoint hashes, sample counts, smoke runs and re-runs stay separate. Percent signs are used only for fractions; loss and router statistics keep their native units.',''])
            if not cr:page.extend(['**No downstream/probe evaluation result was found for this checkpoint.** Training-loop observations, when available, are listed above. It remains in the inventory; no metric has been invented.',''])
            for eid in c['evaluation_ids']:
                er=[r for r in cr if r['evaluation_id']==eid];e=evalmap.get(eid)
                page.extend([f'### Evaluation `{eid}`',''])
                if e:
                    page.extend([f"Protocol: `{e['protocol_id']}` ({e['protocol_status']}); original token summary: `{e['reported_tokens_B']}` B.",'',
                        mdlink(root,e['source'],cpfile,'Original result')+' · '+mdlink(root,'reports/checkpoint_audit/evaluation_versions.json',cpfile,'Full version manifest'), ''])
                else:page.extend(['Recovered from a log; no matching complete JSON summary was found for these metric values.',''])
                page.extend(['| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |','|---|---|---:|---|---:|---|---|---|'])
                for r in er:
                    count='—' if r['passed_recounted'] is None else f"{r['passed_recounted']}/{r['n_recounted']}"
                    page.append(f"| {r['task']} | {r['metric']} | {display_value(r)} | {r['unit']} | {r['n'] if r['n'] is not None else '—'} | {r['coverage']} | {count} | {r['verification']} |")
                page.append('')
            page.extend(['## Provenance',''])
            page.extend('- '+mdlink(root,s,cpfile,s) for s in c['sources'])
            cpfile.write_text('\n'.join(page)+'\n')
            if c['missing_primary_tasks']:
                gaprows.append({'variant':variant,'step':c['step'],'weights_present_at_snapshot':c['weights_present'],'storage_status':c['storage_status'],'missing_full_count_task_families':c['missing_primary_tasks'],'candidate_paths':[l['path'] for l in c['locations'] if l['model_file_present']]})
        lines.append('')
    index.write_text('\n'.join(lines)+'\n');jd(out/'evaluation_gaps.json',gaprows)
    gap=['# Evaluation gaps','',f"Snapshot: {summary['snapshot_utc']}",'',
         'This is a missing-evidence list, not a job submission. Availability must be checked again before any rerun: automatic pruning can remove old weights. Full-count task coverage does not certify that the historical harness was correct.','',
         '| Variant | Step | Model file present | Missing full-count task families |','|---|---:|---|---|']
    for g in gaprows:gap.append(f"| {g['variant']} | {g['step']} | {g['weights_present_at_snapshot']} | {', '.join(g['missing_full_count_task_families'])} |")
    gap.extend(['','[Machine-readable gaps](evaluation_gaps.json) · [All checkpoints](CHECKPOINTS.md)',''])
    (out/'GAPS.md').write_text('\n'.join(gap))
    issue_lines=['# Evidence and protocol issues','',
        'Raw files have not been edited. Discrepancies below describe the archived evidence. No score is selected because it is the highest.','']
    for i in issues:issue_lines.extend(['### '+i['kind'],'','```json',json.dumps(i,indent=2),'```',''])
    if not issues:issue_lines.append('No arithmetic or token-accounting discrepancy was detected by the implemented checks. This does not prove that every original benchmark implementation was correct.')
    issue_lines.extend(['','## Global limitations','',
        'Multiple-choice summaries do not include a complete versioned evaluation manifest or per-example predictions. MMLU `mmlu_cf` and `mmlu_mc5` are distinct custom archived modes, not interchangeable labels for an official leaderboard number. GSM8K summaries lack full per-example scoring evidence in the captured suite detail files.','',
        'Old manifests reference different evaluator and generation code hashes. Curves across these versions are not controlled comparisons. Complete source-hash resolution is listed in evaluation_versions.json.','',
        'Some legacy and smoke evaluations contain only a handful of examples. They are preserved, explicitly labeled, and not substituted for full-run values.',''])
    (out/'ISSUES.md').write_text('\n'.join(issue_lines))
    (out/'METHOD.md').write_text('''# Audit method and limits

The audit reads `provenance/checkpoint_audit_sources.json` and verifies SHA-256 before parsing any source. Rebuild with `python scripts/audit_checkpoints.py --root .`.

## Search and identity

The scope is the four current publication repositories, their live ParaCloud run/source roots and resolved backup targets, the original code3b-pretraining repository and its three branches, and all four publication histories. It is not a claim to have searched every unrelated older model project or external cloud account. The filesystem inventory is a point-in-time snapshot.

A checkpoint enters the main inventory through saved metadata, an evaluation result, a retained model-file listing, or an explicitly labeled resumed-start inference. Other path references are retained in checkpoint_references.csv and do not prove that a scheduled checkpoint was ever written. Variant, step, architecture metadata and declared checkpoint hashes are retained. Distinct hashes are flagged instead of silently merged.

## Results and arithmetic

Every JSON version is preserved by content hash. Repeated copies are source aliases, not extra runs. The result parser includes legacy summaries, nested FIM metrics, correct LiveCodeBench slice denominators, cross-entropy greater than one, all routing-layer scalars and permutation delta loss. Raw summaries remain untouched. Log lines corroborate existing values; otherwise they are explicitly marked as log-only recovery.

HumanEval, MBPP and CRUXEval Boolean judgments are re-counted when a matching complete details file is present. Counts, duplicate identifiers, Plus/base consistency and agreement with four-decimal summaries are checked. This is an independent arithmetic aggregation of saved flags, not an independent rerun of the program or benchmark harness. Sparse legacy example snippets are not treated as complete sample evidence. Optional Wilson intervals in the numeric table describe binomial sampling uncertainty only, not protocol bias or contamination.

## Coverage and units

Full-count labels use sample counts observed in the archived full-run protocol. A count alone does not prove the right item IDs or prompt implementation were used. Smoke/subset/full-count records remain distinct. Fractions may be displayed as percentages; loss, delta loss, characters, tokens and timing remain in native units. Missing tasks are absent or null, never a fabricated zero. Optional FIM, LiveCodeBench and domain probes are not required to call the primary task-family count complete; all their observed values remain visible.

## Tokens and training traces

A saved samples_consumed count times sequence length is distinguished from a reconstruction using the single constant observed production token-per-step schedule. Reconstructed values remain explicitly not metadata-verified. Counts describe logical stream progress, not total compute including replayed steps after rollback. Original rounded token claims are retained and discrepancies reported. Training observations keep their original job, source line and observed step; no near-by value is silently relabeled as a checkpoint measurement.

## No changes to training

This audit did not launch GPU evaluation, deserialize weights, hash tensor payloads, modify checkpoint retention, restart training, or mutate the original logs. File presence and optimizer shard counts are not a loadability test. Old metadata or scores cannot reconstruct deleted weights. Available paths must be rechecked before a rerun because training and pruning continue.
''')

def write_schedule_coverage(root:Path,cps:list,inventory:dict,observations:list)->list:
    """Show possible scheduled slots without pretending a schedule proves a saved file."""
    slots=[];known={(c['variant'],c['step']) for c in cps}
    for variant in VARIANTS[inventory['checkpoints'][0]['repository']] if inventory['checkpoints'] else []:
        intervals={e.get('metadata',{}).get('args',{}).get('ckpt_every') for e in inventory['checkpoints'] if e.get('metadata',{}).get('args',{}).get('preset')==variant}
        intervals={int(x) for x in intervals if finite(x) and x>0}
        obs=[x for x in observations if x['variant']==variant]
        if not intervals or not obs:continue
        last=max(x['step'] for x in obs)
        candidates=sorted({step for every in intervals for step in range(every-1,last+1,every)})
        for step in candidates:
            at=[x for x in obs if x['step']==step and finite(x.get('val_loss'))]
            latest=max(at,key=lambda x:x['job_id']) if at else None
            slots.append({'variant':variant,'step':step,'observed_save_intervals':sorted(intervals),'has_direct_or_resume_evidence':(variant,step) in known,'exact_validation_loss':latest['val_loss'] if latest else None,'validation_source':latest['source'] if latest else None,'validation_source_line':latest['source_line'] if latest else None,'interpretation':'recorded checkpoint entry' if (variant,step) in known else 'schedule-derived candidate only; file creation and retention unconfirmed'})
    out=root/'reports/checkpoint_audit';wc(out/'schedule_coverage.csv',slots);jd(out/'schedule_coverage.json',slots)
    lines=['# Scheduled checkpoint coverage — unconfirmed slots stay unconfirmed','',
        'This is the union of periodic save intervals observed in retained checkpoint metadata, bounded by recorded production progress. It is not an exact schedule-change chronology. A candidate slot does not prove that a save succeeded, or even that the same interval applied at that early step. These candidates are not added to the count of located checkpoint entries.','',
        'Validation loss can exist at a slot even when checkpoint metadata and weights were not retained. Such loss values remain training/evaluation observations, not proof of a surviving checkpoint file.','',
        '| Variant | Candidate step | Save intervals observed | Other checkpoint evidence | Exact-step validation loss |','|---|---:|---|---|---:|']
    for x in slots:
        loss='—' if x['exact_validation_loss'] is None else f"{x['exact_validation_loss']:.6f}"
        lines.append(f"| {x['variant']} | {x['step']} | {x['observed_save_intervals']} | {x['has_direct_or_resume_evidence']} | {loss} |")
    lines+=['','[Machine-readable slots](schedule_coverage.csv) · [Located checkpoint entries](CHECKPOINTS.md)','']
    (out/'SCHEDULE_COVERAGE.md').write_text('\n'.join(lines))
    return slots

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));a=p.parse_args()
    print(json.dumps(build(a.root.resolve()),indent=2))
if __name__=='__main__':main()
