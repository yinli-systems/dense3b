#!/usr/bin/env python3
"""Validate every archived file against its recorded SHA-256 without using a GPU."""
from __future__ import annotations
import argparse, ast, hashlib, json
from pathlib import Path

def validate(root: Path) -> dict:
    records=[]
    snapshot=json.loads((root/'provenance/snapshot.json').read_text())
    records.extend(snapshot['records'])
    for name in ['imported-results.json','supplemental.json']:
        p=root/'provenance'/name
        if p.exists():
            data=json.loads(p.read_text())
            records.extend(data['records'] if isinstance(data,dict) else data)
    failures=[];checked=0
    for rec in records:
        rel=Path(rec['repo_path'])
        if rel.is_absolute() or '..' in rel.parts:
            failures.append({'path':str(rel),'reason':'unsafe inventory path'});continue
        path=root/rel
        if not path.is_file():
            failures.append({'path':str(rel),'reason':'missing'});continue
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=rec['sha256']:
            failures.append({'path':str(rel),'reason':'SHA-256 differs'});continue
        checked+=1
    syntax_checked=0
    for path in root.rglob('*.py'):
        if any(part in ['.git','.venv','.venv-report','__pycache__'] for part in path.parts):continue
        try:ast.parse(path.read_text(),filename=str(path));syntax_checked+=1
        except (SyntaxError,UnicodeError) as exc:failures.append({'path':str(path.relative_to(root)),'reason':'Python syntax: '+str(exc)})
    result={'snapshot_utc':snapshot['snapshot_utc'],'inventory_records':len(records),'verified_records':checked,
            'python_sources_parsed':syntax_checked,'failures':failures,'cuda_or_training_executed':False}
    return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=validate(args.root.resolve());text=json.dumps(result,indent=2)+'\n';print(text)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    raise SystemExit(1 if result['failures'] else 0)
