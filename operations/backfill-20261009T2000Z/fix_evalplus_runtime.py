from pathlib import Path
import shutil,hashlib,json,difflib
C=Path('/data/run01/scxi253/checkpoint-eval-backfill-20261009T2000Z');E=Path('/data/run01/scxi253/code3b-train/evalplus-venv/lib/python3.10/site-packages');dest=C/'score-runtime/evalplus'
if dest.exists():raise SystemExit('Overlay exists; refusing blind overwrite')
shutil.copytree(E/'evalplus',dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
p=dest/'eval/__init__.py';raw=p.read_bytes();text=raw.decode();newline='\r\n' if '\r\n' in text else '\n';normalized=text.replace('\r\n','\n')
old='''                        if "find_zero" == entry_point:
                            assert abs(_poly(*inp, out)) <= atol
                            continue'''
new='''                        if "find_zero" == entry_point:
                            assert abs(_poly(*inp, out)) <= atol
                            # Record the successful test before this special-oracle early continue.
                            details[i] = True
                            progress.value += 1
                            continue'''
assert normalized.count(old)==1
fixed=normalized.replace(old,new);out=fixed.replace('\n',newline).encode();p.write_bytes(out)
patch=''.join(difflib.unified_diff(normalized.splitlines(True),fixed.splitlines(True),fromfile='official-installed/evalplus/eval/__init__.py',tofile='score-runtime/evalplus/eval/__init__.py'))
(C/'evalplus-find-zero-progress.patch').write_text(patch)
info={'source':str(E/'evalplus/eval/__init__.py'),'original_sha256':hashlib.sha256(raw).hexdigest(),'patched_sha256':hashlib.sha256(out).hexdigest(),'change':'Record True and increment progress for successful find_zero special-oracle tests; assertions, tolerance, inputs and model outputs unchanged','installed_environment_modified':False}
(C/'evalplus-patch-provenance.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
