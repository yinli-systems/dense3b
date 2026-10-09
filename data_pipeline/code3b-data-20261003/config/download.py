"""Resumable mirror download of the non-gated code/math sources for the 3B code model. Raw files only."""
import json, os, sys, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download
ROOT = "/data/run01/scxi253/code3b-data-20261003"
JOBS = [  # (repo, allow_patterns)
    ("tokyotech-llm/swallow-code", ["train/*", "README.md"]),
    ("common-pile/stackv2_edu_filtered", ["*.json.gz", "README.md"]),
    ("OpenCoder-LLM/opc-annealing-corpus", None),
    ("OpenCoder-LLM/opc-fineweb-math-corpus", None),
    ("tokyotech-llm/swallow-math", None),
    ("OpenCoder-LLM/opc-fineweb-code-corpus", None),
]
for repo, pat in JOBS:
    dest = os.path.join(ROOT, "raw", repo.replace("/", "__"))
    t = time.time(); print(json.dumps({"start": repo, "dest": dest}), flush=True)
    for attempt in range(20):
        try:
            p = snapshot_download(repo, repo_type="dataset", local_dir=dest, allow_patterns=pat, max_workers=8)
            break
        except Exception as e:
            print(json.dumps({"retry": repo, "attempt": attempt, "error": repr(e)[:300]}), flush=True); time.sleep(30)
    else:
        print(json.dumps({"FAILED": repo}), flush=True); continue
    size = sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(dest) for f in fs)
    print(json.dumps({"done": repo, "bytes": size, "seconds": round(time.time() - t)}), flush=True)
print(json.dumps({"ALL_DONE": True}), flush=True)
