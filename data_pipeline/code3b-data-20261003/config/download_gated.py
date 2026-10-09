"""Phase 2 (gated, token from ~/.cache/huggingface/token): high-quality subsets only. Raw files only."""
import json, os, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download
ROOT = "/data/run01/scxi253/code3b-data-20261003"
SC_LANGS = ["python","java","javascript","typescript","c","cpp","c-sharp","go","rust","php","ruby","kotlin","scala","shell","sql",
            "swift","lua","haskell","julia","dart","jupyter-scripts-dedup-filtered","jupyter-structured-clean-dedup",
            "github-issues-filtered-structured","git-commits-cleaned"]
JOBS = [
    # Nemotron-Pretraining-Code-v1 and Nemotron-CC-Code-v1: added after NVIDIA approves access
    ("nvidia/Nemotron-CC-Math-v1", ["4plus/*", "4plus_MIND/*", "README.md"]),
    ("bigcode/starcoderdata", [l + "/*" for l in SC_LANGS] + ["README.md"]),
]
for repo, pat in JOBS:
    dest = os.path.join(ROOT, "raw", repo.replace("/", "__")); t = time.time()
    print(json.dumps({"start": repo}), flush=True)
    for attempt in range(20):
        try:
            snapshot_download(repo, repo_type="dataset", local_dir=dest, allow_patterns=pat, max_workers=16); break
        except Exception as e:
            print(json.dumps({"retry": repo, "attempt": attempt, "error": repr(e)[:300]}), flush=True); time.sleep(30)
    else:
        print(json.dumps({"FAILED": repo}), flush=True); continue
    size = sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(dest) for f in fs)
    print(json.dumps({"done": repo, "bytes": size, "seconds": round(time.time() - t)}), flush=True)
print(json.dumps({"ALL_DONE": True}), flush=True)
