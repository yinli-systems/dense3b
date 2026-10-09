"""Fast, verified dataset download for the cluster login node.

Paths, sizes and sha256 come from the Hugging Face listing (via hf-mirror). Each file is fetched from the
domestic ModelScope CDN when ModelScope serves a copy of exactly the same size, otherwise from hf-mirror.
Downloads are resumable (curl -C -), run in parallel, and a file is only moved into place after its sha256
matches the Hugging Face LFS digest. Already-complete files are verified and skipped."""
import argparse, concurrent.futures as cf, fnmatch, hashlib, json, os, subprocess, sys, threading, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import HfApi, get_token

LOCK = threading.Lock()


def log(**kw):
    with LOCK:
        print(json.dumps(kw), flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def head_len(url):
    r = subprocess.run(["curl", "-s", "-L", "-I", "-m", "30", "-A", "curl/8", url], capture_output=True, text=True)
    lens = [l.split(":", 1)[1].strip() for l in r.stdout.splitlines() if l.lower().startswith("content-length")]
    codes = [l for l in r.stdout.splitlines() if l.startswith("HTTP/")]
    ok = codes and codes[-1].split()[1] == "200"
    return int(lens[-1]) if ok and lens else None


def fetch(job, f, root, token):
    repo, ms = job["repo"], job.get("modelscope")
    dest = os.path.join(root, "raw", repo.replace("/", "__"), f["path"])
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) == f["size"]:
        if f["sha256"] is None or os.path.exists(dest + ".sha256ok") or sha256(dest) == f["sha256"]:
            open(dest + ".sha256ok", "w").close()
            return "skip", 0
    src = None
    if ms:
        url = f"https://www.modelscope.cn/api/v1/datasets/{ms}/repo?Revision=master&FilePath={f['path']}"
        if head_len(url) == f["size"]:
            src, hdr = url, []
    if src is None:
        src = f"https://hf-mirror.com/datasets/{repo}/resolve/main/{f['path']}"
        hdr = ["-H", "Authorization: Bearer " + token] if token else []
    part = dest + ".part"; t = time.time()
    for attempt in range(30):
        r = subprocess.run(["curl", "-s", "-S", "-L", "-A", "curl/8", "-C", "-", "--retry", "5", "--retry-delay", "10",
                            "--speed-limit", "20000", "--speed-time", "120", "-o", part] + hdr + [src],
                           capture_output=True, text=True)
        size = os.path.getsize(part) if os.path.exists(part) else 0
        if size == f["size"]:
            break
        if size > f["size"]:
            os.remove(part)
        time.sleep(10)
    else:
        raise RuntimeError(f"incomplete after retries: {f['path']} ({size}/{f['size']}) {r.stderr[-200:]}")
    if f["sha256"] and sha256(part) != f["sha256"]:
        os.remove(part)
        raise RuntimeError("sha256 mismatch: " + f["path"])
    os.replace(part, dest); open(dest + ".sha256ok", "w").close()
    return ("modelscope" if "modelscope" in src else "hf-mirror"), f["size"] / max(1e-9, time.time() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="JSON list of {repo, patterns, modelscope?}")
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs)); token = get_token(); api = HfApi(); todo = []
    for job in jobs:
        files = [x for x in api.list_repo_tree(job["repo"], repo_type="dataset", recursive=True)
                 if getattr(x, "size", None) is not None and any(fnmatch.fnmatch(x.path, p) for p in job["patterns"])]
        for x in files:
            lfs = getattr(x, "lfs", None)
            todo.append((job, dict(path=x.path, size=x.size, sha256=(lfs.sha256 if lfs else None))))
        log(listed=job["repo"], files=len(files), gb=round(sum(x.size for x in files) / 1e9, 1))
    todo.sort(key=lambda jf: -jf[1]["size"])
    total = sum(f["size"] for _, f in todo); done = 0; fetched = 0; started = time.time(); failed = []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch, j, f, a.root, token): (j, f) for j, f in todo}
        for fu in cf.as_completed(futs):
            j, f = futs[fu]
            try:
                how, rate = fu.result()
            except Exception as e:
                failed.append(f["path"]); log(failed=j["repo"], path=f["path"], error=str(e)[:300]); continue
            done += f["size"]; fetched += f["size"] if how != "skip" else 0
            log(ok=j["repo"], path=f["path"], via=how, file_mb_s=round(rate / 1e6, 1),
                progress_gb=round(done / 1e9, 1), total_gb=round(total / 1e9, 1),
                avg_download_mb_s=round(fetched / 1e6 / max(1, time.time() - started), 1))
    log(ALL_DONE=True, failed=len(failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
