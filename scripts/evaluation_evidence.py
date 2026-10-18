#!/usr/bin/env python3
"""Recover, check and package *existing* Dense3B evaluation artifacts (CPU only).

No model loading, scoring-code execution, network requests, fitted answers, or
inferred correct counts. A successful check establishes artifact consistency,
not an independent model rerun or authentication of historical timestamps.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LIMIT = 128 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}\Z")
TASKS = {
    "humaneval": (164, {"humaneval": "base", "humaneval_plus": "plus"}),
    "mbpp": (378, {"mbpp": "base", "mbpp_plus": "plus"}),
    "cruxeval": (800, {"cruxeval_i": "ok_i", "cruxeval_o": "ok_o"}),
    "gsm8k": (1319, {"gsm8k": "correct_relaxed", "strict_accuracy": "correct_strict"}),
    "arc_challenge": (1172, {"acc": "correct", "acc_norm": "correct_norm"}),
    "mmlu_mc5": (14042, {"acc": "correct", "acc_norm": "correct_norm"}),
    "mmlu_cf": (14042, {"acc": "correct", "acc_norm": "correct_norm"}),
    "arc_easy": (2376, {"acc": "correct", "acc_norm": "correct_norm"}),
    "hellaswag": (10042, {"acc": "correct", "acc_norm": "correct_norm"}),
    "winogrande": (1267, {"acc": "correct", "acc_norm": "correct_norm"}),
    "boolq": (3270, {"acc": "correct", "acc_norm": "correct_norm"}),
    "openbookqa": (500, {"acc": "correct", "acc_norm": "correct_norm"}),
    "sciq": (1000, {"acc": "correct", "acc_norm": "correct_norm"}),
}
PRIMARY = ("humaneval", "mbpp", "cruxeval", "gsm8k", "arc_challenge", "mmlu_mc5")
STAGES = ("6p82B", "20B", "40B", "60B", "80B", "100B")
BLOCKED = "raw_evaluation_artifacts_not_attached"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def contained(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative), "Empty artifact path")
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe artifact path")
    base = root.resolve(strict=True)
    path = base
    for part in rel.parts:
        path = path / part
        require(not path.is_symlink(), "Symlink artifact is not accepted")
    target = path.resolve(strict=True)
    require(base in target.parents and target.is_file(), "Not a regular file inside root")
    return target


def read_bytes(root: Path, relative: str, *, decoded: bool = False) -> bytes:
    p = contained(root, relative)
    before = p.stat()
    require(before.st_size <= LIMIT, "Artifact exceeds size limit")
    with p.open("rb") as f:
        data = f.read(LIMIT + 1)
    require(len(data) <= LIMIT, "Artifact exceeds read limit")
    after = p.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), "Artifact changed during read")
    if decoded and p.suffix == ".gz":
        import io
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
            data = f.read(LIMIT + 1)
        require(len(data) <= LIMIT, "Decompressed artifact exceeds limit")
    return data


def parse(data: bytes) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "Duplicate JSON key: " + key)
            out[key] = value
        return out
    def constant(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def read_json(root: Path, rel: str) -> Any:
    return parse(read_bytes(root, rel, decoded=True))


def model_hash(summary: dict) -> str:
    manifest = summary.get("manifest")
    require(isinstance(manifest, dict), "Missing evaluator manifest")
    h = manifest.get("checkpoint_model_sha256", manifest.get("model_sha256"))
    require(isinstance(h, str) and SHA.fullmatch(h) is not None, "Missing model SHA-256")
    require(bool(manifest.get("protocol")), "Missing evaluator protocol")
    return h


def verify_task(summary: dict, detail: dict, task: str, expected_n: int | None = None) -> dict:
    require(task in TASKS, "Unsupported task: " + task)
    raw = summary.get("tasks", {}).get(task)
    require(isinstance(raw, dict), "Task missing from summary")
    require(not raw.get("error") and not raw.get("skipped"), "Task failed or skipped")
    h = model_hash(summary)
    require(detail.get("checkpoint_sha256") == h, "Detail belongs to a different checkpoint")
    require(detail.get("task") == task, "Task label mismatch")
    n = raw.get("n")
    require(type(n) is int and n > 0, "Invalid summary sample count")
    target = expected_n if expected_n is not None else TASKS[task][0]
    require(type(target) is int and target > 0 and n == target, "Incomplete/wrong benchmark count")
    items = detail.get("items")
    require(type(detail.get("n")) is int and detail["n"] == n, "Detail count mismatch")
    require(isinstance(items, list) and len(items) == n, "Missing per-item records")
    ids = []
    for item in items:
        require(isinstance(item, dict), "Non-object per-item record")
        ident = next((item[k] for k in ("task", "id", "index") if k in item), None)
        require(type(ident) in (str, int) and ident != "", "Missing/invalid item identity")
        ids.append(str(ident))
    require(len(set(ids)) == n, "Duplicate item IDs")
    scores = {}
    for metric, flag in TASKS[task][1].items():
        require(metric in raw, "Missing score field: " + metric)
        value = raw[metric]
        require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1,
                "Invalid accuracy fraction")
        flags = [item.get(flag) for item in items]
        require(all(type(x) is bool for x in flags), "Missing/non-Boolean outcome flags")
        correct = sum(flags)
        tolerance = 0.00005001 if task == "cruxeval" else 1e-10
        require(abs(correct / n - value) <= tolerance, "Summary and per-item recount disagree")
        scores[metric] = {"correct": correct, "total": n, "fraction_recounted": correct / n}
    if task in ("humaneval", "mbpp"):
        require(all(not item["plus"] or item["base"] for item in items), "Plus passed but base failed")
    return {"task": task, "n": n, "checkpoint_sha256": h,
            "protocol": summary["manifest"]["protocol"], "scores": scores,
            "item_fields": sorted(set().union(*(set(x) for x in items)))}


def audit_archive(root: Path) -> dict:
    """Recheck archived bytes and per-item flags; do not copy author scores into logs."""
    records = read_json(root, "reports/sample_verification.json")
    require(isinstance(records, list), "Invalid verification index")
    matches = [r for r in records if r.get("variant") == "dense3b" and r.get("step") == 6499]
    require(matches, "No step-6499 per-item archive found")
    tasks, seen, model_hashes = [], set(), set()
    for row in sorted(matches, key=lambda r: r["task"]):
        task = row["task"]
        require(task not in seen, "Ambiguous duplicated task/version in step-6499 index")
        seen.add(task)
        detail_bytes = read_bytes(root, row["details_source"], decoded=True)
        require(sha(detail_bytes) == row["details_sha256"], "Archived detail hash mismatch")
        summary_bytes = read_bytes(root, row["summary_source"], decoded=True)
        require(sha(summary_bytes).startswith(row["evaluation_id"]), "Archived summary hash mismatch")
        summary = parse(summary_bytes)
        require(summary.get("step") == 6499, "Archive checkpoint step mismatch")
        result = verify_task(summary, parse(detail_bytes), task)
        require(result["n"] == row["n"], "Archive index count mismatch")
        for metric, score in result["scores"].items():
            old = row["checks"].get(metric, {})
            require(score["correct"] == old.get("passed_recounted") and
                    score["total"] == old.get("n_recounted"), "Archive index recount mismatch")
        model_hashes.add(result["checkpoint_sha256"])
        result.update(summary_source=row["summary_source"], details_source=row["details_source"],
                      summary_sha256=sha(read_bytes(root, row["summary_source"])),
                      details_decoded_sha256=sha(detail_bytes),
                      details_stored_sha256=sha(read_bytes(root, row["details_source"])))
        tasks.append(result)
    require(len(model_hashes) == 1, "Different checkpoint weights mixed across tasks")
    require(set(PRIMARY) <= seen, "Missing primary task families at early checkpoint")
    reported = read_json(root, "results/measured/2026-10-18.json")
    require([s["id"] for s in reported["stages"]] == list(STAGES), "Unexpected reported milestones")
    stages = [{"stage": "6p82B", "checkpoint_step": 6499,
               "status": "archived_artifacts_recounted", "tasks": tasks}]
    for stage in STAGES[1:]:
        attached = [v for b in reported["benchmarks"] for v in b["values"]
                    if v["stage_id"] == stage and v.get("raw_outputs_attached")]
        require(not attached, "New raw-evidence claim requires explicit archive binding")
        stages.append({"stage": stage, "checkpoint_step": None, "status": BLOCKED, "tasks": []})
    return {"schema_version": 1, "kind": "artifact_consistency_audit",
            "model_evaluation_executed": False, "synthetic_outcomes_created": False,
            "checkpoint_weight_hashes": sorted(model_hashes),
            "early_task_families": len(tasks), "early_item_records": sum(x["n"] for x in tasks),
            "archive_index_sha256": sha(read_bytes(root, "reports/sample_verification.json")),
            "score_source_sha256": sha(read_bytes(root, "results/measured/2026-10-18.json")),
            "stages": stages}


def verify_bundle(root: Path, expected_hash: str, required: tuple[str, ...] = PRIMARY) -> dict:
    """Validate a recovered run folder (suite.json / mc.json + task.details.json[.gz])."""
    require(bool(SHA.fullmatch(expected_hash)), "Expected model hash is required")
    summaries = {}
    artifacts = set()
    for filename in ("suite.json", "mc.json"):
        if (root / filename).exists():
            summary = read_json(root, filename)
            require(model_hash(summary) == expected_hash, "Wrong checkpoint weight hash")
            artifacts.add(filename)
            for task in summary.get("tasks", {}):
                if task in TASKS:
                    require(task not in summaries, "Duplicate task in summary files")
                    summaries[task] = summary
    require(all(t in summaries for t in required), "Missing required task summaries")
    checked = []
    for task in sorted(summaries):
        files = [x for x in (task + ".details.json", task + ".details.json.gz") if (root / x).exists()]
        require(len(files) == 1, "Missing or ambiguous detail file: " + task)
        checked.append(verify_task(summaries[task], read_json(root, files[0]), task))
        artifacts.add(files[0])
    optional = ["mc-manifest.json", "generation-manifest.json", "mc-status.json", "generation-status.json"]
    optional += [t + suffix for t in ("humaneval", "mbpp")
                 for suffix in (".samples.jsonl", ".generation.json")]
    optional += ["run_capture.json", "runner.stdout.log", "runner.stderr.log"]
    optional += [t + ".samples_eval_results.json" for t in ("humaneval", "mbpp")]
    optional += ["scoring/" + t + "-scoring.log" for t in ("humaneval", "mbpp")]
    for f in optional:
        if (root / f).exists():
            read_bytes(root, f)
            artifacts.add(f)
    records = [{"path": f, "sha256": sha(read_bytes(root, f)), "bytes": len(read_bytes(root, f))}
               for f in sorted(artifacts)]
    return {"schema_version": 1, "checkpoint_sha256": expected_hash, "tasks": checked,
            "artifacts": records, "verification": "hashes_and_recorded_outcomes_only",
            "model_evaluation_executed": False}


def pack(root: Path, destination: Path, expected_hash: str, stage: str, origin: str) -> dict:
    require(stage in STAGES, "Unknown milestone label")
    require(origin in ("recovered_original", "new_rerun"), "Origin must be explicit")
    require(not destination.exists(), "Output must be a new directory")
    source = root.resolve(strict=True)
    target = destination.resolve()
    require(source != target and source not in target.parents, "Output cannot be inside source")
    receipt = verify_bundle(source, expected_hash)
    receipt.update(stage=stage, origin=origin, packaged_at_utc=datetime.now(timezone.utc).isoformat(),
                   note="Package time is not checkpoint creation or evaluation time. Review for secrets before publication.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".evidence-", dir=destination.parent))
    try:
        for entry in receipt["artifacts"]:
            data = read_bytes(source, entry["path"])
            require(sha(data) == entry["sha256"], "Source changed while packing")
            (temp / entry["path"]).parent.mkdir(parents=True, exist_ok=True)
            (temp / entry["path"]).write_bytes(data)
        (temp / "evidence_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        destination.mkdir(exist_ok=False)
        for item in temp.iterdir():
            if item.name != "evidence_receipt.json":
                shutil.move(str(item), str(destination / item.name))
        shutil.move(str(temp / "evidence_receipt.json"), str(destination / "evidence_receipt.json"))
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return receipt


def inventory(roots: list[Path], max_entries: int = 50000) -> dict:
    require(max_entries > 0, "Positive inventory limit required")
    files, issues, visited = [], [], 0
    accepted = {"meta.json", "suite.json", "mc.json", "metrics.jsonl", "model.pt", "model.safetensors"}
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            issues.append({"root": str(root), "reason": "missing_directory_or_symlink"})
            continue
        def error(exc):
            issues.append({"path": str(exc.filename), "reason": "unreadable_directory"})
        for directory, dirs, names in os.walk(root, followlinks=False, onerror=error):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (Path(directory)/d).is_symlink())
            for name in sorted(names):
                visited += 1
                if visited > max_entries:
                    return {"complete": False, "reason": "entry_limit", "files": files, "issues": issues}
                if name not in accepted and not name.endswith((".details.json", ".details.json.gz")):
                    continue
                p = Path(directory) / name
                if p.is_symlink():
                    issues.append({"path": str(p), "reason": "skipped_symlink"})
                    continue
                try:
                    st = p.stat()
                    if p.is_file():
                        files.append({"path": str(p), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns})
                except OSError:
                    issues.append({"path": str(p), "reason": "unreadable_file"})
    return {"complete": not issues, "files": files, "issues": issues,
            "note": "Inventory only; names and mtime do not prove token counts, ancestry, or original evaluation time."}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    a.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("verify")
    a.add_argument("--source", type=Path, required=True)
    a.add_argument("--checkpoint-sha256", required=True)
    a = sub.add_parser("pack")
    a.add_argument("--source", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--checkpoint-sha256", required=True)
    a.add_argument("--stage", choices=STAGES, required=True)
    a.add_argument("--origin", choices=("recovered_original", "new_rerun"), required=True)
    a = sub.add_parser("inventory")
    a.add_argument("--roots", type=Path, nargs="+", required=True)
    a.add_argument("--max-entries", type=int, default=50000)
    a.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.command == "audit":
        result = audit_archive(args.root)
    elif args.command == "inventory":
        result = inventory(args.roots, args.max_entries)
    elif args.command == "verify":
        result = verify_bundle(args.source, args.checkpoint_sha256)
    else:
        result = pack(args.source, args.out, args.checkpoint_sha256, args.stage, args.origin)
    if args.command in ("audit", "inventory"):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")
    concise = {k: v for k, v in result.items() if k not in ("stages", "tasks", "artifacts", "files")}
    if "stages" in result:
        concise["stage_status"] = {x["stage"]: x["status"] for x in result["stages"]}
        concise["early_recounts"] = [{"task": x["task"], "n": x["n"], "scores": x["scores"]}
                                     for x in result["stages"][0]["tasks"]]
    print(json.dumps(concise, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
