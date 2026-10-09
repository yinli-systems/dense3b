"""Checkpoint retention: keep the newest `keep_last` complete checkpoints in full (resume); for every step that
is a multiple of `milestone_every` keep only the BF16 model.pt + meta.json (evaluation); delete everything else.

A slimmed milestone loses its COMPLETE marker (gets MILESTONE instead), so resume never picks it.

  python -m bench.prune_ckpt /data/run01/scxi253/code3b-train/ckpt-v0 [--keep-last 3] [--milestone-every 5000] [--loop 1800]
"""
import argparse
import glob
import json
import os
import shutil
import time


def step_of(d):
    return int(os.path.basename(d).split('_')[1])


def prune(ckpt_dir, keep_last=3, milestone_every=5000, dry_run=False):
    complete = sorted((d for d in glob.glob(os.path.join(ckpt_dir, 'step_*'))
                       if os.path.exists(os.path.join(d, 'COMPLETE'))), key=step_of)
    keep_full = set(complete[-keep_last:])
    actions = []
    for d in complete:
        if d in keep_full:
            continue
        if (step_of(d) + 1) % milestone_every == 0:
            actions.append(('slim', d))
        else:
            actions.append(('delete', d))
    for act, d in actions:
        if dry_run:
            continue
        if act == 'delete':
            os.remove(os.path.join(d, 'COMPLETE'))          # first: the directory is no longer resumable
            shutil.rmtree(d)
        else:
            open(os.path.join(d, 'MILESTONE'), 'w').close()
            os.remove(os.path.join(d, 'COMPLETE'))
            for f in glob.glob(os.path.join(d, 'rank*.pt')):
                os.remove(f)
    return [(a, os.path.basename(d)) for a, d in actions]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt_dir')
    ap.add_argument('--keep-last', type=int, default=3)
    ap.add_argument('--milestone-every', type=int, default=5000)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--loop', type=int, default=0, help='repeat every N seconds')
    a = ap.parse_args(argv)
    while True:
        done = prune(a.ckpt_dir, a.keep_last, a.milestone_every, a.dry_run)
        if done:
            print(json.dumps(dict(time=time.strftime('%FT%T'), actions=done)), flush=True)
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == '__main__':
    main()
