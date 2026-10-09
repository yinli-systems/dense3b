"""Auditable mechanics for a non-destructive, document-wide scoring census."""
import hashlib
import json
import math
import os
import uuid
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with tmp.open('x') as f:
        json.dump(data, f, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def windows_for_length(length, width):
    if length < 0 or width < 1:
        raise ValueError('Invalid context dimensions')
    starts = [0, max(0, (length-width)//2), max(0, length-width)]
    intervals = [(s, min(length, s+width)) for s in starts]
    covered = 0; end = 0
    for lo, hi in sorted(set(intervals)):
        covered += max(0, hi-max(lo, end)); end = max(end, hi)
    return intervals, covered


def score_scope(source):
    if source in {'fineweb_edu', 'dclm_baseline'}:
        return 'web_educational_value_not_correctness'
    if source in {'finemath', 'olmoe_science_reference'}:
        return 'auxiliary_web_education_out_of_domain_not_correctness'
    raise ValueError('No calibrated non-code model for this source')


def numeric_gate(reference, reduced):
    if not reference or len(reference) != len(reduced):
        raise ValueError('Missing numerical calibration')
    if not all(math.isfinite(x) for x in reference + reduced):
        raise ValueError('Nonfinite numerical calibration')
    errors = [abs(x-y) for x,y in zip(reference,reduced)]
    passed = max(errors) <= .05 and sum(errors)/len(errors) <= .01
    return dict(passed=passed, max_abs=max(errors), mean_abs=sum(errors)/len(errors),
                max_abs_limit=.05, mean_abs_limit=.01, windows=len(errors),
                purpose='numerical_check_not_quality_calibration')


def check_mask(raw, documents, selected):
    if len(raw) != (documents+7)//8:
        raise ValueError('Mask size mismatch')
    if documents%8 and raw and raw[-1] >> (documents%8):
        raise ValueError('Nonzero mask padding')
    if sum(int(x).bit_count() for x in raw) != selected:
        raise ValueError('Mask population mismatch')


def verified_text(raw, eos, row, decoder):
    if len(raw) != 2*row['tokens'] or eos != b'\x67\xc4':
        raise ValueError('Payload length or EOS mismatch')
    if hashlib.sha256(raw).digest() != row['token_hash']:
        raise ValueError('Document token hash mismatch')
    from array import array
    import sys
    ids = array('H'); ids.frombytes(raw)
    if sys.byteorder != 'little': ids.byteswap()
    text = decoder.decode(ids.tolist(), skip_special_tokens=False)
    if hashlib.sha256(text.encode('utf-8')).digest() != row['text_hash']:
        return None
    return text
