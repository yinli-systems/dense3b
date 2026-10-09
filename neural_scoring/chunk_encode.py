"""Exact, cached BERT (uncased WordPiece) encoding.

BertNormalizer + BertPreTokenizer + WordPiece never let a token cross a BERT-whitespace character
(' ', \\t, \\n, \\r, Unicode Zs), so a text split on exactly those characters can be encoded chunk by
chunk and concatenated with identical ids. Repeated chunks (most words) hit a bounded cache.
Exactness is asserted against the reference tokenizer in test_chunk_encode.py."""
import bisect
import itertools
import re
import sys
import unicodedata

_ZS = ''.join(chr(c) for c in range(sys.maxunicode + 1) if unicodedata.category(chr(c)) == 'Zs')
_SPLIT = re.compile('[\t\n\r' + re.escape(_ZS) + ']+')
# Characters str.split() treats as whitespace but BERT does not; when absent, str.split() is exact.
_PY_ONLY = ''.join(c for c in map(chr, range(sys.maxunicode + 1))
                   if c.isspace() and not (c in '\t\n\r' or unicodedata.category(c) == 'Zs'))
_PY_ONLY_RE = re.compile('[' + re.escape(_PY_ONLY) + ']')


class ChunkEncoder:
    def __init__(self, rust_tokenizer, max_entries=4_000_000):
        # Private copy without the padding/truncation that tokenizer.json may carry: encode_batch would
        # otherwise pad every chunk to the longest one in the batch.
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_str(rust_tokenizer.to_str())
        self.t.no_padding(); self.t.no_truncation()
        self.cache = {}
        self.max_entries = max_entries
        self.hits = self.misses = 0

    def _chunks(self, text):
        if _PY_ONLY_RE.search(text):
            return [c for c in _SPLIT.split(text) if c]
        return text.split()

    def _values(self, chunks):
        cache = self.cache; get = cache.get
        vals = [get(c) for c in chunks]
        if None in vals:
            missing = list({c for c, v in zip(chunks, vals) if v is None})
            new = {c: e.ids for c, e in zip(missing, self.t.encode_batch(missing, add_special_tokens=False))}
            if len(cache) + len(new) > self.max_entries:
                cache.clear()
            cache.update(new)
            self.misses += len(missing)
            vals = [v if v is not None else new[c] for c, v in zip(chunks, vals)]
        self.hits += len(chunks)
        return vals

    def encode(self, text):
        return list(itertools.chain.from_iterable(self._values(self._chunks(text))))

    def length_and_slices(self, text, spans_fn):
        """spans_fn(length) -> list of (lo, hi); returns (length, [ids[lo:hi] for each span]) without
        materialising the whole id list."""
        vals = self._values(self._chunks(text))
        ends = list(itertools.accumulate(map(len, vals)))
        length = ends[-1] if ends else 0
        spans = spans_fn(length)
        out = []
        for lo, hi in spans:
            if hi <= lo:
                out.append([]); continue
            i = bisect.bisect_right(ends, lo)
            start = ends[i - 1] if i else 0
            j = bisect.bisect_left(ends, hi)
            piece = list(itertools.chain.from_iterable(vals[i:j + 1]))
            out.append(piece[lo - start:hi - start])
        return length, spans, out
