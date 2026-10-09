"""Deterministic loader over packed uint16 token shards.

A shard is a raw little-endian uint16 file of concatenated documents, each followed by EOS (50279).
Training windows are seq_len+1 tokens at stride seq_len inside a shard (documents pack across window
boundaries; EOS separates them). The sample for (step, dp_rank, micro) is a pure function of the seed,
so resume needs only the step and the order is identical for any number of restarts. A shard listed
as `path:R` is repeated R times (quality upsampling, see design doc), with each copy in a different
position of the epoch permutation.
"""
import glob
import numpy as np
import torch

EOS, PAD = 50279, 1


def write_shard(path, docs, eos=EOS):
    arr = np.concatenate([np.append(np.asarray(d, dtype=np.uint16), np.uint16(eos)) for d in docs])
    arr.astype('<u2').tofile(path)
    return arr.size


class LocalCache:
    """Concurrent, persistent node-local cache for packed token shards."""
    def __init__(self, paths, root, seed=0):
        import hashlib, os, socket, threading
        self.root, self.local, self._mu = root, {}, threading.Lock()
        os.makedirs(root, exist_ok=True)
        uniq = sorted(set(paths))
        h = int.from_bytes(hashlib.sha1(socket.gethostname().encode()).digest()[:8], 'little')
        rng = np.random.default_rng([int(seed), h & 0xffffffff, h >> 32])
        self.todo = [uniq[i] for i in rng.permutation(len(uniq))]
        for p in uniq:
            self._adopt(p)
        if os.environ.get('LOCAL_CACHE_DISABLE_COPY') != '1':
            threading.Thread(target=self._run, daemon=True, name='shard-cache').start()

    def _lp(self, p):
        import hashlib, os
        return os.path.join(self.root, hashlib.sha1(p.encode()).hexdigest()[:20] + '.u16')

    def _adopt(self, p):
        import os
        lp = self._lp(p)
        try:
            if os.path.getsize(lp) != os.path.getsize(p):
                return False
            mm = np.memmap(lp, dtype='<u2', mode='r')
        except OSError:
            return False
        with self._mu:
            self.local[p] = mm
        return True

    def _run(self):
        import collections, os, shutil, time, uuid
        q, retries = collections.deque(self.todo), collections.Counter()
        while q:
            p = q.popleft()
            if p in self.local or self._adopt(p):
                continue
            lp, lock = self._lp(p), self._lp(p) + '.lock'
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if self._adopt(p):
                    continue
                q.append(p)
                time.sleep(0.05)
                continue
            tmp = lp + '.tmp-' + uuid.uuid4().hex
            try:
                size = os.path.getsize(p)
                if shutil.disk_usage(self.root).free < size + (8 << 30):
                    print(f'LOCAL_CACHE_LOW_SPACE root={self.root}', flush=True)
                    return
                shutil.copyfile(p, tmp)
                if os.path.getsize(tmp) != size:
                    raise OSError('short cache copy')
                os.replace(tmp, lp)
                self._adopt(p)
            except OSError as e:
                retries[p] += 1
                print(f'LOCAL_CACHE_RETRY n={retries[p]} path={p} err={e}', flush=True)
                if retries[p] <= 8:
                    q.append(p)
                    time.sleep(min(2 ** retries[p], 30))
            finally:
                try:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                except OSError:
                    pass
                try:
                    os.close(fd)
                except OSError:
                    pass
                try:
                    os.unlink(lock)
                except OSError:
                    pass

    def get(self, p, shared):
        mm = self.local.get(p)
        if mm is not None:
            return mm
        if self._adopt(p):
            return self.local[p]
        return shared


class PackedTokens:
    def __init__(self, shards, seq_len, seed=0, local_cache=None):
        self.seq_len, self.seed = seq_len, seed
        self.maps, counts, self.paths = [], [], []
        for spec in shards:
            spec = str(spec)
            path, rep = (spec.rsplit(':', 1) if spec.rsplit(':', 1)[-1].isdigit() else (spec, ''))
            mm = np.memmap(path, dtype='<u2', mode='r')
            n = (mm.size - 1) // seq_len
            for _ in range(int(rep or 1)):
                self.maps.append(mm)
                self.paths.append(path)
                counts.append(n)
        self.cache = LocalCache(self.paths, local_cache, seed) if local_cache else None
        self.cum = np.cumsum([0] + counts)
        self.n = int(self.cum[-1])
        if self.n == 0:
            raise ValueError('shards hold no full window')
        self._perm_epoch, self._perm = None, None

    def __len__(self):
        return self.n

    def _perm_for(self, epoch):
        if self._perm_epoch != epoch:
            rng = np.random.Generator(np.random.PCG64([self.seed, epoch]))
            self._perm = rng.permutation(self.n).astype(np.int64)
            self._perm_epoch = epoch
        return self._perm

    def window(self, global_pos):
        epoch, i = divmod(int(global_pos), self.n)
        j = int(self._perm_for(epoch)[i])
        s = int(np.searchsorted(self.cum, j, side='right') - 1)
        off = (j - int(self.cum[s])) * self.seq_len
        mm = self.cache.get(self.paths[s], self.maps[s]) if self.cache is not None else self.maps[s]
        return np.asarray(mm[off:off + self.seq_len + 1], dtype=np.int64)

    def batch(self, step, dp_rank, dp_size, micro, micro_bs, global_bs):
        """Micro-batch `micro` of data-parallel rank `dp_rank` at optimizer step `step`."""
        per_rank = global_bs // dp_size
        base = step * global_bs + dp_rank * per_rank + micro * micro_bs
        w = torch.from_numpy(np.stack([self.window(base + i) for i in range(micro_bs)]))
        x, y = w[:, :-1], w[:, 1:].clone()
        y[y == PAD] = -100
        return x, y


def shard_paths(specs):
    """Comma-separated globs, each optionally suffixed `:R` (repeat factor) -> sorted shard specs."""
    out = []
    for spec in specs.split(','):
        pat, _, rep = spec.rpartition(':') if spec.rsplit(':', 1)[-1].isdigit() else (spec, '', '')
        out += [f'{p}:{rep}' if rep else p for p in sorted(glob.glob(pat))]
    return out
