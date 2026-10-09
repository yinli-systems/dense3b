from pathlib import Path
import re, shutil, sys, time
CACHE = r'''class LocalCache:
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
'''

def patch_cache(path):
    p = Path(path).resolve()
    bak = p.with_name(p.name + '.bak-cachefix-' + time.strftime('%Y%m%d-%H%M%S'))
    shutil.copy2(p, bak)
    s = p.read_text()
    s2, n = re.subn(r'class LocalCache:.*?\n\nclass PackedTokens:', CACHE + '\n\nclass PackedTokens:', s, count=1, flags=re.S)
    if n != 1:
        raise SystemExit(f'LocalCache block not found: {p}')
    p.write_text(s2)
    print('patched', p, 'backup', bak)

if __name__ == '__main__':
    for x in sys.argv[1:]:
        patch_cache(x)
