"""How fast does Kaggle read your weights, and does it scale with readers? CPU notebook, no pair needed.
Attach the Model (or Dataset) and push with:  push_test.py probes/probe_read.py read --model-source <user/model/Framework/variation/version>
Reads four cold quarters of one file with 1, 4, 16 and 1 threads (the last one shows whether the disk itself sped up).
If 16 readers beat one by a wide margin, call warm(file) first thing in the run."""
import glob, os, time
from concurrent.futures import ThreadPoolExecutor
FILE = None  # None: the biggest file under /kaggle/input/models, else under /kaggle/input/datasets
CHUNK = 64 * 2**20
def say(m): print(m, flush=True)
if FILE is None:
    for root in ("models", "datasets"):
        found = [f for f in glob.glob(f"/kaggle/input/{root}/**/*", recursive=True) if os.path.isfile(f)]
        if found:
            FILE = max(found, key=os.path.getsize)
            break
size = os.path.getsize(FILE)
def read(lo, hi, threads):
    """Read bytes [lo, hi) once with N threads, return MB/s. Each range is read once, so every call is a cold read."""
    def part(o):
        fd = os.open(FILE, os.O_RDONLY)
        try: os.pread(fd, min(CHUNK, hi - o), o)
        finally: os.close(fd)
    t = time.time()
    with ThreadPoolExecutor(threads) as ex: list(ex.map(part, range(lo, hi, CHUNK)))
    return (hi - lo) / 2**20 / (time.time() - t)
q = max(size // 4 // CHUNK, 1) * CHUNK
say(f"{size/2**30:.2f} GiB {FILE}")
for lo, hi, n in [(0, q, 1), (q, 2 * q, 4), (2 * q, 3 * q, 16), (3 * q, size, 1)]:
    if lo < hi <= size:
        mbs = read(lo, hi, n)
        say(f"cold read {lo/2**30:.1f}-{hi/2**30:.1f} GiB, {n:2d} threads: {mbs:.0f} MB/s (whole file at this rate: {size/2**20/mbs:.0f}s)")
print("ALL DONE")
