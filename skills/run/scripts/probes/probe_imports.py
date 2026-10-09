"""What makes a pair's imports slow? CPU notebook, Internet off (as the real run), `wheels` dataset attached.
Edit the three constants, then:  push_test.py probes/probe_imports.py imports
Reading the result:
  - first run much slower than the second: bytecode compile or cold files. The loader compiles at install; if the gap is
    still there, the slow part is outside the venv.
  - the "again" runs are warm. They can prove a fix for a wait (same seconds every run) and nothing about cold costs:
    for those, push the probe twice, one variant per fresh session, and compare the two "first run" lines.
  - one module body with tens of seconds of own time, the same on every run: it waits (network, lock), it does not work.
    Find its off switch, add it to VARIANTS to prove the gain, then set it in the loader."""
import os, re, subprocess, time
PAIR = "latentsync15"
STATEMENT = "import scripts.inference"  # the real entry point's imports, not a guess at them
REPO = "LatentSync"                     # run from this pinned checkout of the pair; None to run from /kaggle/working
VARIANTS = {"as is": {}}                # name -> extra env to try, e.g. {"no update check": {"SOME_LIB_NO_UPDATE": "1"}}
def say(m): print(m, flush=True)
t = time.time(); p = load(PAIR, need_model=False); say(f"load (venv + install): {time.time()-t:.1f}s")
cwd = str(p.source(REPO)) if REPO else None
def go(tag, env):
    t = time.time()
    r = subprocess.run([p.python, "-X", "importtime", "-c", STATEMENT], capture_output=True, text=True, env=dict(os.environ, **env), cwd=cwd)
    # importtime line: "import time: <own us> | <cumulative us> | <indent><module>"; top-level modules have one space
    rows = [(int(m[1]), int(m[2]), len(m[3]) - len(m[3].lstrip()), m[3].strip())
            for m in re.finditer(r"import time:\s+(\d+) \|\s+(\d+) \|(.*)", r.stderr)]
    err = "" if r.returncode == 0 else " | ".join(l for l in r.stderr.splitlines() if not l.startswith("import time:"))[-600:]
    say(f"{tag}: {time.time()-t:.1f}s rc={r.returncode} {err}")  # rc != 0: the import died part-way, the times are not a full import
    say("  slowest module bodies, own s: " + ", ".join(f"{n} {s/1e6:.1f}" for s, c, d, n in sorted(rows, reverse=True)[:14]))
    say("  slowest top-level imports, total s: " + ", ".join(f"{n} {c/1e6:.1f}" for s, c, d, n in sorted((x for x in rows if x[2] <= 1), key=lambda x: -x[1])[:8]))
go("first run", {})
for tag, env in VARIANTS.items():
    go(f"again, {tag}", env)
print("ALL DONE")
