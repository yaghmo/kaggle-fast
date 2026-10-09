"""Where does building + loading a model go, and does the fast path give the same model? CPU notebook: everything but
the copy to the GPU. Replace make() and CKPT with your model (attach its Model, load its pair, import its class); as
shipped it runs on a throwaway stack of Linear layers, so you can push it once to see the method work:
  push_test.py probes/probe_build.py build
Reports: cProfile of the constructor, then normal load against no_init() + mmap + assign, then a tensor-for-tensor
comparison of the two models. Do not ship the fast path for a model until that comparison says 0 differ."""
import cProfile, io, os, pstats, resource, time
import torch
def say(m): print(m, flush=True)
def ram(): return f"peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2:.1f} GiB"

CKPT = "/kaggle/tmp/probe_build_demo.pt"
def make():
    return torch.nn.Sequential(*[torch.nn.Linear(8192, 8192) for _ in range(12)])  # 3 GiB of fp32, all of it random init
if not os.path.exists(CKPT):  # demo only: a real checkpoint already exists
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)  # /kaggle/tmp is not there until something creates it
    torch.save(make().state_dict(), CKPT)

pr = cProfile.Profile(); t = time.time(); pr.enable(); make(); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
say(f"constructor {time.time()-t:.1f}s, by own time:\n" + "\n".join(l[:170] for l in s.getvalue().splitlines() if "{" in l or "ncalls" in l or ".py" in l))

def build(fast):
    t = time.time()
    if fast:
        with no_init():
            m = make()
    else:
        m = make()
    t_make = time.time() - t; t = time.time()
    sd = torch.load(CKPT, weights_only=True, map_location="cpu", mmap=fast)
    t_load = time.time() - t; t = time.time()
    r = torch.nn.Module.load_state_dict(m, sd, strict=False, assign=fast)  # base class: an override may lack assign=
    assert not r.missing_keys and not r.unexpected_keys, r  # weights that come from elsewhere: filter them here, by name
    say(f"fast={fast}: construct {t_make:.1f}s, torch.load {t_load:.1f}s, load_state_dict {time.time()-t:.1f}s, {ram()}")
    return m.eval()

fast = build(True)   # first: its weights live in the page cache, so both models fit in RAM
slow = build(False)
a, b = dict(slow.state_dict()), dict(fast.state_dict())
a.update(slow.named_buffers()); b.update(fast.named_buffers())  # non-persistent buffers are not in a checkpoint: compare them too
bad = [k for k in a if k not in b or a[k].dtype != b[k].dtype or a[k].shape != b[k].shape
       or not torch.equal(a[k].nan_to_num(), b[k].nan_to_num())]
say(f"fast vs normal: {len(a)} tensors compared, {len(b) - len(a)} extra, {len(bad)} differ {bad[:8]}")
print("ALL DONE")
