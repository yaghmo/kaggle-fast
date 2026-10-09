# Startup speedups: what to measure, what to change, what it gave

Each entry: the symptom, the probe that proves it, the fix, and the gain measured on one real case. The gains are from
three pairs (SAM-Audio large fp16, LatentSync 1.5, Fish Audio S2 Pro) on Kaggle T4 notebooks. Your numbers will differ:
the method transfers, the seconds do not. Measure before applying anything that is not already in the loader.

## Contents

1. Weights read with one thread
2. Random init of weights that are then overwritten
3. `torch.load` copies the whole checkpoint
4. Dtype cast on the CPU during the move to the GPU
5. A library waits on the network at import
6. Kaggle's tensorflow imported by accident
7. Recursive globs over `/kaggle/input`
8. Tried, did not help
9. How to measure without fooling yourself

---

## 1. Weights read with one thread

**Symptom.** Long gap before the model is on the GPU; file is gigabytes and sits on `/kaggle/input`.

**Probe.** `scripts/probes/probe_read.py`. It reads four cold quarters of the file with 1, 4, 16 and 1 threads.

**Fix.** `warm(path)` from the loader: a background thread pool that reads the file into the page cache. Call it as the
first line of the run, before the venv is built and before the imports, so the read overlaps work that is happening
anyway. `load()` already does it for the pair's own Model folder. It returns at once and holds no reference to the data:
the page cache is reclaimable, so it cannot cause an out-of-memory kill.

**Measured.** 6.9 GiB checkpoint: 1 thread 120-240 MB/s (about 30-55 s), 16 threads 770-950 MB/s (about 8 s).

## 2. Random init of weights that are then overwritten

**Symptom.** Constructing the model object takes tens of seconds before any file is read.

**Probe.** `scripts/probes/probe_build.py` (cProfile around the constructor). Look for `uniform_`, `normal_`,
`trunc_normal_`, `kaiming_uniform_` at the top.

**Fix.** `no_init()` from the loader around the constructor only:

```python
with no_init():
    model = Model(config)
```

When the constructor is inside the model's own code and runs in another process (a server, a CLI), rewrite that one
statement in the pair's writable checkout instead:

```python
no_init_patch(f"{repo}/pkg/models/loader.py", "model = model_cls(config)")
```

Both turn `Tensor.uniform_`, `Tensor.normal_` and `Tensor.erfinv_` into no-ops for the duration. They patch `torch.Tensor`
and not `torch.nn.init` on purpose: `transformers.from_pretrained` puts the `torch.nn.init` originals back, so a
`torch.nn.init` patch dies as soon as the constructor loads a Hugging Face sub-model.

Safe only when every weight the constructor initialises is then loaded from a checkpoint. Prove it for your model: build it
both ways in a CPU notebook and compare every parameter and buffer with `torch.equal` (see `runs` example in
`probe_build.py`). A weight that is in the model but not in the checkpoint would be left as uninitialised memory.

**Measured.** SAM-Audio constructor 47.2 s -> 1.2 s. 1449 tensors compared, 0 differ.
Fish Audio S2 Pro (4.5 billion parameters, built inside its own API server): LLM constructor 69 s -> 0.2 s in the server's
log, codec load 29 s -> 8 s; 361 and 541 tensors compared, 0 differ.

## 3. `torch.load` copies the whole checkpoint

**Symptom.** `torch.load` of a `.pt` takes tens of seconds even when the file is already in the page cache, and host RAM
peaks at twice the model size.

**Fix.**

```python
state = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
result = model.load_state_dict(state, strict=False, assign=True)
# strict=False returns the mismatches instead of raising: check them yourself
assert not result.unexpected_keys and not result.missing_keys, result
model.to("cuda")
```

`mmap=True` maps the file instead of reading it; `assign=True` makes the mapped tensors the model's weights instead of
copying into the randomly initialised ones. If the model class overrides `load_state_dict` without `assign`, call
`torch.nn.Module.load_state_dict(model, state, strict=False, assign=True)` and reproduce the override's own key checks.
Needs a checkpoint in torch's zip format (the default since torch 1.6). Combine with `warm()` so the mapped pages are
already in RAM.

**Measured.** SAM-Audio, file already in page cache: `torch.load` 45.1 s -> 0.2 s, `load_state_dict` 1.0 s -> 0.0 s,
peak RAM 16.0 GiB -> 2.1 GiB.

## 4. Dtype cast on the CPU during the move to the GPU

**Symptom.** `model.to(device, dtype=...)` takes tens of seconds although the weights are already in RAM, and the stored
dtype differs from the one requested (bf16 weights loaded with `--half`, fp32 weights run in fp16).

**Why.** For a CPU-to-GPU copy with a dtype change, torch converts on the CPU first, on one core.

**Fix.** Copy, then convert on the GPU: `model.to(device).to(dtype)`. In third-party code, replace that one statement in
the pair's writable checkout. Peak VRAM rises by one tensor, not by the model.

**Measured.** S2 Pro, 8.5 GiB of bf16 to fp16 on a T4: 34 s -> 2.6 s. The whole server start, with this and `no_init_patch`:
186 s -> 54 s. Not compared tensor by tensor (it needs a GPU session); the same seven requests returned 200 before and after.

## 5. A library waits on the network at import

**Symptom.** Imports take about the same long time on every run, warm or cold, with Internet off.

**Probe.** `scripts/probes/probe_imports.py` prints the slowest module bodies by own time. A module whose own time is tens
of seconds and identical between runs is waiting, not working. Search its source for `urlopen`, `requests`, `version`.

**Fix.** The library's own off switch, set before anything is imported. The loader sets `NO_ALBUMENTATIONS_UPDATE=1`.
Add others to the loader as you find them, each with the measurement that justified it.

**Measured.** LatentSync entry point imports: 72.9 s -> 10.0 s.

## 6. Kaggle's tensorflow imported by accident

**Symptom.** `probe_imports.py` lists `tensorflow...` among the slowest module bodies of a model that does not use it. It
comes in through `torch.utils.tensorboard` (audiotools, lightning) or `transformers`.

**Fix.** `no_tensorflow()` from the loader, around the run's imports and nothing else:

```python
with no_tensorflow():
    import torch
    from the_model import Model
```

Inside the block `sys.modules['tensorflow']` is `None`, so `import tensorflow` fails at once and the callers fall back. On
exit the entry is removed again.

**Do not make the block permanent.** A first version left the `None` entry in place for the whole run (a `.pth` in the
venv). Imports got faster and inference crashed: `einops` treats a `tensorflow` key in `sys.modules` as "tensorflow is
loaded", then imports it for real (`ModuleNotFoundError: import of tensorflow halted; None in sys.modules`). For a model
that runs in its own process there is no clean place for the block; leave it out there.

**Measured.** Two fresh CPU sessions, same imports: 49.4 s as is, 38.8 s blocked. One sample each: the direction is
clear, the exact gain is not. A warm in-session comparison showed 0.2 s and nearly got this thrown away.

## 7. Recursive globs over `/kaggle/input`

**Symptom.** A `glob("/kaggle/input/**/...")` line takes seconds to minutes, or returns files from packages and demo
folders.

**Fix.** Glob the exact mount: `/kaggle/input/datasets/*/<dataset>/*.mp4`, `/kaggle/input/models/**/checkpoint.pt`.

**Measured.** Pair lookup 32-144 s -> 0.1 s after replacing an eager `rglob` with known mount paths tried lazily.

## 8. Tried, did not help

Keep these so nobody spends a run on them again.

- **Reading Kaggle's own libraries (torch, tensorflow, numpy, scipy...) into the page cache during the venv build.** Two
  fresh CPU sessions, same imports: without 13.5 s load + 44.3 s imports, with 20.1 s load + 37.7 s imports. Both reached
  "imports done" at 60.1 s: the warm-up only moved the time. Reading whole package folders is too much; it was still
  running when the imports finished.
- **`USE_TF=0` / `USE_FLAX=0` for speed.** No difference in import time. Still needed for correctness when Kaggle's
  tensorflow and your protobuf disagree.
- **Building the model in fp16 or fp32.** Same construction time (47.4 s against 48.5 s): the cost is the random init,
  not the dtype.
- **A prebuilt, bytecode-compiled site-packages tree in a notebook output**, attached instead of installing wheels. Reading
  tens of thousands of small files from the input mount cost more than installing locally: slower for two of three pairs,
  and it needs a second attached input and a build notebook. Wheels plus `--compile-bytecode` won.
- **`uv pip install --compile-bytecode`.** In a CPU probe it looked like a win (first import 42.9 -> 12.2 s), but the
  "after" import ran second in the same session, with every file already in the page cache. In a real fresh GPU session:
  install 9 -> 45 s, imports 53 -> 51 s. A net loss of 34 s.

## 9. How to measure without fooling yourself

- **One cold shot per session.** The second time anything is read or imported in a session it is warm. A before/after
  inside one session measures the page cache, not your change. Compare two fresh sessions, or accept only the first
  number of each session.
- **Do not trust `posix_fadvise(DONTNEED)` to make files cold again.** After evicting 189,509 files that way, imports
  ran as fast as warm ones (11.1 s against 32.7 s truly cold). Either it does nothing on these disks or the cost is
  elsewhere; in both cases an A/B built on it is worthless.
- **CPU notebooks are free and start in a minute; GPU notebooks cost quota and queue.** Find the cause on CPU, confirm the
  total once on GPU. The CPU image is not identical to the GPU one, so a CPU win is a lead, not a result.
- **Stamp every phase** (`[t+12s] pair loaded`) in the run script itself. The gap between two stamps is the only number
  that counts; everything else is a guess about it.
- **Change one thing per GPU run.** Two changes at once and a slower run tells you nothing about either.
