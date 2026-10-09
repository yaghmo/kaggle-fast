---
name: run
description: "Take a Hugging Face model to a fast, offline Kaggle GPU notebook run: check that it fits Kaggle's free tier, package its weights as a private Kaggle Model and its dependencies as pinned wheels (a 'pair'), then measure and cut the time between session start and the first inference, and after that the per-input work itself. Use for /kaggle-fast:run <hf repo>, and whenever the user wants to run, host, port or speed up a model on Kaggle, asks whether a model fits Kaggle or a T4, complains that a Kaggle notebook is slow to start, load a model or import, pip-installs or downloads weights at the top of a Kaggle notebook, or wants to stop wasting weekly GPU quota, even when they do not say 'skill' or 'pair'."
---

# /kaggle-fast:run

Kaggle gives 30 GPU hours a week and counts every second of a GPU session, including the minutes spent installing
packages, downloading weights and importing. This skill moves all of that out of the GPU session (a prebuilt **pair**: one
Kaggle Model with the weights, one folder of pinned wheels) and then cuts what is left by measuring it.

```
/kaggle-fast:run <org/repo>            full flow: fit check -> confirm -> build the pair -> run script -> squeeze startup -> squeeze the work
/kaggle-fast:run fit <org/repo>        only the fit check
/kaggle-fast:run squeeze <run script>  only the squeeze (startup, then the work), for a pair that already exists
```

Scripts are in `scripts/` next to this file; call them by their full path. Work in the user's project folder: specs, run
scripts and `kaggle_out/` belong there, not in the skill.

Read `references/kaggle-facts.md` before writing any notebook code. It is short, and every line in it cost a failed run.

## Rules of the road

- **CPU notebooks are free, GPU notebooks are quota.** Find causes on CPU. Use GPU to confirm a total, one change per run.
- **Ask before the first upload and before the first GPU run.** State what will be created under the user's account
  (Model name, dataset, sizes) or how many GPU minutes you expect, and wait for a yes. A yes covers the tuning loop that
  follows, not a different model.
- **One cold shot per session.** The second read or import in a session is warm. A before/after inside one session
  measures the page cache. This mistake produced a false 30 s "win" here once (see `references/speedups.md`, section 9).
- **Report with the log lines.** Every claim about a run quotes the stamp lines that prove it. If a run failed, say so and
  show the error line.
- **Never touch Kaggle's own packages**, and never ship torch or anything compiled against it. The pair lives in its own uv
  venv.

## 0. Preflight

- `kaggle --version` and a logged-in CLI (`~/.kaggle/`). On Windows prefix kaggle commands with `PYTHONUTF8=1`.
- `uv --version` locally (used for the free dependency check).
- If `references/kaggle-facts.md` looks stale, or the user is on a different accelerator, refresh the numbers:
  `kaggle_push.py run scripts/probes/probe_specs.py specs --no-wheels` (add `--gpu` only with the user's yes).

## 1. Fit check

```
python scripts/hf_fit.py <org/repo>
```

It prints the weight files, licence, gating and a verdict per limit. Then do the part a script cannot:

1. **Read the model card and the inference code** (the repo's README, then the GitHub repo it points to). Find: which
   weight files inference really needs, what else gets downloaded at run time (text encoders, VAEs, face detectors,
   tokenizers), the dependency list, and the entry point.
2. **Check the dependencies resolve for Kaggle's Python**, locally:
   `uv pip compile deps.in --python-version 3.13 --python-platform x86_64-manylinux_2_28 --no-build`
3. **Judge VRAM honestly.** The script's number is weights times 1.5. Models with large activations (video, long audio,
   high resolution) need more; say so. If it only fits in fp16 or split over two GPUs, that is a finding, not a detail.
4. **See what Kaggle already has, and borrow it**: `python scripts/kaggle_push.py have <model name>`. It lists the user's
   Models file by file, the pairs in their `wheels` dataset with their pins, and public uploads that match the name.
   - A weight file already in one of the user's Models (a VAE, a text encoder, a face detector) stays out of the spec:
     attach that Model with a second `--model-source` and point the code at its mount.
   - A pair whose pins cover the dependencies (count what Kaggle's image brings too) is reused with `wheels_from` in the
     spec: no wheels are built.
   - Other people's uploads are a lead, not a source. They can change or vanish, and a `.pth` is a pickle. Borrow one only
     with the user's yes, after a CPU notebook has matched its sha256 with Hugging Face.

Give the user a short verdict: fits / fits with conditions / does not fit, the reasons, what is borrowed, what would be
uploaded (file list and total size), and anything that blocks (gated repo, no wheel for a dependency, licence that forbids redistribution even
privately). **Stop here and ask** before building. If it does not fit, say what would make it fit and do not build.

## 2. Build the pair

Write `<name>.json` in the project following `references/pair-spec.md`, then:

```
python scripts/kaggle_push.py build <name>.json
```

A CPU notebook downloads the weights straight into Kaggle (nothing passes through the user's connection), builds the
wheels, uploads the Model and a new version of the `wheels` dataset. Other pairs already in `wheels` are carried over.
A Model or a pair's wheels that the account already has are kept and not built again; `--rebuild` replaces them (needed
after the spec's files or deps change).
The command prints the tail of `setup.log`; `ALL DONE` is the proof. Typical build: a few minutes plus upload.

## 3. Write the run script

One Python file, pushed with the loader pasted in front (so `load`, `warm`, `no_init`, `no_tensorflow` are simply there):

```python
import time
T0 = time.time()
def stamp(m): print(f"[t+{time.time()-T0:.0f}s] {m}", flush=True)
stamp("start")
pair = load("<name>")                    # venv from the wheels; pair.model is already being read into RAM
stamp("pair loaded")
# in-process:  pair.activate(); import torch, <package>
# or its own process:  subprocess.run([pair.python, "-m", "<module>", ...], cwd=pair.source("<Repo>"))
stamp("imports done")
# build the model from pair.model, move it to the GPU
stamp("model on GPU")
# inference over every input in /kaggle/input/datasets/<user>/<inputs dataset>/
stamp("ALL DONE")
```

- Stamp every phase. The squeeze in step 4 works on the gaps between stamps and on nothing else.
- Point every `from_pretrained("org/name")` and hub download at a folder in `pair.model`. The input mount is read-only:
  when a file must be patched (a config that names a hub repo), copy it to `/kaggle/tmp` and symlink the rest.
- Glob exact mounts (`/kaggle/input/datasets/*/<dataset>/*`), never `/kaggle/input/**`.
- Loop over all inputs in one run. Startup is paid once per session, so ten files in one run cost one startup.
- End with an explicit marker: a script notebook reports `complete` even after a crash.

First run on CPU if the model can start there at all (imports and model construction do not need a GPU): it is free and
catches most mistakes. Then, with the user's yes:

```
python scripts/kaggle_push.py run run_<name>.py <tag> --gpu --model-source <user>/<model_slug>/PyTorch/default/1 --dataset-source <user>/<inputs>
```

## 4. Squeeze the startup

Take the stamps from the first GPU run and write the phase table: pair load, imports, model on GPU, first inference.
Work on the largest gap first. For each gap: run the probe on CPU, apply the fix, and only then spend a GPU run to confirm
the total.

| Gap | Probe (CPU, free) | What it usually is | Fix |
|---|---|---|---|
| model on GPU | `probes/probe_read.py` | weights read by one thread | `warm(path)` first thing in the run; `load()` does it for `pair.model` |
| model on GPU | `probes/probe_build.py` | random init of weights that the checkpoint then overwrites | `with no_init():` around the constructor; `no_init_patch()` when it is built inside another process |
| model on GPU | the model's own log | dtype cast on the CPU during the move to the GPU | `.to(device)` first, then `.to(dtype)` |
| model on GPU | `probes/probe_build.py` | `torch.load` copying the whole file, twice the RAM | `torch.load(mmap=True)` + `load_state_dict(assign=True)` |
| imports | `probes/probe_imports.py` | a library waiting on the network with Internet off | its off switch, set in the loader |
| imports | `probes/probe_imports.py` | Kaggle's tensorflow pulled in by a side door | `with no_tensorflow():` around the imports only |
| pair load | stamps inside `load()` | a recursive glob, or contention with a warm-up | exact mount paths |

`references/speedups.md` has, for each row, how to read the probe, the code, the gain measured on a real model, and a list
of things that were tried and did not help. Read the relevant section before applying a fix, and read the "did not help"
list before inventing a new one.

Two of the fixes change how weights get into the model (`no_init`, `mmap` + `assign`). They are only right when the two
load paths produce the same model: `probe_build.py` compares every tensor. Do not ship them for a model until it prints
`0 differ` for that model.

Stop when the remaining gaps are small against the inference itself, or when a probe shows the time is real work. Say
what is left and why. Then go on to step 5: on anything longer than a demo clip, the work after "model on GPU" is most
of the session.

## 5. Squeeze the work

The seconds after the models are ready scale with the input; startup does not. A model that takes 0.6 s per frame needs
8 GPU hours for a 30 minute video, whatever its startup. Same method as step 4, finer tools:

```
python scripts/phases.py kaggle_out/<tag>/pair-run-<tag>.log [word ...]
```

It splits a run you already paid for: seconds between stamps, and for every tqdm loop the items, seconds per item and the
time of the first item. Then, for each long phase:

1. **Stamp inside it.** Wrap the model's own functions from the run script (no edit of its code) so each step of the
   pipeline prints a stamp. A phase named "preprocessing" hid a 21 s audio load here.
2. **Time the functions inside the loops** in one diagnostic GPU run: count, total, per call and first call, with
   `torch.cuda.synchronize()` around each, because without it the time lands on whichever later line waits for the GPU.
   `references/speedups.md`, section 10, has the wrapper.
3. **Sort what you find:**

| What the timers show | What it usually is | Fix | Free CPU probe? |
|---|---|---|---|
| seconds before a loop, no GPU use | a heavy import done lazily at first call (`librosa.load`) | call what it wraps (`soundfile` + `soxr`); prove the arrays equal | yes |
| frame or audio files written and read back | PNG's zlib | BMP, or PNG level 0; prove the pixels equal | yes |
| first call of a function far above the average | `cudnn.benchmark` left on by some library, searching per new shape | switch it off after the code that wants it; judge any steady gain on a long loop | no |
| a per-item rate that is all network time | real arithmetic | fp16, a smaller input, or nothing: see below | no |
| a Python loop around a small network | per-item overhead | batch it, if the code allows | partly |

4. **Separate what keeps the result from what changes it.** Lossless container formats, load paths that compare equal and
   cudnn settings keep it: apply them, with the proof next to the patch. fp16, a faster encoder preset, a smaller detector
   input or fewer steps change the output: measure the gain, say what changes, and let the user decide. Never switch one of
   those on by default.
5. **Fingerprint what the fast path must not move** (crop boxes, token ids, a checksum of the first output) and print it
   in every run, so a later change can be checked against an earlier run for free.

Changes in different stamped phases can share one GPU run: each phase's gap still has one cause. Two changes in the same
phase cannot.

Stop when what is left is network time in the precision the user chose. Give the per-item cost and what it means for
their real input length (frames x seconds per frame, against the 12 h session and the weekly quota).

## Report

End with the phase table, before and after, in seconds, with the stamp lines under it, the GPU minutes the session used,
and what was tried without effect. Example shape:

| phase | before | after |
|---|---|---|
| pair load | 9 s | 9 s |
| imports | 53 s | 51 s |
| model on GPU | 96 s | 4 s |

If the skill taught you something new about Kaggle (a new network wait, a changed limit, a fix that stopped working), add
it to `references/kaggle-facts.md` or `references/speedups.md` with the measurement, so the next run starts from it.
