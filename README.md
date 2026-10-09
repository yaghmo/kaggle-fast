# kaggle-fast: 96 seconds of model loading down to 4

A skill for Claude Code that makes a Hugging Face model start fast on a free Kaggle GPU notebook. It measures where the
time between "session started" and "first inference" goes, and removes it.

Kaggle gives 30 GPU hours a week and counts every second a GPU session is alive. The minutes a notebook spends on
`pip install`, downloading weights, importing and loading the model are paid from that budget on every single run.

## The numbers: SAM-Audio large (fp16, 6.9 GiB checkpoint) on a T4

Time from the start of the script to the model sitting on the GPU, same notebook type, same inputs:

| phase | before | after | what changed |
|---|---|---|---|
| pair load (venv from wheels) | 9 s | 29 s | slower: the checkpoint is being read in the background on the same disk |
| imports | 53 s | 34 s | Kaggle's tensorflow no longer imported by accident |
| build model + load weights + copy to GPU | 96 s | 4 s | see below |
| **start to model on GPU** | **158 s** | **67 s** | |
| host RAM peak | 15.9 GiB | 8.5 GiB | |

A second run of the same script: pair load 20 s, imports 41 s, model on the GPU at 65 s. The phases trade time with
each other from run to run; the total is the number to read.

The 96 seconds were three things, each found with a probe on a free CPU notebook:

| cause | measured | fix | after |
|---|---|---|---|
| random init of weights the checkpoint then overwrites | 47.2 s (`Tensor.uniform_`: 37.6 s) | `with no_init():` around the constructor | 1.2 s |
| `torch.load` copying the whole file | 45.1 s, file already in RAM | `mmap=True` + `load_state_dict(assign=True)` | 0.2 s |
| checkpoint read by one thread | 135-157 MB/s | 16 readers in the background, started at second 0 | 769 MB/s |

The fast path gives the same model: `1449 tensors compared, 0 extra, 0 differ`.

Log lines from the GPU run:

```
[t+0s] start
[t+29s] pair loaded
[t+63s] imports done
[t+64s] model built, weights mapped
[t+67s] model on GPU
loaded 4s, VRAM 7.3 GiB, host RAM peak 8.5 GiB
[kyky] separate 14s, peak VRAM 11.1 GiB
[t+94s] ALL DONE
```

## Second example: Fish Audio S2 Pro (4.5 billion parameters, its own API server) on 2x T4

Here the model is built inside the project's server, in another process, so the fixes are one-line rewrites of its code
in the run's writable checkout. Times are from the server's own log.

| step | before | after | what changed |
|---|---|---|---|
| LLM constructor | 69 s | 0.2 s | `no_init_patch()` on the line that builds the model |
| move to GPU | 34 s | 2.6 s | the bf16 to fp16 cast was running on one CPU core: copy first, cast on the GPU |
| codec load | 29 s | 8 s | `no_init_patch()` |
| **server ready** | **186 s** | **54 s** | |
| whole run, 7 requests | 403 s | 232 s | |

Same tensors with and without the skipped init: `LLM ... 361 tensors compared, 0 extra, 0 differ`,
`codec ... 541 tensors compared, 0 extra, 0 differ`. All seven requests returned 200 before and after.

Other models, same method:

- **LatentSync 1.5**: imports of the real entry point 72.9 s -> 10.0 s. `albumentations` was waiting for a network timeout
  on every start with Internet off.
- **Any venv with its own protobuf**: `onnx` refused to load (`gencode 6.31.1 runtime 5.29.5`). Kaggle registers its own
  `google` package before your code runs; the loader fixes the lookup.

## Use

```
/kaggle-fast:run <org/repo>            fit check -> you confirm -> build -> run script -> squeeze startup
/kaggle-fast:run fit <org/repo>        only the fit check
/kaggle-fast:run squeeze <run script>  only the startup squeeze
```

## What it does

1. **Fit check.** Reads the repo's metadata (no download): weight files, licence, gating, VRAM estimate against a T4, host
   RAM, the same weights shipped in several formats. Resolves the dependencies for Kaggle's Python on your machine, so a
   package with no wheel fails in seconds instead of in a notebook.
2. **Asks before it uploads anything**, and before the first GPU run.
3. **Builds a "pair"** in a free CPU notebook: the weights become a private Kaggle Model, the dependencies become pinned
   wheels in one private dataset. Weights go from Hugging Face straight to Kaggle; nothing passes through your connection.
4. **Writes the run script** around a small loader: a uv venv built from the wheels, Kaggle's own torch and CUDA left
   untouched, a timestamp printed at every phase.
5. **Squeezes the startup.** Free CPU notebooks find the cause of each slow phase; one GPU run confirms the fix.

## What did not help

Kept in the skill so nobody spends a run on them again (`skills/run/references/speedups.md` has the numbers):

- Compiling bytecode at install. It looked like 30 s saved in a CPU test and cost 34 s in a real GPU session.
- A prebuilt site-packages tree mounted from a notebook output instead of installing wheels.
- Reading Kaggle's torch and CUDA libraries into the page cache up front.
- Building the model in fp16 instead of fp32.
- Blocking tensorflow for the whole run instead of just the imports. Imports got faster, then inference crashed in
  `einops`.
- Starting the background weight read later, after the install. The time moved between phases; the total did not (65 s
  against 67 s).

The first of these is why the skill has a rule: **one cold measurement per session**. The second import in a session is
warm, so a before/after inside one session measures the page cache, not the change.

## Install

In Claude Code:

```
/plugin marketplace add yaghmo/kaggle-fast
/plugin install kaggle-fast@yaghmo
```

Needs: the `kaggle` CLI, logged in; `uv`; Python 3.10 or newer on your machine. No other dependencies.

## What it touches in your Kaggle account

Everything is created private: one Kaggle Model per pair, one dataset named `wheels`, and script notebooks named
`pair-build-*` and `pair-run-*`. Builds and probes run on CPU notebooks, which are free. GPU runs spend your weekly
quota; the skill says how many minutes it expects and waits for a yes.

## Limits, stated plainly

- Measured on Kaggle in October 2026, Python 3.13 image, T4 notebooks. Kaggle changes; `probes/probe_specs.py` re-reads the
  platform and the skill is told to correct its own notes when a run disagrees with them.
- The VRAM verdict is weights times 1.5, a rule of thumb from one model. The run decides.
- Gated and private Hugging Face repos are not supported: the build notebook cannot read Kaggle secrets.
- `no_init` and `mmap` + `assign` change how weights get into the model. The skill refuses to ship them for a model until
  a probe has compared every tensor of both load paths.
- Not affiliated with Kaggle, Google or Hugging Face.

## Licence

MIT.
