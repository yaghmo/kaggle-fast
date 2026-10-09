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
   It first lists what your account already holds and borrows it: a weight file in another of your Models is attached
   instead of uploaded again, and a pair whose pins cover the dependencies lends its wheels.
4. **Writes the run script** around a small loader: a uv venv built from the wheels, Kaggle's own torch and CUDA left
   untouched, a timestamp printed at every phase.
5. **Squeezes the startup.** Free CPU notebooks find the cause of each slow phase; one GPU run confirms the fix.

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

## Limits

- Measured on Kaggle in October 2026, T4 notebooks. Your numbers will differ; the method is what transfers.
- The VRAM verdict is an estimate (weights times 1.5). The run decides.
- Gated and private Hugging Face repos are not supported.
- Not affiliated with Kaggle, Google or Hugging Face.

## Licence

MIT.
