# Kaggle facts that shape the design

Everything here was observed on Kaggle in October 2026 (Python 3.13 image, T4 notebooks) unless marked otherwise. The
platform changes: when a run contradicts a line below, trust the run and fix the line.

## The machine

- A session is one container with the accelerator chosen at start. Changing the accelerator starts a new machine. Sessions
  share nothing: no RAM, no page cache, no `/kaggle/tmp`. A warm CPU session cannot hand anything to a GPU session.
- GPU quota is wall-clock time of a GPU-enabled session, busy or idle. Startup seconds are quota.
- A run is capped at 12 h. Not tested here: taken from Kaggle's documentation.
- One GPU notebook at a time per account in practice; the GPU queue took about 10 minutes. CPU notebooks start in about a
  minute and several run at once.
- CPU notebooks: 4 cores, 31 GiB RAM, torch `2.11.0+cpu`. GPU notebooks: 2x T4 with 15 GiB each, torch `+cu128`. The
  two images differ, so a CPU measurement is a lead and a GPU run is the result.
- The image ships its own torch, torchvision, torchaudio, torchcodec, built against its CUDA. Never ship or reinstall
  these: they are ABI-coupled to each other and to the driver. Read the versions from a probe run, do not assume them.
- Wheels built in a CPU notebook fit the GPU notebook: same image, same Python.

## Disks

- `/kaggle/input` is read-only and slow for one reader, fast for many. Measured on a 6.9 GiB file, cold:
  1 thread 120-240 MB/s, 4 threads 370-560 MB/s, 16 threads 770-950 MB/s.
- `/kaggle/working` is the notebook output and is capped at 20 GB. Put scratch in `/kaggle/tmp`.
- Mount layout: datasets at `/kaggle/input/datasets/<user>/<slug>/`, models at
  `/kaggle/input/models/<user>/<slug>/<framework>/<variation>/<version>/`, another notebook's output at
  `/kaggle/input/notebooks/<user>/<slug>/`.
- Never glob `/kaggle/input/**` blindly. With a large input attached, one recursive walk took 144 s, and it matches files
  you did not mean (package test audio, demo assets). Glob the exact mount you want.

## Datasets, models, notebook outputs

- `kagglehub.dataset_upload` zips a folder with more than 50 files. Kaggle unpacks it server-side, so the mount shows plain
  files. It also unpacks nested `.tar.gz`: ship a repo as a folder, not as a tarball.
- A dataset with tens of thousands of loose files never finished processing. Keep datasets to wheels and a few source
  trees.
- A new dataset version is not mounted until its status is `ready`. Wait for it before pushing a notebook that attaches it,
  or the notebook silently gets the previous version.
- A new dataset version replaces the old one. A build that rebuilds one pair must carry the other pairs over from the
  attached current version, or they are gone.
- That carry-over failed once: a build notebook had `wheels` in its sources, found no other pair under `/kaggle/input`, and
  uploaded a version with one pair instead of four. Cause not found. The builder now refuses to upload when a pair the
  dataset holds would be dropped.
- An old dataset version cannot be reached from a script notebook. Tried: `kagglehub.dataset_download(".../versions/12")`
  (`New Datasets cannot be attached in non-interactive sessions`), the same over HTTP with `DISABLE_KAGGLE_CACHE` (404 on a
  private dataset), and `owner/slug/versions/12` in `dataset_sources` (the CLI rejects it; the API accepts it and mounts
  the latest). What does work, from your own machine: `ListDatasetFiles` and `DownloadDataset` with
  `datasetVersionNumber`, so a lost pair can be rebuilt from the old version's `requirements.txt` as exact pins.
- `kaggle kernels output` downloads everything, which can be gigabytes. Pass `--file-pattern`. Listing an output with tens
  of thousands of files runs into `429 Too Many Requests`.

## Auth and network

- `kagglehub` authenticates inside a notebook with no secret.
- A script notebook pushed through the API could not read Kaggle secrets: `ConnectionError` after about 30 s. Do not call
  `UserSecretsClient` on the hot path.
- With Internet off, any library that phones home at import waits for its timeout. Found so far: `albumentations` (pulled
  in by `insightface`) checks for a newer release, about 63 s. `NO_ALBUMENTATIONS_UPDATE=1` turns it off.
- A script notebook reports `complete` even when the script crashed. Print an explicit end marker and check for it.
- The CLI's OAuth login (`kaggle auth login`) expires and was not renewed by itself: 12 h after login every call failed
  with `Permission 'kernels.get' was denied` or `Authentication required`, in the middle of a run. The notebook kept
  running; only the local wait died. `kaggle auth login` then says "already logged-in"; `kaggle auth login --force`
  fixes it, and the token it issued lasted 3 h. For anything unattended, use an API token instead of the OAuth login.

## Python packaging traps

- Kaggle registers its own `google` package at interpreter start (`sys.modules['google']` exists before your code runs,
  `__path__` pointing at Kaggle's dist-packages). A newer `protobuf` in a venv is therefore never found, whatever the
  `sys.path` order, and `onnx` and friends fail with `Detected mismatched Protobuf Gencode/Runtime major versions`. Fix: a
  `.pth` file in the venv that inserts the venv's `google/` at the front of `google.__path__`. The loader does this.
  Adding `google/__init__.py` to the venv does not help.
  The `.pth` only acts in the pair's own interpreter (`pair.python`). After `pair.activate()` in the notebook's process
  you still get Kaggle's protobuf: run protobuf-dependent code (onnx, mediapipe) through `pair.python`.
- `transformers.from_pretrained` restores the original `torch.nn.init` functions when it finishes. Any patch of
  `torch.nn.init` made before it is silently undone for everything constructed after it.
- `onnxruntime-gpu` from PyPI is built for CUDA 13 from 1.27 on; Kaggle's torch is `+cu128`. An unpinned install
  (1.31.0 here) cannot load its CUDA provider (`libcublasLt.so.13: cannot open shared object file`) and runs every
  session on the CPU, with a warning and no error. Seen in a GPU run: `Applied providers: ['CPUExecutionProvider']`.
  Pin `onnxruntime-gpu<1.27` (CUDA 12.8 builds) and print `session.get_providers()` in the run script.
- Kaggle's torch loads checkpoints with `weights_only=True`, which refuses files in torch's old `.tar` format
  (torchvision's `resnet18-5c106cde.pth`): `Cannot use weights_only=True with files saved in the legacy .tar format`.
  Patch that one `torch.load` to `weights_only=False`, and only for a file whose hash you checked.
- Notebooks pushed through the API are single files: paste the loader in front of the run script. Tags with `_` end up
  with `-` in the notebook's slug.
- On Windows, run the `kaggle` CLI with `PYTHONUTF8=1` and read logs as UTF-8.
