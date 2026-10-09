# The pair spec

One JSON file per model. `kaggle_push.py build spec.json` turns it into a private Kaggle Model (the weights) and a folder in
your private `wheels` dataset (pinned wheels + pinned source checkouts). Keep the specs in the project, next to the run
scripts: they are the record of what was built.

```json
{
  "name": "latentsync15",
  "model_slug": "latentsync-1-5",
  "license": "Other",
  "hf": [
    ["ByteDance/LatentSync-1.5", ["latentsync_unet.pt", "whisper/tiny.pt"], ""],
    ["stabilityai/sd-vae-ft-mse", ["config.json", "diffusion_pytorch_model.safetensors"], "sd-vae-ft-mse"]
  ],
  "zips": {"auxiliary/models/buffalo_l": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"},
  "license_url": "https://raw.githubusercontent.com/bytedance/LatentSync/main/LICENSE",
  "deps": ["diffusers==0.32.2", "transformers==4.48.0", "decord", "omegaconf", "insightface==0.7.3", "onnxruntime-gpu"],
  "excludes": [],
  "repos": {"LatentSync": "https://github.com/bytedance/LatentSync.git"}
}
```

| key | meaning |
|---|---|
| `name` | The pair's name: `load("<name>")` in the run script, folder name in the `wheels` dataset. Short, no dashes. |
| `model_slug` | Kaggle Model slug (lowercase, dashes). The Model is `<you>/<model_slug>/pyTorch/default`. |
| `license` | Kaggle licence name for the Model. `"Other"` unless you checked the exact Kaggle name. Everything is uploaded private. |
| `hf` | List of `[repo, files, subfolder]`. `files: null` takes the whole repo minus `.md`. Name the files: repos carry training leftovers and the same weights in several formats. Omit `hf` for a wheels-only pair. |
| `zips` | `{folder in the Model: URL}` for weights that are not on Hugging Face. Unzipped into that folder. |
| `license_url` | Fetched into the Model as `LICENSE`. Optional. |
| `deps` | What you would `pip install`. Pin what the model's code pins. `pkg @ git+https://...` works and is pinned to the commit built. |
| `excludes` | Packages to leave to Kaggle, on top of torch, torchvision, torchaudio, torchcodec, triton, pillow, requests (always excluded). Add `numpy` when the code is happy with Kaggle's. |
| `repos` | `{folder: git URL}` cloned at build time, commit recorded in `SOURCES.txt`, read at run time with `pair.source("<folder>")`. |

## Getting the spec right

- **Everything the code fetches at run time has to be in the pair**, or the run needs Internet and pays the download every
  time. Read the model's inference code for `from_pretrained("org/name")`, `hf_hub_download`, `torch.hub`, `urlretrieve`
  and library defaults that download on first use (insightface model packs, whisper, tokenizers, VAEs, text encoders). Add
  each to `hf` or `zips`, then point the code at the local copy in the run script.
- **Check the deps resolve for Kaggle's Python before building**, locally and for free:
  `uv pip compile deps.in --python-version 3.13 --python-platform x86_64-manylinux_2_28 --no-build`
  (take the Python version from `probes/probe_specs.py`). A package with no wheel for that Python fails here in seconds
  instead of in the build notebook in minutes. `--no-build` failures on a package that is sdist-only are fine: the build
  notebook compiles those.
- **An old numpy pin usually has to go**: Python 3.13 has no numpy 1.26 wheel. Drop the pin and let the run tell you.
- **Do not add torch, CUDA or anything compiled against them** (xformers, flash-attn, torch-scatter). Kaggle's torch stays;
  a wheel built against another torch breaks at import.
- After a build, read the tail of `setup.log`: the pinned source commits and wheel counts are printed there. `ALL DONE` is
  the only proof it finished.
