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
| `wheels_from` | Name of a pair already in `wheels` whose pins cover this model. Replaces `deps` and `excludes`: no wheels are built, the folder holds only `repos`, and `load("<name>")` builds the venv from that pair. Rebuilding that pair with other pins changes this one too. |
| `excludes` | Packages to leave to Kaggle, on top of torch, torchvision, torchaudio, torchcodec, triton, pillow, requests (always excluded). Add `numpy` when the code is happy with Kaggle's. |
| `repos` | `{folder: git URL}` cloned at build time, commit recorded in `SOURCES.txt`, read at run time with `pair.source("<folder>")`. |
| `quantize` | `{component folder: "int8" \| "nf4"}`, e.g. `{"transformer": "int8", "text_encoder": "int8"}`. Those folders go into the Model already quantized with bitsandbytes, so the run loads them with a plain `from_pretrained` (the quantization config is in their `config.json`) and reads half (int8) or a quarter (nf4) of the bytes. Needs `bitsandbytes` in `deps` (Kaggle's image has none) and `build --gpu`: the Model is made in a GPU notebook after the wheels. Each folder needs a `config.json` naming its class (`_class_name` for diffusers, `architectures` for transformers). |
| `quantize_dtype` | Dtype of the layers bitsandbytes leaves alone and of its compute. `"float16"` (default: a T4 has no native bf16). `"bfloat16"` for a model that overflows to black or NaN output in fp16. |

## Getting the spec right

- **Borrow before building.** `kaggle_push.py have <model name>` shows what the account already holds. Leave out of `hf`
  any file one of your Models already has (attach that Model at run time), and use `wheels_from` when a pair's pins cover
  the deps. `build` skips a Model or a pair that already exists, so after changing a spec pass `--rebuild`.

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
- **Quantizing changes the output**, unlike the startup fixes: build it when the model does not fit a T4 otherwise (see
  "Fitting a T4" in `SKILL.md`), compare a few outputs with the full-precision ones where you can, and let the user decide.
- **A gated repo** (`hf_fit.py` says so) needs its terms accepted on Hugging Face by the token's account, then
  `build --hf-token`.
- After a build, read the tail of `setup.log`: the pinned source commits and wheel counts are printed there. `ALL DONE` is
  the only proof it finished.
