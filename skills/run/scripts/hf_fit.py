"""Does a Hugging Face model fit a free Kaggle notebook? Local, no dependencies, downloads nothing but metadata.

  python hf_fit.py <org/repo> [--json]        HF_TOKEN in the environment is used for private / gated repos

Reports the weight files, their size on disk and in VRAM, licence and gating, and a verdict per Kaggle limit. The VRAM
verdict is an estimate from the weights alone: the run decides. A repo often holds the same weights several times
(safetensors + bin, several precisions, several quantisations): read the file list and pick what the pair really needs.
"""
import json, os, sys, urllib.request

GIB = 2**30
# Kaggle limits as observed in October 2026. Refresh with probes/probe_specs.py: the platform changes.
T4_VRAM = 15 * GIB        # one T4; a GPU notebook has two
HOST_RAM = 30 * GIB
BIGGEST_UPLOADED = 6.9    # GiB: the largest Kaggle Model this tooling has uploaded. The real cap was not found in the docs.
ACTIVATIONS = 1.5         # peak VRAM / weights, measured on one 7.3 GiB model (11.1 GiB peak). A rule of thumb, no more.
WEIGHTS = (".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".gguf", ".onnx", ".msgpack", ".h5")
BYTES = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "F16": 2, "BF16": 2, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def fit(repo: str) -> dict:
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}?blobs=true")
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    info = json.load(urllib.request.urlopen(req, timeout=30))
    files = sorted(((s.get("size") or 0, s["rfilename"]) for s in info.get("siblings", [])), reverse=True)
    weights = [(n, f) for n, f in files if f.lower().endswith(WEIGHTS)]
    twins = {}  # same size to 10 MiB, different file: almost always one set of weights saved in several formats
    for n, f in weights:
        if n > 50 * 2**20:
            twins.setdefault(n // (10 * 2**20), []).append(f)
    params = (info.get("safetensors") or {}).get("parameters") or {}
    stored = sum(BYTES.get(k, 4) * v for k, v in params.items()) or sum(n for n, _ in weights)
    fp16 = sum(min(BYTES.get(k, 4), 2) * v for k, v in params.items()) if params else None
    card = info.get("cardData") or {}
    return {
        "repo": repo, "gated": info.get("gated", False), "private": info.get("private", False),
        "license": card.get("license") or next((t[8:] for t in info.get("tags", []) if t.startswith("license:")), "unknown"),
        "library": info.get("library_name"), "task": info.get("pipeline_tag"),
        "repo_gib": sum(n for n, _ in files) / GIB, "weights_gib": sum(n for n, _ in weights) / GIB,
        "weight_files": [(round(n / GIB, 2), f) for n, f in weights[:25]],
        "same_weights_twice": sorted(v for v in twins.values() if len(v) > 1),
        "params_by_dtype": params, "vram_stored_gib": stored / GIB, "vram_fp16_gib": fp16 / GIB if fp16 else None,
        "vram_from": "parameter counts in the safetensors metadata" if params else "the size of every weight file added up",
    }


def verdicts(r: dict) -> list:
    out = []
    def gpu(label, gib):
        peak = gib * ACTIVATIONS
        if peak <= T4_VRAM / GIB: where = "fits one T4"
        elif peak <= 2 * T4_VRAM / GIB: where = "too big for one T4, fits only if the model can be split over the two"
        else: where = "does not fit the two T4s"
        out.append(f"VRAM, {label}: weights {gib:.1f} GiB, peak about {peak:.1f} GiB (x{ACTIVATIONS}) -> {where}")
    gpu("as stored", r["vram_stored_gib"])
    if r["vram_fp16_gib"] and r["vram_fp16_gib"] < r["vram_stored_gib"] * 0.99:
        gpu("converted to fp16", r["vram_fp16_gib"])
    out.append(f"Host RAM ({HOST_RAM/GIB:.0f} GiB): " + ("fine" if r["vram_stored_gib"] * 2 <= HOST_RAM / GIB else
               "a plain torch.load keeps two copies and will not fit: load with mmap + assign (references/speedups.md)"))
    out.append(f"Upload: {r['weights_gib']:.1f} GiB of weight files in the repo; the largest Model uploaded so far was "
               f"{BIGGEST_UPLOADED} GiB" + ("" if r["weights_gib"] <= BIGGEST_UPLOADED else ": beyond what was tested, pick files or expect surprises"))
    if r["gated"] or r["private"]:
        out.append("BLOCKER: gated or private repo. The build notebook cannot read Kaggle secrets, so it cannot download it.")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    r = fit(args[0])
    if "--json" in sys.argv:
        print(json.dumps({**r, "verdicts": verdicts(r)}, indent=2))
    else:
        print(f"{r['repo']}  task={r['task']}  library={r['library']}  licence={r['license']}  gated={r['gated']}")
        print(f"repo {r['repo_gib']:.2f} GiB, weight files {r['weights_gib']:.2f} GiB; VRAM estimate from {r['vram_from']}")
        for gib, f in r["weight_files"]:
            print(f"  {gib:7.2f} GiB  {f}")
        if r["same_weights_twice"]:
            print("same weights in more than one format (ship one):", r["same_weights_twice"][:6])
        for v in verdicts(r):
            print("-", v)
