"""Drive Kaggle from this machine, API only. Needs the logged-in kaggle CLI. Nothing big comes back by itself.

  python kaggle_push.py have [search words]
      What is already on Kaggle: your Models with their files, the pairs in your `wheels` dataset, and (with search
      words) other people's public models and datasets. Run it before writing a spec: what exists is borrowed, not rebuilt.
  python kaggle_push.py build pair.json [more.json ...] [--no-upload] [--no-models] [--rebuild] [--hf-token] [--gpu]
                        [--source-kernel=<notebook>,<notebook>]
      CPU notebook: weights -> Kaggle Model, wheels -> the `wheels` Kaggle Dataset. Prints the tail of the build log.
      A Model or a pair's wheels that already exist on your account are kept as they are; --rebuild replaces them.
      A spec with "quantize": its Model is made in a GPU notebook after the wheels (bitsandbytes needs CUDA), so it
      needs --gpu: GPU time is weekly quota. --hf-token: for a gated repo, your local Hugging Face token goes up as a
      private dataset for the length of the build and is deleted after; it never appears in the notebook source.
      --source-kernel: your notebooks whose outputs already hold some of the spec's files (an older cache): those
      files are linked from there, not downloaded, so a gated repo needs no token and a GPU build no download time.
  python kaggle_push.py run script.py <tag> [--gpu] [--internet] [--no-wheels]
                        [--model-source user/model/Framework/variation/version]... [--dataset-source user/dataset]...
      Pushes script.py with kaggle_pair.py pasted in front (notebooks are single files), waits, downloads the output into
      ./kaggle_out/<tag>/ and prints the log. CPU unless --gpu: GPU time is weekly quota.
"""
import json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path.cwd() / "kaggle_out"
KAGGLE = shutil.which("kaggle") or str(Path.home() / ".local/bin/kaggle")


def kaggle(*args: str) -> str:
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    if r.returncode:
        sys.exit(f"kaggle {' '.join(args[:3])} failed:\n{r.stdout}{r.stderr}")
    return r.stdout


def user() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    for f in ("kaggle.json", "credentials.json"):
        p = Path.home() / ".kaggle" / f
        if p.exists() and json.loads(p.read_text()).get("username"):
            return json.loads(p.read_text())["username"]
    sys.exit("Kaggle username not found: set KAGGLE_USERNAME or log the kaggle CLI in.")


def push(slug: str, code: str, gpu: bool, internet: bool, datasets: list, models: list, kernels=()) -> str:
    """Push one script notebook, wait for it to end, return the final status line."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / f"{slug}.py").write_text(code, encoding="utf-8")
        (Path(d) / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user()}/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python", "kernel_type": "script",
            "is_private": "true", "enable_gpu": str(gpu).lower(), "enable_internet": str(internet).lower(),
            "machine_shape": "NvidiaTeslaT4" if gpu else "", "dataset_sources": datasets, "model_sources": models,
            "kernel_sources": [f"{user()}/{k}" for k in kernels], "competition_sources": []}))
        print(kaggle("kernels", "push", "-p", d).strip(), flush=True)
    t0, warned = time.time(), False
    while True:
        # A failed status call must not end the wait: the notebook runs on. It happened when the CLI's login expired
        # mid-run ("Permission 'kernels.get' was denied"); after `kaggle auth login --force` the same loop carries on.
        r = subprocess.run([KAGGLE, "kernels", "status", f"{user()}/{slug}"], capture_output=True, text=True, encoding="utf-8",
                           env={**os.environ, "PYTHONUTF8": "1"})
        status = r.stdout.lower().strip()
        if not r.returncode and any(s in status for s in ("complete", "error", "cancel")):
            return f"{status} after {time.time()-t0:.0f}s"
        if r.returncode and not warned:
            warned = True
            print(f"status call failed, still waiting (the notebook keeps running): {(r.stdout + r.stderr).strip()[:300]}\n"
                  "If it says denied or authentication: run `kaggle auth login --force` in another terminal.", flush=True)
        time.sleep(30)


def wheels_ready() -> list:
    """[the wheels dataset] once its newest version is mounted-ready, [] when it does not exist yet."""
    ref = f"{user()}/wheels"
    if subprocess.run([KAGGLE, "datasets", "status", ref], capture_output=True).returncode:
        return []
    while "ready" not in kaggle("datasets", "status", ref).lower():  # else the notebook silently attaches the old version
        time.sleep(10)
    return [ref]


def rows(*args: str) -> list:
    """Every page of a kaggle listing. The CLI prints 'Next Page Token = ...' above the JSON while there is more."""
    out, token = [], ""
    while True:
        text = kaggle(*args, "--format", "json", "--page-size", "200", *(["--page-token", token] if token else []))
        token = ""
        if text.startswith("Next Page Token = "):
            head, text = text.split("\n", 1)
            token = head.split(" = ", 1)[1].strip()
        out += json.loads(text) if text.lstrip().startswith("[") else []  # "No models found" is not JSON
        if not token:
            return out


def wheel_pairs() -> dict:
    """{pair: [its wheel files]} for every pair in the `wheels` dataset; {} when the dataset does not exist."""
    ref = f"{user()}/wheels"
    if subprocess.run([KAGGLE, "datasets", "status", ref], capture_output=True).returncode:
        return {}
    files = [r["name"] for r in rows("datasets", "files", ref)]
    names = [f.split("/")[0] for f in files if f.count("/") == 1 and f.endswith("_WHEELS") and "/PAIR_" in f]
    return {n: sorted(f.split("/")[1] for f in files if f.startswith(n + "/") and f.endswith(".whl")) for n in names}


def model_files(ref: str) -> dict:
    """{handle to pass as --model-source: [(file, bytes)]} for every variation of one Kaggle Model."""
    out = {}
    for i in json.loads(kaggle("models", "get", ref)).get("instances", []):
        handle = f"{ref}/{'/'.join(i['url'].split('/')[-2:])}/{i['versionNumber']}"
        out[handle] = [(r["name"], r["size"]) for r in rows("models", "instances", "versions", "files", handle)]
    return out


def have(words: list):
    me = user()
    print(f"Your Kaggle Models ({me}). Attach one with --model-source <handle>:")
    for m in rows("models", "list", "--owner", me):
        for handle, files in model_files(m["ref"]).items():
            print(f"  {handle}")
            for f, n in files:
                print(f"    {n/2**30:7.2f} GiB  {f}")
    print(f"Pairs in {me}/wheels:")
    for name, whl in wheel_pairs().items():
        print(f"  {name}: " + (" ".join("==".join(w.split("-")[:2]) for w in whl) or "source only, runs on another pair's wheels"))
    if words:
        print(f"Public on Kaggle for '{' '.join(words)}'. Other people's uploads: they can change or vanish, and a name is no "
              "proof of content. Compare sizes and sha256 with Hugging Face before borrowing:")
        for kind in ("models", "datasets"):
            text = kaggle(kind, "list", "--search", " ".join(words), "--format", "json")
            for r in json.loads(text) if text.lstrip().startswith("[") else []:
                print(f"  {kind[:-1]:7} {r['ref']}" + (f"  {r['size']/2**30:.2f} GiB" if r.get("size") else ""))


TOKEN_DS = "pair-hf-token"


def upload_token() -> list:
    """[the token dataset]: HF_TOKEN or the `hf auth login` token, uploaded private. A pushed notebook cannot read Kaggle
    secrets, so this is the way in for a gated repo. build() deletes it in a finally."""
    t = os.environ.get("HF_TOKEN") or next((f.read_text().strip() for f in [Path.home() / ".cache/huggingface/token"]
                                            if f.exists()), "")
    if not t:
        sys.exit("--hf-token: set HF_TOKEN or run `hf auth login` (and accept the model's terms on its HF page).")
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "hf_token").write_text(t, encoding="utf-8")
        (Path(d) / "dataset-metadata.json").write_text(json.dumps(
            {"title": TOKEN_DS, "id": f"{user()}/{TOKEN_DS}", "licenses": [{"name": "CC0-1.0"}]}))
        kaggle("datasets", "create", "-p", d)  # private by default
    for _ in range(60):
        # a status call right after create was refused (403 Forbidden, 2026-10-17): the new dataset was not visible yet
        r = subprocess.run([KAGGLE, "datasets", "status", f"{user()}/{TOKEN_DS}"], capture_output=True, text=True,
                           encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
        if "ready" in r.stdout.lower():
            return [f"{user()}/{TOKEN_DS}"]
        time.sleep(10)
    sys.exit(f"{TOKEN_DS} was not ready after 10 minutes.")


def build_log(slug: str, what: str):
    out = OUT / slug
    log = out / "setup.log"
    log.unlink(missing_ok=True)  # a notebook that never ran writes none: the last run's log was shown as this one's
    kaggle("kernels", "output", f"{user()}/{slug}", "-p", str(out), "--force", "--file-pattern", r"setup\.log")
    text = log.read_text(encoding="utf-8") if log.exists() else "(no setup.log: the script crashed before logging)"
    print("\n".join(text.splitlines()[-25:]))
    subprocess.run([KAGGLE, "kernels", "delete", f"{user()}/{slug}", "--yes"], capture_output=True)  # log is local now
    if "ALL DONE" not in text:  # script notebooks report "complete" even when the script crashed: trust the marker
        sys.exit(f"Build of {what} did not finish. Full log: {log}")


def build(specs: list, flags: list):
    try:  # the upload is inside: a token dataset created and then left by a failed wait was found on Kaggle once
        token = upload_token() if "--hf-token" in flags else []
        src = [k for f in flags if f.startswith("--source-kernel=") for k in f.split("=", 1)[1].split(",")]
        _build(specs, [f for f in flags if f not in ("--hf-token", "--gpu") and not f.startswith("--source-kernel=")],
               token, "--gpu" in flags, src)
    finally:
        if "--hf-token" in flags:
            subprocess.run([KAGGLE, "datasets", "delete", f"{user()}/{TOKEN_DS}", "--yes"], capture_output=True)


def _build(specs: list, flags: list, token: list, gpu: bool, src: list = ()):
    pairs, built = {}, wheel_pairs()
    for f in specs:
        spec = json.loads(Path(f).read_text(encoding="utf-8"))
        name, wf = spec["name"], spec.get("wheels_from")  # not `src`: that is the --source-kernel list
        if wf and wf not in built:
            sys.exit(f"{name}: wheels_from names pair '{wf}', which is not in the wheels dataset ({sorted(built)}).")
        skip = []
        if "--rebuild" not in flags:  # borrow what is already on the account
            if name in built:
                skip.append("wheels")
            if spec.get("hf") and not subprocess.run([KAGGLE, "models", "get", f"{user()}/{spec['model_slug']}"],
                                                     capture_output=True).returncode:
                skip.append("model")
        for s in skip:
            print(f"{name}: {s} already on Kaggle, kept as is (--rebuild to replace)")
        if len(skip) == 1 + bool(spec.get("hf")):
            continue
        pairs[name] = {**spec, "skip": skip}
    if not pairs:
        return print("Nothing to build.")
    # a quantized Model is made in a GPU notebook, after the wheels it quantizes with are in the dataset
    quant = {n: p for n, p in pairs.items() if p.get("quantize") and p.get("hf") and "model" not in p["skip"]
             and "--no-models" not in flags}
    if quant and not gpu:
        sys.exit(f"{sorted(quant)}: spec key quantize needs a GPU notebook. Add --gpu (weekly quota: the download of the "
                 "full-precision weights plus a few minutes per component).")
    cpu = {n: {**p, "skip": p["skip"] + ["model"] * (n in quant)} for n, p in pairs.items()}
    if any(len(p["skip"]) < 1 + bool(p.get("hf")) for p in cpu.values()):
        name = "-".join(cpu)
        code = (f"import sys\nsys.argv = ['kaggle_setup.py', *{flags!r}]\nPAIRS_JSON = {json.dumps(cpu)!r}\n"
                f"KEEP = {sorted(built)!r}\n" + (HERE / "kaggle_setup.py").read_text(encoding="utf-8"))
        # current wheels attached: other pairs are carried over
        print(push(f"pair-build-{name}", code, gpu=False, internet=True, datasets=wheels_ready() + token, models=[], kernels=src))
        build_log(f"pair-build-{name}", name)
    for name in quant:
        # Right after an upload, `datasets status` can still say ready about the version before: a GPU notebook pushed
        # then mounted it and stopped with "PAIR_klein9b_WHEELS not found" after downloading 34 GB (2026-10-17).
        for _ in range(60):
            if name in wheel_pairs():
                break
            time.sleep(30)
        else:
            sys.exit(f"Pair {name} is not in the wheels dataset 30 minutes after its build.")
    for name, p in quant.items():  # the loader goes in front: quantize() builds the pair's venv with load()
        code = (f"import sys\nsys.argv = ['kaggle_setup.py', '--quantize', *{flags!r}]\n"
                f"PAIRS_JSON = {json.dumps({name: {**p, 'skip': ['wheels']}})!r}\nKEEP = []\n"
                + (HERE / "kaggle_pair.py").read_text(encoding="utf-8") + "\n"
                + (HERE / "kaggle_setup.py").read_text(encoding="utf-8"))
        print(push(f"pair-quant-{name}", code, gpu=True, internet=True, datasets=wheels_ready() + token, models=[], kernels=src))
        build_log(f"pair-quant-{name}", name)


def run(script: str, tag: str, argv: list):
    def opt(flag): return [argv[i + 1] for i, a in enumerate(argv) if a == flag]
    slug = f"pair-run-{tag}".replace("_", "-")  # Kaggle turns _ into - in the slug, and the status call then misses it
    code = (HERE / "kaggle_pair.py").read_text(encoding="utf-8") + "\n" + Path(script).read_text(encoding="utf-8")
    datasets = ([] if "--no-wheels" in argv else wheels_ready()) + opt("--dataset-source")
    print(push(slug, code, "--gpu" in argv, "--internet" in argv, datasets, opt("--model-source")))
    out = OUT / tag
    kaggle("kernels", "output", f"{user()}/{slug}", "-p", str(out), "--force")
    log = out / f"{slug}.log"
    for r in json.loads(log.read_text(encoding="utf-8")) if log.exists() else []:
        d = r.get("data", "").rstrip()
        if d and "SyntaxWarning" not in d and "NbConvert" not in d:
            print(d[:400])
    print(f"output in {out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "have":
        have(a[1:])
    elif len(a) >= 2 and a[0] == "build":
        build([x for x in a[1:] if not x.startswith("--")], [x for x in a[1:] if x.startswith("--")])
    elif len(a) >= 3 and a[0] == "run":
        run(a[1], a[2], a[3:])
    else:
        sys.exit(__doc__)
