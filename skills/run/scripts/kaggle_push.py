"""Drive Kaggle from this machine, API only. Needs the logged-in kaggle CLI. Nothing big comes back by itself.

  python kaggle_push.py build pair.json [more.json ...] [--no-upload] [--no-models]
      CPU notebook: weights -> Kaggle Model, wheels -> the `wheels` Kaggle Dataset. Prints the tail of the build log.
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


def push(slug: str, code: str, gpu: bool, internet: bool, datasets: list, models: list) -> str:
    """Push one script notebook, wait for it to end, return the final status line."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / f"{slug}.py").write_text(code, encoding="utf-8")
        (Path(d) / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user()}/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python", "kernel_type": "script",
            "is_private": "true", "enable_gpu": str(gpu).lower(), "enable_internet": str(internet).lower(),
            "machine_shape": "NvidiaTeslaT4" if gpu else "", "dataset_sources": datasets, "model_sources": models,
            "kernel_sources": [], "competition_sources": []}))
        print(kaggle("kernels", "push", "-p", d).strip(), flush=True)
    t0 = time.time()
    while True:
        status = kaggle("kernels", "status", f"{user()}/{slug}").lower().strip()
        if any(s in status for s in ("complete", "error", "cancel")):
            return f"{status} after {time.time()-t0:.0f}s"
        time.sleep(30)


def wheels_ready() -> list:
    """[the wheels dataset] once its newest version is mounted-ready, [] when it does not exist yet."""
    ref = f"{user()}/wheels"
    if subprocess.run([KAGGLE, "datasets", "status", ref], capture_output=True).returncode:
        return []
    while "ready" not in kaggle("datasets", "status", ref).lower():  # else the notebook silently attaches the old version
        time.sleep(10)
    return [ref]


def build(specs: list, flags: list):
    pairs = {}
    for f in specs:
        spec = json.loads(Path(f).read_text(encoding="utf-8"))
        pairs[spec["name"]] = spec
    name = "-".join(pairs)
    slug = f"pair-build-{name}"
    code = (f"import sys\nsys.argv = ['kaggle_setup.py', *{flags!r}]\nPAIRS_JSON = {json.dumps(pairs)!r}\n"
            + (HERE / "kaggle_setup.py").read_text(encoding="utf-8"))
    print(push(slug, code, gpu=False, internet=True, datasets=wheels_ready(), models=[]))  # current wheels attached: other pairs are carried over
    out = OUT / slug
    kaggle("kernels", "output", f"{user()}/{slug}", "-p", str(out), "--force", "--file-pattern", r"setup\.log")
    log = out / "setup.log"
    text = log.read_text(encoding="utf-8") if log.exists() else "(no setup.log: the script crashed before logging)"
    print("\n".join(text.splitlines()[-25:]))
    if "ALL DONE" not in text:  # script notebooks report "complete" even when the script crashed: trust the marker
        sys.exit(f"Build of {name} did not finish. Full log: {log}")


def run(script: str, tag: str, argv: list):
    def opt(flag): return [argv[i + 1] for i, a in enumerate(argv) if a == flag]
    slug = f"pair-run-{tag}"
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
    if len(a) >= 2 and a[0] == "build":
        build([x for x in a[1:] if not x.startswith("--")], [x for x in a[1:] if x.startswith("--")])
    elif len(a) >= 3 and a[0] == "run":
        run(a[1], a[2], a[3:])
    else:
        sys.exit(__doc__)
