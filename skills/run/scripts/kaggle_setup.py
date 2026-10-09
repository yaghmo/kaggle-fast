"""Build model "pairs" on Kaggle and save them to your own account (private). Runs inside a CPU Kaggle notebook with
Internet on; kaggle_push.py pushes it and injects PAIRS_JSON (the pair specs, see references/pair-spec.md).

  Model   <you>/<model_slug>/pyTorch/default   weights only (+ LICENSE), marker file PAIR_<name>_MODEL
  Dataset <you>/wheels                         ONE dataset, one folder per pair: <name>/ holds that pair's pinned wheels,
                                               requirements.txt and pinned source checkouts (marker PAIR_<name>_WHEELS);
                                               `uv` binary at the root

A run attaches the Model + the `wheels` dataset, loads the pair into its own uv venv and starts: nothing is downloaded,
and Kaggle's preinstalled packages are never touched (kaggle_pair.py). Same Python as the GPU runtime, so wheels built
here fit the notebooks that load them. Building one pair keeps the others, as long as the current `wheels` dataset is
attached (kaggle_push.py does that).

Flags: --no-upload (build, upload nothing: a smoke test)   --no-models (wheels only)
"""
import json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

WORK = Path("/kaggle/tmp")           # /kaggle/working is capped at 20 GB
KEEP_OUT = Path("/kaggle/working")   # only the log goes here
T0 = time.time()
DATASET = "wheels"
# Never ship these: Kaggle's builds match its CUDA driver and each other (torchcodec is built against its exact torch).
KAGGLE_OWN = ["torch", "torchvision", "torchaudio", "torchcodec", "triton", "pillow", "requests"]
PAIRS = json.loads(globals().get("PAIRS_JSON", "{}"))


class _Tee:
    def __init__(s, *f): s.f = f
    def write(s, x): [f.write(x) or f.flush() for f in s.f]
    def flush(s): [f.flush() for f in s.f]


def stamp(m): print(f"[t+{time.time()-T0:.0f}s] {m}", flush=True)


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def uv() -> str:
    exe = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    if not Path(exe).exists():
        sh("curl -LsSf https://astral.sh/uv/install.sh | sh")
    return exe


def fetch(url: str, dst: Path, want: int | None = None, token: str | None = None):
    """curl straight to dst and verify the size. HF's Xet client once left a 0-byte file while exiting 0."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    auth = f'-H "Authorization: Bearer {token}"' if token else ""
    for attempt in range(1, 4):
        sh(f'curl -fsSL {auth} --retry 5 --retry-delay 5 --retry-all-errors -C - -o "{dst}" "{url}"')
        got = dst.stat().st_size if dst.exists() else 0
        print(f"  {dst.name}: {got}/{want or '?'} bytes (attempt {attempt})", flush=True)
        if want is None and got > 0 or got == want:
            return
    raise RuntimeError(f"{dst}: got {got} bytes, expected {want}")


def build_model(p: dict, out: Path):
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    for repo, files, sub in p.get("hf", []):
        sizes = {s.rfilename: s.size for s in HfApi().model_info(repo, files_metadata=True, token=token).siblings}
        for f in files or [f for f in sizes if not f.endswith((".md", ".gitattributes"))]:
            fetch(f"https://huggingface.co/{repo}/resolve/main/{f}", out / sub / f, sizes[f], token)
    for rel, url in p.get("zips", {}).items():
        z = WORK / "tmp.zip"
        fetch(url, z)
        zipfile.ZipFile(z).extractall(out / rel)
        z.unlink()
    if p.get("license_url"):
        fetch(p["license_url"], out / "LICENSE")


def build_wheels(name: str, p: dict, out: Path):
    out.mkdir(parents=True)
    (WORK / "excludes.txt").write_text("\n".join(KAGGLE_OWN + p.get("excludes", [])))
    (WORK / "deps.in").write_text("\n".join(p["deps"]))
    pins = WORK / "pins.txt"
    sh([uv(), "pip", "compile", "-q", "--python", sys.executable, "--excludes", WORK / "excludes.txt",
        WORK / "deps.in", "-o", pins])  # git deps come out as `pkg @ git+url@<commit>`: pinned to what's built now
    # Kaggle's image is a pre-existing non-uv env: pip wheel is the one command that downloads wheels AND builds
    # the sdist-only / git ones into wheels.
    sh([sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "-w", out, "-r", pins])
    # pins.txt names git deps by URL; after pip wheel they are plain wheels in the folder, so install by name
    names = []
    for line in pins.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line.split(" @ ")[0] if " @ " in line else line)
    (out / "requirements.txt").write_text("\n".join(names) + "\n")
    src = {}
    for repo, url in p.get("repos", {}).items():
        d = WORK / "repos" / repo
        sh(["git", "clone", "-q", url, d])
        src[repo] = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        (out / "src").mkdir(exist_ok=True)
        shutil.copytree(d, out / "src" / repo, ignore=shutil.ignore_patterns(".git"))  # a plain folder: Kaggle unpacks nested .tar.gz anyway
    (out / "SOURCES.txt").write_text(json.dumps(src, indent=2) + "\n")
    (out / f"PAIR_{name}_WHEELS").write_text(f"Python {sys.version.split()[0]}\n{json.dumps(src)}\n")
    print(f"{name}: {len(names)} wheels, python {sys.version.split()[0]}, sources {src}", flush=True)


def seed_others(names: list[str], root: Path):
    """A new dataset version replaces the old one, so carry the other pairs over from the attached current version."""
    for m in Path("/kaggle/input").glob("datasets/*/*/*/PAIR_*_WHEELS"):
        d = m.parent
        if d.name not in names and not (root / d.name).exists():
            shutil.copytree(d, root / d.name)
            print(f"kept pair {d.name}", flush=True)


def main():
    names = list(PAIRS)
    if not names:
        sys.exit(__doc__)
    upload = "--no-upload" not in sys.argv
    KEEP_OUT.mkdir(exist_ok=True)
    sys.stdout = sys.stderr = _Tee(sys.__stdout__, open(KEEP_OUT / "setup.log", "w"))  # tracebacks too
    print("python", sys.version, flush=True)
    subprocess.run([uv(), "pip", "install", "--system", "-q", "kagglehub", "huggingface_hub"], check=True)
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    subprocess.run(f"df -h {WORK} {KEEP_OUT}", shell=True)
    import kagglehub
    me = kagglehub.whoami(verbose=False)["username"] if upload else "local"  # authenticates by itself inside a notebook

    root = WORK / DATASET
    root.mkdir()
    shutil.copy2(uv(), root / "uv")  # runs work with Internet off
    for name in names:
        p = PAIRS[name]
        build_wheels(name, p, root / name)
        stamp(f"{name}: wheels built, {sum(f.stat().st_size for f in (root / name).rglob('*') if f.is_file())/2**30:.2f} GiB")
        if p.get("hf") and "--no-models" not in sys.argv:
            model = WORK / p["model_slug"]
            build_model(p, model)
            (model / f"PAIR_{name}_MODEL").write_text(name)
            stamp(f"{name}: model built, {sum(f.stat().st_size for f in model.rglob('*') if f.is_file())/2**30:.2f} GiB")
            if upload:
                h = f"{me}/{p['model_slug']}/pyTorch/default"
                kagglehub.model_upload(h, str(model), license_name=p.get("license", "Other"), version_notes=f"pair {name}")
                stamp(f"uploaded model {h}")
            shutil.rmtree(model)  # free disk for the next model
    if upload:
        seed_others(names, root)
        h = f"{me}/{DATASET}"
        kagglehub.dataset_upload(h, str(root), version_notes=f"pairs {names}, python {sys.version.split()[0]}")
        stamp(f"uploaded dataset {h}: {sorted(d.name for d in root.iterdir() if d.is_dir())}")
    stamp("ALL DONE")


if __name__ == "__main__":
    main()
