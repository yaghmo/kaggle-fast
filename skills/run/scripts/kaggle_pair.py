"""Load a model pair (built by kaggle_setup.py) inside a Kaggle notebook. Attach the Model and the `wheels` dataset.
Internet can stay off.

    warm(big_file)                    # optional: any other large input, first thing in the run
    p = load("mymodel")               # venv with the pair's wheels; p.model = weights folder, already being read into RAM
    subprocess.run([p.python, "-m", "scripts.inference", ...], cwd=p.source("TheRepo"))   # run in the pair's venv
    p.activate()                      # or: use the venv in this process, before importing torch & co

Every wheel is installed with uv into a venv of its own (`--system-site-packages`: Kaggle's torch/CUDA stay visible, the
venv's pins win over the preinstalled ones). Kaggle's own site-packages is never modified.

Notebooks pushed through the API are single files: kaggle_push.py pastes this file in front of the run script.
"""
import contextlib, os, shutil, subprocess, sys, threading, zipfile
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

INPUT, TMP = Path("/kaggle/input"), Path("/kaggle/tmp")


def _find(name: str, kind: str):
    """Directory holding PAIR_<name>_<kind>; falls back to unzipping archive.zip (kagglehub zips big uploads)."""
    marker = f"PAIR_{name}_{kind}"
    # Known mount layouts first, lazily: a recursive walk of /kaggle/input took 144 s with one large input attached
    for m in chain(INPUT.glob(f"datasets/*/*/{name}/{marker}"), INPUT.glob(f"models/*/*/*/*/*/{marker}"), INPUT.rglob(marker)):
        return m.parent
    for z in INPUT.rglob("archive.zip"):
        with zipfile.ZipFile(z) as f:
            hit = next((n for n in f.namelist() if n.endswith(marker)), None)
            if hit:
                prefix = hit[: -len(marker)]  # "<name>/" for the wheels dataset, "" for a model
                dest = TMP / f"zip_{kind}_{name}"
                f.extractall(dest, [n for n in f.namelist() if n.startswith(prefix) or n == "uv"])
                return dest / prefix
    sys.exit(f"{marker} not found under /kaggle/input: attach the {kind.lower()} for pair '{name}'.")


def warm(*paths, threads: int = 16) -> threading.Thread:
    """Read files / folders into the page cache in the background; returns at once. Measured on Kaggle's input mount:
    one reader gets 120-240 MB/s, 16 get 770-950 MB/s, and torch.load reads with one. Call it first, so the weights are in
    RAM by the time the venv is built and the imports are done. load() already does it for the pair's Model."""
    step = 64 * 2**20
    def parts():
        for p in map(Path, paths):
            for f in [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file()):
                for o in range(0, f.stat().st_size, step):
                    yield f, o
    def read(part):
        fd = os.open(part[0], os.O_RDONLY)
        try:
            os.pread(fd, step, part[1])
        finally:
            os.close(fd)
    def run():
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(read, parts()))
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


@contextlib.contextmanager
def no_init():
    """Random weight init does nothing inside this block. Wrap the model constructor only, and only when every weight it
    initialises is then loaded from a checkpoint (prove it: probes/probe_build.py compares the two models).
    Measured: a 7 GiB model's constructor 47.2 -> 1.2 s. Patched on torch.Tensor, not on torch.nn.init:
    transformers' from_pretrained puts the torch.nn.init originals back, which undoes such a patch mid-constructor."""
    import torch
    names = ("uniform_", "normal_", "erfinv_")
    assert not any(n in vars(torch.Tensor) for n in names)  # inherited from TensorBase: delattr below restores them
    for n in names:
        setattr(torch.Tensor, n, lambda x, *a, **k: x)
    try:
        yield
    finally:
        for n in names:
            delattr(torch.Tensor, n)


@contextlib.contextmanager
def no_tensorflow():
    """`import tensorflow` fails at once inside this block. Put the run's imports in it when the model does not use
    tensorflow: Kaggle's copy (1.2 GiB of libraries) gets pulled in through side doors such as torch.utils.tensorboard.
    Measured in two fresh sessions, same imports: 49.4 s as is, 38.8 s blocked.
    Only around the imports, never for the whole run: einops (and others) treat a 'tensorflow' key in sys.modules as
    "tensorflow is loaded" and then import it for real, which is how a permanent block broke inference here."""
    blocked = "tensorflow" not in sys.modules
    if blocked:
        sys.modules["tensorflow"] = None
    try:
        yield
    finally:
        if blocked and sys.modules.get("tensorflow", 0) is None:
            del sys.modules["tensorflow"]


def no_init_patch(file, line: str):
    """no_init() for a constructor inside the model's own code that runs in another process (a server, a CLI): rewrites
    `line`, the statement that builds the model as it stands in `file` (a writable checkout), to run with random init off.
    Same condition as no_init(): every weight it initialises is then loaded from a checkpoint."""
    p = Path(file)
    s = p.read_text()
    hit = [l for l in s.splitlines() if l.strip() == line.strip()]
    assert len(hit) == 1, f"{line!r} found {len(hit)} times in {file}"
    pad = hit[0][: len(hit[0]) - len(hit[0].lstrip())]
    new = [  # torch.Tensor, not torch.nn.init: see no_init()
        "import torch as _t", "_names = ('uniform_', 'normal_', 'erfinv_')",
        "for _n in _names: setattr(_t.Tensor, _n, lambda x, *a, **k: x)",
        "try:", "    " + line.strip(), "finally:", "    for _n in _names: delattr(_t.Tensor, _n)"]
    p.write_text(s.replace(hit[0] + chr(10), chr(10).join(pad + x for x in new) + chr(10)))


def load(name: str, need_model: bool = True) -> SimpleNamespace:
    """need_model=False for a wheels-only pair (the weights are some other attached Model: warm() them yourself)."""
    TMP.mkdir(parents=True, exist_ok=True)
    # albumentations (pulled in by insightface) looks for a newer release at import and, with Internet off, waits for the
    # timeout. Measured: one model's imports 72.9 -> 10.0 s. Set here so the pair's subprocesses inherit it.
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    # Kaggle's tensorflow needs Kaggle's protobuf; a pair that brings its own makes `import transformers` fail when
    # transformers tries tensorflow. No pair uses Kaggle's tensorflow, so tell transformers not to look.
    os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("USE_FLAX", "0")
    model = _find(name, "MODEL") if need_model else None
    if model:
        warm(model)
    src = wheels = _find(name, "WHEELS")
    if (src / "WHEELS_FROM").exists():  # spec key wheels_from: this pair brings its source and runs on another pair's wheels
        wheels = _find((src / "WHEELS_FROM").read_text().strip(), "WHEELS")
    uv = TMP / "uv"
    shutil.copy(next(p for p in (wheels / "uv", wheels.parent / "uv") if p.exists()), uv)  # mounts are read-only
    uv.chmod(0o755)
    venv = TMP / f"venv_{wheels.name}"
    py = venv / "bin" / "python"
    if not py.exists():
        subprocess.run([str(uv), "venv", "-q", "--system-site-packages", "--python", sys.executable, str(venv)], check=True)
        # No --compile-bytecode: in a real fresh GPU session it made the install 9 -> 45 s and the imports 53 -> 51 s.
        subprocess.run([str(uv), "pip", "install", "-q", "--python", str(py), "--offline", "--no-index", "--no-deps",
                        "--find-links", str(wheels), "-r", str(wheels / "requirements.txt")], check=True)
    site = next((venv / "lib").glob("python*/site-packages"))
    if (site / "google").is_dir():
        # Kaggle registers its own google/ at interpreter start, so the venv's is never looked at whatever the sys.path
        # order (measured: onnx from the venv got Kaggle's protobuf and refused to load). A .pth in the venv runs first
        # and puts ours at the front of google.__path__.
        (site / "pair_google.pth").write_text("import sys, types; sys.modules.setdefault('google', types.ModuleType('google'))"
                                              f".__dict__.setdefault('__path__', []).insert(0, {str(site / 'google')!r})" + chr(10))
    def source(repo: str) -> Path:
        """Writable copy of the pinned checkout (the input mount is read-only and run scripts usually patch files)."""
        shutil.copytree(src / "src" / repo, TMP / repo, dirs_exist_ok=True)
        return TMP / repo

    def activate():
        sys.path.insert(0, site.as_posix())

    return SimpleNamespace(model=model, wheels=wheels, python=str(py), source=source, activate=activate)
