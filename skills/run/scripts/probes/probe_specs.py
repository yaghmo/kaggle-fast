"""What does a Kaggle notebook give you today? Push once on CPU and once with --gpu (the GPU one costs about a minute of
quota):  kaggle_push.py run probes/probe_specs.py specs --no-wheels [--gpu]
Put the numbers into references/kaggle-facts.md and the limits at the top of hf_fit.py when they have moved."""
import os, subprocess, sys
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
print("python", sys.version.split()[0], "| cores", os.cpu_count(), "| RAM GiB", round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1))
print(sh("mkdir -p /kaggle/tmp; df -h /kaggle/tmp /kaggle/working /kaggle/input 2>&1 | cut -c1-110"))  # /kaggle/tmp is scratch, created on demand
print("GPU:", sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1") or "none")
print(sh("pip list 2>/dev/null | grep -iE '^(torch|torchvision|torchaudio|torchcodec|numpy|transformers|protobuf|tensorflow|onnx|huggingface.hub) '"))
print("google registered at start:", "google" in sys.modules, getattr(sys.modules.get("google"), "__path__", None))
print("ALL DONE")
