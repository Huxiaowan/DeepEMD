#!/usr/bin/env python3
"""Report dependencies and CUDA visibility without changing the environment."""
import importlib
import argparse
from pathlib import Path
import platform
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is unavailable or a small GPU operation fails.")
parser.add_argument("--project-imports", action="store_true", help="Also check DeepEMD's Bernoulli, INR and PDF modules.")
args = parser.parse_args()
failed = []
print("Python:", platform.python_version(), sys.executable)
if sys.version_info < (3, 10):
    failed.append("Python>=3.10")
for name in ("torch", "torchvision", "numpy", "PIL", "yaml", "tqdm", "matplotlib", "skimage", "tifffile"):
    try:
        module = importlib.import_module(name)
        print(name + ":", getattr(module, "__version__", "installed"))
    except Exception as exc:
        failed.append(name)
        print(name + ": ERROR", str(exc))
if "torch" not in failed:
    import torch
    if int(torch.__version__.split(".")[0]) < 2:
        failed.append("torch>=2.0")
    print("CUDA available:", torch.cuda.is_available(), "| PyTorch CUDA:", torch.version.cuda)
    for index in range(torch.cuda.device_count()):
        print("GPU", index, torch.cuda.get_device_name(index))
    if args.require_cuda:
        try:
            if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
                raise RuntimeError("CUDA unavailable. Check the NVIDIA driver, PyTorch wheel and CUDA_VISIBLE_DEVICES.")
            sample = torch.ones((8, 8), device="cuda")
            assert (sample @ sample).sum().item() == 512
            torch.cuda.synchronize()
            print("CUDA operation: OK")
        except Exception as exc:
            failed.append("CUDA")
            print("CUDA operation: ERROR", str(exc))
if args.project_imports:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    for name in ("deepemd.engine", "deepemd.inr_engine", "deepemd.completed"):
        try:
            importlib.import_module(name)
            print(name + ": OK")
        except Exception as exc:
            failed.append(name)
            print(name + ": ERROR", str(exc))
if failed:
    print("[error] Checks failed:", ", ".join(failed))
raise SystemExit(1 if failed else 0)
