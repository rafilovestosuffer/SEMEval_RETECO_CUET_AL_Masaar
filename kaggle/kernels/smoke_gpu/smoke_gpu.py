"""Kaggle GPU smoke test — proves the local -> Kaggle control path works.

Runs in well under a minute, so it costs essentially nothing against the ~30 GPU-hrs/week
quota. It answers four questions before any real job depends on them:

  1. Does the kernel get a GPU at all, and which one (T4 vs P100)?
  2. Does the preinstalled torch see CUDA, and does a tensor op actually execute on it?
  3. How much GPU memory, host RAM and working disk does this session really have?
  4. Are attached datasets visible under /kaggle/input?

Everything is printed to the console log and also written to
/kaggle/working/smoke_gpu_result.json, so `pull_output.py` brings both back.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def run(cmd: list[str]) -> str:
    """Run a command, returning its output or a readable failure string."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (proc.stdout or proc.stderr or "").strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<failed: {exc}>"


def main() -> int:
    result: dict[str, object] = {}

    section("1. Host")
    result["python"] = sys.version.split()[0]
    result["platform"] = platform.platform()
    result["cpu_count"] = os.cpu_count()
    for key in ("python", "platform", "cpu_count"):
        print(f"{key:20s}: {result[key]}")

    total, used, free = shutil.disk_usage(WORKING if WORKING.is_dir() else Path("/"))
    result["working_disk_free_gb"] = round(free / 1024**3, 1)
    print(f"{'working disk free':20s}: {result['working_disk_free_gb']} GB")

    try:
        meminfo = Path("/proc/meminfo").read_text()
        total_kb = int(next(line for line in meminfo.splitlines() if line.startswith("MemTotal")).split()[1])
        result["host_ram_gb"] = round(total_kb / 1024**2, 1)
        print(f"{'host RAM':20s}: {result['host_ram_gb']} GB")
    except (OSError, StopIteration, ValueError):
        result["host_ram_gb"] = None

    section("2. nvidia-smi")
    smi = run(["nvidia-smi"])
    print(smi)
    result["nvidia_smi_ok"] = "NVIDIA-SMI" in smi

    section("3. torch / CUDA")
    try:
        import torch
    except ImportError as exc:
        print(f"torch not importable: {exc}")
        result["torch"] = None
        result["cuda_available"] = False
    else:
        result["torch"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        print(f"{'torch':20s}: {torch.__version__}")
        print(f"{'cuda available':20s}: {result['cuda_available']}")
        if torch.cuda.is_available():
            result["cuda_version"] = torch.version.cuda
            result["device_count"] = torch.cuda.device_count()
            result["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            result["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
            result["capability"] = f"{props.major}.{props.minor}"
            for key in ("cuda_version", "device_count", "device_name", "gpu_memory_gb", "capability"):
                print(f"{key:20s}: {result[key]}")

            # bf16 needs compute capability >= 8.0; a T4 is 7.5, a P100 is 6.0.
            # This decides whether later embedding jobs use bf16 or fall back to fp16.
            result["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            print(f"{'bf16 supported':20s}: {result['bf16_supported']}")

            # Actually execute on the device — is_available() alone has lied before.
            a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
            matmul = (a @ a).sum().item()
            result["matmul_executed"] = True
            result["matmul_finite"] = bool(matmul == matmul)  # NaN-safe check
            print(f"{'fp16 matmul':20s}: executed, sum={matmul:.4g}")

    section("4. Attached datasets (/kaggle/input)")
    if INPUT.is_dir():
        entries = sorted(p.name for p in INPUT.iterdir())
        result["input_datasets"] = entries
        print("\n".join(f"  {e}" for e in entries) if entries else "  (none attached)")
    else:
        result["input_datasets"] = []
        print("  /kaggle/input does not exist")

    section("5. Result")
    result["verdict"] = "GPU OK" if result.get("cuda_available") else "NO GPU"
    print(json.dumps(result, indent=2))

    out = WORKING / "smoke_gpu_result.json" if WORKING.is_dir() else Path("smoke_gpu_result.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")

    return 0 if result.get("cuda_available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
