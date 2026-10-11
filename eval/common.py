"""Shared helpers: secrets, seeding, JSONL I/O, environment fingerprint."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure_python_headers() -> None:
    """Triton (torch.compile, Liger, flash-linear-attention) compiles helpers against Python.h. This machine has
    no python3.12-dev and no sudo, so the headers are unpacked to ~/.local/pydev (README, setup). Point gcc at
    them when the system copy is missing; a no-op on machines that have the headers."""
    import sysconfig

    if (Path(sysconfig.get_paths()["include"]) / "Python.h").exists():
        return
    local = Path.home() / ".local" / "pydev" / "root" / "usr" / "include"
    if (local / f"python{sys.version_info.major}.{sys.version_info.minor}" / "Python.h").exists():
        inc = [str(local / f"python{sys.version_info.major}.{sys.version_info.minor}"), str(local)]
        os.environ["CPATH"] = os.pathsep.join(inc + [p for p in os.environ.get("CPATH", "").split(os.pathsep) if p])


def load_env(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env; map hf_token -> HF_TOKEN for huggingface_hub."""
    ensure_python_headers()
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = (s.strip().strip('"').strip("'") for s in line.split("=", 1))
        if key.lower() in ("hf_token", "hugging_face_hub_token"):
            key = "HF_TOKEN"
        os.environ.setdefault(key, val)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except ImportError:
        pass


def cap_gpu_memory(gb: float | None = None) -> None:
    """Limit PyTorch's CUDA allocator (default 100 GB, env SMMLU_GPU_MEM_GB).

    The DGX Spark shares one 128 GB pool between CPU and GPU: an unbounded GPU job does not get a CUDA
    out-of-memory error, it starves the system and the kernel OOM killer takes down the desktop session and
    tmux (10 Oct, 11:15). With the cap, the job fails with torch.OutOfMemoryError and everything else survives.
    """
    import torch

    if not torch.cuda.is_available():
        return
    gb = float(gb or os.environ.get("SMMLU_GPU_MEM_GB", 100))
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(gb * 1024**3 / total, 1.0), 0)


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def env_fingerprint() -> dict:
    """Versions and code revision, stored with every run for reproducibility."""
    info = {"python": platform.python_version(), "machine": platform.machine()}
    for mod in ("torch", "transformers", "datasets", "peft", "accelerate", "huggingface_hub"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
        info["git_commit"] = rev.stdout.strip() or None
        info["git_dirty"] = bool(dirty.stdout.strip())
    except FileNotFoundError:
        pass
    return info
