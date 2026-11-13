"""Seeding, environment capture, GPU-aware timing, and atomic file writes."""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
from PIL import Image


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU + all GPUs)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit(path: Optional[str]) -> Optional[str]:
    if not path or not Path(path).exists():
        return None
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def environment_info(repo_paths: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Everything needed to say *where* a number came from: versions, GPU, commits."""
    import torch

    info: Dict[str, Any] = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(gpu_name=props.name,
                    gpu_total_mem_gb=round(props.total_memory / 1024 ** 3, 2),
                    gpu_capability=f"{props.major}.{props.minor}")
    for name, path in (repo_paths or {}).items():
        info[f"git_{name}"] = _git_commit(path)
    return info


class GpuTimer:
    """Wall-clock timer that synchronizes CUDA so GPU work is actually counted, plus peak VRAM."""

    def __init__(self) -> None:
        self.seconds: float = float("nan")
        self.peak_vram_gb: Optional[float] = None

    @contextmanager
    def measure(self) -> Iterator["GpuTimer"]:
        import torch

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            if cuda:
                torch.cuda.synchronize()
                self.peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
            self.seconds = time.perf_counter() - t0


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via a temp file + rename, so an interrupted session never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, default=str).encode("utf-8"))


def save_png_atomic(img_chw01, path: Path) -> None:
    """Save a (C, H, W) tensor/array in [0, 1] as an 8-bit PNG, atomically."""
    import io

    arr = img_chw01.detach().cpu().float().numpy() if hasattr(img_chw01, "detach") else np.asarray(img_chw01)
    arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 0)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    atomic_write_bytes(Path(path), buf.getvalue())


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
