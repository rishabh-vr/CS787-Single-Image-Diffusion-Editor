"""Kaggle helpers: carry results from one session to the next, and keep the saved output small.

Kaggle keeps ``/kaggle/working`` only as the *output of a saved version*. To continue in a later session
(or to evaluate), that output is attached as an input (``/kaggle/input/...``, read-only) and merged back:

* ``gather_inputs()`` finds, anywhere under /kaggle/input:
    - ``cs787_*.zip`` archives written by ``pack()`` -> extracted into /kaggle/working;
    - folders named ``outputs`` / ``checkpoints`` / ``tables`` / ``eval`` -> merged into /kaggle/working/<name>;
    - experiment folders (``phase*`` containing ``*/job.json``, e.g. ``phase4_reproduction`` downloaded
      from Google Drive and uploaded as a Kaggle dataset) -> merged into /kaggle/working/outputs/<name>.
* ``pack()`` zips the result folders into one ``cs787_<name>.zip`` (stored, no compression: PNGs are
  already compressed) and removes the loose files, so the version's output is a single download.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, List

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
KINDS = ("outputs", "checkpoints", "tables", "eval")


def on_kaggle() -> bool:
    return INPUT.parent.exists() and Path("/kaggle/working").exists()


def _count_files(p: Path) -> int:
    return sum(len(f) for _, _, f in os.walk(p))


def gather_inputs(work: Path = WORK, input_root: Path = INPUT, kinds: Iterable[str] = KINDS) -> List[str]:
    work, input_root = Path(work), Path(input_root)
    kinds = tuple(kinds)
    merged: List[str] = []
    if not input_root.exists():
        return merged
    for z in sorted(input_root.rglob("cs787_*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(work)
        merged.append(f"{z} -> {work}")
    done_roots: List[Path] = []
    for d in sorted(input_root.rglob("*")):
        if not d.is_dir() or any(r in d.parents for r in done_roots):
            continue
        if d.name in kinds:
            target = work / d.name
        elif d.name.startswith("phase") and any(d.glob("*/job.json")):
            target = work / "outputs" / d.name
        else:
            continue
        shutil.copytree(d, target, dirs_exist_ok=True)
        done_roots.append(d)
        merged.append(f"{d} -> {target} ({_count_files(d)} files)")
    for m in merged:
        print("[resume]", m)
    if not merged:
        print("[resume] nothing attached under /kaggle/input: starting from scratch")
    return merged


def unpack_own(work: Path = WORK) -> None:
    """Interactive reruns: if this session already packed its results, extract them again so a rerun of
    the run cell sees the finished work (the loose folders were removed by ``pack``)."""
    work = Path(work)
    for z in sorted(work.glob("cs787_*.zip")):
        with zipfile.ZipFile(z) as zf:
            missing = [n for n in zf.namelist() if not (work / n).exists()]
            for n in missing:
                zf.extract(n, work)
        if missing:
            print(f"[resume] re-extracted {len(missing)} files from {z.name}")


def pack(name: str, work: Path = WORK, kinds: Iterable[str] = KINDS, remove: bool = True) -> Path:
    """Zip work/<kind> folders into work/cs787_<name>.zip and (by default) delete the loose copies."""
    work = Path(work)
    dst = work / f"cs787_{name}.zip"
    tmp = dst.with_suffix(".zip.tmp")
    n = 0
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for kind in kinds:
            root = work / kind
            if not root.is_dir():
                continue
            for dirpath, _, files in os.walk(root):
                for f in files:
                    if f.endswith(".tmp"):
                        continue
                    p = Path(dirpath) / f
                    zf.write(p, p.relative_to(work).as_posix())
                    n += 1
    os.replace(tmp, dst)
    if remove:
        for kind in kinds:
            shutil.rmtree(work / kind, ignore_errors=True)
    print(f"[pack] {n} files -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst
