"""Resumable, time-budgeted job queue — the workhorse for free Colab sessions.

A *job file* (YAML) lists jobs; each job is (method + params) × images × seeds. The runner walks
every (job, image, seed), **skips samples whose PNG already exists**, and stops cleanly before the
time budget runs out. Rerunning the same command after a disconnect continues where it stopped.

Output layout (identical for every method)::

    <out_root>/<experiment>/<job_name>__<cfg_hash>/
        job.json                     method description + environment of the first session
        <image_id>/
            setup.json               per-image setup facts (sizes, #patches, setup time, VRAM)
            seed_0010.png ...        one PNG per seed, written atomically
            samples.jsonl            one record per sample: seed, seconds, peak VRAM, session env id
            errors.jsonl             failures (e.g. CUDA OOM), if any

``cfg_hash`` covers the method name, its params *and the content of any referenced config file*,
so changing a setting can never silently reuse old outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from sidx.methods.base import make_method
from sidx.utils.runtime import (GpuTimer, append_jsonl, atomic_write_json, environment_info,
                                save_png_atomic)


# ---------------------------------------------------------------------------
# Job specification
# ---------------------------------------------------------------------------

@dataclass
class ImageSpec:
    id: str
    path: str


@dataclass
class JobSpec:
    name: str
    method: str
    params: Dict[str, Any]
    images: List[ImageSpec]
    seeds: List[int]
    cfg_hash: str = ""


def _fmt(value: Any, variables: Dict[str, str]) -> Any:
    """Substitute {VAR} placeholders in strings (recursively in lists/dicts)."""
    if isinstance(value, str):
        out = value
        for k, v in variables.items():
            out = out.replace("{" + k + "}", str(v))
        return out
    if isinstance(value, list):
        return [_fmt(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: _fmt(v, variables) for k, v in value.items()}
    return value


def _seed_list(spec: Any) -> List[int]:
    if isinstance(spec, list):
        return [int(s) for s in spec]
    if isinstance(spec, dict):
        return list(range(int(spec.get("start", 0)), int(spec.get("start", 0)) + int(spec["count"])))
    raise ValueError(f"seeds must be a list or {{start, count}}, got {spec!r}")


def _read_manifest(path: Path) -> List[ImageSpec]:
    """CSV with at least columns ``id`` and ``path`` (relative paths are relative to the CSV)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = Path(row["path"])
            if not p.is_absolute():
                p = path.parent / p
            rows.append(ImageSpec(id=row["id"], path=str(p)))
    return rows


def config_hash(method: str, params: Dict[str, Any], sid_root: Optional[str] = None) -> str:
    """Stable 8-hex hash of everything that determines the outputs."""
    payload: Dict[str, Any] = {"method": method, "params": params}
    cfg = params.get("config")
    if isinstance(cfg, str):
        p = Path(cfg)
        if not p.is_absolute() and (params.get("sid_root") or sid_root):
            p = Path(params.get("sid_root") or sid_root) / p
        if p.exists():
            payload["config_file_sha1"] = hashlib.sha1(p.read_bytes()).hexdigest()
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]


def load_jobs(job_file: Path, variables: Dict[str, str]) -> tuple[str, List[JobSpec]]:
    raw = yaml.safe_load(Path(job_file).read_text(encoding="utf-8"))
    raw = _fmt(raw, variables)
    experiment = raw["experiment"]
    defaults = raw.get("defaults", {}) or {}
    jobs = []
    for j in raw["jobs"]:
        merged = {**defaults, **j}
        images: List[ImageSpec] = [ImageSpec(id=i["id"], path=i["path"]) for i in merged.get("images", [])]
        if merged.get("images_manifest"):
            images += _read_manifest(Path(merged["images_manifest"]))
        if not images:
            raise ValueError(f"job {merged['name']!r} has no images")
        params = dict(merged.get("params", {}) or {})
        job = JobSpec(name=merged["name"], method=merged["method"], params=params, images=images,
                      seeds=_seed_list(merged["seeds"]))
        job.cfg_hash = config_hash(job.method, job.params, variables.get("SID_ROOT"))
        jobs.append(job)
    return experiment, jobs


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    done: int = 0
    skipped_existing: int = 0
    failed: int = 0
    remaining: int = 0
    stopped_reason: str = "completed"
    elapsed_s: float = 0.0
    per_job: Dict[str, Dict[str, int]] = field(default_factory=dict)


def seed_png(image_dir: Path, seed: int) -> Path:
    return image_dir / f"seed_{seed:04d}.png"


def run_jobs(experiment: str, jobs: Iterable[JobSpec], out_root: Path, budget_min: Optional[float] = None,
             max_samples: Optional[int] = None, repo_paths: Optional[Dict[str, str]] = None,
             safety_factor: float = 1.25, verbose: bool = True) -> RunSummary:
    """Run every missing (job, image, seed) until done, the time budget, or ``max_samples``.

    The budget check happens before starting each sample: if the previous sample's duration
    (× ``safety_factor``) would overrun the budget, stop now rather than get cut off mid-write.
    """
    t0 = time.perf_counter()
    budget_s = None if budget_min is None else budget_min * 60.0
    env = environment_info(repo_paths)
    session_id = env["timestamp"]
    summary = RunSummary()
    log = print if verbose else (lambda *a, **k: None)
    stop = False
    jobs = list(jobs)

    def out_of_time(next_cost_s: float) -> bool:
        return budget_s is not None and (time.perf_counter() - t0) + next_cost_s * safety_factor > budget_s

    for job in jobs:
        job_dir = Path(out_root) / experiment / f"{job.name}__{job.cfg_hash}"
        counts = summary.per_job.setdefault(job.name, {"done": 0, "skipped": 0, "failed": 0, "remaining": 0})
        if not (job_dir / "job.json").exists():
            atomic_write_json(job_dir / "job.json", {
                "experiment": experiment, "job": job.name, "method": job.method, "params": job.params,
                "cfg_hash": job.cfg_hash, "seeds": job.seeds, "images": [i.__dict__ for i in job.images],
                "first_session_env": env,
            })

        method = None
        for image in job.images:
            image_dir = job_dir / image.id
            todo = [s for s in job.seeds if not seed_png(image_dir, s).exists()]
            n_existing = len(job.seeds) - len(todo)
            summary.skipped_existing += n_existing
            counts["skipped"] += n_existing
            if not todo:
                continue
            if stop or out_of_time(0.0):
                stop = True
                summary.remaining += len(todo)
                counts["remaining"] += len(todo)
                continue

            # --- per-image setup -------------------------------------------------------
            try:
                method = make_method(job.method, **job.params)
                setup_info = method.setup(image.path)
                if not (image_dir / "setup.json").exists():
                    atomic_write_json(image_dir / "setup.json", {
                        "image": image.__dict__, "setup": setup_info, "method": method.describe(), "env": env})
            except Exception as e:  # e.g. CUDA OOM while building denoisers
                summary.failed += len(todo)
                counts["failed"] += len(todo)
                append_jsonl(image_dir / "errors.jsonl", {"stage": "setup", "error": repr(e),
                                                          "traceback": traceback.format_exc(),
                                                          "session": session_id})
                log(f"[runner] SETUP FAILED {job.name}/{image.id}: {e!r}")
                if method is not None:
                    method.teardown()
                continue

            last_cost = 0.0
            for k, seed in enumerate(todo):
                if (max_samples is not None and summary.done >= max_samples) or out_of_time(last_cost):
                    stop = True
                    summary.remaining += len(todo) - k
                    counts["remaining"] += len(todo) - k
                    break
                timer = GpuTimer()
                try:
                    with timer.measure():
                        img = method.sample(seed)
                    save_png_atomic(img, seed_png(image_dir, seed))
                except Exception as e:
                    summary.failed += 1
                    counts["failed"] += 1
                    append_jsonl(image_dir / "errors.jsonl", {"stage": "sample", "seed": seed, "error": repr(e),
                                                              "traceback": traceback.format_exc(),
                                                              "session": session_id})
                    log(f"[runner] SAMPLE FAILED {job.name}/{image.id}/seed {seed}: {e!r}")
                    continue
                last_cost = timer.seconds
                append_jsonl(image_dir / "samples.jsonl", {
                    "seed": seed, "seconds": round(timer.seconds, 4), "peak_vram_gb": timer.peak_vram_gb,
                    "session": session_id, "gpu": env.get("gpu_name"), "torch": env["torch"]})
                summary.done += 1
                counts["done"] += 1
                log(f"[runner] {job.name}/{image.id} seed {seed}: {timer.seconds:.2f}s"
                    + (f", peak {timer.peak_vram_gb} GB" if timer.peak_vram_gb is not None else ""))
            method.teardown()
            method = None

    summary.elapsed_s = round(time.perf_counter() - t0, 2)
    if stop:
        summary.stopped_reason = "max_samples" if (max_samples is not None and summary.done >= max_samples) \
            else "time_budget"
    log(f"[runner] done={summary.done} skipped(existing)={summary.skipped_existing} failed={summary.failed} "
        f"remaining={summary.remaining} stop={summary.stopped_reason} elapsed={summary.elapsed_s}s")
    if summary.remaining:
        log("[runner] Not finished - rerun the same command to resume.")
    return summary
