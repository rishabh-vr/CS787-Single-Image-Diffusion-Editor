"""Run a list of shell tasks on all GPUs at once (Kaggle gives two T4s), with live progress bars.

Each *slot* runs one task at a time as a subprocess with ``CUDA_VISIBLE_DEVICES`` set to slot % n_gpus,
its output going to ``<log_dir>/<label>.log``. When a task ends, the slot takes the next one from the queue
(so fast and slow tasks balance out). A user-supplied ``poll()`` returns the progress to display, e.g. the
number of finished sample PNGs on disk, so the bars show real work done, independent of the task logs.

``job_tasks`` turns runner job files into such tasks: every job is split into image chunks (so that a single
big job can use both GPUs), and each chunk is one ``scripts/run_queue.py --only <job> --images <ids>`` call.
Everything stays resumable: a stopped task leaves finished PNGs on disk and is simply rerun next session.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from sidx.progress import Bars, tail

Argv = Union[List[str], Callable[[], List[str]]]
PollResult = Dict[str, Tuple[int, int]]


@dataclass
class Task:
    label: str
    argv: Argv                                   # a list, or a function returning it at launch time
    cwd: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    cost: float = 1.0                            # only used for ordering by the task builders


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        return sum(1 for l in out.splitlines() if l.startswith("GPU "))
    except Exception:
        return 0


def _safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)[:120]


def run_pool(tasks: Sequence[Task], poll: Callable[[], PollResult], *, slots: Optional[int] = None,
             n_gpus: Optional[int] = None, log_dir: Union[str, Path] = "logs", deadline: Optional[float] = None,
             unit: str = "sample", poll_every: float = 10.0, print_every: float = 120.0,
             extra_postfix: Optional[Callable[[], Dict[str, str]]] = None) -> Dict[str, Optional[int]]:
    """Run ``tasks`` on ``slots`` parallel workers; return {label: exit code, or None if never started}.

    deadline: a time.time() value after which no new task is started (running ones stop by their own budget).
    """
    n_gpus = gpu_count() if n_gpus is None else n_gpus
    slots = slots or max(1, n_gpus)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    queue: List[Task] = list(tasks)
    running: Dict[int, Tuple[Task, subprocess.Popen, object, float]] = {}
    results: Dict[str, Optional[int]] = {t.label: None for t in tasks}
    seconds: Dict[str, float] = {}
    bars = Bars(unit=unit, every_s=print_every,
                title=f"[pool] {len(queue)} tasks on {slots} slot(s), {n_gpus} GPU(s); logs in {log_dir}")
    n_finished = 0

    def launch(slot: int, task: Task) -> None:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", **task.env}
        if n_gpus > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(slot % n_gpus)
        argv = task.argv() if callable(task.argv) else task.argv
        log = open(log_dir / f"{_safe(task.label)}.log", "a", encoding="utf-8")
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(map(str, argv))}\n")
        log.flush()
        p = subprocess.Popen([str(a) for a in argv], cwd=task.cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        running[slot] = (task, p, log, time.time())

    def refresh(force: bool = False) -> None:
        extra = extra_postfix() if extra_postfix else {}
        try:
            values = poll()
        except Exception as e:                                   # never let the monitor kill the run
            values = {}
            print(f"[pool] progress poll failed: {e!r}", flush=True)
        for name, (done, total) in values.items():
            bars.set(name, done, total, extra.get(name, ""))
        busy = ", ".join(f"{f'GPU{s % n_gpus}' if n_gpus else f'worker{s}'}: {t.label} ({(time.time() - t0) / 60:.0f} min)"
                         for s, (t, _, _, t0) in sorted(running.items()))
        bars.status(f"tasks {n_finished}/{len(results)} finished | running: {busy or '-'}")
        bars.maybe_print(force)

    try:
        last = 0.0
        while queue or running:
            for slot in range(slots):
                if slot not in running and queue:
                    if deadline is not None and time.time() > deadline:
                        break
                    launch(slot, queue.pop(0))
            if not running:                                       # deadline passed with tasks left
                break
            for slot, (task, p, log, t0) in list(running.items()):
                rc = p.poll()
                if rc is None:
                    continue
                log.close()
                del running[slot]
                results[task.label] = rc
                seconds[task.label] = round(time.time() - t0, 1)
                n_finished += 1
                if rc != 0:
                    print(f"\n[pool] FAILED (exit {rc}): {task.label}  -- last lines of its log:\n"
                          f"{tail(str(log_dir / (_safe(task.label) + '.log')))}\n", flush=True)
            if time.time() - last >= poll_every:
                refresh()
                last = time.time()
            time.sleep(2)
    except KeyboardInterrupt:
        print("[pool] interrupted: stopping running tasks", flush=True)
        for _, p, log, _ in running.values():
            p.terminate()
        for _, p, log, _ in running.values():
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
            log.close()
        raise
    finally:
        refresh(force=True)
        bars.close()
    summary_file = log_dir / "_summary.json"
    old = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    old.update({k: {"rc": v, "seconds": seconds.get(k)} for k, v in results.items() if v is not None})
    summary_file.write_text(json.dumps(old, indent=1))
    failed = [k for k, v in results.items() if v not in (0, None)]
    not_started = [k for k, v in results.items() if v is None]
    print(f"[pool] finished: {sum(v == 0 for v in results.values())} ok, {len(failed)} failed, "
          f"{len(not_started)} not started" + (f" (failed: {failed})" if failed else ""), flush=True)
    return results


# --------------------------------------------------------------------------------------------------
# Runner job files -> tasks
# --------------------------------------------------------------------------------------------------

@contextmanager
def _cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _image_cost(path: str) -> float:
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        return (w * h) ** 1.5                                     # dense patch matching ~ (#patches)^1.5..2
    except Exception:
        return (256 * 256) ** 1.5


def _steps_factor(params: dict) -> float:
    m = re.search(r"num_steps\W+(\d+)", json.dumps(params))
    return int(m.group(1)) / 10.0 if m else 1.0


@dataclass
class _Plan:
    experiment: str
    job_dir: Path
    image_id: str
    seeds: List[int]


def job_tasks(job_files: Sequence[str], out_root: str, project_root: str, variables: Dict[str, str],
              n_gpus: int, deadline: Optional[float] = None, chunks_target: Optional[int] = None,
              python: str = sys.executable) -> Tuple[List[Task], Callable[[], PollResult]]:
    """Tasks for every pending (job, image chunk) of the job files, in file order, biggest first within a
    file, plus a ``poll()`` that counts finished samples per experiment on disk."""
    from sidx.runner import load_jobs, seed_png

    out_root, project_root = Path(out_root), Path(project_root)
    chunks_target = chunks_target or 4 * max(1, n_gpus)
    tasks: List[Task] = []
    plans: List[_Plan] = []
    var_args = [a for k, v in variables.items() if k != "PROJECT_ROOT" for a in ("--var", f"{k}={v}")]
    for jf in job_files:
        with _cwd(project_root):                                  # manifest paths are relative to the repo
            experiment, jobs = load_jobs(project_root / jf, {"PROJECT_ROOT": str(project_root), **variables})
        file_tasks = []
        per_job = max(1, math.ceil(chunks_target / max(1, len(jobs))))
        for job in jobs:
            job_dir = out_root / experiment / f"{job.name}__{job.cfg_hash}"
            sf = _steps_factor(job.params)
            pend = []
            for im in job.images:
                plans.append(_Plan(experiment, job_dir, im.id, job.seeds))
                n = sum(1 for s in job.seeds if not seed_png(job_dir / im.id, s).exists())
                if n:
                    path = im.path if os.path.isabs(im.path) else str(project_root / im.path)
                    pend.append((n * sf * _image_cost(path), im.id))
            if not pend:
                continue
            k = min(per_job, len(pend))
            bins: List[List] = [[0.0, []] for _ in range(k)]
            for cost, iid in sorted(pend, reverse=True):          # greedy balance of the chunks
                b = min(bins, key=lambda x: x[0])
                b[0] += cost
                b[1].append(iid)
            for i, (cost, ids) in enumerate(bins):
                def argv(jf=jf, job=job.name, ids=tuple(ids)):
                    a = [python, str(project_root / "scripts" / "run_queue.py"), str(project_root / jf),
                         "--out-root", str(out_root), "--only", job, "--images", ",".join(ids), "--no-progress",
                         *var_args]
                    if deadline is not None:
                        a += ["--budget-min", f"{max(1.0, (deadline - time.time()) / 60):.1f}"]
                    return a
                label = f"{experiment}:{job.name}" + (f"[{i + 1}/{k}]" if k > 1 else "")
                file_tasks.append(Task(label, argv, cwd=str(project_root), cost=cost))
        tasks += sorted(file_tasks, key=lambda t: -t.cost)

    def poll() -> PollResult:
        res: Dict[str, List[int]] = {}
        for p in plans:
            d = p.job_dir / p.image_id
            have = set(os.listdir(d)) if d.is_dir() else set()
            done = sum(1 for s in p.seeds if f"seed_{s:04d}.png" in have)
            r = res.setdefault(p.experiment, [0, 0])
            r[0] += done
            r[1] += len(p.seeds)
        return {k: (v[0], v[1]) for k, v in res.items()}

    return tasks, poll
