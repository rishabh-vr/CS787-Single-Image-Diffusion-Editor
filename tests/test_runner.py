"""CPU tests for the resumable job runner (uses the dummy method; no GPU needed)."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from sidx.runner import config_hash, load_jobs, run_jobs, seed_png


@pytest.fixture()
def tiny_image(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    p = tmp_path / "img.png"
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(p)
    return p


def _job_file(tmp_path: Path, image: Path, count: int = 5, sleep_s: float = 0.0, block: int = 16) -> Path:
    spec = {
        "experiment": "t",
        "jobs": [{
            "name": "j", "method": "dummy_shuffle", "params": {"block": block, "sleep_s": sleep_s},
            "images": [{"id": "a", "path": str(image)}, {"id": "b", "path": str(image)}],
            "seeds": {"start": 0, "count": count},
        }],
    }
    f = tmp_path / "jobs.yaml"
    f.write_text(yaml.safe_dump(spec))
    return f


def _outputs(out: Path, exp: str, job) -> list:
    d = out / exp / f"{job.name}__{job.cfg_hash}"
    return sorted(p for p in d.rglob("seed_*.png"))


def test_runs_everything_and_layout(tmp_path, tiny_image):
    exp, jobs = load_jobs(_job_file(tmp_path, tiny_image, count=3), {})
    s = run_jobs(exp, jobs, tmp_path / "out", verbose=False)
    assert (s.done, s.remaining, s.failed) == (6, 0, 0)
    job_dir = tmp_path / "out" / exp / f"j__{jobs[0].cfg_hash}"
    assert (job_dir / "job.json").exists()
    for im in ("a", "b"):
        assert (job_dir / im / "setup.json").exists()
        assert len((job_dir / im / "samples.jsonl").read_text().strip().splitlines()) == 3
    assert not list(job_dir.rglob("*.tmp"))


def test_interrupt_and_resume_does_not_redo_work(tmp_path, tiny_image):
    exp, jobs = load_jobs(_job_file(tmp_path, tiny_image, count=5), {})
    out = tmp_path / "out"
    s1 = run_jobs(exp, jobs, out, max_samples=4, verbose=False)        # "session 1" cut short
    assert s1.done == 4 and s1.remaining == 6 and s1.stopped_reason == "max_samples"
    first = {p: p.stat().st_mtime_ns for p in _outputs(out, exp, jobs[0])}
    s2 = run_jobs(exp, jobs, out, verbose=False)                        # "session 2" resumes
    assert s2.done == 6 and s2.skipped_existing == 4 and s2.remaining == 0
    assert all(p.stat().st_mtime_ns == t for p, t in first.items())    # untouched
    assert len(_outputs(out, exp, jobs[0])) == 10


def test_time_budget_stops_cleanly(tmp_path, tiny_image):
    exp, jobs = load_jobs(_job_file(tmp_path, tiny_image, count=20, sleep_s=0.05), {})
    s = run_jobs(exp, jobs, tmp_path / "out", budget_min=0.4 / 60, verbose=False)  # 0.4 s budget
    assert s.stopped_reason == "time_budget"
    assert 0 < s.done < 40 and s.done + s.remaining == 40


def test_samples_are_deterministic_per_seed(tmp_path, tiny_image):
    exp, jobs = load_jobs(_job_file(tmp_path, tiny_image, count=2), {})
    run_jobs(exp, jobs, tmp_path / "o1", verbose=False)
    run_jobs(exp, jobs, tmp_path / "o2", verbose=False)
    for p1, p2 in zip(_outputs(tmp_path / "o1", exp, jobs[0]), _outputs(tmp_path / "o2", exp, jobs[0])):
        assert np.array_equal(np.asarray(Image.open(p1)), np.asarray(Image.open(p2)))


def test_hash_changes_with_params_and_config_content(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("a: 1\n")
    h1 = config_hash("m", {"config": str(cfg), "overrides": []})
    assert h1 == config_hash("m", {"config": str(cfg), "overrides": []})
    assert h1 != config_hash("m", {"config": str(cfg), "overrides": ["x=1"]})
    cfg.write_text("a: 2\n")
    assert h1 != config_hash("m", {"config": str(cfg), "overrides": []})


def test_failed_setup_is_logged_and_skipped(tmp_path, tiny_image):
    spec = {"experiment": "t", "jobs": [{
        "name": "j", "method": "dummy_shuffle", "params": {},
        "images": [{"id": "missing", "path": str(tmp_path / "nope.png")}, {"id": "ok", "path": str(tiny_image)}],
        "seeds": [0, 1]}]}
    f = tmp_path / "jobs.yaml"
    f.write_text(yaml.safe_dump(spec))
    exp, jobs = load_jobs(f, {})
    s = run_jobs(exp, jobs, tmp_path / "out", verbose=False)
    assert s.failed == 2 and s.done == 2
    job_dir = tmp_path / "out" / exp / f"j__{jobs[0].cfg_hash}"
    assert (job_dir / "missing" / "errors.jsonl").exists()
    assert seed_png(job_dir / "ok", 1).exists()
