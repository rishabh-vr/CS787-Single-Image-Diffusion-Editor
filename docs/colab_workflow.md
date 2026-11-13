# Free-Colab Workflow (Phase 2)

Everything GPU-related runs on **free Google Colab (T4)**. Everything else runs on the laptop (CPU).
This page is the how-to; the reasons behind it are in PROJECT_PLAN.md §5–§6.

## 1. One-time setup

1. **Code on GitHub.** The notebooks clone `https://github.com/rishabh-vr/CS787-training-free-single-image-diffusion` (private). The project files are at the repo root.
   - The `.gitignore` keeps `.venv/`, `external/` clones and outputs out of git.
   - Because the repo is **private**, create a fine-grained, **read-only** GitHub token for that repo (GitHub → Settings → Developer settings → Fine-grained tokens → *Only select repositories* → *Contents: Read-only*). In Colab, add it as a secret named `GH_TOKEN` (key icon in the left sidebar, then enable notebook access).
   - Alternative: upload `Project/` (without `.venv`/`external`) to `MyDrive/CS787/code/Project` and set `CODE_SOURCE = "drive"` in the notebook's first cell.
2. **Google Drive layout** (created automatically): `MyDrive/CS787/{outputs, checkpoints, tables, env_logs, data}`.
3. **Optional:** a free Hugging Face account, with the FLUX.1-schnell terms accepted. This is only needed for the latent-space ablation AB5.

## 2. Phase-2 notebooks (run once each, in this order)

| Notebook | GPU time [est.] | What it settles |
|---|---|---|
| `notebooks/00_setup_smoke.ipynb` | 20–30 min | Official code runs on a T4; D2/D3 precision questions; **T4 timing table** → `tables/phase2_t4_timing.csv`; faiss import |
| `notebooks/05_sinddm_port.ipynb` | 25–35 min | Ported SinDDM samples the 3 pretrained models; **seconds per 1 000 training steps** → `tables/phase2_sinddm_timing.json`; stop-and-resume works |
| `notebooks/06_lora_smoke.ipynb` | 15–20 min | SD 1.5 LoRA fits on a T4; train and sample timings → `tables/phase2_lora_timing.json` |

In Colab: **File → Open notebook → GitHub tab** (or upload the file), then **Runtime → Change runtime type → T4 GPU**, then **Run all**.

Notebooks are generated from `scripts/make_notebooks.py`. Edit that script, then regenerate with `python scripts/make_notebooks.py`.

## 3. Running experiments: the job runner

Every experiment is a YAML **job file** in `configs/jobs/`, run through `scripts/run_queue.py`:

```bash
python scripts/run_queue.py configs/jobs/<file>.yaml --out-root /content/drive/MyDrive/CS787/outputs \
    --budget-min 100 --var SID_ROOT=/content/efficient-SID-code
```

- **Resumable.** A sample whose PNG already exists on Drive is skipped. After a disconnect, rerun the same command.
- **Time budget.** Before each sample, the runner checks whether another sample fits within `--budget-min` (with a 25 % margin). If not, it stops cleanly.
- **`--dry-run`** lists every job, its config hash and the number of pending samples.
- **Changing any setting changes the config hash**, which means a new output folder. Old results are never silently reused.

Output layout:

```
outputs/<experiment>/<job>__<hash>/job.json
                                 /<image_id>/setup.json, seed_0010.png, ..., samples.jsonl, errors.jsonl
```

## 4. Free-tier etiquette

- Keep sessions short (≤ 100-min budgets) and spread the work across days.
- Close the runtime when finished (**Runtime → Disconnect and delete runtime**) so you don't burn quota.
- Each teammate uses **their own** account; split the work by lane (PROJECT_PLAN §13).
- Run metric evaluation on a **CPU runtime** or the laptop, never on GPU quota.

## 5. Laptop (CPU) environment

```powershell
python -m venv --system-site-packages .venv          # reuses the installed CPU torch
.venv\Scripts\python -m pip install -r env\requirements-local.txt
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pytest -q                    # runner tests (6) — all CPU
```

## 6. Known setup pitfalls (found in Phase 2)

| Pitfall | Fix |
|---|---|
| The official `config.py` imports the CLIP app, so **`clip` is required even for unconditional runs** | Install `clip-anytorch==2.6.0` (in `env/requirements-colab.txt`) |
| `clip` does `from pkg_resources import packaging`, which fails with setuptools ≥ 70 (the official env pins only `<80`) | Pin `setuptools<70` |
| Editable installs aren't visible to an already-running kernel | The notebooks also add the paths to `sys.path`; scripts run as subprocesses, which are unaffected |
| Exact fp32 kernel at ~250 px holds ~2 dense N×N tensors (~13 GB), which is borderline on a T4 | Use `image_denoiser.patch_denoiser.query_chunks=2` (4 at 256×256), or `exact_flash_attn` |
| SinDDM's pinned torch 1.13 can't be installed on Colab | Use the port in `patches/sinddm_port/` (copied over the pinned clone by notebook 05) |
