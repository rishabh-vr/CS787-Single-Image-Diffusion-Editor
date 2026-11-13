# Free-Colab Workflow (Phase 2)

Everything GPU-related runs on **free Google Colab (T4)**. Everything else runs on the laptop (CPU).
This page is the how-to; the reasons behind it are in PROJECT_PLAN.md §5–§6.

## 1. One-time setup

1. **Code on GitHub.** The notebooks clone `https://github.com/rishabh-vr/CS787-Single-Image-Diffusion-Editor` (**public**, so no token is needed). The project files are at the repo root.
   - The `.gitignore` keeps `.venv/`, `external/` clones and outputs out of git.
   - If the repo is ever made private again: create a fine-grained, read-only token and add it in Colab as the secret `GH_TOKEN`. The notebooks already use it when present.
   - Alternative: upload `Project/` (without `.venv`/`external`) to `MyDrive/CS787/code/Project` and set `CODE_SOURCE = "drive"` in the notebook's first cell.
2. **Google Drive layout** (created automatically): `MyDrive/CS787/{outputs, checkpoints, tables, env_logs, data}`.
3. **Optional:** a free Hugging Face account, with the FLUX.1-schnell terms accepted. This is only needed for the latent-space ablation AB5.

## 2. Notebooks (Phase 2: run 00, 05, 06 once each; then 10 for the experiments)

| Notebook | GPU time [est.] | What it settles |
|---|---|---|
| `notebooks/00_setup_smoke.ipynb` | 20–30 min | Official code runs on a T4; D2/D3 precision questions; **T4 timing table** → `tables/phase2_t4_timing.csv`; faiss import |
| `notebooks/05_sinddm_port.ipynb` | 25–35 min | Ported SinDDM samples the 3 pretrained models; **seconds per 1 000 training steps** → `tables/phase2_sinddm_timing.json`; stop-and-resume works |
| `notebooks/06_lora_smoke.ipynb` | 15–20 min | SD 1.5 LoRA fits on a T4; train and sample timings → `tables/phase2_lora_timing.json` |
| `notebooks/10_run_queue.ipynb` *(Phase 4+, rerun every session)* | ~100 min per session | Runs a job file through the resumable runner (default: `configs/jobs/phase4_reproduction.yaml`, the Table-1 reproduction). Fetches the sinddm15 images and checks them against the committed manifest first |
| `notebooks/07_tpu_sinddm.ipynb` *(optional, **TPU runtime**)* | 30–45 min | Is the free **v5e-1 TPU** ≥ 2× faster than the T4 at SinDDM training? → `tables/phase2_sinddm_timing_tpu.json` + a printed decision. **Run after 05.** |

### About the free TPU
- The free Colab TPU is a **v5e-1** (one chip, 16 GB). The older v2-8 was retired in 2025.
- Like the T4, its availability is not guaranteed.
- Only **SinDDM training** is worth trying on it, because that's the project's only real compute bottleneck. Notebook 07 measures it with exactly the same method as notebook 05:
  - steady-state seconds per 1 000 steps;
  - the cost of compiling the XLA graph is cancelled out.
- Our SinDDM port runs on the TPU with `--device xla`:
  - no device-to-host syncs inside the loop;
  - one XLA graph per step;
  - checkpoints saved as CPU tensors, so a TPU-trained model samples on the T4.
- **Kept off the TPU:**
  - the paper's method — nothing to train, and the code is CUDA-only;
  - LoRA — short already, and diffusers' LoRA script targets CUDA.

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
| On the TPU runtime, `import torch` fails with "Failed to load the backend extension: torch_tpu" after installing torch_xla | The Colab TPU image ships `torch_tpu` (for torch ≥ 2.11), which torch auto-loads and which crashes with torch_xla's older torch. Notebook 07 uninstalls `torch_tpu` and sets `TORCH_DEVICE_BACKEND_AUTOLOAD=0` |
