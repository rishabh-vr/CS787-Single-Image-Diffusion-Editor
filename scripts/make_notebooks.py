#!/usr/bin/env python
"""Generate the free-Colab notebooks in notebooks/ from this single source.

    python scripts/make_notebooks.py

Edit the cell text here, not the .ipynb files: regenerating overwrites them.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

SID_COMMIT = "85381e3c6f29fc07144da854d86be07150404716"
SINDDM_COMMIT = "e7107d9fd198928f26e278f99ec428e261726a45"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


# ---------------------------------------------------------------------------
# Shared cells
# ---------------------------------------------------------------------------

SETTINGS = f'''
# ==== SETTINGS (edit once) ====================================================================
CODE_SOURCE   = "github"                                      # "github" or "drive"
GITHUB_REPO   = "https://github.com/rishabh-vr/CS787-Single-Image-Diffusion-Editor.git"
GITHUB_BRANCH = "main"
REPO_SUBDIR   = ""                                            # project files are at the repo root
DRIVE_CODE    = "/content/drive/MyDrive/CS787/code/Project"   # used when CODE_SOURCE == "drive"
DRIVE_ROOT    = "/content/drive/MyDrive/CS787"                # outputs, checkpoints, tables, env logs
SID_COMMIT    = "{SID_COMMIT}"     # official code, audited commit
SINDDM_COMMIT = "{SINDDM_COMMIT}"  # SinDDM, audited commit
# If the GitHub repo is private: add a Colab secret named GH_TOKEN (key icon in the left bar)
# holding a fine-grained, read-only token for that repo.
'''

GPU_CHECK = '''
import torch, subprocess
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU (then rerun)."
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                     capture_output=True, text=True).stdout)
print("torch", torch.__version__, "| CUDA", torch.version.cuda,
      "| bf16 native:", torch.cuda.get_device_capability(0) >= (8, 0))
'''

MOUNT_AND_CODE = '''
import os, shutil, subprocess, sys
from google.colab import drive
drive.mount("/content/drive")
for sub in ("outputs", "checkpoints", "tables", "env_logs", "data"):
    os.makedirs(f"{DRIVE_ROOT}/{sub}", exist_ok=True)

PROJECT = "/content/project"
if os.path.exists(PROJECT):
    shutil.rmtree(PROJECT)
if CODE_SOURCE == "github":
    url = GITHUB_REPO
    try:
        from google.colab import userdata
        token = userdata.get("GH_TOKEN")
        url = GITHUB_REPO.replace("https://", f"https://x-access-token:{token}@")
    except Exception:
        pass  # public repo: no token needed
    subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", GITHUB_BRANCH, url, "/content/repo"], check=True)
    shutil.copytree(os.path.join("/content/repo", REPO_SUBDIR), PROJECT,
                    ignore=shutil.ignore_patterns(".venv", "external", "outputs*"))
else:
    shutil.copytree(DRIVE_CODE, PROJECT, ignore=shutil.ignore_patterns(".venv", "external", "outputs*"))
print("project code at", PROJECT, "->", sorted(os.listdir(PROJECT)))
'''

INSTALL_SID = '''
SID_ROOT = "/content/efficient-SID-code"
if not os.path.exists(SID_ROOT):
    subprocess.run(["git", "clone", "-q", "https://github.com/haojunqiu/efficient-SID-code", SID_ROOT], check=True)
subprocess.run(["git", "-C", SID_ROOT, "checkout", "-q", SID_COMMIT], check=True)
!pip install -q -r {PROJECT}/env/requirements-colab.txt
!pip install -q --no-deps -e {SID_ROOT}
!pip install -q -e {PROJECT}
os.environ["SID_ROOT"] = SID_ROOT
# editable installs are only picked up by *new* processes; make them importable in this kernel too
for p in (SID_ROOT, f"{PROJECT}/src"):
    if p not in sys.path:
        sys.path.insert(0, p)
import efficient_sid, sidx
print("efficient_sid OK | sidx", sidx.__version__)
'''

FREEZE = '''
import datetime
stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
!pip freeze > {DRIVE_ROOT}/env_logs/{stamp}_{NOTEBOOK}_freeze.txt
print("saved", f"{DRIVE_ROOT}/env_logs/{stamp}_{NOTEBOOK}_freeze.txt")
'''


# ---------------------------------------------------------------------------
# 00 — setup + smoke test + T4 timing
# ---------------------------------------------------------------------------

def nb_setup_smoke():
    cells = [
        md('''
# 00 · Setup, smoke test and T4 timing calibration (Phase 2)

**Runtime → Change runtime type → T4 GPU.** Then *Run all*. About 20–30 min of GPU time.

This notebook:
1. installs the official efficient-SID code at the audited commit, plus our `sidx` package;
2. answers open item **D3**: what does `autocast(dtype=float32)` do to bf16 tensors on this GPU?
3. runs `configs/jobs/phase2_t4_timing.yaml` through the resumable runner (outputs on Drive);
4. writes the **timing table** that recalibrates PROJECT_PLAN §5.3;
5. checks that `faiss-gpu-cu12` imports (needed only for ablation AB6).

If the session disconnects, rerun everything: finished samples are skipped.
'''),
        code(SETTINGS + '\nNOTEBOOK = "00_setup_smoke"'),
        code(GPU_CHECK),
        code(MOUNT_AND_CODE),
        code(INSTALL_SID),
        md('''
## D3 — what precision does the "float32 bottleneck" really use?
The Table-1 preset keeps tensors in bf16 and wraps the kernel in `torch.autocast("cuda", dtype=float32)`.
(See docs/technical_notes.md §4, D3.)
'''),
        code('''
import torch, torch.nn.functional as F
x = torch.randn(8, 675, device="cuda", dtype=torch.bfloat16)
y = torch.randn(16, 675, device="cuda", dtype=torch.bfloat16)
a = torch.tensor(0.5, device="cuda")                       # 0-dim fp32, like scheduler.alphas[t]
with torch.autocast(device_type="cuda", dtype=torch.float32):
    ay = a * y                                             # elementwise
    logits = x @ ay.T                                      # matmul
    w = F.softmax(logits, dim=1)
    out = torch.einsum("bn,nd->bd", w, y)
print("alpha*y:", ay.dtype, "| matmul:", logits.dtype, "| softmax:", w.dtype, "| einsum:", out.dtype)
'''),
        md('## Timing calibration (resumable; outputs → Drive)'),
        code('''
JOBS = f"{PROJECT}/configs/jobs/phase2_t4_timing.yaml"
OUT  = f"{DRIVE_ROOT}/outputs"
!python {PROJECT}/scripts/run_queue.py {JOBS} --out-root {OUT} --var SID_ROOT={SID_ROOT} --dry-run
!python {PROJECT}/scripts/run_queue.py {JOBS} --out-root {OUT} --var SID_ROOT={SID_ROOT} --budget-min 60
'''),
        code('''
import json, glob, pandas as pd
rows = []
for jsonl in glob.glob(f"{OUT}/phase2_t4_timing/*/*/samples.jsonl"):
    job = jsonl.split("/")[-3].split("__")[0]
    setup = json.load(open(jsonl.replace("samples.jsonl", "setup.json")))["setup"]
    recs = [json.loads(l) for l in open(jsonl)]
    secs = [r["seconds"] for r in recs]
    rows.append(dict(job=job, input_hw=setup["input_hw"], n_patches_finest=setup["num_patches_per_scale"][0],
                     setup_s=setup["setup_seconds"], samples=len(secs),
                     sec_per_sample_median=round(sorted(secs)[len(secs)//2], 2),
                     sec_first_sample=round(secs[0], 2), peak_vram_gb=max(r["peak_vram_gb"] or 0 for r in recs),
                     gpu=recs[0]["gpu"]))
df = pd.DataFrame(rows).sort_values("job")
display(df)
df.to_csv(f"{DRIVE_ROOT}/tables/phase2_t4_timing.csv", index=False)
'''),
        md('''
### D2 — does the shipped bf16 mode change the output?
Same seed, fp32 vs as-shipped. Mean absolute difference in grey levels (0–255).
'''),
        code('''
import numpy as np
from PIL import Image
def first(job, seed=10):
    return np.asarray(Image.open(glob.glob(f"{OUT}/phase2_t4_timing/{job}__*/balloons/seed_{seed:04d}.png")[0]), np.float32)
for seed in (10, 11, 12):
    try:
        a, b = first("paper_fp32_native", seed), first("paper_shipped_bf16_native", seed)
        c = first("flash_fp32_native", seed)
        print(f"seed {seed}: |fp32 - bf16 shipped| = {np.abs(a-b).mean():.2f}   |exact - flash (fp32)| = {np.abs(a-c).mean():.3f}")
    except IndexError:
        print(f"seed {seed}: not generated yet")
'''),
        md('## Optional: faiss (ablation AB6 only)'),
        code('''
!pip install -q faiss-gpu-cu12
try:
    import faiss; print("faiss", faiss.__version__, "| GPUs:", faiss.get_num_gpus())
except Exception as e:
    print("faiss NOT usable:", repr(e), "-> AB6 stays optional")
'''),
        code(FREEZE),
        md('''
**Done when:** the timing table exists on Drive, D2/D3 are answered, and a rerun of the runner cell
prints `done=0 skipped(existing)=18`. Copy the numbers into PROJECT_PLAN §5.3 and docs/technical_notes.md.
'''),
    ]
    return cells


# ---------------------------------------------------------------------------
# 05 — SinDDM port: pretrained sampling, training speed, resume
# ---------------------------------------------------------------------------

def nb_sinddm():
    return [
        md('''
# 05 · SinDDM on Colab: port check, training speed, resume (Phase 2)

**T4 GPU runtime.** About 25–35 min.

SinDDM pins Python 3.8 / torch 1.13, which Colab cannot install, so we run our port:
`patches/sinddm_port/` replaces 3 files (see `CHANGES.diff`):
- a device option;
- resumable, time-budgeted training;
- safe weights-only checkpoint loading;
- a scikit-image API fix.

This notebook:
1. samples the 3 official pretrained models;
2. measures **seconds per 1 000 training steps** at the native size and at a 128-px short side;
3. tests stop-and-resume.
'''),
        code(SETTINGS + '\nNOTEBOOK = "05_sinddm_port"'),
        code(GPU_CHECK),
        code(MOUNT_AND_CODE),
        code('''
SINDDM = "/content/SinDDM"
if not os.path.exists(SINDDM):
    subprocess.run(["git", "clone", "-q", "https://github.com/fallenshock/SinDDM", SINDDM], check=True)
subprocess.run(["git", "-C", SINDDM, "checkout", "-q", "--", "."], check=True)
subprocess.run(["git", "-C", SINDDM, "checkout", "-q", SINDDM_COMMIT], check=True)
shutil.copytree(f"{PROJECT}/patches/sinddm_port", SINDDM, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("CHANGES.diff"))
!pip install -q einops ftfy
print(subprocess.run(["git", "-C", SINDDM, "status", "--short"], capture_output=True, text=True).stdout)
'''),
        md('## 1 · Sample the 3 official pretrained models (validates the port)'),
        code('''
%cd {SINDDM}
for scope in ("forest", "night_sky", "pyramids"):
    cmd = (f"python main.py --scope {scope} --mode sample --dataset_folder ./datasets/{scope}/ "
           f"--image_name {scope}.png --results_folder ./results/ --load_milestone 12 "
           f"--sample_batch_size 4 --seed 0 2>&1 | grep -v -i warn | tail -3")
    !{cmd}
# If loading fails with "Unsupported global", the official checkpoint holds another type:
# add it to the allow-list in patches/sinddm_port/SinDDM/trainer.py (load) and rerun.
'''),
        code('''
import glob
from IPython.display import Image as IPImage, display
for scope in ("forest", "night_sky", "pyramids"):
    grid = sorted(glob.glob(f"{SINDDM}/results/{scope}/final_samples/*_out_s*"))
    if grid:
        print(scope); display(IPImage(grid[-1], width=600))
'''),
        md('## 2 · Training speed (seconds per 1 000 steps)'),
        code('''
import time, json
from PIL import Image
def make_input(name, short_side=None):
    d = f"/content/sinddm_bench/{name}/"; os.makedirs(d, exist_ok=True)
    im = Image.open(f"{SINDDM}/datasets/balloons/balloons.png").convert("RGB")
    if short_side:
        w, h = im.size; s = short_side / min(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    im.save(d + f"{name}.png"); return d, im.size

results = {}
for name, short in (("bench_native", None), ("bench_128", 128)):
    folder, size = make_input(name, short)
    cmd = (f"python main.py --scope {name} --mode train --dataset_folder {folder} --image_name {name}.png "
           f"--results_folder /content/sinddm_bench/results/ --train_num_steps 1000 --save_and_sample_every 1000000 "
           f"--time_budget_min 20 --checkpoint_every_min 100 --sample_batch_size 1 --seed 0 "
           f"2>&1 | grep -E 'step:|port|completed' | tail -3")
    t0 = time.time()
    !{cmd}
    results[name] = {"input_wh": size, "seconds_per_1000_steps_incl_final_sampling": round(time.time() - t0, 1)}
print(json.dumps(results, indent=2))
full = 120_000
for k, v in results.items():
    print(f"{k}: full 120k-step schedule would take ~{v['seconds_per_1000_steps_incl_final_sampling'] * full / 1000 / 3600:.1f} h on this GPU (upper bound)")
json.dump(results, open(f"{DRIVE_ROOT}/tables/phase2_sinddm_timing.json", "w"), indent=2)
'''),
        md('## 3 · Stop-and-resume test (checkpoints on Drive)'),
        code('''
ck = f"{DRIVE_ROOT}/checkpoints/sinddm_resume_test/"
folder, _ = make_input("resume_test", 128)
cmd = (f"python main.py --scope resume_test --mode train --dataset_folder {folder} --image_name resume_test.png "
       f"--results_folder {ck} --train_num_steps 400 --save_and_sample_every 1000000 --resume "
       f"--time_budget_min 0.5 --checkpoint_every_min 0.25 --sample_batch_size 1 --seed 0 "
       f"2>&1 | grep -E 'port|completed' | tail -3")
for attempt in (1, 2):
    print(f"--- run {attempt}")
    !{cmd}
print("expected: run 1 stops on the time budget; run 2 says 'resumed from step N' with N > 0")
'''),
        code(FREEZE),
    ]


# ---------------------------------------------------------------------------
# 06 — LoRA smoke test
# ---------------------------------------------------------------------------

def nb_lora():
    return [
        md('''
# 06 · DreamBooth-LoRA smoke test on a T4 (Phase 2)

**T4 GPU runtime.** About 15–20 min (most of it is downloading SD 1.5, ~4 GB, into `/content` — not Drive).

This notebook:
1. runs a 50-step SD 1.5 DreamBooth-LoRA training with fp16, gradient checkpointing and 8-bit Adam;
2. samples 2 images;
3. records time and VRAM.

The input image is a placeholder (the balloons example). The real images come in Phase 8.
'''),
        code(SETTINGS + '\nNOTEBOOK = "06_lora_smoke"'),
        code(GPU_CHECK),
        code(MOUNT_AND_CODE),
        code('''
DIFFUSERS_VERSION = "0.35.1"
!pip install -q diffusers=={DIFFUSERS_VERSION} transformers==4.53.2 accelerate==1.9.0 peft bitsandbytes
!wget -q -O /content/train_dreambooth_lora.py https://raw.githubusercontent.com/huggingface/diffusers/v{DIFFUSERS_VERSION}/examples/dreambooth/train_dreambooth_lora.py
!accelerate config default
os.makedirs("/content/lora_inst", exist_ok=True)
!wget -q -O /content/lora_inst/img.png https://raw.githubusercontent.com/haojunqiu/efficient-SID-code/{SID_COMMIT}/examples/uncond/balloons.png
'''),
        code('''
import time, json, torch
cmd = ("accelerate launch /content/train_dreambooth_lora.py "
       "--pretrained_model_name_or_path=stable-diffusion-v1-5/stable-diffusion-v1-5 "
       "--instance_data_dir=/content/lora_inst --instance_prompt='a photo in the style of sks' "
       "--output_dir=/content/lora_out --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 "
       "--learning_rate=1e-4 --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=50 --rank=8 "
       "--mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam --checkpointing_steps=1000 --seed=0")
t0 = time.time()
!{cmd}
train_s = time.time() - t0
print(f"50 steps (incl. model download): {train_s:.0f} s")
'''),
        code('''
from diffusers import StableDiffusionPipeline
torch.cuda.reset_peak_memory_stats()
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",
                                               torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.load_lora_weights("/content/lora_out")
t0 = time.time()
imgs = [pipe("a photo in the style of sks", num_inference_steps=30, guidance_scale=7.5,
             generator=torch.Generator("cuda").manual_seed(s)).images[0] for s in (0, 1)]
sample_s = (time.time() - t0) / 2
res = {"train_50_steps_s_incl_download": round(train_s, 1), "sample_s_per_image": round(sample_s, 2),
       "sampling_peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}
print(res); json.dump(res, open(f"{DRIVE_ROOT}/tables/phase2_lora_timing.json", "w"), indent=2)
display(imgs[0])
'''),
        code(FREEZE),
    ]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"}, "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"}}
    for fname, cells in (("00_setup_smoke.ipynb", nb_setup_smoke()),
                         ("05_sinddm_port.ipynb", nb_sinddm()),
                         ("06_lora_smoke.ipynb", nb_lora())):
        nb = nbf.v4.new_notebook(cells=cells, metadata=meta)
        nbf.validate(nb)
        nbf.write(nb, OUT / fname)
        print("wrote", OUT / fname)


if __name__ == "__main__":
    main()
