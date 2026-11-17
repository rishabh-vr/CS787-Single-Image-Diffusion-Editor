#!/usr/bin/env python
"""Generate the Kaggle notebooks in notebooks/kaggle/ (steps 3 onwards of the run plan).

    python scripts/make_kaggle_notebooks.py            # all
    python scripts/make_kaggle_notebooks.py K11        # only names starting with K11

Why Kaggle: free GPU quota of ~30 h/week with **two T4s per session** and sessions of up to 12 h that keep
running in the background ("Save Version -> Save & Run All"). Every notebook here:
  * uses both GPUs (sidx.gpupool), with live progress bars (sidx.progress; plain status lines in the log
    of a background run);
  * stores its results in /kaggle/working as one zip (cs787_<name>.zip), which becomes the version's output;
  * resumes: attach the previous version's output as an input and it continues where it stopped.
Edit the cells here, not the .ipynb files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle"
SID_COMMIT = "85381e3c6f29fc07144da854d86be07150404716"
SINDDM_COMMIT = "e7107d9fd198928f26e278f99ec428e261726a45"
REPO = "https://github.com/rishabh-vr/CS787-Single-Image-Diffusion-Editor.git"


def md(t):
    return nbf.v4.new_markdown_cell(t.strip())


def code(t):
    return nbf.v4.new_code_cell(t.strip())


HOWTO = '''
**How to run (Kaggle):**
1. Right panel → *Session options*: **Accelerator = __ACCEL__**, **Internet = On**.
2. Top right **Save Version** → *Save & Run All (Commit)* → Save. It runs in the background (you can close the tab);
   watch it under *View Active Events* / the version's **Logs** (progress lines every 2 min).
3. When it finishes, the result is the version's **Output**: `cs787___NAME__.zip`.
4. **Not finished?** (it stops itself before 2 hours) → *Add Input* → *Your Work* → this notebook's
   latest version → then *Save Version* again. It continues where it stopped.
'''


def header(title, name, accel, body):
    return md(f"# {title}\n\n{body.strip()}\n" + HOWTO.replace("__ACCEL__", accel).replace("__NAME__", name))


def settings(name, extra):
    return code(f'''
import time
T_START = time.time()                      # session clock: work stops before SESSION_HOURS
NOTEBOOK = "{name}"
GITHUB_REPO   = "{REPO}"
GITHUB_BRANCH = "main"
SID_COMMIT    = "{SID_COMMIT}"     # official code, audited commit
SINDDM_COMMIT = "{SINDDM_COMMIT}"  # SinDDM, audited commit
SESSION_HOURS = 1.9                        # hard stop: everything (incl. zipping) ends before 2 h
''' + extra)


CODE = code('''
import os, sys, subprocess, shutil, glob, json
if not os.path.isdir("/kaggle/working"):
    raise SystemExit("This notebook is written for Kaggle (kaggle.com/code).")
WORK, TMP = "/kaggle/working", "/tmp/cs787"
PROJECT = f"{TMP}/project"
shutil.rmtree(PROJECT, ignore_errors=True)
os.makedirs(TMP, exist_ok=True)
r = subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", GITHUB_BRANCH, GITHUB_REPO, PROJECT],
                   capture_output=True, text=True)
if r.returncode:
    raise SystemExit("Could not download the code from GitHub. Turn Internet ON: right panel -> Session options -> "
                     "Internet (needs a phone-verified Kaggle account), then run again.\\n" + r.stderr)
sys.path.insert(0, f"{PROJECT}/src")
os.environ["PYTHONPATH"] = f"{PROJECT}/src" + os.pathsep + os.environ.get("PYTHONPATH", "")
from sidx.kaggle import gather_inputs, unpack_own, pack
from sidx.progress import Bars
print("code:", subprocess.run(["git", "-C", PROJECT, "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout)
gather_inputs()          # results of earlier versions attached under /kaggle/input (resume)
''')

GPU = code('''
import torch
N_GPUS = torch.cuda.device_count()
if N_GPUS == 0:
    raise SystemExit("No GPU. Right panel -> Session options -> Accelerator -> 'GPU T4 x2', then run again.")
cap = torch.cuda.get_device_capability(0)
if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
    raise SystemExit(f"{torch.cuda.get_device_name(0)} is not supported by this PyTorch build: choose 'GPU T4 x2' (not P100).")
print(subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "|", N_GPUS, "GPU(s) -> all of them are used")
''')

INSTALL_SID = code('''
SID_ROOT = f"{TMP}/efficient-SID-code"
if not os.path.exists(SID_ROOT):
    subprocess.run(["git", "clone", "-q", "https://github.com/haojunqiu/efficient-SID-code", SID_ROOT], check=True)
subprocess.run(["git", "-C", SID_ROOT, "checkout", "-q", SID_COMMIT], check=True)
!pip install -q -r {PROJECT}/env/requirements-colab.txt
!pip install -q --no-deps -e {SID_ROOT}
!pip install -q --no-deps -e {PROJECT}
os.environ["SID_ROOT"] = SID_ROOT
sys.path.insert(0, SID_ROOT)
import efficient_sid, sidx
print("efficient_sid OK | sidx", sidx.__version__)
''')

NATURAL = code('''
# natural images (balloons, forest, ...) are third-party and not in the repo: fetch + check sha256 against the manifests
r = subprocess.run([sys.executable, "scripts/prepare_natural.py", "--verify", "--sets", "uncond/sinddm15", "--sid-root", SID_ROOT],
                   cwd=PROJECT, capture_output=True, text=True)
print(r.stdout[-1200:], r.stderr[-1200:])
assert r.returncode == 0, "data check failed (see above)"
''')

SINDDM_CLONE = code('''
SINDDM = f"{TMP}/SinDDM"
if not os.path.exists(SINDDM):
    subprocess.run(["git", "clone", "-q", "https://github.com/fallenshock/SinDDM", SINDDM], check=True)
subprocess.run(["git", "-C", SINDDM, "checkout", "-q", "--", "."], check=True)
subprocess.run(["git", "-C", SINDDM, "checkout", "-q", SINDDM_COMMIT], check=True)
shutil.copytree(f"{PROJECT}/patches/sinddm_port", SINDDM, dirs_exist_ok=True, ignore=shutil.ignore_patterns("CHANGES.diff"))
!pip install -q einops ftfy scikit-image
print(subprocess.run(["git", "-C", SINDDM, "status", "--short"], capture_output=True, text=True).stdout)
''')

FREEZE = code('''
os.makedirs(f"{WORK}/logs", exist_ok=True)
!pip freeze > {WORK}/logs/{NOTEBOOK}_pip_freeze.txt
print("finished; output files:", sorted(os.listdir(WORK)))
''')


# --------------------------------------------------------------------------------------------------
def k10():
    return [
        header("K10 · Experiment queue on 2 GPUs (steps 3: OOD, ablations, ours on the baseline images)", "k10",
               "GPU T4 x2", '''
Runs the job files below with the official code / our core, **both T4s in parallel** (each job is split into
image chunks; a free GPU takes the next chunk). Same outputs and the same config hashes as the Colab runner.

**Reduced to finish in < 2 h** (`JOB_LIMITS`; the full plan was ≈ 14 GPU-hours):

| Job file | Reduced run | Samples | ≈ GPU-h |
|---|---|---|---|
| `phase7_ood_gpu.yaml` | Set B (24 synthetic, 128 px) and Set A (24 real OOD, ≤ 256 px), **6 seeds** each | 294 | ≈ 1.0 |
| `phase9_ablations.yaml` | AB1–AB4, AB7 **without the two T = 40 jobs**, on **4 images** (2 OOD types + circuit + balloons), **4 seeds** | 208 | ≈ 1.3 |
| `phase8_ours_on_baseline_images.yaml` | our method on the SinDDM / LoRA images, **10 seeds** | 80 | ≈ 0.4 |

≈ 2.7 GPU-hours = **≈ 80 min on 2 GPUs** + ≈ 10 min setup. Rerunning with bigger limits only adds the missing samples.
Progress: one bar per job file (samples finished).
'''),
        settings("K10_run_jobs", '''
# job file -> limits of this (reduced) run: first N seeds, only these images, without these jobs
JOB_LIMITS = {
    "configs/jobs/phase7_ood_gpu.yaml": {"seeds": 6},
    "configs/jobs/phase9_ablations.yaml": {"seeds": 4, "skip": ["ab3_T40_eta0", "ab3_T40_eta1"],
                                           "images": ["fp_layout", "pat_p4", "ckt_24aa", "balloons"]},
    "configs/jobs/phase8_ours_on_baseline_images.yaml": {"seeds": 10},
}
JOB_FILES = list(JOB_LIMITS)
'''),
        CODE, GPU, INSTALL_SID, NATURAL,
        md("## Plan: what is still pending (resumes automatically)"),
        code('''
from sidx.gpupool import job_tasks, run_pool
unpack_own()
OUT_ROOT = f"{WORK}/outputs"
DEADLINE = T_START + SESSION_HOURS * 3600 - 8 * 60         # keep 8 min to zip the results
tasks, poll = job_tasks(JOB_FILES, OUT_ROOT, PROJECT, {"SID_ROOT": SID_ROOT}, N_GPUS, deadline=DEADLINE,
                        limits=JOB_LIMITS)
for exp, (d, t) in poll().items():
    print(f"{exp:22s} {d:5d}/{t} samples already done")
print(len(tasks), "tasks to run:", ", ".join(t.label for t in tasks[:12]), "..." if len(tasks) > 12 else "")
'''),
        md("## Run on all GPUs (live bars; per-task logs in /kaggle/working/logs)"),
        code('''
try:
    run_pool(tasks, poll, n_gpus=N_GPUS, log_dir=f"{WORK}/logs/{NOTEBOOK}", deadline=DEADLINE, unit="sample")
finally:
    for exp, (d, t) in poll().items():
        print(f"{exp:22s} {d:5d}/{t} samples" + ("  DONE" if d == t else "  -> not finished: attach this output and run again"))
    pack("k10", kinds=("outputs",))
'''),
        FREEZE,
    ]


# --------------------------------------------------------------------------------------------------
def k11():
    return [
        header("K11 · SinDDM baseline: training + sampling on 2 GPUs (step 4)", "k11", "GPU T4 x2", '''
Trains SinDDM on the 3 OOD images (128 px), **all three at the same time** (two on GPU 0, one on GPU 1),
for `STEPS = 3000` each (**reduced to finish in < 2 h**: ≈ 1 T4-hour per image, the same order of compute as our
method's whole run; the full 120k-step schedule would take ~24 T4-h per image), with the LR milestones compressed in
proportion. Then samples 20 images from the 1.5k- and 3k-step checkpoints and from the 3 **official pretrained** models
(fully trained: the converged reference). So the comparison is "SinDDM at a matched, small compute budget" plus
"SinDDM fully trained" on its own images.

Progress: one bar per image (training steps, with loss and speed), then one bar per sampling job.
Checkpoints are saved every 10 min, so an interrupted run resumes (attach the output, run again).
'''),
        settings("K11_sinddm", '''
IMAGES       = ["sd_b0", "sd_fp_house2", "sd_pat_p4"]   # data/sinddm/*.png, 128 px
STEPS        = 3000
SAVE_EVERY   = 1500
SAMPLE_AFTER = [1, 2]         # checkpoints to sample: 1 = 1.5k, 2 = 3k steps
LR_K_MILESTONES = ["0.45", "1.05", "1.8", "1.95", "2.25", "2.7"]   # official 20/40/70/80/90/110k, scaled to 3k steps
N_SAMPLES    = 20
SLOTS        = 3              # trainings at the same time (the model is small; 2 share one T4)
SAMPLING_RESERVE_MIN = 22     # session minutes kept for sampling + zipping
'''),
        CODE, GPU, SINDDM_CLONE,
        md("## Inputs and resume state"),
        code('''
import collections
from sidx.gpupool import Task, run_pool
unpack_own()
DATA = f"{TMP}/sinddm_data"
for i in IMAGES:
    os.makedirs(f"{DATA}/{i}", exist_ok=True)
    shutil.copy(f"{PROJECT}/data/sinddm/{i}.png", f"{DATA}/{i}/{i}.png")
CK = f"{WORK}/checkpoints/sinddm"
OUT_ROOT = f"{WORK}/outputs"

def trained_steps(i):
    p = f"{CK}/{i}/model-last.pt"            # main.py writes to <results_folder>/<scope>/
    if not os.path.exists(p):
        return 0
    with torch.serialization.safe_globals([collections.Counter]):
        return int(torch.load(p, map_location="cpu", weights_only=True)["step"])

START = {i: trained_steps(i) for i in IMAGES}
print("steps trained so far:", START)
'''),
        md("## Train all images in parallel (live bar per image: steps, loss, speed)"),
        code('''
TRAIN_DEADLINE = T_START + SESSION_HOURS * 3600 - SAMPLING_RESERVE_MIN * 60

def train_argv(i):
    def argv():
        budget = max(1.0, (TRAIN_DEADLINE - time.time()) / 60)
        return [sys.executable, "main.py", "--scope", i, "--mode", "train", "--dataset_folder", f"{DATA}/{i}/",
                "--image_name", f"{i}.png", "--results_folder", CK, "--train_num_steps", str(STEPS + 1),
                "--save_and_sample_every", str(SAVE_EVERY), "--sched_k_milestones", *LR_K_MILESTONES,
                "--resume", "--time_budget_min", f"{budget:.1f}", "--checkpoint_every_min", "5", "--seed", "0",
                "--skip_final_sample"]
    return argv

def read_progress(i):
    try:
        return json.load(open(f"{CK}/{i}/progress.json"))
    except Exception:
        return {}

def poll_train():
    return {i: (min(STEPS, max(START[i], read_progress(i).get("step", 0))), STEPS) for i in IMAGES}

def train_postfix():
    out = {}
    for i in IMAGES:
        p = read_progress(i)
        if p.get("loss") is not None:
            out[i] = f"loss {p['loss']:.4f} | {p.get('steps_per_s', 0):.2f} steps/s"
    return out

tasks = [Task(f"train_{i}", train_argv(i), cwd=SINDDM) for i in IMAGES if START[i] < STEPS]
if tasks:
    run_pool(tasks, poll_train, slots=SLOTS, n_gpus=N_GPUS, log_dir=f"{WORK}/logs/{NOTEBOOK}", deadline=TRAIN_DEADLINE,
             unit="step", extra_postfix=train_postfix)
print("steps trained now:", {i: trained_steps(i) for i in IMAGES})
'''),
        md("## Sample every finished checkpoint + the 3 pretrained models (30 images each, standard output layout)"),
        code('''
import shlex
def collect_cmd(job, scope, data_dir, image_name, results_folder, milestone, image_id, image_path):
    return [sys.executable, f"{PROJECT}/scripts/sinddm_collect.py", "--sinddm-root", SINDDM, "--scope", scope,
            "--dataset-folder", data_dir, "--image-name", image_name, "--results-folder", results_folder,
            "--milestone", str(milestone), "--out-root", OUT_ROOT, "--job", job, "--image-id", image_id,
            "--image-path", image_path, "--n", str(N_SAMPLES), "--batch", "10"]

expected, tasks = [], []
for i in IMAGES:                     # one task per image (its checkpoints one after another: same scratch folder)
    cmds = []
    for k in SAMPLE_AFTER:
        if os.path.exists(f"{CK}/{i}/model-{k}.pt"):
            job = f"sinddm_steps{k * SAVE_EVERY}"
            cmds.append(collect_cmd(job, i, f"{DATA}/{i}/", f"{i}.png", CK, k, i, f"{PROJECT}/data/sinddm/{i}.png"))
            expected.append((job, i))
        else:
            print(i, "checkpoint", k * SAVE_EVERY, "not trained yet")
    if cmds:
        tasks.append(Task(f"sample_{i}", ["bash", "-c", " && ".join(shlex.join(c) for c in cmds)], cwd=SINDDM))
for scope in ("forest", "night_sky", "pyramids"):
    tasks.append(Task(f"sample_pretrained_{scope}", collect_cmd("sinddm_pretrained_full", scope, f"{SINDDM}/datasets/{scope}/",
                      f"{scope}.png", f"{SINDDM}/results", "pretrained", scope, f"{SINDDM}/datasets/{scope}/{scope}.png"), cwd=SINDDM))
    expected.append(("sinddm_pretrained_full", scope))

def poll_sample():
    res = {}
    for job, i in expected:
        n = len(glob.glob(f"{OUT_ROOT}/phase8_sinddm/{job}__sinddm/{i}/seed_*.png"))
        d, t = res.get(job, (0, 0))
        res[job] = (d + min(n, N_SAMPLES), t + N_SAMPLES)
    return res

try:
    run_pool(tasks, poll_sample, n_gpus=N_GPUS, log_dir=f"{WORK}/logs/{NOTEBOOK}", unit="image",
             deadline=T_START + SESSION_HOURS * 3600 - 6 * 60)
finally:
    print(poll_sample())
    pack("k11", kinds=("outputs", "checkpoints"))
'''),
        FREEZE,
    ]


# --------------------------------------------------------------------------------------------------
def k12():
    return [
        header("K12 · DreamBooth-LoRA baseline on Stable Diffusion 1.5 (step 5)", "k12", "GPU T4 x2", '''
One LoRA per image (rank 8, lr 1e-4, **400 steps** (reduced from 600 to finish in < 2 h), batch 1, 512 px, fp16, 8-bit Adam, no prior preservation) for the
8 images in `data/manifests/lora8.csv` (3 natural + 5 OOD), **two trainings at a time (one per T4)**. Then **15** samples
per image (30 DDIM steps, CFG 7.5, seeds 10..39), resized back to the input size.

Progress: one bar per image while training (the diffusers step counter), then one bar per image while sampling.
Limitation (documented): no dev-set tuning of steps / lr on the free tier; the diffusers example's defaults.
'''),
        settings("K12_lora", '''
STEPS     = 400
N_SAMPLES = 15
SEED0     = 10
DIFFUSERS_VERSION = "0.35.1"
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
'''),
        CODE, GPU,
        code('''
!pip uninstall -q -y torchao
!pip install -q diffusers=={DIFFUSERS_VERSION} transformers==4.53.2 accelerate==1.9.0 peft==0.17.1 bitsandbytes pandas
!wget -q -O {TMP}/train_dreambooth_lora.py https://raw.githubusercontent.com/huggingface/diffusers/v{DIFFUSERS_VERSION}/examples/dreambooth/train_dreambooth_lora.py
from huggingface_hub import snapshot_download      # download once, before two trainings start at the same time
snapshot_download(MODEL, allow_patterns=["*.json", "*.txt", "tokenizer/*", "text_encoder/model.safetensors",
                                         "unet/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors"])
print("SD 1.5 cached")
'''),
        md("## Train one LoRA per image, two at a time (skips images already trained)"),
        code('''
import csv
from sidx.gpupool import Task, run_pool, _safe
from sidx.progress import last_fraction
unpack_own()
rows = list(csv.DictReader(open(f"{PROJECT}/data/manifests/lora8.csv", encoding="utf-8")))
LORA = f"{WORK}/checkpoints/lora"
LOGS = f"{WORK}/logs/{NOTEBOOK}"
abs_path = lambda r: os.path.normpath(os.path.join(f"{PROJECT}/data/manifests", r["path"]))
weights_ready = lambda i: bool(glob.glob(f"{LORA}/{i}/pytorch_lora_weights.*"))

tasks = []
for r in rows:
    i = r["id"]
    if weights_ready(i):
        continue
    inst = f"{TMP}/lora_inst/{i}"; os.makedirs(inst, exist_ok=True)
    shutil.copy(abs_path(r), f"{inst}/img.png")
    tasks.append(Task(f"train_{i}", [sys.executable, f"{TMP}/train_dreambooth_lora.py", f"--pretrained_model_name_or_path={MODEL}",
        f"--instance_data_dir={inst}", f"--instance_prompt={r['prompt']}", f"--output_dir={TMP}/lora_out/{i}", "--resolution=512",
        "--train_batch_size=1", "--gradient_accumulation_steps=1", "--learning_rate=1e-4", "--lr_scheduler=constant",
        "--lr_warmup_steps=0", f"--max_train_steps={STEPS}", "--rank=8", "--mixed_precision=fp16", "--gradient_checkpointing",
        "--use_8bit_adam", "--checkpointing_steps=100000", "--seed=0"]))

def poll_train():
    res = {}
    for r in rows:
        i = r["id"]
        if weights_ready(i) or glob.glob(f"{TMP}/lora_out/{i}/pytorch_lora_weights.*"):
            res[i] = (STEPS, STEPS)
        else:
            f = last_fraction(f"{LOGS}/{_safe('train_' + i)}.log")
            res[i] = (min(f[0], STEPS), STEPS) if f else (0, STEPS)
    return res

print(len(tasks), "LoRAs to train")
if tasks:
    run_pool(tasks, poll_train, n_gpus=N_GPUS, log_dir=LOGS, unit="step", deadline=T_START + SESSION_HOURS * 3600 - 35 * 60)
for r in rows:                                  # keep the weights in the output
    i = r["id"]
    for f in glob.glob(f"{TMP}/lora_out/{i}/pytorch_lora_weights.*"):
        os.makedirs(f"{LORA}/{i}", exist_ok=True); shutil.copy(f, f"{LORA}/{i}/")
print({r["id"]: weights_ready(r["id"]) for r in rows})
'''),
        md("## Sample 30 images per LoRA (live bar per image)"),
        code('''
import pandas as pd
from diffusers import StableDiffusionPipeline
from PIL import Image
OUT = f"{WORK}/outputs/phase8_lora/lora_r8_s{STEPS}__lora"
os.makedirs(OUT, exist_ok=True)
if not os.path.exists(f"{OUT}/job.json"):
    json.dump({"experiment": "phase8_lora", "job": f"lora_r8_s{STEPS}", "method": "lora_sd15",
               "params": {"rank": 8, "lr": 1e-4, "steps": STEPS, "resolution": 512, "sampler_steps": 30, "cfg": 7.5},
               "images": [{"id": r["id"], "path": abs_path(r)} for r in rows]}, open(f"{OUT}/job.json", "w"), indent=2)
summary = json.load(open(f"{LOGS}/_summary.json")) if os.path.exists(f"{LOGS}/_summary.json") else {}
cost = {r["id"]: {"train_s": summary.get(f"train_{r['id']}", {}).get("seconds")} for r in rows}
pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.set_progress_bar_config(disable=True)
bars = Bars(unit="image", every_s=120)
try:
    for r in rows:
        i = r["id"]
        d = f"{OUT}/{i}"; os.makedirs(d, exist_ok=True)
        have = lambda: sum(os.path.exists(f"{d}/seed_{s:04d}.png") for s in range(SEED0, SEED0 + N_SAMPLES))
        bars.set(i, have(), N_SAMPLES)
        todo = [s for s in range(SEED0, SEED0 + N_SAMPLES) if not os.path.exists(f"{d}/seed_{s:04d}.png")]
        if not todo or not weights_ready(i) or time.time() > T_START + SESSION_HOURS * 3600 - 6 * 60:
            continue
        W, H = Image.open(abs_path(r)).size
        pipe.load_lora_weights(f"{LORA}/{i}")
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        for s in todo:
            im = pipe(r["prompt"], num_inference_steps=30, guidance_scale=7.5, generator=torch.Generator("cuda").manual_seed(s)).images[0]
            im.resize((W, H), Image.LANCZOS).save(f"{d}/seed_{s:04d}.png")
            bars.set(i, have(), N_SAMPLES); bars.maybe_print()
        cost[i].update(sample_s_per_img=round((time.time() - t0) / len(todo), 2),
                       sample_peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2))
        pipe.unload_lora_weights()
finally:
    bars.close()
    os.makedirs(f"{WORK}/tables", exist_ok=True)
    json.dump(cost, open(f"{WORK}/tables/phase8_lora_cost.json", "w"), indent=2)
    pack("k12", kinds=("outputs", "checkpoints", "tables"))
pd.DataFrame(cost).T
'''),
        FREEZE,
    ]


# --------------------------------------------------------------------------------------------------
def k13():
    return [
        header("K13 · Evaluate all samples (step 6, CPU — uses no GPU quota)", "k13_eval", "None (CPU)", '''
Computes the metric tables (`scripts/evaluate.py`: SIFID, diversity, NIQE/MUSIQ/NIMA, patch statistics, copying,
structure) for every experiment it finds, several jobs in parallel. Results: `eval/<experiment>/<job>/per_image.csv`
and `per_sample.csv`, all in `cs787_k13_eval.zip`.

**Inputs to attach first** (*Add Input*):
- *Your Work* → the latest versions of **K10**, **K11**, **K12** (their `cs787_*.zip` are found automatically);
- the Colab results of step 2: download the folder `MyDrive/CS787/outputs/phase4_reproduction` from Google Drive
  (right-click → Download, gives a zip), then *Add Input* → *Upload* → *New Dataset* → drop the zip → Create.
Already-evaluated jobs are skipped, so you can rerun it after attaching more results.
'''),
        settings("K13_evaluate", '''
GROUPS      = "paper,patch,copy,structure"   # "patch,copy,structure" = fast pass without pretrained networks
MAX_SAMPLES = 20                             # per image (keeps the run < 2 h)
SLOTS       = 2                              # jobs evaluated at the same time (Kaggle CPU sessions have 4 cores)
# experiment folder -> manifests that contain its input images
EVAL = {
    "phase4_reproduction": ["uncond_sinddm15"],
    "phase7_ood":          ["ood_setA", "synthetic_setB"],
    "phase8_ours":         ["sinddm_ood3", "lora8"],
    "phase8_sinddm":       ["uncond_sinddm15", "sinddm_ood3"],
    "phase8_lora":         ["lora8"],
    "phase9_ablations":    ["ablation_subset"],
    "phase9_ab6_ann":      ["ablation_subset"],
}
'''),
        CODE,
        code('''
!pip install -q lpips pytorch-fid pyiqa opencv-python-headless scikit-image scipy pandas
!pip install -q --no-deps -e {PROJECT}
SID_ROOT = f"{TMP}/efficient-SID-code"
if not os.path.exists(SID_ROOT):
    subprocess.run(["git", "clone", "-q", "https://github.com/haojunqiu/efficient-SID-code", SID_ROOT], check=True)
'''),
        NATURAL,
        md("## Evaluate (parallel; one bar per experiment, counted in images)"),
        code('''
from sidx.gpupool import Task, run_pool
unpack_own()
OUT_ROOT, EVAL_ROOT = f"{WORK}/outputs", f"{WORK}/eval"
threads = max(1, (os.cpu_count() or 2) // SLOTS)
tasks, plan = [], []
for exp, mans in EVAL.items():
    for jj in sorted(glob.glob(f"{OUT_ROOT}/{exp}/*/job.json")):
        job_dir = os.path.dirname(jj); job = os.path.basename(job_dir)
        n_img = sum(1 for d in glob.glob(f"{job_dir}/*/") if glob.glob(f"{d}seed_*.png"))
        plan.append((exp, job, n_img))
        cmd = [sys.executable, "scripts/evaluate.py", f"{OUT_ROOT}/{exp}", "--only-job", job, "--groups", GROUPS,
               "--max-samples", str(MAX_SAMPLES), "--eval-root", EVAL_ROOT, "--no-progress"]
        for m in mans:
            cmd += ["--manifest", f"{PROJECT}/data/manifests/{m}.csv"]
        tasks.append(Task(f"{exp}:{job}", cmd, cwd=PROJECT, env={"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}))
print({e: sum(1 for p in plan if p[0] == e) for e in EVAL}, "jobs found")

def poll():
    res = {}
    for exp, job, n_img in plan:
        d = f"{EVAL_ROOT}/{exp}/{job}"
        try:
            done = json.load(open(f"{d}/progress.json"))["done"]
        except Exception:
            done = n_img if os.path.exists(f"{d}/stamp.json") else 0
        a, b = res.get(exp, (0, 0))
        res[exp] = (a + min(done, n_img), b + n_img)
    return res

try:
    run_pool(tasks, poll, slots=SLOTS, n_gpus=0, log_dir=f"{WORK}/logs/{NOTEBOOK}", unit="image",
             deadline=T_START + SESSION_HOURS * 3600 - 10 * 60)
finally:
    pack("k13_eval", kinds=("eval",), remove=False)
'''),
        md("## Quick look: per-job means of the main metrics"),
        code('''
import pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
rows = []
for f in sorted(glob.glob(f"{EVAL_ROOT}/*/*/per_image.csv")):
    df = pd.read_csv(f)
    exp, job = f.split("/")[-3], f.split("/")[-2]
    rows.append({"experiment": exp, "job": job.split("__")[0], "images": len(df),
                 **df.select_dtypes("number").mean().to_dict()})
summary = pd.DataFrame(rows)
summary.to_csv(f"{WORK}/eval_summary.csv", index=False)
summary.round(4)
'''),
        FREEZE,
    ]


# --------------------------------------------------------------------------------------------------
def k14():
    return [
        header("K14 · Qualitative applications on OOD images (step 7, optional)", "k14", "GPU T4 x2", '''
Official application scripts on a few Set A images, two at a time (one per T4):
- **Retargeting** to 1:2 (wide) and 2:1 (tall): does structure survive a change of shape?
- **Tileable generation** on the patterns, then tiled 2×2 to show the seams.
Uses the fused-attention presets (no N×N matrix). Qualitative only: a contact sheet is shown at the end.
'''),
        settings("K14_applications", '''
RETARGET = ["fp_house2", "fp_layout", "pat_p4", "pat_cmm", "ckt_24aa", "dia_life"]
TILEABLE = ["pat_p4", "pat_cmm", "pat_p2", "pat_demi1"]
'''),
        CODE, GPU, INSTALL_SID,
        code('''
from PIL import Image
from sidx.gpupool import Task, run_pool
unpack_own()
OUT = f"{WORK}/outputs/qualitative"; os.makedirs(OUT, exist_ok=True)
IMG = lambda i: f"{PROJECT}/data/ood/setA/{i}.png"
FP32 = ["precision.dtype=float32", "image_denoiser.patch_denoiser.bottleneck_dtype=float32"]
tasks, targets = [], []
for i in RETARGET:
    W, H = Image.open(IMG(i)).size
    for tag, size in (("wide", f"({H},{2 * W})"), ("tall", f"({2 * H},{W})")):
        dst = f"{OUT}/retarget_{i}_{tag}.png"; targets.append(dst)
        if not os.path.exists(dst):
            tasks.append(Task(f"retarget_{i}_{tag}", [sys.executable, "scripts/sample_retarget.py", "--config",
                "configs/retarget/pixel_exact_flash_attn.yaml", f"image_path={IMG(i)}", f"output_size={size}",
                f"final_output_path={dst}", *FP32, "image_denoiser.patch_denoiser.query_chunks=4"], cwd=SID_ROOT))
for i in TILEABLE:
    dst = f"{OUT}/tileable_{i}.png"; targets.append(dst)
    if not os.path.exists(dst):
        tasks.append(Task(f"tileable_{i}", [sys.executable, "scripts/sample_tileable.py", "--config",
            "configs/tileable/pixel_exact_flash_attn.yaml", f"image_path={IMG(i)}", f"final_output_path={dst}", *FP32], cwd=SID_ROOT))
poll = lambda: {"qualitative": (sum(os.path.exists(t) for t in targets), len(targets))}
run_pool(tasks, poll, n_gpus=N_GPUS, log_dir=f"{WORK}/logs/{NOTEBOOK}", unit="image")
for i in TILEABLE:
    p = f"{OUT}/tileable_{i}.png"
    if os.path.exists(p):
        im = Image.open(p).convert("RGB"); t = Image.new("RGB", (2 * im.width, 2 * im.height))
        for a in range(2):
            for b in range(2):
                t.paste(im, (a * im.width, b * im.height))
        t.save(f"{OUT}/tileable_{i}_2x2.png")
'''),
        code('''
from IPython.display import display
for f in sorted(glob.glob(f"{OUT}/retarget_*.png")) + sorted(glob.glob(f"{OUT}/tileable_*_2x2.png")):
    im = Image.open(f).convert("RGB"); w = min(600, im.width)
    print(os.path.basename(f)); display(im.resize((w, int(im.height * w / im.width))))
pack("k14", kinds=("outputs",))
'''),
        FREEZE,
    ]


def main() -> None:
    only = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    gpu_meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                "language_info": {"name": "python"},
                "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True,
                           "language": "python", "sourceType": "notebook"}}
    cpu_meta = {**gpu_meta, "kaggle": {**gpu_meta["kaggle"], "accelerator": "none", "isGpuEnabled": False}}
    for fname, cells, meta in (("K10_run_jobs.ipynb", k10(), gpu_meta), ("K11_sinddm.ipynb", k11(), gpu_meta),
                               ("K12_lora.ipynb", k12(), gpu_meta), ("K13_evaluate.ipynb", k13(), cpu_meta),
                               ("K14_applications.ipynb", k14(), gpu_meta)):
        if only and not any(fname.startswith(p) for p in only):
            continue
        nb = nbf.v4.new_notebook(cells=cells, metadata=meta)
        nbf.validate(nb)
        nbf.write(nb, OUT / fname)
        print("wrote", OUT / fname)


if __name__ == "__main__":
    main()
