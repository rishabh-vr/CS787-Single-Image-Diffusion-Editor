"""Notebook builders for Phase 8 (11 SinDDM, 12 LoRA) and evaluation (13). Imported by make_notebooks.py,
which provides md/code and the shared cells (SETTINGS, GPU_CHECK, MOUNT_AND_CODE, SINDDM_CLONE, FREEZE)."""

from __future__ import annotations


def build(md, code, SETTINGS, GPU_CHECK, MOUNT_AND_CODE, SINDDM_CLONE, FREEZE):
    def nb_sinddm_train():
        return [
            md('''
# 11 · SinDDM baseline: reduced-budget training + sampling (Phase 8)

**T4 GPU runtime.** Trains SinDDM on the 3 OOD images (128 px) and samples it at several training
lengths, giving a **quality-vs-training-time curve** (the full 120k-step schedule would take ~24 T4-hours
per image, notebook 05).

- Reduced schedule: `STEPS = 20000` with the LR milestones compressed in proportion (3, 7, 12, 13, 15, 18 ×1000).
- **Resumable:** checkpoints are on Drive every 10 min, and `model-1..4.pt` at 5k/10k/15k/20k steps.
  Each session trains for at most `SESSION_MIN` minutes; rerun this notebook until all 3 images show 20000 steps.
- Sampling: 30 images per checkpoint (seeds 10..39) stored in the standard output layout, plus the 3 **official pretrained**
  models (forest, night_sky, pyramids; full training) as the converged reference.
- Evaluate afterwards with notebook 13 (CPU runtime).
'''),
            code(SETTINGS + '''
NOTEBOOK     = "11_sinddm_train"
IMAGES       = ["sd_b0", "sd_fp_house2", "sd_pat_p4"]   # data/sinddm/*.png, 128 px
STEPS        = 20000
SAVE_EVERY   = 5000
SESSION_MIN  = 100                  # training minutes this session (all images together)
SAMPLE_AFTER = [1, 2, 4]            # milestones to sample: 5k, 10k, 20k steps (1..4 = 5k..20k)
N_SAMPLES    = 30
'''),
            code(GPU_CHECK),
            code(MOUNT_AND_CODE),
            code(SINDDM_CLONE),
            md("## Inputs"),
            code('''
import torch, collections, time, glob
DATA = "/content/sinddm_data"
for i in IMAGES:
    os.makedirs(f"{DATA}/{i}", exist_ok=True)
    shutil.copy(f"{PROJECT}/data/sinddm/{i}.png", f"{DATA}/{i}/{i}.png")
CK = f"{DRIVE_ROOT}/checkpoints/sinddm"

def trained_steps(i):
    p = f"{CK}/{i}/{i}/model-last.pt"
    if not os.path.exists(p):
        return 0
    with torch.serialization.safe_globals([collections.Counter]):
        return int(torch.load(p, map_location="cpu", weights_only=True)["step"])

print({i: trained_steps(i) for i in IMAGES})
'''),
            md("## Train (resumes where the last session stopped; stops cleanly at the session budget)"),
            code('''
remaining = float(SESSION_MIN)
for i in IMAGES:
    done = trained_steps(i)
    if done >= STEPS:
        print(i, "already at", done); continue
    if remaining < 3:
        print("session budget used up; rerun later"); break
    cmd = (f"python main.py --scope {i} --mode train --dataset_folder {DATA}/{i}/ --image_name {i}.png "
           f"--results_folder {CK} --train_num_steps {STEPS + 1} --save_and_sample_every {SAVE_EVERY} "
           f"--sched_k_milestones 3 7 12 13 15 18 --resume --time_budget_min {remaining:.1f} "
           f"--checkpoint_every_min 10 --seed 0 --skip_final_sample 2>&1 | grep -E 'port|step:' | tail -4")
    t0 = time.time(); print(f"--- {i}: from step {done}")
    os.chdir(SINDDM)
    !{cmd}
    remaining -= (time.time() - t0) / 60
    print(i, "now at step", trained_steps(i), f"| {remaining:.0f} session-min left")
'''),
            md("## Sample the checkpoints that exist (30 images each) into the standard output layout"),
            code('''
OUT = f"{DRIVE_ROOT}/outputs"
def collect(job, scope, data_dir, image_name, results_folder, milestone, image_id, image_path):
    cmd = (f"python {PROJECT}/scripts/sinddm_collect.py --sinddm-root {SINDDM} --scope {scope} --dataset-folder {data_dir} "
           f"--image-name {image_name} --results-folder {results_folder} --milestone {milestone} --out-root {OUT} "
           f"--job {job} --image-id {image_id} --image-path {image_path} --n {N_SAMPLES} --batch 10 2>&1 | tail -2")
    !{cmd}

for i in IMAGES:
    for k in SAMPLE_AFTER:
        if os.path.exists(f"{CK}/{i}/{i}/model-{k}.pt"):
            collect(f"sinddm_steps{k * SAVE_EVERY}", i, f"{DATA}/{i}/", f"{i}.png", CK, k, i, f"data/sinddm/{i}.png")
        else:
            print(i, "milestone", k, "not trained yet")
'''),
            md("## The 3 official pretrained models (full training) as the converged reference"),
            code('''
for scope in ("forest", "night_sky", "pyramids"):
    collect("sinddm_pretrained_full", scope, f"{SINDDM}/datasets/{scope}/", f"{scope}.png", f"{SINDDM}/results", "pretrained",
            scope, f"{SINDDM}/datasets/{scope}/{scope}.png")
'''),
            code(FREEZE),
        ]

    def nb_lora_train():
        return [
            md('''
# 12 · DreamBooth-LoRA baseline on Stable Diffusion 1.5 (Phase 8)

**T4 GPU runtime.** One LoRA per image (rank 8, lr 1e-4, **600 steps**, batch 1, 512 px, fp16, 8-bit Adam,
no prior preservation), for the 8 images in `data/manifests/lora8.csv` (3 natural + 5 OOD).
Then 30 samples per image (30 DDIM steps, CFG 7.5, seeds 10..39), resized back to the input's size for evaluation.

- **Resumable:** a LoRA already on Drive is not retrained; finished samples are skipped.
- **Limitation (documented):** the plan's 2-image dev set for tuning steps/lr is dropped on the free tier;
  these are the defaults of the diffusers example, untuned. No regularisation images.
- Costs per image (train s, sample s/img, peak VRAM) go to `tables/phase8_lora_cost.json`.
'''),
            code(SETTINGS + '''
NOTEBOOK    = "12_lora_train"
STEPS       = 600
SESSION_MIN = 100
N_SAMPLES   = 30
SEED0       = 10
'''),
            code(GPU_CHECK),
            code(MOUNT_AND_CODE),
            code('''
DIFFUSERS_VERSION = "0.35.1"
# same pins as notebook 06 (newest peft rejects Colab's torchao; we never use torchao)
!pip uninstall -q -y torchao
!pip install -q diffusers=={DIFFUSERS_VERSION} transformers==4.53.2 accelerate==1.9.0 peft==0.17.1 bitsandbytes pandas
!wget -q -O /content/train_dreambooth_lora.py https://raw.githubusercontent.com/huggingface/diffusers/v{DIFFUSERS_VERSION}/examples/dreambooth/train_dreambooth_lora.py
!accelerate config default
'''),
            md("## Train one LoRA per image (skips images already trained)"),
            code('''
import time, json, glob, csv, subprocess, torch, pandas as pd
rows = list(csv.DictReader(open(f"{PROJECT}/data/manifests/lora8.csv", encoding="utf-8")))
LORA = f"{DRIVE_ROOT}/checkpoints/lora"
COST = f"{DRIVE_ROOT}/tables/phase8_lora_cost.json"
cost = json.load(open(COST)) if os.path.exists(COST) else {}
def abs_path(r):                                   # manifest paths are relative to the manifest folder
    return os.path.normpath(os.path.join(f"{PROJECT}/data/manifests", r["path"]))
def weights_ready(i):
    return bool(glob.glob(f"{LORA}/{i}/pytorch_lora_weights.*"))

remaining = float(SESSION_MIN) * 60
for r in rows:
    i = r["id"]
    if weights_ready(i):
        print(i, "already trained"); continue
    if remaining < 600:
        print("session budget used up; rerun later"); break
    inst = f"/content/lora_inst/{i}"; os.makedirs(inst, exist_ok=True)
    shutil.copy(abs_path(r), f"{inst}/img.png")
    out = f"/content/lora_out/{i}"
    cmd = (f"accelerate launch /content/train_dreambooth_lora.py --pretrained_model_name_or_path=stable-diffusion-v1-5/stable-diffusion-v1-5 "
           f"--instance_data_dir={inst} --instance_prompt='{r['prompt']}' --output_dir={out} --resolution=512 "
           f"--train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=1e-4 --lr_scheduler=constant --lr_warmup_steps=0 "
           f"--max_train_steps={STEPS} --rank=8 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam "
           f"--checkpointing_steps=100000 --seed=0")
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    ret = subprocess.run(cmd, shell=True).returncode
    dt = time.time() - t0; remaining -= dt
    assert ret == 0 and glob.glob(f"{out}/pytorch_lora_weights.*"), f"LoRA training FAILED for {i} (see the error above)"
    os.makedirs(f"{LORA}/{i}", exist_ok=True)
    for f in glob.glob(f"{out}/pytorch_lora_weights.*"):
        shutil.copy(f, f"{LORA}/{i}/")
    cost.setdefault(i, {})["train_s"] = round(dt, 1)
    json.dump(cost, open(COST, "w"), indent=2)
    print(f"{i}: trained in {dt/60:.1f} min")
'''),
            md("## Sample 30 images per trained LoRA into the standard output layout (resized back to the input size)"),
            code('''
from diffusers import StableDiffusionPipeline
from PIL import Image
OUT = f"{DRIVE_ROOT}/outputs/phase8_lora/lora_r8_s{STEPS}__lora"
os.makedirs(OUT, exist_ok=True)
if not os.path.exists(f"{OUT}/job.json"):
    json.dump({"experiment": "phase8_lora", "job": f"lora_r8_s{STEPS}", "method": "lora_sd15",
               "params": {"rank": 8, "lr": 1e-4, "steps": STEPS, "resolution": 512, "sampler_steps": 30, "cfg": 7.5},
               "images": [{"id": r["id"], "path": abs_path(r)} for r in rows]}, open(f"{OUT}/job.json", "w"), indent=2)
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16,
                                               safety_checker=None).to("cuda")
pipe.set_progress_bar_config(disable=True)
for r in rows:
    i = r["id"]
    if not weights_ready(i):
        print(i, "no LoRA yet"); continue
    d = f"{OUT}/{i}"; os.makedirs(d, exist_ok=True)
    todo = [s for s in range(SEED0, SEED0 + N_SAMPLES) if not os.path.exists(f"{d}/seed_{s:04d}.png")]
    if not todo:
        continue
    W, H = Image.open(abs_path(r)).size
    pipe.load_lora_weights(f"{LORA}/{i}")
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for s in todo:
        im = pipe(r["prompt"], num_inference_steps=30, guidance_scale=7.5,
                  generator=torch.Generator("cuda").manual_seed(s)).images[0]
        im.resize((W, H), Image.LANCZOS).save(f"{d}/seed_{s:04d}.png")
    cost.setdefault(i, {}).update(sample_s_per_img=round((time.time() - t0) / len(todo), 2),
                                  sample_peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2))
    json.dump(cost, open(COST, "w"), indent=2)
    pipe.unload_lora_weights()
    print(i, len(todo), "samples", cost[i])
pd.DataFrame(cost).T
'''),
            code(FREEZE),
        ]

    def nb_evaluate():
        return [
            md('''
# 13 · Evaluate generated samples (CPU runtime, no GPU quota)

**Runtime → Change runtime type → CPU.** Computes the metric tables (`scripts/evaluate.py`) for the experiments listed in
`EVAL` and writes `per_sample.csv` / `per_image.csv` to `MyDrive/CS787/eval/<experiment>/<job>/`. Already-evaluated
jobs are skipped, so rerun after each generation session. About 5–10 s per sample for the full metric set on Colab CPU;
use `GROUPS = "patch,copy,structure"` for a fast pass (no pretrained networks).
'''),
            code(SETTINGS + '''
NOTEBOOK    = "13_evaluate"
GROUPS      = "paper,patch,copy,structure"
MAX_SAMPLES = 50
# experiment folder -> manifests that contain its input images (ids must not collide, so list only the relevant ones)
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
            code(MOUNT_AND_CODE),
            code('''
!pip install -q lpips pytorch-fid pyiqa opencv-python-headless scikit-image scipy pandas
!pip install -q -e {PROJECT}
sys.path.insert(0, f"{PROJECT}/src")
# the natural-image sets are third-party and not in the repo: fetch them and check against the manifests
SID_ROOT = "/content/efficient-SID-code"
if not os.path.exists(SID_ROOT):
    subprocess.run(["git", "clone", "-q", "https://github.com/haojunqiu/efficient-SID-code", SID_ROOT], check=True)
r = subprocess.run([sys.executable, "scripts/prepare_natural.py", "--verify", "--sets", "uncond/sinddm15", "--sid-root", SID_ROOT],
                   cwd=PROJECT, capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-800:])
assert r.returncode == 0, "natural images do not match the manifest"
'''),
            code('''
OUT = f"{DRIVE_ROOT}/outputs"
for exp, mans in EVAL.items():
    if not os.path.isdir(f"{OUT}/{exp}"):
        print(exp, "-> no outputs yet"); continue
    cmd = [sys.executable, "scripts/evaluate.py", f"{OUT}/{exp}", "--groups", GROUPS, "--max-samples", str(MAX_SAMPLES),
           "--eval-root", f"{DRIVE_ROOT}/eval"]
    for m in mans:
        cmd += ["--manifest", f"{PROJECT}/data/manifests/{m}.csv"]
    print("=====", exp)
    subprocess.run(cmd, cwd=PROJECT)
'''),
            code(FREEZE),
        ]

    def nb_apps():
        return [
            md('''
# 14 · Qualitative applications on OOD images (Phase 5 R9 / Phase 7 tasks ii–iii)

**T4 GPU runtime, ~20–30 min.** Runs the official application scripts on a few Set A images:
- **Retargeting** to a 1:2 (wide) and 2:1 (tall) canvas: how does structure survive a change of shape?
- **Tileable generation** on the patterns: is the periodicity preserved across the wrap-around?

Uses the fused-attention (`exact_flash_attn`) presets, which never build the N×N matrix, so the large targets fit the T4.
Outputs go to `MyDrive/CS787/outputs/qualitative/` and a contact sheet is shown below (qualitative only, no metrics).
'''),
            code(SETTINGS + '''
NOTEBOOK = "14_applications"
RETARGET = ["fp_house2", "fp_layout", "pat_p4", "pat_cmm", "ckt_24aa", "dia_life"]
TILEABLE = ["pat_p4", "pat_cmm", "pat_p2", "pat_demi1"]
'''),
            code(GPU_CHECK),
            code(MOUNT_AND_CODE),
            code('''
SID_ROOT = "/content/efficient-SID-code"
if not os.path.exists(SID_ROOT):
    subprocess.run(["git", "clone", "-q", "https://github.com/haojunqiu/efficient-SID-code", SID_ROOT], check=True)
subprocess.run(["git", "-C", SID_ROOT, "checkout", "-q", SID_COMMIT], check=True)
!pip install -q -r {PROJECT}/env/requirements-colab.txt
!pip install -q --no-deps -e {SID_ROOT}
from PIL import Image
OUT = f"{DRIVE_ROOT}/outputs/qualitative"; os.makedirs(OUT, exist_ok=True)
IMG = lambda i: f"{PROJECT}/data/ood/setA/{i}.png"
'''),
            md("## Retargeting: 1:2 (wide) and 2:1 (tall)"),
            code('''
FP32 = "precision.dtype=float32 image_denoiser.patch_denoiser.bottleneck_dtype=float32"
for i in RETARGET:
    W, H = Image.open(IMG(i)).size
    for tag, size in (("wide", f"({H},{2 * W})"), ("tall", f"({2 * H},{W})")):
        dst = f"{OUT}/retarget_{i}_{tag}.png"
        if os.path.exists(dst):
            continue
        cmd = (f"cd {SID_ROOT} && python scripts/sample_retarget.py --config configs/retarget/pixel_exact_flash_attn.yaml "
               f"image_path={IMG(i)} output_size='{size}' final_output_path={dst} {FP32} image_denoiser.patch_denoiser.query_chunks=4 "
               f"2>&1 | tail -1")
        print(i, tag, size)
        !{cmd}
'''),
            md("## Tileable generation (then tiled 2×2 to show the seams)"),
            code('''
for i in TILEABLE:
    dst = f"{OUT}/tileable_{i}.png"
    if not os.path.exists(dst):
        cmd = (f"cd {SID_ROOT} && python scripts/sample_tileable.py --config configs/tileable/pixel_exact_flash_attn.yaml "
               f"image_path={IMG(i)} final_output_path={dst} {FP32} 2>&1 | tail -1")
        print(i)
        !{cmd}
    im = Image.open(dst).convert("RGB")
    t = Image.new("RGB", (2 * im.width, 2 * im.height))
    for a in range(2):
        for b in range(2):
            t.paste(im, (a * im.width, b * im.height))
    t.save(f"{OUT}/tileable_{i}_2x2.png")
'''),
            code('''
import glob
from IPython.display import display
for f in sorted(glob.glob(f"{OUT}/retarget_*.png")) + sorted(glob.glob(f"{OUT}/tileable_*_2x2.png")):
    print(os.path.basename(f)); display(Image.open(f).convert("RGB").copy().resize((min(600, Image.open(f).width), int(Image.open(f).height * min(600, Image.open(f).width) / Image.open(f).width))))
'''),
            code(FREEZE),
        ]

    return nb_sinddm_train, nb_lora_train, nb_evaluate, nb_apps
