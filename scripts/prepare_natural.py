"""Phase 3: fetch the paper's natural-image data, verify it, and write manifests + descriptors.

Uses the official download script (pinned checkout), then copies the images into our layout:

    data/natural/<set_name>/<id>.png               images (gitignored: third-party, non-commercial)
    data/manifests/<set_name>.csv                  id, path, size, sha256, source, licence, descriptors

The manifest is committed; the sha256 column lets any machine (Colab included) check that it has
exactly the same images.

    python scripts/prepare_natural.py                          # sinddm15 + the small figure sets
    python scripts/prepare_natural.py --sets uncond/sinddm15 --sid-root /content/efficient-SID-code
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sidx.core.sampler import load_image  # noqa: E402
from sidx.data.descriptors import compute_descriptors  # noqa: E402

DEFAULT_SETS = ["uncond/sinddm15", "retarget/small", "symmetric/small", "tileable/small",
                "structural_analogy/small", "text_style/small"]

#: provenance per set, from the official datasets/README.md (checked 2026-11-13)
PROVENANCE = {
    "uncond/sinddm15": ("SinDDM repo datasets/ (Kulikov et al. 2023); several from SinGAN Input/Images",
                        "MIT (repositories); photo rights undocumented; research use"),
    "retarget/small": ("classic retargeting test images (SinGAN/InGAN/GPNN codebases)",
                       "repository licences; photo rights undocumented; research use"),
    "symmetric/small": ("efficient-SID release data-v1", "see efficient-SID datasets/README.md"),
    "tileable/small": ("efficient-SID release data-v1", "see efficient-SID datasets/README.md"),
    "structural_analogy/small": ("efficient-SID release data-v1 (GPDM repo images)",
                                 "see efficient-SID datasets/README.md"),
    "text_style/small": ("efficient-SID release data-v1", "see efficient-SID datasets/README.md"),
}
IMAGE_EXT = {".png", ".jpg", ".jpeg"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS)
    ap.add_argument("--sid-root", default=str(ROOT / "external" / "efficient-SID-code"))
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--no-descriptors", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="check the images against the committed manifests (sha256) instead of rewriting them")
    args = ap.parse_args()

    sid, data = Path(args.sid_root), Path(args.data_root)
    for s in args.sets:
        subprocess.run([sys.executable, str(sid / "datasets" / "download_data.py"), s], cwd=sid, check=True,
                       stdout=subprocess.DEVNULL)
        src_dir = sid / "datasets" / s
        set_name = s.replace("/", "_")
        dst_dir = data / "natural" / set_name
        out = data / "manifests" / f"{set_name}.csv"
        expected = {}
        if args.verify:
            with open(out, newline="", encoding="utf-8") as f:
                expected = {r["id"]: r["sha256"] for r in csv.DictReader(f)}
        rows = []
        for src in sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT):
            rel = src.relative_to(src_dir)
            image_id = "_".join(rel.with_suffix("").parts)
            dst = dst_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            img = load_image(str(dst))
            if args.verify:
                got = sha256(dst)
                if expected.get(image_id) != got:
                    raise SystemExit(f"MISMATCH {set_name}/{image_id}: sha256 {got} != manifest {expected.get(image_id)}")
                rows.append({"id": image_id})
                continue
            # path relative to the manifest file (that is how the job runner resolves it)
            row = {"id": image_id, "set": set_name,
                   "path": Path(os.path.relpath(dst, out.parent)).as_posix(),
                   "height": img.shape[1], "width": img.shape[2], "sha256": sha256(dst),
                   "source": PROVENANCE.get(s, ("efficient-SID release data-v1", ""))[0],
                   "licence": PROVENANCE.get(s, ("", "see efficient-SID datasets/README.md"))[1]}
            if not args.no_descriptors:
                row.update(compute_descriptors(img))
            rows.append(row)
        if args.verify:
            missing = set(expected) - {r["id"] for r in rows}
            if missing:
                raise SystemExit(f"MISSING in {set_name}: {sorted(missing)}")
            print(f"{s}: {len(rows)} images verified against {out.relative_to(ROOT)}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        sizes = sorted({(r["height"], r["width"]) for r in rows})
        print(f"{s}: {len(rows)} images -> {out.relative_to(ROOT)}; sizes {sizes[0]}..{sizes[-1]}")


if __name__ == "__main__":
    main()
