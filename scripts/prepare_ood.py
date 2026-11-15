"""Phase 6: Set A — real out-of-distribution images from Wikimedia Commons, with licences.

Two steps, so a human chooses the images (plan §4.3: "curated by hand"):

  1. candidates:  python scripts/prepare_ood.py candidates
     Lists freely licensed files (PD / CC0 / CC BY / CC BY-SA) in each category below, downloads
     small previews and writes a numbered contact sheet per category to data/ood_candidates/.
  2. build:       python scripts/prepare_ood.py build
     Downloads the files chosen in configs/data/ood_setA.yaml (by Commons title), renders them at
     a fixed width (Commons rasterises SVGs, so line widths are controlled), converts to RGB on
     white, crops/resizes to 256 px, and writes data/ood/setA/*.png + data/manifests/ood_setA.csv
     (licence, author and source URL per image). `--verify` checks sha256 instead of rewriting.

Why Commons instead of FloorPlanCAD / DTD / AI2D (plan §4.2): one API with per-file licence
metadata, SVG rendering at any width, and a few MB of downloads (DTD is a 600 MB archive; the
FloorPlanCAD site is shut down). The image types are the same.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

API = "https://commons.wikimedia.org/w/api.php"
UA = "CS787-course-project-dataset-builder/0.1 (academic research; python-urllib)"
FREE = re.compile(r"^(public domain|pd|cc0|cc[ -]by(-sa)?([ -]\d\.\d)?)", re.I)

#: category -> Commons categories to draw candidates from
CATEGORIES = {
    "floorplan": ["Category:SVG floor plans", "Category:Floor plans of houses"],
    "pattern": ["Category:SVG tessellations", "Category:Islamic geometric patterns", "Category:Tilings"],
    "circuit": ["Category:Electronic circuit diagrams in SVG", "Category:Circuit diagrams"],
    "diagram": ["Category:SVG biology diagrams", "Category:Labelled diagrams"],
}


_LAST = [0.0]
MIN_INTERVAL_S = 1.5          # be polite: at most ~40 requests/minute


def _throttled_open(req, timeout):
    """urlopen with a minimum spacing between requests and Retry-After handling for HTTP 429."""
    for attempt in range(8):
        wait = MIN_INTERVAL_S - (time.time() - _LAST[0])
        if wait > 0:
            time.sleep(wait)
        _LAST[0] = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                delay = float(e.headers.get("retry-after", 20)) + 2 + 5 * attempt
                print(f"[commons] HTTP {e.code}; waiting {delay:.0f}s", flush=True)
                time.sleep(delay)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"request failed after retries: {req.full_url}")


def api(params: dict) -> dict:
    # POST: title batches with accents / apostrophes make GET URLs too long or mangled
    body = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2}).encode("utf-8")
    return json.loads(_throttled_open(urllib.request.Request(API, data=body, headers={"User-Agent": UA}), 60))


def fetch(url: str) -> bytes:
    return _throttled_open(urllib.request.Request(url, headers={"User-Agent": UA}), 120)


def image_info(titles, width: int) -> list:
    out = []
    for i in range(0, len(titles), 20):
        res = api({"action": "query", "titles": "|".join(titles[i:i + 20]), "prop": "imageinfo",
                   "iiprop": "url|size|mime|extmetadata", "iiurlwidth": width})
        for p in res["query"]["pages"]:
            if "imageinfo" not in p:
                continue
            ii = p["imageinfo"][0]
            em = ii.get("extmetadata", {})
            lic = em.get("LicenseShortName", {}).get("value", "")
            artist = re.sub("<[^>]+>", "", em.get("Artist", {}).get("value", "")).strip()
            out.append({"title": p["title"], "licence": lic, "artist": artist[:200],
                        "mime": ii.get("mime"), "width": ii.get("width"), "height": ii.get("height"),
                        "thumb": ii.get("thumburl"), "page": ii.get("descriptionurl")})
    return out


def category_files(cat: str, limit: int = 200, depth: int = 1) -> list:
    """Files of a category and (depth levels of) its subcategories; SVG/PNG only."""
    res = api({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmtype": "file|subcat",
               "cmlimit": limit})
    files, subs = [], []
    for m in res["query"]["categorymembers"]:
        (subs if m["title"].startswith("Category:") else files).append(m["title"])
    if depth > 0:
        for s in subs[:6]:
            try:
                files += category_files(s, 40, depth - 1)
            except RuntimeError:
                pass
    return [f for f in files if f.lower().endswith((".svg", ".png"))]


def to_rgb_on_white(data: bytes):
    from PIL import Image

    im = Image.open(io.BytesIO(data))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def candidates(args) -> None:
    from PIL import Image, ImageDraw

    out = ROOT / "data" / "ood_candidates"
    out.mkdir(parents=True, exist_ok=True)
    for name, cats in CATEGORIES.items():
        titles = []
        for c in cats:
            try:
                titles += category_files(c)
            except RuntimeError as e:
                print(e)
        infos = [i for i in image_info(sorted(set(titles)), 160)
                 if FREE.match(i["licence"] or "") and i["mime"] in ("image/svg+xml", "image/png")
                 and min(i["width"] or 0, i["height"] or 0) >= 200]
        infos = infos[: args.max_per_category]
        sheet = Image.new("RGB", (8 * 170, ((len(infos) + 7) // 8) * 190 or 190), "white")
        d = ImageDraw.Draw(sheet)
        for k, inf in enumerate(infos):
            try:
                im = to_rgb_on_white(fetch(inf["thumb"]))
                im.thumbnail((160, 160))
                sheet.paste(im, ((k % 8) * 170, (k // 8) * 190))
            except Exception as e:
                print("preview failed", inf["title"], e)
            d.text(((k % 8) * 170, (k // 8) * 190 + 165), str(k), fill=(255, 0, 0))
        sheet.save(out / f"{name}.png")
        (out / f"{name}.json").write_text(json.dumps(infos, indent=1))
        print(f"{name}: {len(infos)} free candidates -> {out / (name + '.png')}")


def crop_or_resize(im, mode: str, size: int = 256, crop=None):
    """'fit': resize so the long side is `size`. 'crop': resize so the long side is 2*size, then
    take a size x size crop at `crop` (fractions of the free range) or at the densest window."""
    from PIL import Image
    import numpy as np

    W, H = im.size
    if mode == "fit":
        s = size / max(W, H)
        return im.resize((max(1, round(W * s)), max(1, round(H * s))), Image.LANCZOS), None
    s = 2 * size / max(W, H)
    im = im.resize((max(size, round(W * s)), max(size, round(H * s))), Image.LANCZOS)
    W, H = im.size
    if crop is None:
        g = np.asarray(im.convert("L"), dtype=np.float32) / 255
        ink = (g < 0.6).astype(np.float32)
        best, crop_xy = -1, (0, 0)
        for y in range(0, H - size + 1, 16):
            for x in range(0, W - size + 1, 16):
                v = ink[y:y + size, x:x + size].mean()
                score = -abs(v - 0.12)                    # prefer moderate ink coverage
                if score > best:
                    best, crop_xy = score, (x, y)
    else:
        crop_xy = (round(crop[0] * (W - size)), round(crop[1] * (H - size)))
    x, y = crop_xy
    return im.crop((x, y, x + size, y + size)), crop_xy


def build(args) -> None:
    import yaml

    from sidx.core.sampler import load_image
    from sidx.data.descriptors import compute_descriptors

    spec = yaml.safe_load(open(ROOT / "configs" / "data" / "ood_setA.yaml", encoding="utf-8"))
    out_dir = ROOT / "data" / "ood" / "setA"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "data" / "manifests" / "ood_setA.csv"
    expected = {}
    if args.verify:
        with open(manifest, newline="", encoding="utf-8") as f:
            expected = {r["id"]: r["sha256"] for r in csv.DictReader(f)}
    rows = []
    for item in spec["images"]:
        info = image_info([item["title"]], item.get("render_width", 1024))[0]
        if not FREE.match(info["licence"] or ""):
            raise SystemExit(f"{item['title']}: licence {info['licence']!r} is not in the allowed set")
        im = to_rgb_on_white(fetch(info["thumb"]))
        im, crop_xy = crop_or_resize(im, item.get("mode", "fit"), crop=item.get("crop"))
        p = out_dir / f"{item['id']}.png"
        im.save(p)
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if args.verify:
            if expected.get(item["id"]) != sha:
                raise SystemExit(f"MISMATCH {item['id']}: {sha} != {expected.get(item['id'])}")
            continue
        row = {"id": item["id"], "set": "ood_setA", "category": item["category"],
               "path": Path(os.path.relpath(p, manifest.parent)).as_posix(),
               "height": im.size[1], "width": im.size[0], "sha256": sha,
               "source": info["page"], "commons_title": item["title"], "licence": info["licence"],
               "author": info["artist"], "render_width": item.get("render_width", 1024),
               "mode": item.get("mode", "fit"), "crop_xy": "" if crop_xy is None else f"{crop_xy[0]};{crop_xy[1]}"}
        row.update(compute_descriptors(load_image(str(p))))
        rows.append(row)
        print(f"{item['id']}: {im.size} {info['licence']}")
    if args.verify:
        print(f"{len(spec['images'])} images verified")
        return
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} images -> {manifest.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["candidates", "build"])
    ap.add_argument("--max-per-category", type=int, default=32)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    candidates(args) if args.step == "candidates" else build(args)


if __name__ == "__main__":
    main()
