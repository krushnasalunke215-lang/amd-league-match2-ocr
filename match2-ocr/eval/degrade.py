#!/usr/bin/env python3
"""Make harder variants of the samples, like the graded set's adverse conditions.

    python degrade.py [--seed 0]

Reads eval/samples/ (run extract_samples.py first) and writes
eval/degraded/sample_NN__<variant>.<ext> plus expected.json. Output formats
rotate through PNG/JPEG/TIFF so every format meets every kind of damage.
Deterministic for a given seed. Pillow + numpy.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
FORMATS = [("png", "PNG"), ("jpg", "JPEG"), ("tiff", "TIFF")]


def _arr(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def gblur(img, rng):
    return img.filter(ImageFilter.GaussianBlur(radius=3))


def motion(img, rng, length=11):
    a = _arr(img)
    out = np.zeros_like(a)
    for dx in range(-(length // 2), length // 2 + 1):
        out += np.roll(a, dx, axis=1)
    return _img(out / length)


def noise(img, rng, sigma=40):
    a = _arr(img)
    return _img(a + np.random.default_rng(rng.randrange(1 << 30)).normal(0, sigma, a.shape))


def glare(img, rng, strength=0.75):
    w, h = img.size
    cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - cx, yy - cy) / (0.45 * max(w, h))
    alpha = (strength * np.clip(1 - r, 0, 1) ** 1.5)[..., None]
    return _img(_arr(img) * (1 - alpha) + 255 * alpha)


def lowlight(img, rng):
    a = (_arr(img) / 255.0) ** 2.0 * 0.35 * 255
    return _img(a + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 4, a.shape))


def skew(img, rng, k=0.18):
    # Map a quadrilateral of the source onto the output rectangle: an off-axis view.
    w, h = img.size
    d = k * w
    quad = (-d * rng.random(), 0, 0, h, w, h + d * 0.5, w + d * rng.random(), -d)
    return img.convert("RGB").transform((w, h), Image.QUAD, quad, Image.BICUBIC, fillcolor=(90, 90, 90))


def small(img, rng, long_side=110):
    w, h = img.size
    s = long_side / max(w, h)
    return img.convert("RGB").resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)


def huge(img, rng, long_side=5000):
    w, h = img.size
    s = long_side / max(w, h)
    return img.convert("RGB").resize((round(w * s), round(h * s)), Image.BICUBIC)


VARIANTS = {
    "gblur": gblur, "motion": motion, "noise": noise, "glare": glare,
    "lowlight": lowlight, "skew": skew, "small": small, "huge": huge,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_dir, out_dir = HERE / "samples", HERE / "degraded"
    exp_file = src_dir / "expected.json"
    if not exp_file.is_file():
        print(f"{exp_file} not found; run extract_samples.py first", file=sys.stderr)
        return 1
    expected = json.loads(exp_file.read_text(encoding="utf-8"))
    out_dir.mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    result = {}
    for i, (name, answer) in enumerate(sorted(expected.items())):
        src = Image.open(src_dir / name)
        src.load()
        for j, (vname, fn) in enumerate(VARIANTS.items()):
            ext, fmt = FORMATS[(i + j) % 3]
            out_name = f"{Path(name).stem}__{vname}.{ext}"
            img = fn(src, rng).convert("RGB")
            if fmt == "JPEG":
                img.save(out_dir / out_name, fmt, quality=85)
            elif fmt == "TIFF":
                img.save(out_dir / out_name, fmt, compression="tiff_lzw")
            else:
                img.save(out_dir / out_name, fmt)
            result[out_name] = answer
    (out_dir / "expected.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(result)} images ({len(expected)} samples x {len(VARIANTS)} variants: "
          f"{', '.join(VARIANTS)}) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
