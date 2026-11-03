#!/usr/bin/env python3
"""Write synthetic text images for contract tests (CI, laptop). Never the brief's samples.

    python make_synthetic.py [--out DIR] [--seed 0]

Default DIR is eval/synthetic/ (gitignored). Writes plate-like random text on plain
backgrounds in every accepted extension (.png, .jpg, .jpeg, .tif, .tiff), in RGB,
grayscale and RGBA, including one huge (5000 px) and one tiny (110 px) image, plus
expected.json (filename -> text), so run_local.py and selfcheck_mc2.py can use the folder.
File stems are unique because the output name is <stem>_output.json. Deterministic for a
given seed. Pillow and the standard library only, so it also runs inside the built image.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"
DIGITS = "0123456789"

# (file name, size, mode, background, ink, save options)
IMAGES = [
    ("syn_01.png", (520, 150), "RGB", (255, 255, 255), (20, 20, 20), {}),
    ("syn_02.jpg", (640, 200), "RGB", (235, 235, 230), (10, 10, 60), {"quality": 90}),
    ("syn_03.jpeg", (480, 160), "RGB", (250, 205, 40), (15, 15, 15), {"quality": 70}),
    ("syn_04.tif", (520, 150), "RGB", (255, 255, 255), (150, 20, 20), {"compression": "tiff_lzw"}),
    ("syn_05_gray.tiff", (600, 180), "L", (240, 240, 240), (30, 30, 30), {}),
    ("syn_06_rgba.png", (520, 150), "RGBA", None, (20, 20, 20), {}),
    ("syn_07_large.jpg", (5000, 1440), "RGB", (255, 255, 255), (20, 20, 20), {"quality": 85}),
    ("syn_08_small.tif", (110, 32), "RGB", (255, 255, 255), (20, 20, 20), {"compression": "tiff_lzw"}),
]
FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".tif": "TIFF", ".tiff": "TIFF"}


def plate_text(rng: random.Random) -> str:
    return "".join(rng.choice(LETTERS) for _ in range(3)) + " " + "".join(rng.choice(DIGITS) for _ in range(4))


def font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()  # small bitmap font; glyphs() scales it up


def glyphs(text: str, size: tuple[int, int]) -> Image.Image:
    """The text as an L mask scaled to fill most of an image of `size`."""
    w, h = size
    f = font(max(8, int(h * 0.55)))
    left, top, right, bottom = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=f)
    mask = Image.new("L", (right - left + 8, bottom - top + 8), 0)
    ImageDraw.Draw(mask).text((4 - left, 4 - top), text, font=f, fill=255)
    scale = min(0.85 * w / mask.width, 0.65 * h / mask.height)
    return mask.resize((max(1, round(mask.width * scale)), max(1, round(mask.height * scale))), Image.LANCZOS)


def render(text: str, size: tuple[int, int], mode: str, bg, ink) -> Image.Image:
    w, h = size
    mask = glyphs(text, size)
    x, y = (w - mask.width) // 2, (h - mask.height) // 2
    box = (x, y, x + mask.width, y + mask.height)
    if mode == "RGBA":  # transparent background, opaque text
        img = Image.new("RGBA", size, (255, 255, 255, 0))
        img.paste(ink + (255,), box, mask)
        return img
    img = Image.new("RGB", size, bg)
    ImageDraw.Draw(img).rectangle((2, 2, w - 3, h - 3), outline=ink, width=max(1, h // 40))
    img.paste(ink, box, mask)
    return img.convert(mode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "synthetic", help="output folder (default eval/synthetic)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    expected = {}
    for name, size, mode, bg, ink, opts in IMAGES:
        text = plate_text(rng)
        render(text, size, mode, bg, ink).save(args.out / name, FORMATS[Path(name).suffix], **opts)
        expected[name] = text
    (args.out / "expected.json").write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(expected)} synthetic images (PNG, JPEG, TIFF; RGB, L, RGBA; 5000 px and 110 px) "
          f"+ expected.json to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
