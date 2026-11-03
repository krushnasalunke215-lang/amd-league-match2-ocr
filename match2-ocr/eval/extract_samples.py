#!/usr/bin/env python3
"""Extract the 10 Match 2 sample images from the challenge brief PDF.

    python extract_samples.py ["path/to/AMD AI League — Match 1 & 2 Challenge Brief.pdf"]

Writes (all gitignored, regenerate any time):
  eval/samples/sample_01.png ... sample_10.tiff   each in the format the brief states
  eval/samples/expected.json                      filename -> expected text
  eval/samples_allformats/sample_NN_{png,jpg,tiff}.<ext> + expected.json
      every sample in all three formats (distinct stems, so outputs never collide)

The images are JPEG streams inside the PDF; they are decoded and re-encoded in
the stated format. The page/index -> answer mapping below was checked by viewing
every extracted image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
DEFAULT_PDF = HERE.parents[2] / "AMD AI League — Match 1 & 2 Challenge Brief.pdf"

# (1-based page, image index on that page, expected size, format, expected answer)
SAMPLES = [
    (9, 0, (690, 400), "PNG", "7ABC123"),
    (10, 0, (750, 310), "PNG", "京A·12345"),
    (10, 1, (690, 400), "JPEG", "JHT 2951"),
    (10, 2, (690, 400), "PNG", "5XYZ891"),
    (11, 0, (750, 310), "JPEG", "沪B·88888"),
    (11, 1, (570, 570), "PNG", "STOP"),
    (11, 2, (570, 570), "TIFF", "STOP"),
    (12, 0, (520, 620), "JPEG", "SPEED LIMIT 65"),
    (12, 1, (610, 610), "PNG", "ROAD WORK AHEAD"),
    (13, 0, (480, 480), "TIFF", "35"),
]
EXT = {"PNG": "png", "JPEG": "jpg", "TIFF": "tiff"}


def save(img, path: Path, fmt: str) -> None:
    img = img.convert("RGB")
    if fmt == "JPEG":
        img.save(path, "JPEG", quality=95)
    elif fmt == "TIFF":
        img.save(path, "TIFF", compression="tiff_lzw")
    else:
        img.save(path, "PNG")


def write_expected(folder: Path, expected: dict) -> None:
    (folder / "expected.json").write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF, help="the Match 1 & 2 brief PDF")
    args = ap.parse_args()
    if not args.pdf.is_file():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    reader = PdfReader(str(args.pdf))
    samples_dir, all_dir = HERE / "samples", HERE / "samples_allformats"
    samples_dir.mkdir(exist_ok=True)
    all_dir.mkdir(exist_ok=True)
    expected, expected_all = {}, {}

    for n, (page, index, size, fmt, answer) in enumerate(SAMPLES, start=1):
        img = reader.pages[page - 1].images[index].image
        if img.size != size:
            print(f"sample {n}: page {page} image {index} is {img.size}, expected {size}; "
                  "the PDF layout changed, re-check the mapping", file=sys.stderr)
            return 1
        name = f"sample_{n:02d}.{EXT[fmt]}"
        save(img, samples_dir / name, fmt)
        expected[name] = answer
        for f, ext in EXT.items():
            all_name = f"sample_{n:02d}_{ext}.{ext}"
            save(img, all_dir / all_name, f)
            expected_all[all_name] = answer
        print(f"{name:<16} {fmt:<5} {img.size[0]}x{img.size[1]}  {answer}")

    write_expected(samples_dir, expected)
    write_expected(all_dir, expected_all)
    print(f"wrote {len(expected)} images to {samples_dir} and {len(expected_all)} to {all_dir}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.exit(main())
