#!/usr/bin/env python3
"""Pack the committed repo plus the Match 2 eval images into one zip for the notebook or VM.

    python tools/make_bundle.py [out.zip]        (default: ../amd-ai-league.zip)

Contents: `git archive HEAD` (so only committed files) under amd-ai-league/, plus
match2-ocr/eval/{samples,samples_allformats,degraded}, which are gitignored. Unpack with
`python3 -m zipfile -e amd-ai-league.zip <dir>`. No image reference or secrets are included.
"""
from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIRS = ["match2-ocr/eval/samples", "match2-ocr/eval/samples_allformats", "match2-ocr/eval/degraded"]


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "amd-ai-league.zip"
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        print("note: uncommitted changes are NOT included (bundle = HEAD)", file=sys.stderr)
    archive = subprocess.run(["git", "archive", "--format=zip", "--prefix=amd-ai-league/", "HEAD"],
                             cwd=ROOT, capture_output=True, check=True).stdout
    n_img = 0
    with zipfile.ZipFile(io.BytesIO(archive)) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            dst.writestr(item, src.read(item))
        for d in EVAL_DIRS:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file():
                    dst.write(p, "amd-ai-league/" + p.relative_to(ROOT).as_posix())
                    n_img += 1
    if n_img == 0:
        print("warning: no eval images; run match2-ocr/eval/extract_samples.py and degrade.py first", file=sys.stderr)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): repo at {head} + {n_img} eval files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
