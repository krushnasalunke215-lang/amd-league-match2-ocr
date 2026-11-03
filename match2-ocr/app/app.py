#!/usr/bin/env python3
"""Match 2 (OCR) entry point. The harness runs, once per image, in a NEW process:

    python3 /app/app.py --input-image /app/input/image_01.png

and reads /app/output/image_01_output.json -> {"text": "4KTR902", "confidence": 0.94}.

This is a thin client: the model stays loaded in server.py (started by the
container CMD). We wait for the server if it is still loading, send the image
path, and write the answer atomically. Whatever goes wrong (missing or
unreadable image, server down, timeout, any exception) we still write
{"text": "", "confidence": 0.0}, log to stderr, and exit 0, all within
MC2_DEADLINE_S (default 27 s, under the 30 s per-image limit).

Environment: MC2_OUTPUT_DIR (default /app/output), MC2_SOCKET, MC2_DEADLINE_S.
"""

from __future__ import annotations

import time

T0 = time.monotonic()

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
for _root in (HERE, HERE.parent, HERE.parent.parent):  # /app in the image, the repo root locally
    if (_root / "common").is_dir():
        sys.path.insert(0, str(_root))
        break

from common import contract, resident  # noqa: E402

OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))
SOCKET = os.environ.get("MC2_SOCKET", "/tmp/mc2-ocr.sock")
DEADLINE_S = float(os.environ.get("MC2_DEADLINE_S", "27"))
FALLBACK = {"text": "", "confidence": 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-image", required=True, help="e.g. /app/input/image_01.png")
    args = ap.parse_args()

    out = contract.output_path_for(args.input_image, OUTPUT_DIR)
    with contract.always_write(out, FALLBACK) as res:
        image = Path(args.input_image).resolve()
        with open(image, "rb") as f:  # missing or unreadable -> exception -> fallback output
            f.read(1)
        resp = resident.request(SOCKET, {"op": "ocr", "image": str(image)}, deadline=T0 + DEADLINE_S)
        if not resp.get("ok"):
            raise RuntimeError(f"server error: {resp.get('error')}")
        res.set(contract.ocr_payload(resp.get("text", ""), resp.get("confidence")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
