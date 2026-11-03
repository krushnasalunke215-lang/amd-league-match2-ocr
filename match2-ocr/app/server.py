#!/usr/bin/env python3
"""Resident OCR server. The container CMD starts it once; app.py talks to it.

    python3 /app/server.py            # load model, warm up, then serve on MC2_SOCKET
    python3 /app/server.py --stats    # client mode: print the running server's stats JSON
    python3 /app/server.py --ping     # client mode: exit 0 if the server is ready (Docker HEALTHCHECK)

Loads the backend and runs one warm-up inference BEFORE binding the socket, so
"socket accepts connections" means "ready". If loading fails it writes
MC2_SOCKET + ".failed" (clients then give up at once) and exits; the container
CMD keeps the container alive regardless.

Environment:
  OCR_BACKEND        vlm (default) | fake  (fake: no torch, canned text, laptop contract tests)
  OCR_MODEL_DIR      model snapshot dir (the Dockerfile sets it from MODEL_ID)
  OCR_MAX_NEW_TOKENS default 48
  OCR_SECOND_PASS    1 = also read a contrast-enhanced copy and vote (default off)
  OCR_FAKE_FAIL      load | request: make the fake backend fail, to test failure paths
  MC2_SOCKET         default /tmp/mc2-ocr.sock
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _root in (HERE, HERE.parent, HERE.parent.parent):  # /app in the image, the repo root locally
    if (_root / "common").is_dir():
        sys.path.insert(0, str(_root))
        break
sys.path.insert(0, str(HERE))

from common import resident  # noqa: E402

SOCKET = os.environ.get("MC2_SOCKET", "/tmp/mc2-ocr.sock")
BACKEND = os.environ.get("OCR_BACKEND", "vlm").lower()
MODEL_DIR = os.environ.get("OCR_MODEL_DIR", "/models/Qwen/Qwen2.5-VL-7B-Instruct")


def log(msg: str) -> None:
    print(f"[server {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class FakeBackend:
    """Deterministic stand-in with the same interface as ocr.VlmBackend. Never returns sample answers."""

    def __init__(self) -> None:
        self.fail = os.environ.get("OCR_FAKE_FAIL", "")

    def load(self) -> None:
        if self.fail == "load":
            raise RuntimeError("OCR_FAKE_FAIL=load")

    def read(self, path: str) -> tuple[str, float]:
        if self.fail == "request":
            raise RuntimeError("OCR_FAKE_FAIL=request")
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        return "FAKE", 0.5


def make_backend():
    if BACKEND == "fake":
        return FakeBackend()
    if BACKEND != "vlm":
        raise ValueError(f"unknown OCR_BACKEND={BACKEND!r} (use vlm or fake)")
    import ocr

    return ocr.VlmBackend(
        MODEL_DIR,
        max_new_tokens=int(os.environ.get("OCR_MAX_NEW_TOKENS", "48")),
        second_pass=os.environ.get("OCR_SECOND_PASS", "0") == "1",
    )


def gpu_stats() -> dict:
    if BACKEND != "vlm":
        return {}
    from common import gpu

    return {"device": gpu.device_info(), "vram": gpu.vram_report()}


def main() -> int:
    if sys.argv[1:2] == ["--ping"]:
        return 0 if resident.wait_ready(SOCKET, 3.0) else 1
    if sys.argv[1:2] == ["--stats"]:
        print(json.dumps(resident.request(SOCKET, {"op": "stats"}, deadline=time.monotonic() + 10), indent=2))
        return 0

    resident.clear_markers(SOCKET)
    log(f"backend={BACKEND} model_dir={MODEL_DIR} socket={SOCKET}")
    t0 = time.monotonic()
    try:
        backend = make_backend()
        backend.load()  # includes the warm-up inference for the vlm backend
    except Exception:
        err = traceback.format_exc()
        log("model load FAILED:\n" + err)
        resident.mark_failed(SOCKET, err)
        return 1
    load_s = round(time.monotonic() - t0, 1)
    log(f"loaded and warmed up in {load_s}s {json.dumps(gpu_stats())}")
    started = time.time()
    count = {"ocr": 0, "errors": 0}

    def handler(req: dict) -> dict:
        op = req.get("op")
        if op == "ocr":
            t = time.monotonic()
            try:
                text, conf = backend.read(str(req["image"]))
            except Exception as e:
                count["errors"] += 1
                log(f"ocr {req.get('image')!r} failed: {e!r}")
                raise
            count["ocr"] += 1
            ms = round(1000 * (time.monotonic() - t))
            log(f"ocr {Path(str(req['image'])).name} -> {text!r} conf={conf} {ms}ms")
            return {"ok": True, "text": text, "confidence": conf, "ms": ms}
        if op == "stats":
            return {"ok": True, "backend": BACKEND, "model_dir": MODEL_DIR, "load_s": load_s,
                    "uptime_s": round(time.time() - started), "requests": count, **gpu_stats()}
        raise ValueError(f"unknown op {op!r}")

    resident.serve(SOCKET, handler)


if __name__ == "__main__":
    sys.exit(main())
