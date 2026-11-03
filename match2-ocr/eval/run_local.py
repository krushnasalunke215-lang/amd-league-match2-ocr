#!/usr/bin/env python3
"""Run the Match 2 pipeline the way the harness does, without Docker (AMD notebook or laptop).

    python run_local.py --images eval/samples --model-dir $PERSIST/models/Qwen__Qwen2.5-VL-7B-Instruct
    python run_local.py --images eval/samples_allformats --backend fake     # laptop, no torch

Starts server.py once (logging to <out>/server.log), waits until it is ready,
then runs app.py as a NEW PROCESS per image, exactly like the harness. Prints
each call's time (flagging any over 30 s), the totals against the 10 minute
budget, peak VRAM (amd-smi/rocm-smi sampled every second, plus torch's own
numbers from the server; best effort), and the score from score.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

EVAL = Path(__file__).resolve().parent
APP = EVAL.parent / "app"
sys.path.insert(0, str(EVAL.parents[1]))  # repo root, for common/
sys.path.insert(0, str(EVAL))

from common import gpu, resident  # noqa: E402
import score  # noqa: E402

PER_IMAGE_S = 30
TOTAL_S = 600
VRAM_FLOOR_GIB, VRAM_CEILING_GIB = 1, 48 * 1.01


class VramSampler(threading.Thread):
    def __init__(self, every_s: float = 1.0):
        super().__init__(daemon=True)
        self.every_s, self.peak, self.samples = every_s, None, 0
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            v = gpu.smi_vram_used_gib()
            if v is None and self.samples == 0:
                return  # no amd-smi / rocm-smi here (laptop)
            if v is not None:
                self.samples += 1
                self.peak = v if self.peak is None else max(self.peak, v)
            self.stop.wait(self.every_s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=EVAL / "samples", help="folder with images + expected.json")
    ap.add_argument("--out", type=Path, help="output folder (default eval/outputs/<images folder name>)")
    ap.add_argument("--backend", default=os.environ.get("OCR_BACKEND", "vlm"), choices=["vlm", "fake"])
    ap.add_argument("--model-dir", help="sets OCR_MODEL_DIR for the server")
    ap.add_argument("--python", default=sys.executable, help="interpreter for server.py and app.py")
    ap.add_argument("--server-running", action="store_true", help="use an already running server.py")
    ap.add_argument("--startup-timeout", type=float, default=600)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="backslashreplace")

    expected = json.loads((args.images / "expected.json").read_text(encoding="utf-8"))
    out = (args.out or EVAL / "outputs" / args.images.name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*_output.json"):
        old.unlink()

    default_sock = "/tmp/mc2-ocr.sock" if resident.USE_UNIX else str(Path(tempfile.gettempdir()) / "mc2-ocr.sock")
    sock = os.environ.get("MC2_SOCKET", default_sock)
    env = dict(os.environ, MC2_SOCKET=sock, MC2_OUTPUT_DIR=str(out), OCR_BACKEND=args.backend,
               PYTHONIOENCODING="utf-8")
    if args.model_dir:
        env["OCR_MODEL_DIR"] = args.model_dir

    sampler = VramSampler()
    sampler.start()
    server = None
    try:
        if not args.server_running:
            log = open(out / "server.log", "w", encoding="utf-8")
            t0 = time.monotonic()
            server = subprocess.Popen([args.python, str(APP / "server.py")], env=env, stdout=log,
                                      stderr=subprocess.STDOUT)
            while True:
                if server.poll() is not None:
                    print(f"server exited with {server.returncode} before becoming ready; see {out / 'server.log'}")
                    return 1
                if resident.wait_ready(sock, 2.0):
                    break
                if resident.failed_marker(sock).exists() or time.monotonic() - t0 > args.startup_timeout:
                    print(f"server not ready (failed or > {args.startup_timeout:.0f}s); see {out / 'server.log'}")
                    return 1
            print(f"server ready in {time.monotonic() - t0:.1f}s (startup budget {TOTAL_S}s)")
        elif not resident.wait_ready(sock, 5.0):
            print(f"no server answering on {sock}")
            return 1

        times, slow = [], []
        for name in expected:
            image = args.images / name
            t = time.monotonic()
            try:
                p = subprocess.run([args.python, str(APP / "app.py"), "--input-image", str(image)], env=env,
                                   capture_output=True, text=True, encoding="utf-8", timeout=PER_IMAGE_S + 5)
                rc, err = p.returncode, p.stderr.strip().splitlines()[-1:] if p.stderr.strip() else []
            except subprocess.TimeoutExpired:
                rc, err = "timeout", []
            dt = time.monotonic() - t
            times.append(dt)
            flag = ""
            if dt > PER_IMAGE_S or rc == "timeout":
                slow.append(name)
                flag = "  OVER 30s"
            print(f"{dt:6.2f}s  rc={rc}  {name}{flag}{('  stderr: ' + err[0][:120]) if err else ''}")

        total = sum(times)
        print(f"images {len(times)}  total {total:.1f}s (budget {TOTAL_S}s)  max {max(times, default=0):.2f}s  "
              f"mean {total / max(1, len(times)):.2f}s  over {PER_IMAGE_S}s: {slow or 'none'}")
        try:
            stats = resident.request(sock, {"op": "stats"}, deadline=time.monotonic() + 10)
            print("server stats:", json.dumps({k: v for k, v in stats.items() if k != "ok"}))
        except Exception as e:  # noqa: BLE001
            print(f"server stats unavailable: {e!r}")
        sampler.stop.set()
        if sampler.peak is not None:
            inside = VRAM_FLOOR_GIB <= sampler.peak <= VRAM_CEILING_GIB
            print(f"peak VRAM (smi, device-wide): {sampler.peak:.2f} GiB over {sampler.samples} samples "
                  f"{'OK' if inside else 'OUTSIDE 1-48 GiB'}")
        else:
            print("peak VRAM (smi): unavailable (no amd-smi/rocm-smi)")
        score.score(expected, out)
        return 0
    finally:
        sampler.stop.set()
        if server is not None:
            server.terminate()
            try:
                server.wait(10)
            except subprocess.TimeoutExpired:
                server.kill()
            resident.clear_markers(sock)


if __name__ == "__main__":
    sys.exit(main())
