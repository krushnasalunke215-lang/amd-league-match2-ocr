#!/usr/bin/env python3
"""Check your container against the MC-2 (OCR) submission contract before you submit.

The Match 2 counterpart of the official mc3-starter-kit selfcheck.py, same
style. If any of these fails here it will fail in the harness, and there it
costs you the submission.

    python3 tools/selfcheck_mc2.py your-image:tag match2-ocr/eval/samples

The samples dir must contain at least one PNG, one JPEG and one TIFF. If it has
an expected.json (filename -> text), answers are also scored with the brief's
normalization. Run it on the GPU VM: it starts the container with the ROCm
device flags and --network none.

Without a GPU (GitHub Actions), --fake-backend runs the container with
OCR_BACKEND=fake and no --device flags. Every contract check still runs; the
VRAM bounds and scoring are skipped:

    python3 tools/selfcheck_mc2.py your-image:tag match2-ocr/eval/synthetic --fake-backend
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for common/
from common.gpu import smi_vram_used_gib  # noqa: E402  (stdlib only, no torch)

BASE_IMAGE = "rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0"
SCRIPT = "/app/app.py"
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
SOCKET = "/tmp/mc2-ocr.sock"
SIZE_LIMIT_B = 60 * 1024 ** 3
STARTUP_BUDGET_S = 600
PER_IMAGE_BUDGET_S = 30
TOTAL_BUDGET_S = 600
VRAM_FLOOR_GIB = 1
VRAM_CEILING_GIB = 48 * 1.01
GPU_FLAGS = ["--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "video"]
FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".tif": "TIFF", ".tiff": "TIFF"}


def run(args, timeout=120):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, check=False)


def ok(msg):
    print(f"  [PASS] {msg}")
    return True


def bad(msg):
    print(f"  [FAIL] {msg}")
    return False


def warn(msg):
    print(f"  [WARN] {msg}")


def skip(msg):
    print(f"  [SKIP] {msg}")


def fail(msg):
    """Report a failure that stops the check; returns the exit code 1."""
    bad(msg)
    print()
    print("SOME CHECKS FAILED - fix these before submitting")
    return 1


def normalize(s: str) -> str:
    """The brief's rule: uppercase, all whitespace removed, - . · _ removed."""
    return re.sub(r"\s+", "", str(s).upper()).translate(str.maketrans("", "", "-.\u00b7_"))


def history_bytes(out: str) -> int:
    """Sum `docker history --format {{.Size}}` lines like '12.3GB', '845kB', '0B'."""
    units = {"B": 1, "KB": 1000, "MB": 1000 ** 2, "GB": 1000 ** 3, "TB": 1000 ** 4}
    total = 0.0
    for line in out.split():
        m = re.fullmatch(r"([\d.]+)\s*([kKMGT]?B)", line.strip())
        if m:
            total += float(m.group(1)) * units[m.group(2).upper()]
    return int(total)


class VramSampler(threading.Thread):
    """Host-side, device-wide used VRAM every second; the harness samples every 3 s and keeps the peak."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak = None
        self.samples = 0
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            v = smi_vram_used_gib()
            if v is not None:
                self.samples += 1
                self.peak = v if self.peak is None else max(self.peak, v)
            elif self.samples == 0:
                return
            self.stop.wait(1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="your built image, e.g. mc2-local:v1 or user/repo:tag")
    ap.add_argument("samples", type=Path, help="a folder of sample images (PNG, JPEG and TIFF)")
    ap.add_argument("--no-gpu-flags", action="store_true", help="do not pass --device /dev/kfd /dev/dri")
    ap.add_argument("--cap-drop", action="store_true", help="also run with --cap-drop DAC_OVERRIDE")
    ap.add_argument("--keep", action="store_true", help="leave the container running afterwards")
    ap.add_argument("--fake-backend", action="store_true",
                    help="no GPU (CI): run with -e OCR_BACKEND=fake and no --device flags; "
                         "skips the VRAM bounds and scoring")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="backslashreplace")

    if not args.samples.is_dir():
        return fail(f"{args.samples} is not a directory")
    images = sorted(p for p in args.samples.iterdir() if p.suffix.lower() in FORMATS)
    have = {FORMATS[p.suffix.lower()] for p in images}
    if have != {"PNG", "JPEG", "TIFF"}:
        return fail(f"{args.samples} needs a PNG, a JPEG and a TIFF; found {sorted(have) or 'none'}")
    expected_file = args.samples / "expected.json"
    expected = json.loads(expected_file.read_text(encoding="utf-8")) if expected_file.is_file() else {}
    if args.fake_backend:
        expected = {}  # the fake backend answers FAKE for every image; nothing to score

    passed = True
    name = f"selfcheck-{uuid.uuid4().hex[:8]}"

    print("1. size, uncompressed, and the mandated base")
    try:
        hist = run(["docker", "history", "--no-trunc", "--format", "{{.Size}}", args.image])
    except OSError as e:
        return fail(f"could not read image history (is docker installed?): {e}")
    if hist.returncode != 0:
        return fail(f"could not read image history: {hist.stderr.strip()}")
    size_hist = history_bytes(hist.stdout)
    insp = run(["docker", "image", "inspect", "--format", "{{.Size}}", args.image])
    size_insp = int(insp.stdout.strip()) if insp.returncode == 0 and insp.stdout.strip().isdigit() else 0
    size = max(size_hist, size_insp)
    gib = size / 1024 ** 3
    if size > SIZE_LIMIT_B:
        passed = bad(f"{gib:.2f} GiB uncompressed is over the 60 GiB limit")
    else:
        ok(f"{gib:.2f} GiB uncompressed (history sum {size_hist / 1024 ** 3:.2f}, inspect {size_insp / 1024 ** 3:.2f}; "
           "limit 60 GiB)")
    layers = run(["docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", args.image])
    base = run(["docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", BASE_IMAGE])
    if base.returncode != 0:
        warn(f"base image not present locally, cannot compare layers: docker pull {BASE_IMAGE}")
    elif layers.returncode == 0:
        mine, theirs = json.loads(layers.stdout), json.loads(base.stdout)
        if mine[: len(theirs)] == theirs:
            ok(f"the lower {len(theirs)} layers are the mandated base's layers")
        else:
            passed = bad("lower layers do not match the mandated base (wrong FROM, or squashed/flattened)")

    print("2. torch and torchvision inside the image are the ROCm builds")
    tv = run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "python3", args.image,
              "-c", "import torch, torchvision; print(torch.__version__, torchvision.__version__)"], timeout=300)
    versions = tv.stdout.split()
    if tv.returncode == 0 and len(versions) == 2 and all("+rocm" in v for v in versions):
        ok(f"torch {versions[0]}, torchvision {versions[1]}")
    else:
        passed = bad(f"torch/torchvision not both ROCm builds: {tv.stdout.strip() or tv.stderr.strip()[-300:]}")

    print("3. the container starts WITH NO NETWORK, and the first image works straight away")
    cmd = ["docker", "run", "-d", "--network", "none", "--name", name]
    if not args.no_gpu_flags and not args.fake_backend:
        cmd += GPU_FLAGS
    if args.fake_backend:
        cmd += ["-e", "OCR_BACKEND=fake"]
    if args.cap_drop:
        cmd += ["--cap-drop", "DAC_OVERRIDE"]
    sampler = VramSampler()
    if not args.fake_backend:
        sampler.start()
    started = time.time()
    up = run(cmd + [args.image])
    if up.returncode != 0:
        sampler.stop.set()
        return fail(f"container did not start: {up.stderr.strip()}")

    try:
        run(["docker", "exec", name, "mkdir", "-p", INPUT_DIR, OUTPUT_DIR], timeout=60)
        first = images[0]
        run(["docker", "cp", str(first), f"{name}:{INPUT_DIR}/{first.name}"], timeout=120)
        t0 = time.time()
        try:
            proc = run(["docker", "exec", name, "python3", SCRIPT, "--input-image", f"{INPUT_DIR}/{first.name}"],
                       timeout=PER_IMAGE_BUDGET_S)
            dt = time.time() - t0
            if proc.returncode != 0:
                passed = bad(f"first call exited {proc.returncode}: {(proc.stderr or proc.stdout).strip()[-300:]}")
            else:
                ok(f"first call, {t0 - started:.1f}s after start, returned in {dt:.1f}s "
                   "(it must write an output file even while the model is still loading)")
        except subprocess.TimeoutExpired:
            passed = bad(f"first call did not return within {PER_IMAGE_BUDGET_S}s")

        print("4. the model server becomes ready inside the startup budget")
        ready = load_failed = False
        while time.time() - started < STARTUP_BUDGET_S:
            failed = run(["docker", "exec", name, "cat", SOCKET + ".failed"], timeout=30)
            if failed.returncode == 0:
                load_failed = True
                passed = bad("server reported a load failure:\n" + failed.stdout.strip()[-1500:])
                break
            st = run(["docker", "exec", name, "python3", "/app/server.py", "--stats"], timeout=30)
            if st.returncode == 0:
                ready = True
                break
            time.sleep(2)
        if ready:
            ok(f"ready {time.time() - started:.1f}s after start (budget {STARTUP_BUDGET_S}s)")
        elif not load_failed:
            passed = bad(f"server not ready within {STARTUP_BUDGET_S}s; see docker exec {name} cat /tmp/server.log")

        print("5. every image answers in time, in the right file, in the right shape")
        total, scored = 0.0, 0
        for img in images:
            cp = run(["docker", "cp", str(img), f"{name}:{INPUT_DIR}/{img.name}"], timeout=120)
            if cp.returncode != 0:
                passed = bad(f"could not copy {img.name} in: {cp.stderr.strip()}")
                continue
            t0 = time.time()
            try:
                proc = run(["docker", "exec", name, "python3", SCRIPT, "--input-image", f"{INPUT_DIR}/{img.name}"],
                           timeout=PER_IMAGE_BUDGET_S)
            except subprocess.TimeoutExpired:
                passed = bad(f"{img.name}: no answer within the {PER_IMAGE_BUDGET_S}s per-image budget")
                total += PER_IMAGE_BUDGET_S
                continue
            dt = time.time() - t0
            total += dt
            if proc.returncode != 0:
                passed = bad(f"{img.name}: {SCRIPT} exited {proc.returncode}: "
                             f"{(proc.stderr or proc.stdout).strip()[-300:]}")
                continue
            want = f"{OUTPUT_DIR}/{img.stem}_output.json"
            cat = run(["docker", "exec", name, "cat", want], timeout=60)
            if cat.returncode != 0:
                passed = bad(f"no {want}. The name is the input filename with _output.json in place of its extension")
                continue
            try:
                payload = json.loads(cat.stdout)
            except ValueError:
                passed = bad(f"{want} is not valid JSON")
                continue
            if not isinstance(payload, dict):
                passed = bad(f"{want}: the JSON must be an object")
                continue
            if not isinstance(payload.get("text"), str):
                passed = bad(f"{want}: missing a string 'text'")
                continue
            conf = payload.get("confidence")
            if conf is not None and not (isinstance(conf, (int, float)) and 0 <= conf <= 1):
                warn(f"{img.name}: confidence {conf!r} is not a number in 0..1")
            verdict = ""
            if img.name in expected:
                hit = normalize(payload["text"]) == normalize(expected[img.name])
                scored += hit
                verdict = f"  {'MATCH' if hit else 'MISS'} (want {expected[img.name]!r})"
            ok(f"{img.name} ({FORMATS[img.suffix.lower()]}) {dt:.1f}s text={payload['text']!r}{verdict}")
            if not payload["text"] and args.fake_backend:
                passed = bad(f"{img.name}: empty text (the fallback output). The fake backend answers FAKE, so "
                             "app.py did not get an answer from the server")
            elif not payload["text"]:
                warn(f"{img.name}: empty text (the fallback output); check docker exec {name} cat /tmp/server.log")
        if args.fake_backend:
            skip("score: skipped: no GPU / fake backend")
        elif expected:
            n = sum(1 for img in images if img.name in expected)
            print(f"     score {scored}/{n} against {expected_file}")

        print("6. totals and VRAM")
        if total > TOTAL_BUDGET_S:
            passed = bad(f"{total:.1f}s of inference is over the {TOTAL_BUDGET_S}s whole-run budget")
        else:
            ok(f"{total:.1f}s for {len(images)} images (whole-run budget {TOTAL_BUDGET_S}s for 10)")
        st = run(["docker", "exec", name, "python3", "/app/server.py", "--stats"], timeout=30)
        if st.returncode == 0:
            print("     server stats:", " ".join(st.stdout.split()))
        sampler.stop.set()
        if args.fake_backend:
            skip("VRAM bounds (1-48 GiB): skipped: no GPU / fake backend")
        elif sampler.peak is None:
            warn("could not read VRAM from rocm-smi or amd-smi. Watch it by hand while this runs:")
            print("       watch -n1 'amd-smi metric --mem | grep USED_VRAM'   # or: rocm-smi --showmeminfo vram")
        elif VRAM_FLOOR_GIB <= sampler.peak <= VRAM_CEILING_GIB:
            ok(f"peak VRAM {sampler.peak:.2f} GiB over {sampler.samples} samples (allowed 1-48 GiB)")
        else:
            passed = bad(f"peak VRAM {sampler.peak:.2f} GiB is outside 1-48 GiB. Under the floor means the model "
                         "ran on the CPU")

        print("7. hygiene")
        state = run(["docker", "inspect", "-f", "{{.State.Running}}", name])
        if state.stdout.strip() == "true":
            ok("container still running after all calls")
        else:
            passed = bad("container stopped during the run")
        health = run(["docker", "inspect", "-f", "{{if .State.Health}}{{.State.Health.Status}}{{end}}", name])
        if health.stdout.strip() == "healthy":
            ok("Docker HEALTHCHECK reports healthy")
        else:
            warn(f"Docker health status: {health.stdout.strip() or 'no HEALTHCHECK in the image'}")
        if run(["docker", "exec", name, "sh", "-c", "test ! -e /app/.env"]).returncode != 0:
            passed = bad("/app/.env exists in the image. No secrets in the image")
        secret = run(["docker", "history", "--no-trunc", "--format", "{{.CreatedBy}}", args.image])
        if re.search(r"token|password|secret|HF_TOKEN", secret.stdout, re.IGNORECASE):
            warn("a build step mentions token/password/secret; make sure nothing secret is baked in")
    finally:
        sampler.stop.set()
        if args.keep:
            print(f"     container left running: {name}")
        else:
            run(["docker", "rm", "-f", name], timeout=120)

    print()
    print("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED - fix these before submitting")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
