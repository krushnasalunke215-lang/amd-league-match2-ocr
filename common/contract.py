"""Output-file side of the harness contract.

The harness reads one JSON file per call. A half-written file parses as invalid
JSON and scores zero, and a missing file scores zero too, so every write here
is atomic and `always_write` guarantees exactly one file per invocation.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def output_path_for(input_path: str | Path, out_dir: Path) -> Path:
    """image_01.png -> <out_dir>/image_01_output.json (last extension replaced)."""
    return Path(out_dir) / (Path(input_path).stem + "_output.json")


def write_json_atomic(path: Path, obj: dict) -> None:
    """Write `obj` as UTF-8 JSON: temp file in the same dir, fsync, os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    )
    try:
        with tmp:
            json.dump(obj, tmp, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, path)
    except BaseException:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


def ocr_payload(text, confidence=None) -> dict:
    """{"text": ...} plus "confidence" when it is a finite number, clamped to [0, 1]."""
    out = {"text": "" if text is None else str(text)}
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return out
    if math.isfinite(c):
        out["confidence"] = round(min(1.0, max(0.0, c)), 4)
    return out


class _Result:
    def __init__(self) -> None:
        self.payload: dict | None = None

    def set(self, payload: dict) -> None:
        self.payload = payload


@contextmanager
def always_write(path: Path, fallback: dict) -> Iterator[_Result]:
    """Write exactly one output file, whatever happens inside the block.

        with always_write(out, {"text": "", "confidence": 0.0}) as res:
            res.set(do_work())

    If the block raises (or never calls set), `fallback` is written instead and
    the exception is logged to stderr and swallowed, so the caller can exit 0.
    """
    res = _Result()
    try:
        yield res
    except Exception:
        traceback.print_exc(file=sys.stderr)
        res.payload = None
    payload = res.payload if isinstance(res.payload, dict) else fallback
    try:
        write_json_atomic(path, payload)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        if payload is not fallback:
            write_json_atomic(path, fallback)
