#!/usr/bin/env python3
"""Score OCR outputs the way the Match 2 brief does.

    python score.py --images eval/samples --outputs eval/outputs/samples
    python score.py --self-test

`--images` is a folder with an expected.json mapping image filename -> expected
text. For image `sample_01.png` the answer is read from
`<outputs>/sample_01_output.json`, field "text".

Normalization is exactly the brief's: uppercase, ALL whitespace removed, and
the characters - . · _ removed. Nothing else. Each image is pass/fail.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_DROP = str.maketrans("", "", "-.\u00b7_")


def normalize(s: str) -> str:
    return re.sub(r"\s+", "", str(s).upper()).translate(_DROP)


def read_answer(outputs: Path, image_name: str) -> tuple[str | None, str]:
    """(text, problem). text is None when the output is missing or malformed."""
    p = outputs / (Path(image_name).stem + "_output.json")
    if not p.is_file():
        return None, f"missing {p.name}"
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        return None, f"bad JSON: {e}"
    if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
        return None, "no string 'text'"
    return obj["text"], ""


def score(expected: dict[str, str], outputs: Path, quiet: bool = False) -> tuple[int, int]:
    passed = 0
    for name, want in expected.items():
        got, problem = read_answer(outputs, name)
        ok = got is not None and normalize(got) == normalize(want)
        passed += ok
        if not quiet:
            shown = problem if got is None else repr(got)
            print(f"{'PASS' if ok else 'FAIL'}  {name:<34} got={shown:<28} want={want!r}")
    if not quiet:
        print(f"SCORE {passed}/{len(expected)}  ({passed * 20} of {len(expected) * 20} points at 20 each)")
    return passed, len(expected)


def self_test() -> int:
    same = [
        ("7ABC123", "7ABC123"), ("7abc123", "7ABC123"), ("7-ABC-123", "7ABC123"), ("7 ABC 123", "7ABC123"),
        ("京A12345", "京A·12345"), ("京A·12345", "京A·12345"), ("ROAD\nWORK  AHEAD", "ROAD WORK AHEAD"),
        ("speed_limit.65", "SPEED LIMIT 65"),
    ]
    differ = [
        ("7ABC128", "7ABC123"), ("CALIFORNIA 7ABC123", "7ABC123"), ("PLATE: 7ABC123", "7ABC123"),
        ("12345", "京A·12345"), ("35 MPH", "35"),
    ]
    bad = [(g, w, "should match") for g, w in same if normalize(g) != normalize(w)]
    bad += [(g, w, "should NOT match") for g, w in differ if normalize(g) == normalize(w)]
    for g, w, why in bad:
        print(f"self-test FAIL: {g!r} vs {w!r} {why}")
    print("self-test OK" if not bad else "self-test FAILED")
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, help="folder containing expected.json")
    ap.add_argument("--expected", type=Path, help="expected.json (default: <images>/expected.json)")
    ap.add_argument("--outputs", type=Path, help="folder with <stem>_output.json files")
    ap.add_argument("--self-test", action="store_true", help="check normalize() against the brief's examples")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="backslashreplace")  # Chinese answers on a cp1252 Windows console
    if args.self_test:
        return self_test()
    expected_file = args.expected or (args.images / "expected.json" if args.images else None)
    if expected_file is None or args.outputs is None:
        ap.error("need --images (or --expected) and --outputs")
    expected = json.loads(expected_file.read_text(encoding="utf-8"))
    score(expected, args.outputs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
