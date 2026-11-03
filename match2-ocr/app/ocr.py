#!/usr/bin/env python3
"""OCR core: image loading, the prompt, post-processing, and the VLM backend.

Everything above `class VlmBackend` is pure Python + Pillow and runs on a
laptop. torch/transformers are imported only inside VlmBackend.load().

Swapping the model is configuration only: point OCR_MODEL_DIR (server.py) at a
different snapshot. Tested candidates: Qwen2.5-VL-7B-Instruct (default),
Qwen3-VL-8B-Instruct, Qwen2.5-VL-3B-Instruct. The code uses only
AutoProcessor + AutoModelForImageTextToText and the chat template.

    python ocr.py --self-test              # post-processing cases
    python ocr.py --load-check DIR [DIR]   # load + prepare every image, print mode/size
    python ocr.py --processor-check MODEL_DIR   # CPU: processor + chat template + encode (build time)
"""

from __future__ import annotations

import math
import re
import sys
import time
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

# Image bounds. The area cap is what bounds visual tokens (and so latency):
# ~1.0 MP is ~1280 tokens for Qwen2.5-VL (28 px per token side).
MIN_SIDE = 640
MAX_SIDE = 2048
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
MAX_ASPECT = 20.0

PROMPT = (
    "Transcribe the text in this image exactly as printed, in reading order.\n"
    "Rules:\n"
    "- Licence plate: output only the registration number. Do not output text printed around it, "
    "such as a state or province name (CALIFORNIA, TEXAS, NEW YORK), a slogan or motto, "
    "a website, or dealer frame text.\n"
    "- Chinese licence plate: the leading Chinese province character and the letter after it are part "
    "of the number. Keep them, for example 粤B·K6079.\n"
    "- Road sign: output every word and number printed on the sign, for example YIELD, DETOUR, "
    "ICY BRIDGE, or SPEED LIMIT followed by its number. An advisory plaque with only a number "
    "is just that number.\n"
    "- Text on several lines: read top to bottom and join the lines with single spaces.\n"
    "- Never add units, labels, quotes, explanations or anything not printed in the image.\n"
    "Output only the text."
)

# ---------------------------------------------------------------- image loading


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode.startswith("I;16") or img.mode in ("I", "F"):
        if img.mode not in ("I", "F"):
            img = img.convert("I")
        lo, hi = img.getextrema()
        if img.mode == "F" or hi > 255 or lo < 0:  # stretch 16-bit / float range to 8-bit
            span = (hi - lo) or 1
            img = img.point(lambda v: (v - lo) * (255.0 / span))
        return img.convert("L").convert("RGB")
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA", "PA", "RGBa", "La"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return img.convert("RGB")


def load_image(path: str | Path) -> Image.Image:
    """Open PNG/JPEG/TIFF (16-bit, float, multi-page -> first frame), apply EXIF rotation, return RGB."""
    with Image.open(path) as im:
        im.seek(0)
        im.load()
        img = ImageOps.exif_transpose(im) or im
        return _to_rgb(img.copy())


def prepare(img: Image.Image) -> Image.Image:
    """Bound the image: pad extreme aspect ratios, upscale tiny, downscale huge."""
    w, h = img.size
    if max(w, h) / max(1, min(w, h)) > MAX_ASPECT:
        side = math.ceil(max(w, h) / MAX_ASPECT)
        canvas = Image.new("RGB", (max(w, side), max(h, side)), (255, 255, 255))
        canvas.paste(img, ((canvas.width - w) // 2, (canvas.height - h) // 2))
        img, (w, h) = canvas, canvas.size
    scale = 1.0
    if max(w, h) < MIN_SIDE:
        scale = MIN_SIDE / max(w, h)
    scale = min(scale, MAX_SIDE / max(w, h), math.sqrt(MAX_PIXELS / (w * h)))
    if abs(scale - 1.0) > 1e-3:
        new = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = img.resize(new, Image.BICUBIC if scale > 1 else Image.LANCZOS)
    return img


def contrast_variant(img: Image.Image) -> Image.Image:
    """Second-pass input: stretched and equalized, for low light and glare."""
    return ImageOps.equalize(ImageOps.autocontrast(img, cutoff=1))


# --------------------------------------------------------------- post-processing

US_STATES = (
    "ALABAMA ALASKA ARIZONA ARKANSAS CALIFORNIA COLORADO CONNECTICUT DELAWARE FLORIDA GEORGIA HAWAII IDAHO "
    "ILLINOIS INDIANA IOWA KANSAS KENTUCKY LOUISIANA MAINE MARYLAND MASSACHUSETTS MICHIGAN MINNESOTA "
    "MISSISSIPPI MISSOURI MONTANA NEBRASKA NEVADA OHIO OKLAHOMA OREGON PENNSYLVANIA TENNESSEE TEXAS UTAH "
    "VERMONT VIRGINIA WASHINGTON WISCONSIN WYOMING"
).split() + [
    "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK", "NORTH CAROLINA", "NORTH DAKOTA",
    "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA", "WEST VIRGINIA", "DISTRICT OF COLUMBIA",
]
PLATE_SLOGANS = [
    "EXCELSIOR", "EMPIRE STATE", "THE EMPIRE STATE", "THE LONE STAR STATE", "LONE STAR STATE",
    "SUNSHINE STATE", "LAND OF LINCOLN", "GARDEN STATE", "GRAND CANYON STATE", "FIRST IN FLIGHT",
    "FIRST IN FREEDOM", "LIVE FREE OR DIE", "FAMOUS POTATOES", "GREAT LAKES STATE", "PURE MICHIGAN",
    "PEACH STATE", "KEYSTONE STATE", "THE KEYSTONE STATE", "DMV.CA.GOV", "DMV CA GOV", "LAND OF ENCHANTMENT",
    "THE NATURAL STATE", "TAXATION WITHOUT REPRESENTATION", "GREEN MOUNTAIN STATE", "THE OCEAN STATE",
    "OCEAN STATE", "WILD WONDERFUL", "SHOW ME STATE", "BEEHIVE STATE", "BIG SKY COUNTRY", "VACATIONLAND",
    "AMERICA'S DAIRYLAND", "THE FIRST STATE", "ALOHA STATE", "THE GOLDEN STATE", "GOLDEN STATE",
]
# Longest first, so "THE LONE STAR STATE" is tried before "LONE STAR STATE".
BANNERS = sorted(set(US_STATES + PLATE_SLOGANS), key=len, reverse=True)

# Label prefixes, any case: "PLATE: 4KTR902", "Answer: 25", "License plate number: ..."
_LABEL = re.compile(
    r"^(?:the\s+)?(?:licen[cs]e\s+plate(?:\s+number)?|plate(?:\s+number)?|registration(?:\s+number)?|"
    r"number|text|answer|output|result|transcription|sign(?:\s+text)?|ocr)\s*[:：=]\s*",
    re.IGNORECASE,
)
# Model prose is mixed case; sign text is upper case. These only match a lowercase lead-in,
# so a sign such as "THIS LANE IS CLOSED" is never touched.
_PROSE_COLON = re.compile(r"^[A-Za-z][a-z'][^:：\n]{0,80}[:：]\s*")  # "Here is the text: ..."
_PROSE_LEAD = re.compile(r"^(?:The|This|It|the|this|it)\b[^:\n]{0,60}?\b(?:says|reads|shows|is|contains)\s+")
_QUOTES = "\"'`“”‘’«»「」『』"


def _strip_quotes(s: str) -> str:
    s = re.sub(f"([{_QUOTES}])[.,;!]+$", r"\1", s.strip())  # '"YIELD".' -> '"YIELD"'
    return s.strip(_QUOTES + " \t")


def _is_prose(line: str) -> bool:
    """A sentence the model added (mixed case, several words, ends like prose)."""
    words = line.split()
    return len(words) >= 4 and any(c.islower() for c in line) and line.rstrip().endswith((".", "!", ")"))


def _drop_banners(text: str) -> str:
    """Drop a state name/slogan at either end, but only if a digit remains (so signs are untouched)."""
    changed = True
    while changed:
        changed = False
        up = text.upper()
        for b in BANNERS:
            if up.startswith(b + " ") and any(c.isdigit() for c in text[len(b):]):
                text, changed = text[len(b):].strip(), True
                break
            if up.endswith(" " + b) and any(c.isdigit() for c in text[: -len(b)]):
                text, changed = text[: -len(b)].strip(), True
                break
    return text


# The grader removes only whitespace and - . · _ (U+00B7). Models often emit lookalikes
# (• ・ ∙ for the plate dot, – — − for the dash) or full-width letters and digits, which
# would survive that normalization. NFKC folds full-width forms; this map handles the rest.
_SEPARATORS = str.maketrans({c: "·" for c in "•・∙⋅‧･●"} | {c: "-" for c in "‐‑‒–—―−"})


def fold_lookalikes(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_SEPARATORS)


def postprocess(raw: str) -> str:
    """Model output -> answer: no fences, quotes, labels, markdown or prose; lines joined by single spaces."""
    lines = []
    for line in fold_lookalikes(str(raw or "")).replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        line = re.sub(r"^(?:#+|[-*•·]|\d+[.)])\s+", "", line)  # markdown heading / bullet / numbered list
        line = line.replace("**", "").replace("__", "").strip("*`").strip()
        line = _LABEL.sub("", line, count=1)
        line = _PROSE_COLON.sub("", line, count=1)
        line = _PROSE_LEAD.sub("", line, count=1)
        line = _strip_quotes(line)
        if line and not (lines and _is_prose(line)):  # drop a trailing explanation sentence
            lines.append(line)
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    text = _strip_quotes(text)
    return _drop_banners(text)


# ------------------------------------------------------------------ VLM backend


class VlmBackend:
    """A vision-language model held on the GPU. Not thread-safe; server.py serialises calls."""

    def __init__(self, model_dir: str, max_new_tokens: int = 48, second_pass: bool = False):
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self.second_pass = second_pass
        self.model = self.processor = self.torch = None
        self.device = "cuda"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText

        from common import gpu

        gpu.assert_rocm_torch()
        self.device = gpu.require_gpu()
        self.torch = torch
        self.load_processor()
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=self.device
        ).eval()
        dev = next(self.model.parameters()).device
        if dev.type != "cuda":
            raise RuntimeError(f"model landed on {dev}, not the GPU")
        self.warm_up()

    def warm_up(self) -> None:
        """One real inference so kernels are compiled before the first graded image."""
        img = Image.new("RGB", (640, 320), "white")
        ImageDraw.Draw(img).text((40, 130), "WARM UP 123", fill="black")
        self._generate(img)

    def load_processor(self) -> None:
        # Qwen2.5-VL / Qwen3-VL processors always build a video sub-processor, which needs torchvision.
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_dir)

    def encode(self, img: Image.Image):
        """Chat template + processor call: the model inputs for one image (on the CPU)."""
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.processor(
            text=[prompt], images=[img], return_tensors="pt",
            images_kwargs={"size": {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}},
        )

    def _generate(self, img: Image.Image) -> tuple[str, float | None]:
        torch = self.torch
        inputs = self.encode(img).to(self.device)
        with torch.inference_mode():
            # Greedy, and no repetition penalty: Qwen2.5-VL's generation_config sets 1.05, which
            # would penalise repeated characters (77777) and any digit already in the prompt.
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                output_scores=True, return_dict_in_generate=True,
            )
        new_tokens = out.sequences[0, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode([new_tokens], skip_special_tokens=True)[0]
        conf = None
        if out.scores:
            logps = [torch.log_softmax(s[0].float(), dim=-1)[t].item() for s, t in zip(out.scores, new_tokens)]
            conf = math.exp(sum(logps) / len(logps)) if logps else None
        return text, conf

    def read(self, path: str) -> tuple[str, float | None]:
        img = prepare(load_image(path))
        raw, conf = self._generate(img)
        text = postprocess(raw)
        if self.second_pass:
            raw2, conf2 = self._generate(contrast_variant(img))
            text2 = postprocess(raw2)
            same = re.sub(r"[\s\-._·]", "", text.upper()) == re.sub(r"[\s\-._·]", "", text2.upper())
            if not same and (conf2 or 0) > (conf or 0):
                text, conf = text2, conf2
        return text, conf


# ----------------------------------------------------------------- self-checks

# Post-processing fixtures. Deliberately none of the brief's sample answers: this file ships in
# the public image, and the samples are only ever used from eval/.
SELF_TEST = [
    ("PLATE: 4KTR902", "4KTR902"),
    ('"YIELD"', "YIELD"),
    ("BRIDGE\nICES\nBEFORE ROAD", "BRIDGE ICES BEFORE ROAD"),
    ("粤B·K6079", "粤B·K6079"),
    ("苏E·77777", "苏E·77777"),
    ("粤B•K6079", "粤B·K6079"),
    ("苏E・77777", "苏E·77777"),
    ("４ＫＴＲ９０２", "4KTR902"),
    ("KLM–4471", "KLM-4471"),
    ("CALIFORNIA 4KTR902", "4KTR902"),
    ('Plate: "KLM 4471"', "KLM 4471"),
    ("NEW YORK\nKLM 4471\nEXCELSIOR", "KLM 4471"),
    ("SPEED LIMIT 40", "SPEED LIMIT 40"),
    ("```\n25\n```", "25"),
    ("**YIELD**", "YIELD"),
    ("The text in the image is: DETOUR AHEAD", "DETOUR AHEAD"),
    ('The sign reads "SPEED LIMIT 40".', "SPEED LIMIT 40"),
    ("Here is the text:\nBRIDGE\nICES\nBEFORE ROAD", "BRIDGE ICES BEFORE ROAD"),
    ("THIS LANE IS CLOSED", "THIS LANE IS CLOSED"),
    ("LICENSE PLATE: 8MNP204", "8MNP204"),
    ("4KTR902\n\nThis is a California licence plate.", "4KTR902"),
    ("Answer: 25", "25"),
    ("YIELD  ", "YIELD"),
    ("", ""),
]


def _self_test() -> int:
    bad = 0
    for raw, want in SELF_TEST:
        got = postprocess(raw)
        if got != want:
            bad += 1
            print(f"FAIL postprocess({raw!r}) = {got!r}, want {want!r}")
    print("postprocess self-test OK" if not bad else f"postprocess self-test: {bad} failures")
    return 1 if bad else 0


def _load_check(paths: list[str]) -> int:
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")) \
            if p.is_dir() else [p]
    bad = 0
    for f in files:
        t = time.perf_counter()
        try:
            src = load_image(f)
            img = prepare(src)
            print(f"{f.name:<36} {src.size[0]}x{src.size[1]} -> {img.mode} {img.size[0]}x{img.size[1]} "
                  f"{img.size[0] * img.size[1] / 1e6:.2f}MP {1000 * (time.perf_counter() - t):.0f}ms")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"{f.name:<36} ERROR {e!r}")
    print(f"load-check: {len(files) - bad}/{len(files)} ok")
    return 1 if bad else 0


def _processor_check(model_dir: str) -> int:
    """CPU only, no model weights loaded: the processor builds and encodes exactly as in serving.

    Run at docker build time so a missing torchvision or jinja2 fails the build instead of
    producing an image whose server cannot start (every answer would then be empty).
    """
    import torchvision

    backend = VlmBackend(model_dir)
    backend.load_processor()
    print(f"torchvision {torchvision.__version__}, processor {type(backend.processor).__name__}")
    for size in ((640, 320), (4000, 2000)):
        img = Image.new("RGB", size, "white")
        ImageDraw.Draw(img).text((40, size[1] // 2), "CHECK 0", fill="black")
        inputs = backend.encode(prepare(img))
        print(f"  {size[0]}x{size[1]}: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in inputs.items()))
    print("processor-check OK")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="backslashreplace")
    if sys.argv[1:2] == ["--self-test"]:
        sys.exit(_self_test())
    if sys.argv[1:2] == ["--load-check"] and len(sys.argv) > 2:
        sys.exit(_load_check(sys.argv[2:]))
    if sys.argv[1:2] == ["--processor-check"] and len(sys.argv) == 3:
        sys.exit(_processor_check(sys.argv[2]))
    print(__doc__)
    sys.exit(2)
