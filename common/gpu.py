"""GPU guards for ROCm images. torch is imported inside each function, so this
module imports fine on a laptop without torch.

ROCm builds of torch use the device name "cuda". Place modules on it explicitly:
a run that silently falls back to the CPU stays under the 1 GiB VRAM floor and
scores zero.
"""

from __future__ import annotations

GIB = 1024 ** 3


def assert_rocm_torch() -> str:
    """Raise unless torch is a ROCm build. Returns torch.__version__.

    `pip install transformers` without --no-deps can replace the base image's
    ROCm torch with a CUDA wheel; this catches that.
    """
    import torch

    v = torch.__version__
    if "+rocm" not in v or not getattr(torch.version, "hip", None):
        raise RuntimeError(f"torch {v} is not a ROCm build (torch.version.hip={getattr(torch.version, 'hip', None)})")
    return v


def require_gpu() -> str:
    """Return the device string to place modules on; raise if no GPU is visible."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("no GPU visible to torch (check --device /dev/kfd --device /dev/dri)")
    return "cuda"


def device_info() -> dict:
    import torch

    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "arch": getattr(props, "gcnArchName", "?"),
        "total_gib": round(props.total_memory / GIB, 2),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
    }


def vram_report() -> dict:
    """VRAM in GiB: this process's allocator stats plus device-wide used memory."""
    import torch

    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / GIB, 2),
        "reserved_gib": round(torch.cuda.memory_reserved() / GIB, 2),
        "max_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 2),
        "max_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 2),
        "device_used_gib": round((total - free) / GIB, 2),
    }


def reset_peak() -> None:
    import torch

    torch.cuda.reset_peak_memory_stats()


# ---- host-side VRAM sampling (no torch): what the harness measures, device-wide.

_UNIT = {"B": 1, "KB": 1024, "KIB": 1024, "MB": 1024 ** 2, "MIB": 1024 ** 2, "GB": GIB, "GIB": GIB}


def _find_used(obj) -> list[float]:
    """Bytes of every '*used*vram*' field in amd-smi JSON (shape differs between versions)."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if "used" in key and "vram" in key:
                if isinstance(v, dict) and "value" in v:
                    found.append(float(v["value"]) * _UNIT.get(str(v.get("unit", "MB")).upper(), 1024 ** 2))
                elif isinstance(v, (int, float)):
                    found.append(float(v) * 1024 ** 2)
            else:
                found += _find_used(v)
    elif isinstance(obj, list):
        for v in obj:
            found += _find_used(v)
    return found


def smi_vram_used_gib() -> float | None:
    """Highest used VRAM over all visible GPUs in GiB, via rocm-smi or amd-smi. None if neither works."""
    import json
    import re
    import subprocess

    def run(args):
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
            return p.stdout if p.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    out = run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    if out:
        try:
            cards = json.loads(out).values()
            used = [float(c["VRAM Total Used Memory (B)"]) for c in cards if "VRAM Total Used Memory (B)" in c]
            if used:
                return max(used) / GIB
        except (ValueError, TypeError, AttributeError):
            pass
    out = run(["amd-smi", "metric", "--mem", "--json"])
    if out:
        try:
            used = _find_used(json.loads(out))
            if used:
                return max(used) / GIB
        except ValueError:
            pass
    out = run(["amd-smi", "metric", "--mem"])
    if out:
        m = [float(x) for x in re.findall(r"USED_VRAM:\s*([\d.]+)\s*MB", out)]
        if m:
            return max(m) / 1024
    return None
