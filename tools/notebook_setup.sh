#!/usr/bin/env bash
# Set up the AMD hackathon notebook (notebooks.amd.com/hackathon) for Match 2 work.
#
#   bash tools/notebook_setup.sh [MODEL_ID]        e.g. Qwen/Qwen2.5-VL-7B-Instruct
#   source "$PERSIST/mc2env.sh"                    (path printed at the end)
#
# - Persistent storage: /persistent if it exists and is writable (jupyter-hack-* pods),
#   else /workspace (rgapi-hackathon-* pods). Only that survives a session.
# - HF_HOME, a pip --target package folder and model weights all live there, so the next
#   session only re-runs this script (it skips work that is already done).
# - Packages: only what the base image lacks goes into the --target folder (so base numpy etc.
#   are not shadowed), all with --no-deps; transformers/accelerate never pull a CUDA torch.
# - torch AND torchvision must be +rocm (the Qwen-VL processors need torchvision). Only if the
#   session lacks torchvision: export TORCHVISION_SPEC (see match2-ocr/Dockerfile) and re-run.
# Idempotent: safe to run at the start of every session.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,15p' "$0"; exit 0
fi
MODEL_ID="${1:-}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ="$REPO/match2-ocr/app/requirements.txt"
REQ_NODEPS="$REPO/match2-ocr/app/requirements-nodeps.txt"

if [[ -d /persistent && -w /persistent ]]; then
  PERSIST=/persistent; why="/persistent exists and is writable (jupyter-hack-* pod)"
else
  PERSIST=/workspace; why="no writable /persistent, using /workspace (rgapi-hackathon-* pod)"
fi
echo "persistent storage: $PERSIST ($why)"
mkdir -p "$PERSIST/hf" "$PERSIST/pylib" "$PERSIST/models"
export HF_HOME="$PERSIST/hf"
PYLIB="$PERSIST/pylib"

echo "python: $(command -v python3) $(python3 --version 2>&1); $(python3 -m pip --version)"

stamp="$(cat "$REQ" "$REQ_NODEPS" | sha256sum | cut -d' ' -f1)"
if [[ -f "$PYLIB/.stamp" && "$(cat "$PYLIB/.stamp")" == "$stamp" ]]; then
  echo "packages: up to date in $PYLIB"
else
  echo "packages: installing into $PYLIB"
  report="$(mktemp --suffix=.json)"
  # Ask pip what it WOULD add to the base environment, then install exactly those pins.
  env -u PYTHONPATH python3 -m pip install --quiet --dry-run --report "$report" -r "$REQ"
  python3 - "$report" > "$PYLIB.pins" <<'EOF'
import json, sys
for item in json.load(open(sys.argv[1], encoding="utf-8")).get("install", []):
    print(f"{item['metadata']['name']}=={item['metadata']['version']}")
EOF
  if [[ -s "$PYLIB.pins" ]]; then
    python3 -m pip install --quiet --no-cache-dir --upgrade --no-deps --target "$PYLIB" -r "$PYLIB.pins"
  fi
  python3 -m pip install --quiet --no-cache-dir --upgrade --no-deps --target "$PYLIB" -r "$REQ_NODEPS"
  rm -f "$report"
  echo "$stamp" > "$PYLIB/.stamp"
fi
export PYTHONPATH="$PYLIB${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${TORCHVISION_SPEC:-}" ]] && ! python3 -c "import torchvision" 2>/dev/null; then
  echo "torchvision: installing $TORCHVISION_SPEC into $PYLIB (--no-deps)"
  # shellcheck disable=SC2086  # the spec is a requirement plus pip options, split on purpose
  python3 -m pip install --quiet --no-cache-dir --no-deps --target "$PYLIB" $TORCHVISION_SPEC
fi

python3 - <<'EOF'
import torch, transformers
assert "+rocm" in torch.__version__, f"torch {torch.__version__} is not ROCm: something replaced it"
try:
    import torchvision
except ImportError as e:
    raise SystemExit(f"torchvision is missing ({e}). The Qwen-VL processors need it: find the ROCm build "
                     f"matching torch {torch.__version__} (hip {torch.version.hip}), export TORCHVISION_SPEC "
                     "and re-run this script")
assert "+rocm" in torchvision.__version__, f"torchvision {torchvision.__version__} is not ROCm"
print(f"torch {torch.__version__} (hip {torch.version.hip}), torchvision {torchvision.__version__}, "
      f"transformers {transformers.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  arch {getattr(p, 'gcnArchName', '?')}  {p.total_memory / 2**30:.1f} GiB")
else:
    print("WARNING: no GPU visible to torch")
EOF

MODEL_DIR=""
if [[ -n "$MODEL_ID" ]]; then
  MODEL_DIR="$PERSIST/models/${MODEL_ID//\//__}"
  if [[ -f "$MODEL_DIR/.complete" ]]; then
    echo "model: $MODEL_ID already in $MODEL_DIR"
  else
    free_gb="$(df -BG --output=avail "$PERSIST" | tail -1 | tr -dc '0-9')"
    echo "model: downloading $MODEL_ID to $MODEL_DIR (free: ${free_gb} GB; quota ~25 GB holds one 7-8B model)"
    if (( free_gb < 20 )) && [[ ! -d "$MODEL_DIR" ]]; then
      echo "not enough space. Models already here:"; du -sh "$PERSIST"/models/* 2>/dev/null || true
      echo "free one with: rm -rf $PERSIST/models/<name>"; exit 1
    fi
    python3 - "$MODEL_ID" "$MODEL_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.tiktoken"])
EOF
    touch "$MODEL_DIR/.complete"
  fi
  # Same CPU check as the docker build: processor + chat template + encode.
  python3 "$REPO/match2-ocr/app/ocr.py" --processor-check "$MODEL_DIR"
fi

cat > "$PERSIST/mc2env.sh" <<EOF
export PERSIST="$PERSIST"
export HF_HOME="$PERSIST/hf"
export PYTHONPATH="$PYLIB\${PYTHONPATH:+:\$PYTHONPATH}"
export MC2_SOCKET=/tmp/mc2-ocr.sock
${MODEL_DIR:+export OCR_MODEL_DIR="$MODEL_DIR"}
EOF

echo "disk: $(df -h --output=avail,size "$PERSIST" | tail -1 | awk '{print $1 " free of " $2}') on $PERSIST; used here: $(du -sh "$PERSIST" 2>/dev/null | cut -f1)"
echo
echo "done. Now run:  source $PERSIST/mc2env.sh"
