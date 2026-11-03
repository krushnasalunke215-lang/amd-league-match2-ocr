#!/usr/bin/env bash
# Build a Match image from the repo root, check it, and optionally push it.
#
#   bash tools/build_push.sh <match-dir> <MODEL_ID> [--tag v1] [--revision main] [--push]
#   e.g. bash tools/build_push.sh match2-ocr Qwen/Qwen2.5-VL-7B-Instruct --tag v1 --push
#
# Builds <match-dir>-local:<tag> with `docker build -f <match-dir>/Dockerfile .` (never --squash:
# the harness checks the base image's layers), then checks: torch and torchvision are +rocm,
# the lower layers are the mandated base, and the uncompressed size is <= 60 GiB.
# With --push, tags and pushes to the Docker Hub repo from $DOCKERHUB_REPO or the first line
# of the gitignored .image-ref file (e.g. user/name). NEVER commit that reference.
set -euo pipefail

usage() { sed -n '2,11p' "$0"; exit "${1:-2}"; }
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage 0
(( $# >= 2 )) || usage 2
MATCH_DIR="${1%/}"; MODEL_ID="$2"; shift 2
TAG=v1; REVISION=main; PUSH=0
while (( $# )); do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --revision) REVISION="$2"; shift 2 ;;
    --push) PUSH=1; shift ;;
    *) echo "unknown argument: $1"; usage 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f "$MATCH_DIR/Dockerfile" ]] || { echo "no $MATCH_DIR/Dockerfile under $ROOT"; exit 1; }
BASE=rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
LOCAL="$MATCH_DIR-local:$TAG"
LIMIT=$((60 * 1024 * 1024 * 1024))

echo "== build $LOCAL (MODEL_ID=$MODEL_ID@$REVISION)"
extra=()  # only if the base lacks torchvision: export TORCHVISION_SPEC (see match2-ocr/Dockerfile)
[[ -n "${TORCHVISION_SPEC:-}" ]] && extra+=(--build-arg "TORCHVISION_SPEC=$TORCHVISION_SPEC")
docker build -f "$MATCH_DIR/Dockerfile" \
  --build-arg MODEL_ID="$MODEL_ID" --build-arg MODEL_REVISION="$REVISION" "${extra[@]}" \
  -t "$LOCAL" .

echo "== torch and torchvision inside the image"
docker run --rm --network none --entrypoint python3 "$LOCAL" \
  -c "import torch, torchvision as tv; assert '+rocm' in torch.__version__, torch.__version__; assert '+rocm' in tv.__version__, tv.__version__; print('torch', torch.__version__, 'torchvision', tv.__version__)"

echo "== base layers"
if docker image inspect "$BASE" >/dev/null 2>&1; then
  python3 - "$(docker image inspect --format '{{json .RootFS.Layers}}' "$LOCAL")" \
            "$(docker image inspect --format '{{json .RootFS.Layers}}' "$BASE")" <<'EOF'
import json, sys
mine, base = json.loads(sys.argv[1]), json.loads(sys.argv[2])
assert mine[:len(base)] == base, "lower layers are NOT the mandated base (wrong FROM or squashed)"
print(f"ok: the lower {len(base)} of {len(mine)} layers are the mandated base")
EOF
else
  echo "WARNING: $BASE not present locally; skipped (docker pull it to check)"
fi

echo "== uncompressed size"
inspect_b="$(docker image inspect --format '{{.Size}}' "$LOCAL")"
history_b="$(docker history --no-trunc --format '{{.Size}}' "$LOCAL" | python3 -c '
import re, sys
u = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}
print(int(sum(float(m[1]) * u[m[2].upper()] for m in (re.fullmatch(r"([\d.]+)\s*([kKMGT]?B)", l.strip()) for l in sys.stdin) if m)))')"
size_b=$(( inspect_b > history_b ? inspect_b : history_b ))
python3 -c "print(f'image inspect {$inspect_b/2**30:.2f} GiB, history sum {$history_b/2**30:.2f} GiB, limit 60 GiB')"
if (( size_b > LIMIT )); then
  echo "FAIL: over the 60 GiB uncompressed limit"; exit 1
fi

if (( PUSH )); then
  REF="${DOCKERHUB_REPO:-}"
  if [[ -z "$REF" && -f .image-ref ]]; then REF="$(head -n1 .image-ref | tr -d '[:space:]')"; fi
  if [[ -z "$REF" ]]; then
    echo "no push target: export DOCKERHUB_REPO=user/name, or put it in .image-ref (gitignored)"; exit 1
  fi
  [[ "${REF##*/}" == *:* ]] || REF="$REF:$TAG"
  echo "== push"
  docker tag "$LOCAL" "$REF"
  docker push "$REF"
  echo "pushed: $(docker image inspect --format '{{index .RepoDigests 0}}' "$REF" 2>/dev/null || echo "$REF")"
  echo "Make the Docker Hub repo public, check 'docker pull' works logged out, then submit the reference."
fi
echo
echo "Reminder: never commit the Docker Hub image reference (.image-ref is gitignored)."
