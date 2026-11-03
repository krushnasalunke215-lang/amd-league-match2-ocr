#!/usr/bin/env bash
# Prepare a fresh AMD Developer Cloud VM (single MI300X) for building and pushing images.
#
#   export DOCKERHUB_USER=<your docker hub user>
#   bash tools/vm_setup.sh [GITHUB_REPO_URL]
#
# - checks docker, the GPUs and free disk
# - docker login: the Docker Hub access token is read from stdin (typed hidden, or piped);
#   this script never writes it to disk. Docker itself keeps it in ~/.docker/config.json
#   on the VM, which is destroyed with the VM.
# - clones the repo to ~/amd-ai-league when a URL is given (skipped if it already exists)
# - pulls the mandated base image
# Destroy the VM when done (a stopped VM is still billed) and log the hours in tracker/points.md.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,14p' "$0"; exit 0
fi
REPO_URL="${1:-}"
BASE=rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
echo "VM session started $(date '+%Y-%m-%d %H:%M')"

command -v docker >/dev/null || { echo "docker is not installed"; exit 1; }
docker --version
docker info >/dev/null 2>&1 || { echo "docker daemon not reachable (try: sudo usermod -aG docker \$USER, then log in again)"; exit 1; }

if command -v amd-smi >/dev/null; then
  amd-smi list || true
  n_gpu="$(amd-smi list 2>/dev/null | grep -c '^GPU' || true)"
elif command -v rocm-smi >/dev/null; then
  rocm-smi || true
  n_gpu="$(rocm-smi --showid 2>/dev/null | grep -c 'GPU\[' || true)"
else
  echo "WARNING: neither amd-smi nor rocm-smi found"; n_gpu=0
fi
if (( n_gpu > 1 )); then
  echo "WARNING: $n_gpu GPUs. An 8-GPU node burns credits 8x faster; use the single-GPU VM."
fi

docker_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
df -h "$docker_root" 2>/dev/null || df -h /
free_gb="$(df -BG --output=avail "$docker_root" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)"
if (( ${free_gb:-0} < 150 )); then
  echo "WARNING: only ${free_gb} GB free for Docker; base image (~30 GB) + weights + build cache need ~150 GB"
fi

: "${DOCKERHUB_USER:?set DOCKERHUB_USER to your Docker Hub user name first}"
if [[ -t 0 ]]; then
  read -rsp "Docker Hub access token for $DOCKERHUB_USER (hidden): " token; echo
else
  read -r token
fi
printf '%s' "$token" | docker login -u "$DOCKERHUB_USER" --password-stdin
unset token

if [[ -n "$REPO_URL" ]]; then
  if [[ -d "$HOME/amd-ai-league/.git" ]]; then
    echo "repo already at ~/amd-ai-league; pulling"
    git -C "$HOME/amd-ai-league" pull --ff-only
  else
    git clone "$REPO_URL" "$HOME/amd-ai-league"   # private repo: GitHub user + personal access token
  fi
fi

docker pull "$BASE"
echo
echo "ready. Next: cd ~/amd-ai-league && echo '<dockerhub-user>/<repo>' > .image-ref"
echo "Destroy the VM when done; log the hours in tracker/points.md."
