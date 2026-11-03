#!/usr/bin/env bash
# GitHub-hosted Ubuntu runner only: make room for a Match image build (the base is 29.2 GiB,
# the weights ~16 GB; peak Docker use is about 50 GiB). Called by .github/workflows/build-image.yml.
#
#   bash tools/ci_disk.sh [MIN_FREE_GIB]      (default 55)
#
# 1. Deletes preinstalled toolchains the build does not use (.NET, Android SDK, Haskell, CodeQL,
#    the whole tool cache, browsers, cloud CLIs, ...), turns swap off and prunes Docker.
# 2. Pools the free space of / and the runner's second disk at /mnt into one LVM volume (loop
#    files on each disk, like easimon/maximize-build-space) and mounts it at /docker-data. Neither
#    disk alone has enough room after cleanup; together they have ~100 GiB. Without a usable
#    /mnt it uses / alone.
# 3. Points Docker's data-root there and turns the containerd image store off (it keeps a
#    compressed and an unpacked copy of every layer).
# 4. Fails if Docker's storage then has less than MIN_FREE_GIB free.
# It deletes system directories with sudo, so it refuses to run outside GitHub Actions.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then sed -n '2,16p' "$0"; exit 0; fi
MIN_GIB="${1:-55}"
[[ "$MIN_GIB" =~ ^[0-9]+$ ]] || { echo "MIN_FREE_GIB must be a whole number of GiB"; exit 2; }
if [[ "${GITHUB_ACTIONS:-}" != true ]]; then
  echo "refusing to run outside GitHub Actions: this script deletes system directories"; exit 1
fi

ROOT_RESERVE_GIB=5   # left on / for the runner, logs and the checkout
MNT_RESERVE_GIB=2   # df -BG rounds up, so keep a margin for fallocate
POOL_DIR=/docker-data

avail_gib() { df -BG --output=avail "$1" | tail -n1 | tr -dc '0-9'; }

echo "== disk before"
df -h

echo "== delete preinstalled software the build does not use"
docker system prune -af >/dev/null || true
sudo swapoff -a || true
sudo rm -f /swapfile /mnt/swapfile
t0=$SECONDS
# /opt/hostedtoolcache holds setup-* toolchains (Python, Node, Go, ...); the workflow only uses
# the system /usr/bin/python3, which stays.
sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /usr/local/.ghcup \
  /opt/hostedtoolcache /usr/local/share/boost /usr/share/swift /usr/local/share/powershell \
  /usr/local/share/chromium /usr/local/julia* /usr/lib/jvm /opt/az /usr/share/miniconda \
  /usr/local/lib/node_modules /opt/microsoft /opt/google /usr/lib/google-cloud-sdk \
  /usr/share/az_* /opt/pipx /usr/local/share/vcpkg /usr/share/kotlinc /usr/local/aws-cli \
  /usr/local/aws-sam-cli /usr/lib/firefox /opt/mssql-tools /usr/local/n /usr/local/sqlpackage
sudo apt-get clean >/dev/null 2>&1 || true
echo "deleted in $((SECONDS - t0))s"
df -h / /mnt 2>/dev/null || df -h /

root_free="$(avail_gib /)"
root_size=$(( root_free - ROOT_RESERVE_GIB ))
mnt_size=0
if mountpoint -q /mnt && [[ "$(stat -c %d /)" != "$(stat -c %d /mnt)" ]]; then
  mnt_free="$(avail_gib /mnt)"
  mnt_size=$(( mnt_free - MNT_RESERVE_GIB ))
  echo "free: / ${root_free} GiB, /mnt ${mnt_free} GiB (separate disk)"
else
  echo "free: / ${root_free} GiB (no separate disk at /mnt)"
fi

target=/var/lib/docker
if (( mnt_size > 5 && root_size > 5 )); then
  echo "== pool / (${root_size} GiB) and /mnt (${mnt_size} GiB) into one LVM volume at $POOL_DIR"
  command -v vgcreate >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq lvm2; }
  sudo systemctl stop docker docker.socket || true
  sudo fallocate -l "${root_size}G" /pool-root.img
  sudo fallocate -l "${mnt_size}G" /mnt/pool-mnt.img
  loop_root="$(sudo losetup --find --show /pool-root.img)"
  loop_mnt="$(sudo losetup --find --show /mnt/pool-mnt.img)"
  sudo pvcreate -qq -f "$loop_root" "$loop_mnt"
  sudo vgcreate -qq docker_vg "$loop_root" "$loop_mnt"
  sudo lvcreate -qq -y -l 100%FREE -n docker_lv docker_vg
  sudo mkfs.ext4 -q -m 0 -E nodiscard /dev/docker_vg/docker_lv
  sudo mkdir -p "$POOL_DIR"
  sudo mount /dev/docker_vg/docker_lv "$POOL_DIR"
  target="$POOL_DIR"
elif (( mnt_size > root_size )); then
  target=/mnt/docker-data
fi

echo "== Docker data-root -> $target, classic overlay2 store"
sudo mkdir -p "$target"
sudo python3 - "$target" <<'EOF'
import json, os, sys
path = "/etc/docker/daemon.json"
cfg = {}
if os.path.isfile(path) and os.path.getsize(path):
    with open(path) as f:
        cfg = json.load(f)
cfg["data-root"] = sys.argv[1]
cfg.setdefault("features", {})["containerd-snapshotter"] = False
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
print(path, json.dumps(cfg))
EOF
sudo systemctl restart docker
for _ in $(seq 30); do docker info >/dev/null 2>&1 && break; sleep 1; done
root_dir="$(docker info --format '{{.DockerRootDir}}')"
driver="$(docker info --format '{{.Driver}}')"
echo "Docker root dir $root_dir, storage driver $driver"
if [[ "$root_dir" != "$target" || "$driver" != overlay2 ]]; then
  echo "::error::Docker did not take the new storage settings (want $target with overlay2); see /etc/docker/daemon.json"
  exit 1
fi

echo "== disk after"
df -h
free="$(avail_gib "$root_dir")"
if (( free < MIN_GIB )); then
  echo "::error::only ${free} GiB free for Docker at $root_dir; the build needs about ${MIN_GIB} GiB (29.2 GiB base, ~16 GB weights, headroom). Paste the 'disk before' and 'free:' lines of this step."
  exit 1
fi
echo "ok: ${free} GiB free for Docker at $root_dir (need ${MIN_GIB})"
