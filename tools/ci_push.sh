#!/usr/bin/env bash
# Push an image that tools/build_push.sh already built and checked, then prove it pulls without
# credentials. Written for .github/workflows/build-image.yml; works anywhere with docker:
#
#   DOCKERHUB_USER=... DOCKERHUB_TOKEN=... DOCKERHUB_REPO=user/name bash tools/ci_push.sh <local-image> <tag>
#
# Same target rule as build_push.sh --push (DOCKERHUB_REPO, plus :<tag> unless it has a tag), but it
# only tags and pushes, so nothing is rebuilt. In GitHub Actions the reference is masked (***) and this
# script never prints it. The token reaches docker login on stdin and is stored only in a temporary
# DOCKER_CONFIG, deleted on exit. The public check runs logged out, with an empty config.
set -euo pipefail

if (( $# != 2 )); then sed -n '2,10p' "$0"; exit 2; fi
LOCAL="$1"; TAG="$2"
: "${DOCKERHUB_USER:?is not set (repository secret DOCKERHUB_USER)}"
: "${DOCKERHUB_TOKEN:?is not set (repository secret DOCKERHUB_TOKEN: a Docker Hub access token, read/write)}"
: "${DOCKERHUB_REPO:?is not set (repository secret DOCKERHUB_REPO: user/repo-name)}"

REPO="$(printf '%s' "$DOCKERHUB_REPO" | tr -d '[:space:]')"
REPO="${REPO#docker.io/}"
if [[ "${GITHUB_ACTIONS:-}" == true ]]; then
  name="${REPO##*/}"
  echo "::add-mask::$REPO"
  echo "::add-mask::${name%%:*}"
fi
if [[ ! "$REPO" =~ ^[^/]+/[^/]+$ ]]; then
  echo "DOCKERHUB_REPO must be <user>/<repo-name>: one slash, no registry host"; exit 1
fi
REF="$REPO"
[[ "${REF##*/}" == *:* ]] || REF="$REF:$TAG"
TAG="${REF##*:}"  # the tag actually pushed (DOCKERHUB_REPO may carry its own)

DOCKER_CONFIG="$(mktemp -d)"; export DOCKER_CONFIG
log="$(mktemp)"
cleanup() { docker logout >/dev/null 2>&1 || true; rm -rf "$DOCKER_CONFIG" "$log"; }
trap cleanup EXIT

echo "== docker login (token on stdin)"
printf '%s' "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USER" --password-stdin

echo "== tag and push $LOCAL (target from DOCKERHUB_REPO, masked)"
docker tag "$LOCAL" "$REF"
docker push "$REF" | tee "$log"
digest="$(grep -oE 'digest: sha256:[0-9a-f]{64}' "$log" | tail -n1 | cut -d' ' -f2 || true)"
if [[ -z "$digest" ]]; then
  digest="$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$REF" \
            | grep -F "${REF%:*}@" | head -n1 | sed 's/.*@//' || true)"
fi
[[ -n "$digest" ]] || { echo "could not read the pushed digest"; exit 1; }
echo "pushed digest: $digest"

echo "== logged out, empty docker config: can anyone pull it?"
docker logout >/dev/null
rm -rf "$DOCKER_CONFIG"
DOCKER_CONFIG="$(mktemp -d)"; export DOCKER_CONFIG
if manifest="$(docker manifest inspect --verbose "$REF" 2>&1)"; then
  if grep -qF "$digest" <<<"$manifest"; then
    echo "ok: pullable without credentials, registry digest matches"
  else
    echo "WARNING: pullable without credentials, but the registry reports another digest (a second push to the same tag?)"
  fi
else
  printf '%s\n' "$manifest" | tail -n 5
  echo "::error::Pushed, but the image cannot be pulled without credentials. On hub.docker.com open the repository > Settings > Visibility settings > make it Public, then check again with: docker logout; docker manifest inspect <ref>"
  exit 1
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  echo "- pushed tag \`$TAG\` to the DOCKERHUB_REPO repository, digest \`$digest\`, pullable without credentials" \
    >> "$GITHUB_STEP_SUMMARY"
fi
echo "Submit <user>/<repo-name>:$TAG (the DOCKERHUB_REPO secret) on lablab. Never commit or post the reference."
