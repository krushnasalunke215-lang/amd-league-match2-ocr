# ROCm Plate & Sign OCR (AMD AI League, Match 2)

An OCR container for [AMD AI League](https://lablab.ai/ai-hackathons/amd-lablab-ai-academy-challenge) Match 2. It reads licence plates and road signs with an open vision-language model (Qwen2.5-VL-7B-Instruct) running on an AMD GPU through ROCm.

- **Contract:** `python3 /app/app.py --input-image /app/input/image_01.png` writes `/app/output/image_01_output.json` with `{"text": ..., "confidence": ...}`.
- **Design:** the container `CMD` starts a resident server (`match2-ocr/app/server.py`) that loads the model once in bf16 on the GPU, warms up, and serves a unix socket. `app.py` is a thin client, so a new process per image never reloads the model. Weights are baked into `/models`; no network is needed at run time.
- **Image handling and output rules:** `match2-ocr/app/ocr.py` (PNG/JPEG/TIFF incl. 16-bit and multi-page, EXIF rotation, resizing; prompt rules for plates and signs; post-processing that strips labels and folds look-alike separators).
- **Build:** `match2-ocr/Dockerfile` on the mandated `rocm/pytorch` base, built and contract-checked by `.github/workflows/build-image.yml`.

The brief's sample images are not included; `match2-ocr/eval/extract_samples.py` recreates them from the brief PDF. Code is MIT-licensed (see `LICENSE`). The model is Qwen2.5-VL-7B-Instruct (Apache-2.0), downloaded from Hugging Face at build time.

The rest of this README is the working guide used during development.

> **Rules**
> - **Never commit the Docker Hub image reference.** It comes from the env var `DOCKERHUB_REPO` or the gitignored file `.image-ref`, and only on the VM or as the GitHub Actions secret `DOCKERHUB_REPO`.
> - No `.env`, API keys or tokens in the repo or in any image. The images are public.
> - Never build with `--squash` (or flatten the image): the harness checks the base image's layers.
> - Test containers with `--network none`.

## What is in the repo

| Path | What it is |
|---|---|
| `common/resident.py` | Keeps the model loaded between calls. A JSON-lines server over a unix socket (binds only after the model is loaded and warmed up, which is the readiness signal) and a client that waits for readiness up to a deadline. Uses a 127.0.0.1 TCP fallback only on a platform without unix sockets (Windows laptop tests). |
| `common/contract.py` | Atomic JSON output writes and `always_write`, which guarantees an output file even when something raises. |
| `common/gpu.py` | `+rocm` torch check, GPU requirement, device/VRAM reports, and an amd-smi/rocm-smi VRAM reader. torch is imported lazily. |
| `match2-ocr/Dockerfile` | The Match 2 image: mandated ROCm base, deps, baked-in weights, then the code. Build context is the repo root. |
| `match2-ocr/app/app.py` | What the harness runs per image. A thin client: always writes `/app/output/<stem>_output.json` and exits 0. |
| `match2-ocr/app/server.py` | Resident model server started by the container `CMD`. `OCR_BACKEND=vlm` (default) or `fake` (no torch, for laptop tests). |
| `match2-ocr/app/ocr.py` | Image loading, the prompt with the transcription rules, post-processing, the VLM backend. `--self-test`, `--load-check`, `--processor-check MODEL_DIR` (CPU, run at build time). |
| `match2-ocr/app/requirements*.txt` | Non-torch deps (normal install) and transformers/accelerate (`--no-deps`, so the ROCm torch stays). |
| `match2-ocr/eval/extract_samples.py` | Pulls the 10 sample images out of the brief PDF into `eval/samples/` and `eval/samples_allformats/` (+ `expected.json`). |
| `match2-ocr/eval/degrade.py` | Blur, motion blur, noise, glare, low light, skew, tiny and huge variants into `eval/degraded/`. |
| `match2-ocr/eval/score.py` | The brief's normalization and PASS/FAIL per image. |
| `match2-ocr/eval/run_local.py` | Starts `server.py`, runs `app.py` as a new process per image like the harness, times it, reports VRAM, scores. |
| `match2-ocr/eval/make_synthetic.py` | Synthetic plate-like text images in every extension (.png .jpg .jpeg .tif .tiff; RGB, grayscale, RGBA; one 5000 px, one 110 px) + `expected.json` into `eval/synthetic/`. Pillow only. For contract tests without the brief's samples. |
| `.github/workflows/build-image.yml` | Manual GitHub Actions job: build with `build_push.sh`, fake-backend contract check on synthetic images, optional push. See "Build on GitHub Actions". |
| `tools/ci_disk.sh` | Actions runner only: frees disk, puts Docker's storage on the biggest disk, fails if under 60 GiB free. |
| `tools/ci_push.sh` | Actions: tags and pushes the checked image (no rebuild), prints the digest, checks it pulls without credentials. |
| `tools/notebook_setup.sh` | AMD notebook: persistent packages, HF cache and weights; writes `$PERSIST/mc2env.sh`. |
| `tools/vm_setup.sh` | Fresh AMD Developer Cloud VM: checks, `docker login` (token from stdin), clone, pull base image. |
| `tools/build_push.sh` | Build, `+rocm` check, base-layer check, 60 GiB size check, push with `--push`. |
| `tools/selfcheck_mc2.py` | Match 2 version of the official self-check: run it against the built image before submitting. |
| `tools/make_bundle.py` | Laptop: zips the committed repo plus the eval images into `..\amd-ai-league.zip` for upload to the notebook or VM. |

Sample images, degraded images, outputs and weights are gitignored. The scripts recreate them; the brief PDF is not in the repo.

## Laptop (Windows, no GPU, no Docker)

```powershell
$PY = "E:\miniconda\python.exe"
& $PY match2-ocr/eval/extract_samples.py        # default PDF path: ..\AMD AI League — Match 1 & 2 Challenge Brief.pdf
& $PY match2-ocr/eval/degrade.py
& $PY match2-ocr/eval/score.py --self-test
& $PY match2-ocr/app/ocr.py --self-test
& $PY match2-ocr/app/ocr.py --load-check match2-ocr/eval/samples match2-ocr/eval/degraded
& $PY match2-ocr/eval/run_local.py --images match2-ocr/eval/samples_allformats --backend fake
& $PY match2-ocr/eval/make_synthetic.py                 # no PDF needed: eval/synthetic/
& $PY match2-ocr/eval/run_local.py --images match2-ocr/eval/synthetic --backend fake
```

The fake backend answers `FAKE` for everything, so the score is 0; it proves the plumbing, file names and timing.

## Commands you run in the AMD notebook

Open a fresh tab at https://notebooks.amd.com/hackathon every time (the first URL is single-use). Check the pod URL: `jupyter-hack-*` means persistent storage is `/persistent`, `rgapi-hackathon-*` means `/workspace`. In a JupyterLab terminal:

On the laptop, `python tools/make_bundle.py` writes `..\amd-ai-league.zip` (committed code plus the sample, all-format and degraded images). Upload it with the JupyterLab upload button (arrow icon above the file list), then:

```bash
# 1. Code, samples and packages (re-run at the start of every session; it skips finished work)
PERSIST=$([ -w /persistent ] && echo /persistent || echo /workspace)
ZIP=$(ls -t $PERSIST/amd-ai-league.zip ~/amd-ai-league.zip /workspace/amd-ai-league.zip 2>/dev/null | head -1)
cd $PERSIST && python3 -m zipfile -e "$ZIP" . && echo "unpacked $ZIP"
bash $PERSIST/amd-ai-league/tools/notebook_setup.sh Qwen/Qwen2.5-VL-7B-Instruct
source $PERSIST/mc2env.sh
cd $PERSIST/amd-ai-league

# 2. Samples are already in the bundle. (Without the bundle: upload the brief PDF and run
#    match2-ocr/eval/extract_samples.py "<pdf>" then match2-ocr/eval/degrade.py.)

# 3. Evaluate (OCR_MODEL_DIR comes from mc2env.sh)
python3 match2-ocr/eval/run_local.py --images match2-ocr/eval/samples
python3 match2-ocr/eval/run_local.py --images match2-ocr/eval/samples_allformats
python3 match2-ocr/eval/run_local.py --images match2-ocr/eval/degraded
OCR_SECOND_PASS=1 python3 match2-ocr/eval/run_local.py --images match2-ocr/eval/degraded   # optional experiment

# 4. Compare another model (25 GB quota holds one 7-8B model at a time)
rm -rf $PERSIST/models/Qwen__Qwen2.5-VL-7B-Instruct
bash tools/notebook_setup.sh Qwen/Qwen3-VL-8B-Instruct && source $PERSIST/mc2env.sh
python3 match2-ocr/eval/run_local.py --images match2-ocr/eval/samples_allformats
```

Record for each model: score on samples / all formats / degraded, max seconds per image, peak VRAM, GPU arch. Pick by score, then by time per image. Press **Turn-off Session** when done: an idle session still uses the 3 h daily quota.

## Commands you run on the AMD Developer Cloud VM

Create the **single-GPU** MI300X VM (never the 8-GPU node). Then:

```bash
# On the laptop first: python tools/make_bundle.py; scp ..\amd-ai-league.zip root@<vm-ip>:~/
export DOCKERHUB_USER=<docker hub user>
cd ~ && python3 -m zipfile -e amd-ai-league.zip . && cd ~/amd-ai-league   # code + samples
bash tools/vm_setup.sh                       # asks for the Docker Hub access token (hidden)
echo '<dockerhub-user>/<repo-name>' > .image-ref   # VM only; gitignored, never commit it

bash tools/build_push.sh match2-ocr Qwen/Qwen2.5-VL-7B-Instruct --tag v1     # use the model picked in the notebook
python3 tools/selfcheck_mc2.py match2-ocr-local:v1 match2-ocr/eval/samples   # note step 4's cold load time
python3 tools/selfcheck_mc2.py match2-ocr-local:v1 match2-ocr/eval/samples --cap-drop
bash tools/build_push.sh match2-ocr Qwen/Qwen2.5-VL-7B-Instruct --tag v1 --push

# Prove the pushed image works from a clean pull
REF="$(head -n1 .image-ref):v1"
docker rmi match2-ocr-local:v1 "$REF" && docker pull "$REF"
python3 tools/selfcheck_mc2.py "$REF" match2-ocr/eval/samples
```

The exact image build command (what `build_push.sh` runs) is, from the repo root:

```bash
docker build -f match2-ocr/Dockerfile --build-arg MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct --build-arg MODEL_REVISION=main -t match2-ocr-local:v1 .
```

The build fails on purpose if torch or torchvision is not a `+rocm` build, or if the processor cannot encode an image on the CPU (`ocr.py --processor-check`). The base ships torchvision 0.28.0+rocm10.0.0 (Match 3 brief). Only if it is missing, `export TORCHVISION_SPEC="torchvision==<version> --index-url <rocm index>"` before `build_push.sh` (confirm the pair in the notebook first). The image has a Docker `HEALTHCHECK` (`server.py --ping`) that turns healthy once the model is loaded and warmed up; record the cold load time from selfcheck step 4 in your notes.

Destroy the VM at the end of every session (a stopped VM is still billed).

## Build on GitHub Actions (no VM needed)

`.github/workflows/build-image.yml` builds the Match 2 image on a GitHub-hosted runner, checks it without a GPU, and pushes it to Docker Hub when you ask. Use it while the GPU credits are pending. It only runs when you start it by hand.

One-time setup:

1. On github.com: New repository, visibility **Private**, nothing added (no README, .gitignore or licence). Then on the laptop:
   ```powershell
   git remote add origin https://github.com/<github-user>/<repo>.git
   git push -u origin main
   ```
2. On hub.docker.com: Account settings > Personal access tokens > Generate new token, access permissions **Read & Write**. Copy the token.
3. On hub.docker.com: Repositories > Create a repository, visibility **Public** (the harness pulls without credentials).
4. In the GitHub repo: Settings > Secrets and variables > Actions > New repository secret, once for each:
   - `DOCKERHUB_USER`: your Docker Hub user name
   - `DOCKERHUB_TOKEN`: the access token from step 2
   - `DOCKERHUB_REPO`: `<dockerhub-user>/<repo-name>` from step 3, without a tag (the workflow adds the `tag` input)

Each build:

5. Actions tab > **build-image** > Run workflow. Inputs: `match_dir` `match2-ocr`, `model_id` (the model picked in the notebook), `tag` `v1`, `push` off. This first run builds and checks only.
6. When it is green, Run workflow again with the same inputs and `push` on. Each run is a full build on a fresh runner. The push step prints the pushed digest and fails if the image cannot be pulled logged out (then make the Docker Hub repo public and run again). In the log the reference shows as `***`.
7. Submit `<dockerhub-user>/<repo-name>:<tag>` on lablab (Submission steps below).

What the job checks:

- `tools/ci_disk.sh`: deletes preinstalled toolchains the build does not use (Android, .NET, Haskell, CodeQL, ...), puts Docker's storage on the runner disk with the most space, and stops if it has under 60 GiB free (the base is 29.2 GiB, the 7B weights about 16 GB). Runners come with different disk layouts, so a failure here usually passes on a re-run.
- `tools/build_push.sh` (no push): the Dockerfile's own checks (`+rocm` torch and torchvision, `ocr.py --self-test`, `ocr.py --processor-check` on the CPU), then `+rocm` inside the image, the lower layers are the mandated base (the job pulls the base first so this runs), and at most 60 GiB uncompressed.
- `tools/selfcheck_mc2.py --fake-backend` on images from `make_synthetic.py`: the container starts with `--network none` and `OCR_BACKEND=fake`, a call right after start still writes its file, the server becomes ready, every image gets `/app/output/<stem>_output.json` with a string `text` (the fake backend answers `FAKE`, so an empty answer fails), the container keeps running, HEALTHCHECK, no `/app/.env`.
- With `push` on, `tools/ci_push.sh`: tags and pushes the image that was checked (no rebuild), then logs out and runs `docker manifest inspect` with no credentials.

What it cannot check: the runner has no GPU, so nothing here loads the real model on ROCm, measures VRAM (1-48 GiB) or times real inference. Check those in the hackathon notebook with `run_local.py` (score, seconds per image, peak VRAM), and run the full GPU selfcheck (`selfcheck_mc2.py <ref> match2-ocr/eval/samples`) on the VM once the credits arrive.

Cost: private repos get 2,000 free Actions minutes a month on GitHub Free. The job's timeout is 150 minutes, so even two full-length runs fit.

## Submission steps

1. Make the Docker Hub repository public. Check `docker logout && docker pull <ref>` works without credentials.
2. Submit the image reference on lablab (Match 2 deadline: Wed Nov 4). Do not put it in this repo or any public place.
