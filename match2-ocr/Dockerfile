# Match 2 (OCR) submission image.
#
# Build from the REPO ROOT (the context must contain common/):
#
#   docker build -f match2-ocr/Dockerfile --build-arg MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct -t mc2-local:v1 .
#
# or use tools/build_push.sh, which also runs the +rocm, base-layer and size checks.
#
# The final (only) stage MUST start from the mandated base: the harness checks layer
# identity against it. Never build with --squash, buildah squash or a FROM scratch copy.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1. Dependencies. Non-torch packages normally; transformers/accelerate with --no-deps so
#    pip cannot replace the base image's ROCm torch. Then prove torch AND torchvision are ROCm.
#
#    torchvision is required: the Qwen2.5-VL / Qwen3-VL processors always build a video
#    sub-processor that exists only for torchvision. The base already ships
#    torchvision 0.28.0+rocm10.0.0 built against its torch 2.13.0 (measured on the evaluation
#    hardware, Match 3 brief), so by default nothing is installed and the assert below proves it
#    is there. If a base ever lacks it, PyTorch's index has no 0.28.0 build for ROCm 10.0; check
#    the right pair in the notebook first, then pass for example
#      --build-arg TORCHVISION_SPEC="torchvision==0.28.0 --index-url https://download.pytorch.org/whl/rocm7.2"
#    (installed with --no-deps so it cannot touch torch).
ARG TORCHVISION_SPEC=""
COPY match2-ocr/app/requirements.txt match2-ocr/app/requirements-nodeps.txt /app/
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt \
 && python3 -m pip install --no-cache-dir --no-deps -r /app/requirements-nodeps.txt \
 && if [ -n "${TORCHVISION_SPEC}" ]; then python3 -m pip install --no-cache-dir --no-deps ${TORCHVISION_SPEC}; fi \
 && python3 -c "import torch, torchvision as tv; assert '+rocm' in torch.__version__, torch.__version__; assert '+rocm' in tv.__version__, tv.__version__; print('torch', torch.__version__, 'torchvision', tv.__version__)" \
 && python3 -c "import transformers, accelerate, jinja2; print('transformers', transformers.__version__, 'accelerate', accelerate.__version__, 'jinja2', jinja2.__version__)"

# 2. Weights, baked in (no network needed at evaluation). This layer comes before the code
#    so editing app/ or common/ does not re-download or re-push ~16 GB.
#    Changing the model is a build arg only: OCR_MODEL_DIR follows MODEL_ID.
ARG MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
ARG MODEL_REVISION=main
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}', revision='${MODEL_REVISION}', local_dir='/models/${MODEL_ID}', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model', '*.tiktoken'])" \
 && echo "${MODEL_ID}@${MODEL_REVISION}" > "/models/${MODEL_ID}/MODEL_ID" \
 && rm -rf "/models/${MODEL_ID}/.cache" /root/.cache
ENV OCR_MODEL_DIR=/models/${MODEL_ID} \
    OCR_BACKEND=vlm \
    MC2_SOCKET=/tmp/mc2-ocr.sock \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/hf

# 3. Code last: cheap to rebuild.
#    --processor-check runs on the CPU: AutoProcessor + chat template + the exact processor call
#    used in serving, on synthetic images. A missing torchvision/jinja2 or a processor API
#    change fails the build here instead of shipping a server that cannot start.
COPY common/ /app/common/
COPY match2-ocr/app/ /app/
RUN mkdir -p /app/input /app/output \
 && python3 /app/ocr.py --self-test \
 && python3 /app/ocr.py --processor-check "${OCR_MODEL_DIR}"

# Readiness for Docker: healthy once the server answers ping (model loaded and warmed up).
HEALTHCHECK --interval=5s --timeout=5s --start-period=600s --retries=3 \
  CMD ["python3", "/app/server.py", "--ping"]

# The harness execs `python3 /app/app.py --input-image ...` into the running container.
# Start the resident model server in the background (log: /tmp/server.log) and keep the
# container alive with sleep even if the server dies, so app.py can still write its
# fallback output instead of the exec failing. If the server crashes after loading it is
# restarted (up to 3 runs); a load failure leaves $MC2_SOCKET.failed and is not retried,
# so clients give up at once instead of waiting through repeated reloads.
CMD ["/bin/sh", "-c", "(for i in 1 2 3; do python3 /app/server.py; [ -e \"$MC2_SOCKET.failed\" ] && break; sleep 1; done) >> /tmp/server.log 2>&1 & exec sleep infinity"]
