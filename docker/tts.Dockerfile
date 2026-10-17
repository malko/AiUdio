# AiUdio gateway (TTS + API + UI). Official PyTorch/CUDA base image (CUDA 13).
# Overridable: --build-arg BASE_IMAGE=... TORCH_SPEC=... TORCH_INDEX_URL=...
ARG BASE_IMAGE=pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime
FROM ${BASE_IMAGE}

# Packages force-reinstalled to match the base image's CUDA variant. Empty by
# default (the CUDA 13 image ships a compatible torch/torchaudio). Since
# torchaudio 2.11 it is ABI-stable and no longer needs to match the torch minor.
ARG TORCH_SPEC=""
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
# INSTALL_TTS=0 builds a lightweight "gateway" image without the TTS engine.
ARG INSTALL_TTS=1
# INSTALL_CHATTERBOX=1 adds the isolated Chatterbox venv (/opt/cb/venv). Chatterbox
# pins different torch/transformers versions than qwen-tts, so it cannot share the
# main environment; app/chatterbox_worker.py runs under this venv.
ARG INSTALL_CHATTERBOX=1
# Chatterbox package to install. The PyPI release (0.1.7) is multilingual **v2 only**;
# V3 is only on GitHub master, so we install from git by default. Pin a tag/commit
# here for reproducibility (e.g. git+https://github.com/resemble-ai/chatterbox.git@<sha>).
ARG CHATTERBOX_SPEC="git+https://github.com/resemble-ai/chatterbox.git"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    # config.yaml is read-only defaults; runtime settings persist in /app/state.
    CONFIG_DEFAULTS=/app/config.yaml \
    CONFIG_PATH=/app/state/config.yaml

RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg sox libsox-fmt-all git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-tts.txt ./

RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && if [ "$INSTALL_TTS" = "1" ]; then \
        pip install -r requirements-tts.txt \
        && if [ -n "$TORCH_SPEC" ]; then \
             pip install --force-reinstall --no-deps $TORCH_SPEC --index-url ${TORCH_INDEX_URL} ; \
           fi ; \
    fi \
 && pip install pytest ruff

# Chatterbox backend: isolated venv (its own transformers, but reuses the base
# image's CUDA 13 torch). Placed before COPY app so app changes do not invalidate
# this heavy layer. See app/chatterbox_backend.py / app/chatterbox_worker.py.
# `--system-site-packages` exposes the base image's GPU-enabled torch (chatterbox
# pins torch==2.6.0 / CUDA 12.4, which has no kernels for Blackwell/sm_120). We
# then drop the torch/torchaudio and nvidia-* CUDA wheels chatterbox pulled into
# the venv, so the system's CUDA 13 build (and its NCCL/cu13 libs) take over.
RUN if [ "$INSTALL_CHATTERBOX" = "1" ]; then \
        set -e; \
        apt-get update && apt-get install -y --no-install-recommends \
            build-essential python3-venv python3-dev \
        && rm -rf /var/lib/apt/lists/*; \
        python -m venv --system-site-packages /opt/cb/venv; \
        /opt/cb/venv/bin/pip install --upgrade pip; \
        /opt/cb/venv/bin/pip install "${CHATTERBOX_SPEC}"; \
        NVIDIA_PKGS=$(/opt/cb/venv/bin/pip freeze --local | sed -n 's/^\(nvidia-[^=]*\)==.*/\1/p'); \
        /opt/cb/venv/bin/pip uninstall -y torch torchaudio $NVIDIA_PKGS; \
    fi

COPY app ./app
COPY ui ./ui
COPY config.yaml ./config.yaml

RUN mkdir -p outputs logs voices reference_audio hf_cache state

EXPOSE 8004
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8004/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
