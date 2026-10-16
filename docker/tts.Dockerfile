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

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

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
    fi

COPY app ./app
COPY ui ./ui
COPY config.yaml ./config.yaml

RUN mkdir -p outputs logs voices reference_audio hf_cache

EXPOSE 8004
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8004/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
