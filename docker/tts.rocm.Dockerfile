# AMD ROCm variant (official rocm/pytorch base image).
# FlashAttention 2 is not available: use attn_implementation=sdpa (the default).
# See docs/backends.md.
ARG BASE_IMAGE=rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0
FROM ${BASE_IMAGE}

ARG INSTALL_TTS=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HSA_OVERRIDE_GFX_VERSION=11.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-tts.txt ./

RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && if [ "$INSTALL_TTS" = "1" ]; then pip install -r requirements-tts.txt ; fi

COPY app ./app
COPY ui ./ui
COPY config.yaml ./config.yaml
RUN mkdir -p outputs logs voices reference_audio hf_cache

EXPOSE 8004
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
