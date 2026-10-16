# CPU variant (official python:slim base image). Slow for Qwen3-TTS but portable.
FROM python:3.12-slim

ARG TORCH_SPEC="torch==2.5.1 torchaudio==2.5.1"
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG INSTALL_TTS=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg git curl \
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
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
