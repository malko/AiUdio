# STT: NeMo-Speech.cpp is not published to a registry, so it is compiled from the
# official sources (NVIDIA/NeMo-Speech.cpp repository) using the multi-stage build
# below. Selectable backend: cuda (default), cpu, vulkan, metal.
#
# Base OS: Ubuntu 26.04 LTS (NVIDIA CUDA 13.4 images), which ships a recent CMake
# (>= 4.x) required by NeMo-Speech.cpp's CMakePresets. Override the CUDA/OS images
# via CUDA_BUILDER_IMAGE / CUDA_RUNTIME_IMAGE (e.g. CUDA 12.8 + Ubuntu 24.04 for
# older drivers).
#
# The build directory depends on the upstream preset; installation goes through
# `cmake --install` and, as a fallback, a targeted copy of the binary.
ARG BACKEND=cuda
ARG NEMO_SPEECH_REF=main
# CUDA base images for the builder/runtime stages (override for CUDA 12/13).
ARG CUDA_BUILDER_IMAGE=nvidia/cuda:13.4.2-devel-ubuntu26.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:13.4.2-runtime-ubuntu26.04

# --------------------------------------------------------------- builder ----
FROM ${CUDA_BUILDER_IMAGE} AS builder
ARG BACKEND
ARG NEMO_SPEECH_REF
# Optional CUDA arch list (e.g. "75;80;86;89;90"). When empty, a broad portable
# list is used so the build works on any target GPU (the builder has no GPU, so
# ggml's "-arch=native" detection would otherwise fall back to a default arch).
ARG CUDA_ARCHITECTURES=""

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git pkg-config \
      libsentencepiece-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Vulkan dependencies only for backend=vulkan.
RUN if [ "$BACKEND" = "vulkan" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        libvulkan-dev glslc spirv-headers \
      && rm -rf /var/lib/apt/lists/* ; \
    fi

RUN git clone --depth 1 --branch "${NEMO_SPEECH_REF}" \
      https://github.com/NVIDIA/NeMo-Speech.cpp.git /src
WORKDIR /src
RUN git submodule update --init llama.cpp third_party/cpp-httplib

RUN archs="${CUDA_ARCHITECTURES}" ; \
    if [ -z "$archs" ] && [ "$BACKEND" = "cuda" ]; then \
      archs="75;80;86;89;90;100;120" ; \
    fi ; \
    cmake --version \
 && if [ -n "$archs" ]; then \
      scripts/configure.sh ${BACKEND}-server -DCMAKE_CUDA_ARCHITECTURES="$archs" ; \
    else \
      scripts/configure.sh ${BACKEND}-server ; \
    fi \
 && cmake --build --preset ${BACKEND}-server -j"$(nproc)" \
 && (cmake --install build/${BACKEND}-server --prefix /opt/nemo || true)

# Fallback: locate and copy the binary + shared libraries when the install did
# not produce a usable tree.
RUN mkdir -p /opt/nemo/bin /opt/nemo/lib \
 && (cp -a /opt/nemo/bin/. /opt/nemo/bin/ 2>/dev/null || true) \
 && find /src/build -name 'nemo-speech' -type f -perm -u+x \
      -exec cp {} /opt/nemo/bin/nemo-speech \; \
 && find /src/build -name 'libnemo_speech_*.so*' -exec cp {} /opt/nemo/lib/ \; 2>/dev/null || true

# --------------------------------------------------------------- runtime ----
FROM ${CUDA_RUNTIME_IMAGE} AS runtime
ARG BACKEND

# libssl package name differs across Ubuntu releases (libssl3t64 >= 24.04, libssl3
# on 22.04); install whichever exists.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsentencepiece0 libgomp1 ca-certificates \
    && (for p in libssl3t64 libssl3; do apt-get install -y --no-install-recommends "$p" 2>/dev/null && break; done || true) \
    && if [ "$BACKEND" = "vulkan" ]; then \
         apt-get install -y --no-install-recommends libvulkan1 ; \
       fi \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/nemo/bin /usr/local/bin
COPY --from=builder /opt/nemo/lib /usr/local/lib
RUN ldconfig

# Model index (catalog of downloadable models). The binary looks it up next to its
# own prefix (/usr/local/share/nemo-speech/model-index.json); we also set the
# override env var for robustness.
RUN mkdir -p /usr/local/share/nemo-speech
COPY --from=builder /src/models/index.json /usr/local/share/nemo-speech/model-index.json

COPY docker/stt-entrypoint.sh /usr/local/bin/stt-entrypoint.sh
RUN chmod +x /usr/local/bin/stt-entrypoint.sh

ENV NEMO_SPEECH_MODEL_INDEX=/usr/local/share/nemo-speech/model-index.json \
    NEMO_SPEECH_MODEL_DIR=/models \
    STT_ASR_MODEL=nemotron-3.5 \
    STT_PORT=8080 \
    STT_MODELS_DIR=/models

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/stt-entrypoint.sh"]
