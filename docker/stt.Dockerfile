# NeMo-Speech.cpp built from source. Used ONLY by the non-NVIDIA overrides
# (cpu / vulkan / rocm). The default stack uses the official NGC image instead
# (see docker-compose.yml).
#
# Backend: cuda (default), cpu, vulkan, metal.
# Base images and CUDA arch list are configurable via build args.
ARG BACKEND=cpu
ARG NEMO_SPEECH_REF=main
ARG CUDA_BUILDER_IMAGE=nvidia/cuda:13.4.2-devel-ubuntu26.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:13.4.2-runtime-ubuntu26.04

# --------------------------------------------------------------- builder ----
FROM ${CUDA_BUILDER_IMAGE} AS builder
ARG BACKEND
ARG NEMO_SPEECH_REF
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

# Fallback: copy the binary + shared libraries if the install layout differs.
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

# Same wait-for-model entrypoint as the official image.
COPY docker/stt-wait.sh /stt-wait.sh
RUN chmod +x /stt-wait.sh && mkdir -p /hf_cache/nemo-speech/models

ENV NEMO_SPEECH_MODEL_DIR=/hf_cache/nemo-speech/models \
    STT_PORT=8080

ENTRYPOINT ["/bin/sh", "/stt-wait.sh"]
