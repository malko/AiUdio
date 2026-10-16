# STT: NeMo-Speech.cpp is not published to a registry, so it is compiled from the
# official sources (NVIDIA/NeMo-Speech.cpp repository) using the multi-stage build
# below. Selectable backend: cuda (default), cpu, vulkan, metal.
#
# The build directory depends on the upstream preset; installation goes through
# `cmake --install` and, as a fallback, a targeted copy of the binary.
ARG BACKEND=cuda
ARG NEMO_SPEECH_REF=main

# --------------------------------------------------------------- builder ----
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder
ARG BACKEND
ARG NEMO_SPEECH_REF

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

RUN scripts/configure.sh ${BACKEND}-server \
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
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS runtime
ARG BACKEND

RUN apt-get update && apt-get install -y --no-install-recommends \
      libsentencepiece0 libgomp1 libssl3 ca-certificates \
    && if [ "$BACKEND" = "vulkan" ]; then \
         apt-get install -y --no-install-recommends libvulkan1 ; \
       fi \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/nemo/bin /usr/local/bin
COPY --from=builder /opt/nemo/lib /usr/local/lib
RUN ldconfig

COPY docker/stt-entrypoint.sh /usr/local/bin/stt-entrypoint.sh
RUN chmod +x /usr/local/bin/stt-entrypoint.sh

ENV STT_ASR_MODEL=nemotron-3.5 \
    STT_PORT=8080 \
    STT_MODELS_DIR=/models

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/stt-entrypoint.sh"]
