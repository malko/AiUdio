# Official NeMo-Speech.cpp image + a minimal shell for the wait-for-model entrypoint.
#
# The NGC image is distroless (no /bin/sh), so we add busybox (static) and install
# its applets. No compilation — the speech runtime comes from the official image.
ARG STT_BASE_IMAGE=nvcr.io/nvidia/nemo-speech.cpp:0.2.0

FROM busybox:1.36 AS busybox

FROM ${STT_BASE_IMAGE}
ARG STT_BASE_IMAGE
USER root

COPY --from=busybox /bin/busybox /bin/busybox
COPY docker/stt-wait.sh /stt-wait.sh
# Provide /bin/sh, cat, sleep, ... from busybox.
RUN ["/bin/busybox", "--install", "-s", "/bin"]

ENV NEMO_SPEECH_MODEL_DIR=/hf_cache/nemo-speech/models \
    STT_PORT=8080

ENTRYPOINT ["/bin/sh", "/stt-wait.sh"]
