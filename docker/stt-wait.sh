#!/bin/sh
# Wait for a downloaded STT model, then start the NeMo-Speech.cpp HTTP server.
#
# The gateway downloads GGUF models into the shared volume and writes a pointer
# file (".active") with the chosen filename. This works for both the official
# NGC image and the source build (no model index or downloader inside the
# container). Set NEMO_SPEECH_MODEL_DIR to the cache directory, STT_PORT for the
# listening port.
set -eu

DIR="${NEMO_SPEECH_MODEL_DIR:-/hf_cache/nemo-speech/models}"
PORT="${STT_PORT:-8080}"

echo "[stt] waiting for a model in ${DIR} (download it from the UI)."
until [ -s "${DIR}/.active" ]; do
  sleep 3
done

FILE="$(cat "${DIR}/.active")"
until [ -f "${DIR}/${FILE}" ]; do
  echo "[stt] waiting for ${FILE} ..."
  sleep 3
done

ARGS="--asr-model ${DIR}/${FILE}"
if [ -s "${DIR}/.active-diar" ]; then
  DIAR="$(cat "${DIR}/.active-diar")"
  if [ -f "${DIR}/${DIAR}" ]; then
    ARGS="${ARGS} --diar-model ${DIR}/${DIAR}"
  fi
fi

echo "[stt] starting: nemo-speech serve --host 0.0.0.0 --port ${PORT} ${ARGS}"
# shellcheck disable=SC2086
exec nemo-speech serve --host 0.0.0.0 --port "${PORT}" ${ARGS}
