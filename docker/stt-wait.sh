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

# Locate the nemo-speech binary (name on PATH, or common install locations).
BIN="$(command -v nemo-speech 2>/dev/null || true)"
if [ -z "${BIN}" ]; then
  for p in /usr/local/bin/nemo-speech /opt/nemo-speech/bin/nemo-speech /opt/nemo/bin/nemo-speech; do
    if [ -x "${p}" ]; then BIN="${p}"; break; fi
  done
fi
if [ -z "${BIN}" ]; then
  echo "[stt] error: nemo-speech binary not found" >&2
  exit 1
fi

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

echo "[stt] starting: ${BIN} serve --host 0.0.0.0 --port ${PORT} ${ARGS}"
# shellcheck disable=SC2086
exec "${BIN}" serve --host 0.0.0.0 --port "${PORT}" ${ARGS}
