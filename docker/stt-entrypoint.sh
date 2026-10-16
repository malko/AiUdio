#!/usr/bin/env bash
# Entrypoint for the STT container (NeMo-Speech.cpp).
# Environment:
#   STT_ASR_MODEL   ASR model (default: nemotron-3.5; alternative: parakeet-tdt-0.6b-v3)
#   STT_DIAR_MODEL  optional diarization model
#   STT_PORT        listening port (default: 8080)
#   STT_EXTRA_ARGS  extra arguments passed to `nemo-speech serve`
set -euo pipefail

# Model cache: NeMo-Speech.cpp downloads models into $NEMO_SPEECH_MODEL_DIR.
export NEMO_SPEECH_MODEL_DIR="${NEMO_SPEECH_MODEL_DIR:-${STT_MODELS_DIR:-/models}}"

ARGS=(serve --host 0.0.0.0 --port "${STT_PORT:-8080}" --asr-model "${STT_ASR_MODEL:-nemotron-3.5}")
if [[ -n "${STT_DIAR_MODEL:-}" ]]; then
  ARGS+=(--diar-model "${STT_DIAR_MODEL}")
fi
# shellcheck disable=SC2206
if [[ -n "${STT_EXTRA_ARGS:-}" ]]; then
  ARGS+=(${STT_EXTRA_ARGS})
fi

echo "[stt] starting: nemo-speech ${ARGS[*]}"
exec nemo-speech "${ARGS[@]}"
