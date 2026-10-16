#!/usr/bin/env bash
# Entrypoint for the STT container (NeMo-Speech.cpp).
#
# Model downloads are NOT forced at startup: the server only starts once the ASR
# model is present in the cache (downloaded on demand from the UI via the gateway,
# or pre-seeded). Set STT_AUTO_DOWNLOAD=true to let NeMo download it at startup.
#
# Environment:
#   STT_ASR_MODEL           ASR model index name/alias (default: nemotron-3.5) or a .gguf path
#   STT_DIAR_MODEL          optional diarization model (name or path)
#   STT_PORT                listening port (default: 8080)
#   STT_AUTO_DOWNLOAD       "true" lets NeMo download the model at startup (default: false)
#   STT_EXTRA_ARGS          extra arguments passed to `nemo-speech serve`
#   NEMO_SPEECH_MODEL_INDEX model index path
#   NEMO_SPEECH_MODEL_DIR   model cache directory
set -euo pipefail

INDEX="${NEMO_SPEECH_MODEL_INDEX:-/usr/local/share/nemo-speech/model-index.json}"
CACHE="${NEMO_SPEECH_MODEL_DIR:-/hf_cache/nemo-speech/models}"
SHARE_DIR="$(dirname "$CACHE")"
SPEC="${STT_ASR_MODEL:-nemotron-3.5}"
DIAR="${STT_DIAR_MODEL:-}"
PORT="${STT_PORT:-8080}"
AUTO="${STT_AUTO_DOWNLOAD:-false}"

mkdir -p "$CACHE" "$SHARE_DIR"
# Expose the model index on the shared volume so the gateway can offer downloads.
if [ -f "$INDEX" ]; then
  cp -f "$INDEX" "$SHARE_DIR/model-index.json" || true
fi

# Resolve a model spec to its relative cache path (<org>/<repo>/<revision>/<file>).
resolve_rel() {
  local spec="$1" role="$2"
  [ -f "$INDEX" ] || return 0
  jq -r --arg s "$spec" --arg r "$role" '
    .models[]
    | select(((.aliases // []) | index($s)) or (.repo == $s))
    | . as $m
    | ($m.artifacts[] | select(.role == $r and .type == "file"))
    | "\($m.repo)/\((.revision // $m.revision))/\(.filename)"
  ' "$INDEX" 2>/dev/null | head -n1
}

# Print the local path of a model if it is present in the cache.
resolve_path() {
  local spec="$1" role="$2" rel
  [ -z "$spec" ] && return 0
  [ -f "$spec" ] && { printf '%s' "$spec"; return; }
  [ -f "$CACHE/$spec" ] && { printf '%s' "$CACHE/$spec"; return; }
  rel="$(resolve_rel "$spec" "$role" || true)"
  [ -n "$rel" ] && [ -f "$CACHE/$rel" ] && printf '%s' "$CACHE/$rel"
  return 0
}

build_and_run() {
  local args=(serve --host 0.0.0.0 --port "$PORT")
  local asr diar
  asr="$(resolve_path "$SPEC" asr)"
  args+=(--asr-model "${asr:-$SPEC}")
  if [ -n "$DIAR" ]; then
    diar="$(resolve_path "$DIAR" diarization)"
    args+=(--diar-model "${diar:-$DIAR}")
  fi
  # shellcheck disable=SC2206
  if [ -n "${STT_EXTRA_ARGS:-}" ]; then
    args+=(${STT_EXTRA_ARGS})
  fi
  echo "[stt] starting: nemo-speech ${args[*]}"
  exec nemo-speech "${args[@]}"
}

if [ "$AUTO" = "true" ]; then
  build_and_run
fi

while true; do
  if [ -n "$(resolve_path "$SPEC" asr)" ]; then
    build_and_run
  fi
  echo "[stt] ASR model '${SPEC}' not cached yet."
  echo "[stt] download it from the UI (STT tab) or set STT_AUTO_DOWNLOAD=true; retrying in 5s."
  sleep 5
done
