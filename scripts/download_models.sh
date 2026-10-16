#!/usr/bin/env bash
# Pre-download the Qwen3-TTS models into the Hugging Face cache.
# Usage: ./scripts/download_models.sh [directory]
set -euo pipefail

HF_HOME_DIR="${1:-$(pwd)/hf_cache}"
export HF_HOME="${HF_HOME_DIR}"
mkdir -p "${HF_HOME_DIR}"

MODELS=(
  "Qwen/Qwen3-TTS-Tokenizer-12Hz"
  "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
  "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
)

if command -v huggingface-cli >/dev/null 2>&1; then
  CLIENT=(huggingface-cli download)
elif command -v hf >/dev/null 2>&1; then
  CLIENT=(hf download)
else
  echo "❌ huggingface_hub CLI not found. Install with: pip install 'huggingface_hub[cli]'" >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  echo "⬇️  ${model}"
  "${CLIENT[@]}" "${model}" --local-dir "${HF_HOME_DIR}/models/$(basename "${model}")" \
    ${HF_TOKEN:+--token "${HF_TOKEN}"} || echo "⚠️  failed for ${model}"
done

echo "✅ Download complete into ${HF_HOME_DIR}"
