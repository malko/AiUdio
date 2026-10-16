#!/usr/bin/env bash
# Pre-download the Qwen3-TTS models with plain curl (no Python dependency).
#
# It queries the Hugging Face API for the file list of each repo, then fetches
# every file through the public "resolve" endpoint (LFS redirects are followed).
#
# Usage:
#   ./scripts/download_models.sh [destination-dir]
#
# Environment:
#   HF_TOKEN     optional token (private/gated repos or higher rate limits)
#   HF_ENDPOINT  optional API host (default: https://huggingface.co; use a
#                mirror such as https://hf-mirror.com when needed)
set -euo pipefail

DEST="${1:-$(pwd)/hf_cache/models}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_TOKEN="${HF_TOKEN:-}"

MODELS=(
  "Qwen/Qwen3-TTS-Tokenizer-12Hz"
  "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
  "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
)

if ! command -v curl >/dev/null 2>&1; then
  echo "❌ curl is required but was not found in PATH." >&2
  exit 1
fi

auth=()
[[ -n "${HF_TOKEN}" ]] && auth=(-H "Authorization: Bearer ${HF_TOKEN}")

# List the files of a repo (one "rfilename" per line), using the HF API.
list_files() {
  local repo="$1"
  curl -fsSL "${auth[@]}" "${HF_ENDPOINT}/api/models/${repo}" |
    tr ',' '\n' |
    sed -n 's/.*"rfilename"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'
}

# Download a single file, resuming partial transfers and creating subdirectories.
download_file() {
  local repo="$1" file="$2" outdir="$3"
  curl -fL --create-dirs --retry 3 --retry-delay 2 -C - "${auth[@]}" \
    -o "${outdir}/${file}" \
    "${HF_ENDPOINT}/${repo}/resolve/main/${file}"
}

rc=0
for repo in "${MODELS[@]}"; do
  name="${repo##*/}"
  outdir="${DEST}/${name}"
  echo "⬇️  ${repo} -> ${outdir}"

  files="$(list_files "${repo}")" || {
    echo "⚠️  could not list ${repo} (network or token?)" >&2
    rc=1
    continue
  }
  if [[ -z "${files}" ]]; then
    echo "⚠️  no files returned for ${repo}" >&2
    rc=1
    continue
  fi

  while IFS= read -r file; do
    [[ -z "${file}" ]] && continue
    echo "   • ${file}"
    download_file "${repo}" "${file}" "${outdir}" || {
      echo "   ⚠️  failed: ${file}" >&2
      rc=1
    }
  done <<< "${files}"
done

if [[ "${rc}" -eq 0 ]]; then
  echo "✅ Download complete into ${DEST}"
else
  echo "⚠️  Completed with errors (see messages above)." >&2
fi
exit "${rc}"
