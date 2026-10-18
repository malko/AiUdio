# curl examples

```bash
export BASE=http://localhost:8004
# export KEY=aiud_...            # only if auth is enabled
[ -n "$KEY" ] && AUTH=(-H "X-API-Key: $KEY") || AUTH=()
```

## Discovery

```bash
curl -s "$BASE/health" | jq
curl -s "$BASE/ready" | jq
curl -s "${AUTH[@]}" "$BASE/api/capabilities" | jq
curl -s "${AUTH[@]}" "$BASE/api/config" | jq '.models.families'
```

## TTS — one-shot WAV

```bash
curl -s "${AUTH[@]}" "$BASE/api/tts" -H 'Content-Type: application/json' \
  -d '{"text":"Hello world","family":"custom_voice","speaker":"Vivian","language":"English"}' \
  -o out.wav
```

## TTS — streaming PCM (low latency)

```bash
curl -N "${AUTH[@]}" "$BASE/api/tts" -H 'Content-Type: application/json' \
  -d '{"text":"Sentence one. Sentence two.","stream":true,"format":"pcm"}' \
  | ffplay -f s16le -ar 24000 -ch_layout mono -
```

## TTS — clone from a reference clip (base64 data-URL)

```bash
REF="data:audio/wav;base64,$(base64 -w0 ref.wav)"
curl -s "${AUTH[@]}" "$BASE/api/tts" -H 'Content-Type: application/json' \
  -d "{\"text\":\"Cloned voice\",\"family\":\"base\",\"ref_audio\":\"$REF\"}" -o clone.wav
```

## TTS — reusable avatar

```bash
curl -s "${AUTH[@]}" "$BASE/api/tts" -H 'Content-Type: application/json' \
  -d '{"text":"Hello","avatar":"narrator"}' -o out.wav
```

## Voices

```bash
curl -s "${AUTH[@]}" "$BASE/api/voices?family=custom_voice" | jq          # speakers + languages
curl -s "${AUTH[@]}" "$BASE/v1/audio/voices" | jq                        # voices + voice-bank avatars
curl -s "${AUTH[@]}" "$BASE/api/avatars" | jq '.avatars[].id'            # voice-bank ids
```

## OpenAI-compatible

```bash
curl -s "${AUTH[@]}" "$BASE/v1/audio/speech" -H 'Content-Type: application/json' \
  -d '{"input":"Hello there","voice":"Ryan","response_format":"wav"}' -o hello.wav
```

## STT

```bash
curl -s "${AUTH[@]}" "$BASE/v1/audio/transcriptions" \
  -F file=@audio.wav -F response_format=json | jq

curl -s "${AUTH[@]}" "$BASE/api/stt/diarize" -F file=@meeting.wav | jq
```

## Chat (SSE, text only)

```bash
curl -sN "${AUTH[@]}" "$BASE/api/chat" -H 'Content-Type: application/json' \
  -d '{"text":"Hello!"}'
```
