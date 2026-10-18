"""Isolated Chatterbox worker.

This script runs under a **dedicated virtualenv** (see ``docker/tts.Dockerfile``)
because ``chatterbox-tts`` pins different ``torch``/``transformers`` versions than
``qwen-tts`` and the two cannot coexist in one environment. The gateway spawns it
as a subprocess and drives it over a line-delimited JSON protocol on stdin/stdout.

It is deliberately self-contained: it only imports the standard library plus
packages available in its own venv (``chatterbox``, ``soundfile``). It must not
import the gateway package.

Protocol (one JSON object per line):

    -> {"cmd": "load", "family": "chatterbox", "repo": "...", "t3_model": "v3",
        "nano": false, "device": "cuda:0", "watermark": false}
    <- {"ok": true}

    -> {"cmd": "synth", "texts": ["hi"], "language": "fr",
        "audio_prompt": "/path/ref.wav", "params": {"temperature": 0.8},
        "out_dir": "/tmp/xyz"}
    <- {"ok": true, "items": [{"path": "/tmp/xyz/0.wav", "sr": 24000, "parts": 1}]}

A text longer than the model's positional-embedding table cannot be embedded (the
gather kernel asserts and poisons the context), so it is split into ``parts``
pieces and the audio is concatenated — the caller still gets one wav per text.

    -> {"cmd": "languages"}   <- {"ok": true, "languages": ["en", "fr", ...]}
    -> {"cmd": "unload"}      <- {"ok": true}
    -> {"cmd": "shutdown"}    <- {"ok": true}   (then exit)

Errors are answered with ``{"ok": false, "error": "...", "trace": "...", "fatal": bool}``.
``fatal: true`` marks a poisoned CUDA context (device-side assert, illegal access,
OOM...): the worker exits right after replying and the gateway respawns a clean one.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import traceback
from pathlib import Path

# The protocol stream (real stdout). When run as a script we rebind sys.stdout to
# stderr (see ``__main__``) so library prints/logging can never corrupt it.
_OUT = sys.stdout

# Empty HF_* token variables are treated as a real (empty) bearer token by
# huggingface_hub ("Illegal header value b'Bearer '"). Drop them entirely.
for _var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    if _var in os.environ and not os.environ[_var].strip():
        del os.environ[_var]

MODEL = None
FAMILY = ""
SR = 24000

# Supported language name -> ISO. Kept in sync with app/languages.py (the worker
# cannot import the gateway package).
ISO_TO_NAME = {
    "ar": "Arabic", "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French",
    "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "tr": "Turkish", "vi": "Vietnamese", "zh": "Chinese",
}


class _NoopWatermarker:
    """Drop-in replacement for perth's watermarker that does nothing."""

    def apply_watermark(self, wav, sample_rate=None):
        return wav


# A CUDA fault (device-side assert, illegal access, OOM...) leaves the worker's
# context poisoned: the process survives but every later call fails with the very
# same error. Those are reported as `fatal` and make the worker exit, so the
# gateway tears the process down and spawns a clean one instead of reusing a
# wedged context until the container is restarted.
_FATAL_MARKERS = (
    "cuda error",
    "cuda_error",
    "acceleratorerror",
    "device-side assert",
    "illegal memory access",
    "unspecified launch failure",
    "misaligned address",
    "out of memory",
)


def _is_fatal_error(exc: BaseException) -> bool:
    """True when ``exc`` poisoned the CUDA context (worker must not continue)."""
    text = str(exc).lower()
    return any(marker in text for marker in _FATAL_MARKERS)


def _log(msg: str) -> None:
    print(f"[chatterbox-worker] {msg}", file=sys.stderr, flush=True)


def _params(callable_obj):
    try:
        return inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return None


def _accepts(callable_obj, name: str) -> bool:
    """True if ``callable_obj`` accepts keyword ``name`` (or has **kwargs)."""
    params = _params(callable_obj)
    if params is None or name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _filter_kwargs(callable_obj, kwargs: dict) -> dict:
    """Drop keywords not accepted by ``callable_obj``'s signature."""
    params = _params(callable_obj)
    if params is None or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _detect_t3(dirpath: Path, preferred: str) -> str:
    if preferred and (dirpath / preferred).exists():
        return preferred
    for candidate in ("t3_mtl23ls_v3.safetensors", "t3_mtl23ls_v2.safetensors"):
        if (dirpath / candidate).exists():
            return candidate
    found = sorted(p.name for p in dirpath.glob("t3_*.safetensors"))
    return found[0] if found else preferred


# ------------------------------------------------------------------- text length
# The multilingual T3 embeds text through a learned positional table sized
# ``max_text_tokens + 2`` (the +2 is the start/stop pair ``mtl_tts.generate`` pads
# in). A longer text makes that gather read past the table: the CUDA kernel
# asserts, which *also* poisons the worker's context (every later call then fails
# with "device-side assert triggered").
#
# Counting tokens exactly would mean running the multilingual tokenizer — which,
# besides being redundant with `generate()`, triggers heavy per-language
# preprocessing (`pkuseg` + Cangjie for Chinese, `pykakasi` for Japanese, an ONNX
# diacritizer for Hebrew, a stresser for Russian) on the time-to-first-audio path.
# So the text is bounded in **characters**, deliberately more conservative than
# the tokenizer needs to be: `_CHAR_EXPANSION` is how many tokens one character
# can become (checked against the shipped `Cangjie5_TC.json`: 126 610 entries,
# never more than 8 tokens per glyph, worst case `[cj_m][cj_m][cj_h][cj_a][cj_f]
# [cj_1][cj_0][cj_.]`; Japanese/Korean decompositions and Hebrew/Russian marks add
# fewer; plain Latin only splits on NFKD). The resulting budget stays above
# `chunking.max_chars` (default 240) for every language, so the gateway's own
# chunks are never split a second time here.
_SPLIT_SILENCE_MS = 120       # matches `chunking.silence_ms` (default)
_SPLIT_MARGIN_TOKENS = 2      # keeps the start/stop pair off the table's end
_CHAR_EXPANSION = {"zh": 8, "ja": 8, "ko": 8, "he": 8, "ru": 8}
_DEFAULT_EXPANSION = 2


def _text_budget(model):
    """Text tokens the loaded model can embed, or None when it has no limit."""
    t3 = getattr(model, "t3", None)
    if t3 is None or getattr(t3, "text_pos_emb", None) is None:
        return None
    limit = getattr(getattr(t3, "hp", None), "max_text_tokens", None)
    return int(limit) if limit else None


def _char_budget(model, language) -> int | None:
    """Characters that fit the model's text table for ``language`` (None: no limit)."""
    limit = _text_budget(model)
    if limit is None:
        return None
    factor = _CHAR_EXPANSION.get((language or "").lower(), _DEFAULT_EXPANSION)
    return max(1, (limit - _SPLIT_MARGIN_TOKENS) // factor)


# Boundary classes, strongest first (kept in sync with `app/chunking.py`: the
# worker cannot import the gateway package). A piece ends on a sentence, else on
# a clause, else on a line break, else between words — never inside a word.
_BOUNDARY_CLASSES = (
    ".!?…؟۔।。！？",       # end of sentence
    ",;:—–、，；：",        # end of clause
    "\n",                   # end of line / paragraph
    " \t",                  # between words
)
_BOUNDARY_RANK = {
    ch: rank for rank, chars in enumerate(_BOUNDARY_CLASSES) for ch in chars
}
_MIN_LOOKBACK = 24


def _cut_point(text: str, limit: int) -> int:
    """Best cut index at most ``limit``: sentence > clause > line > word."""
    window = max(_MIN_LOOKBACK, limit // 4)
    found = {}
    for idx in range(limit, max(0, limit - window), -1):
        rank = _BOUNDARY_RANK.get(text[idx - 1])
        if rank is not None and rank not in found:
            found[rank] = idx
    if found:
        return found[min(found)]
    # Nothing strong nearby: fall back to the nearest word/line break, without
    # cutting the piece below half the budget. A *single* token longer than the
    # budget is the only case where a cut inside it is unavoidable.
    floor = max(1, limit // 2)
    for idx in range(limit, floor - 1, -1):
        if text[idx - 1] in " \t\n":
            return idx
    return limit


def _split_text(text: str, budget: int) -> list[str]:
    """Split ``text`` into pieces of at most ``budget`` characters."""
    pieces: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= budget:
            pieces.append(remaining.strip())
            break
        cut = _cut_point(remaining, budget)
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    return [p for p in pieces if p]


def _split_for_model(model, text: str, language: str) -> list[str]:
    """Split ``text`` so no piece can exceed the model's text-token table."""
    if not text.strip():
        return [text]
    budget = _char_budget(model, language)
    if budget is None or len(text) <= budget:
        return [text]
    return _split_text(text, budget)

    # Cheap first: below this many characters the tokenizer cannot reach the
    # model's limit, so the (sometimes heavy) tokenization is skipped entirely
    # and the text goes to `generate()` untouched — nothing added to
    # time-to-first-audio in the normal (chunked, short) case.
    factor = _CHAR_EXPANSION.get((language or "").lower(), _DEFAULT_EXPANSION)
    if len(text) * factor <= budget:
        return [text], ""
    return _split_text(text, budget)


def _join_wavs(wavs: list, sample_rate: int):
    """Concatenate pieces with a short silence between them (numpy array)."""
    import numpy as np

    gap = np.zeros(int(sample_rate * _SPLIT_SILENCE_MS / 1000.0), dtype=np.float32)
    parts = []
    for i, wav in enumerate(wavs):
        arr = wav.detach().cpu().numpy() if hasattr(wav, "detach") else wav
        if i:
            parts.append(gap)
        parts.append(np.asarray(arr, dtype=np.float32).reshape(-1))
    return np.concatenate(parts)


def _load(req: dict) -> None:
    global MODEL, FAMILY, SR

    family = req.get("family", "chatterbox")
    repo = req.get("repo") or "ResembleAI/chatterbox"
    device = req.get("device") or "cpu"
    watermark = bool(req.get("watermark", False))

    try:
        import torch

        _log(
            f"torch {torch.__version__} cuda={torch.version.cuda} "
            f"avail={torch.cuda.is_available()} "
            f"cap={torch.cuda.get_device_capability() if torch.cuda.is_available() else None} "
            f"archs={torch.cuda.get_arch_list() if torch.cuda.is_available() else []}"
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        _log(f"torch diagnostics failed: {exc}")

    if family in ("chatterbox_turbo", "chatterbox_nano"):
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        kwargs: dict = {"device": device}
        if _accepts(ChatterboxTurboTTS.from_pretrained, "nano"):
            kwargs["nano"] = family == "chatterbox_nano"
        elif family == "chatterbox_nano":
            _log("installed chatterbox does not support nano=; loading Turbo instead")
        MODEL = ChatterboxTurboTTS.from_pretrained(**kwargs)
    else:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        t3_model = req.get("t3_model") or "v3"
        supports_t3 = _accepts(ChatterboxMultilingualTTS.from_pretrained, "t3_model")
        if repo and repo not in ("ResembleAI/chatterbox", ""):
            # Single Language Pack: its own repo, loaded from a local snapshot.
            from huggingface_hub import snapshot_download

            local = snapshot_download(repo_id=repo, token=os.getenv("HF_TOKEN") or None)
            kwargs = {"device": device}
            if _accepts(ChatterboxMultilingualTTS.from_local, "t3_model"):
                kwargs["t3_model"] = _detect_t3(Path(local), t3_model)
            MODEL = ChatterboxMultilingualTTS.from_local(local, **kwargs)
        else:
            kwargs = {"device": device}
            if supports_t3:
                kwargs["t3_model"] = t3_model
            elif t3_model not in ("", "v2", "t3_mtl23ls_v2"):
                _log(
                    f"installed chatterbox does not support t3_model={t3_model} "
                    "(PyPI 0.1.7 is v2-only); install from git for V3."
                )
            MODEL = ChatterboxMultilingualTTS.from_pretrained(**kwargs)

    if not watermark:
        MODEL.watermarker = _NoopWatermarker()
    FAMILY = family
    SR = int(getattr(MODEL, "sr", 24000) or 24000)
    _log(f"loaded family={family} repo={repo} device={device} watermark={watermark} sr={SR}")


def _synth(req: dict) -> list[dict]:
    if MODEL is None:
        raise RuntimeError("No model loaded")
    import soundfile as sf

    texts = req.get("texts") or []
    language = (req.get("language") or "").strip()
    audio_prompt = req.get("audio_prompt") or None
    params = req.get("params") or {}
    out_dir = Path(req.get("out_dir") or "/tmp")
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = {
        k: params[k]
        for k in ("temperature", "top_p", "repetition_penalty", "exaggeration",
                  "cfg_weight", "min_p", "top_k")
        if params.get(k) is not None
    }
    kwargs = _filter_kwargs(MODEL.generate, candidates)

    items: list[dict] = []
    for idx, text in enumerate(texts):
        gen_kwargs = dict(kwargs)
        gen_kwargs["audio_prompt_path"] = audio_prompt
        if FAMILY not in ("chatterbox_turbo", "chatterbox_nano"):
            gen_kwargs["language_id"] = language or None
        pieces = _split_for_model(MODEL, str(text), language)
        if len(pieces) > 1:
            _log(
                f"text {idx}: {len(pieces)} parts above the model's character budget "
                f"(joined with {_SPLIT_SILENCE_MS} ms silence)"
            )
        wavs = [MODEL.generate(piece, **gen_kwargs) for piece in pieces]
        # A single piece returns the model's own tensor; several are joined into
        # a plain array (the caller below handles both).
        wav = wavs[0] if len(wavs) == 1 else _join_wavs(wavs, SR)
        path = out_dir / f"{idx}.wav"
        arr = wav.squeeze(0).detach().cpu().numpy() if hasattr(wav, "detach") else wav
        sf.write(str(path), arr, SR)
        items.append({"path": str(path), "sr": SR, "parts": len(pieces)})
    return items


def _languages() -> list[str]:
    if FAMILY in ("chatterbox_turbo", "chatterbox_nano"):
        return ["en"]
    getter = getattr(MODEL, "get_supported_languages", None)
    if callable(getter):
        try:
            return sorted((getter() or {}).keys())
        except Exception:
            pass
    return sorted(ISO_TO_NAME)


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:  # malformed input -> report and continue
            _reply({"ok": False, "error": f"bad request: {exc}"})
            continue

        cmd = req.get("cmd")
        try:
            if cmd == "load":
                _load(req)
                _reply({"ok": True})
            elif cmd == "synth":
                _reply({"ok": True, "items": _synth(req)})
            elif cmd == "languages":
                _reply({"ok": True, "languages": _languages()})
            elif cmd == "unload":
                globals()["MODEL"] = None
                _reply({"ok": True})
            elif cmd == "shutdown":
                _reply({"ok": True})
                return 0
            else:
                _reply({"ok": False, "error": f"unknown cmd: {cmd}"})
        except Exception as exc:
            fatal = _is_fatal_error(exc)
            _reply({
                "ok": False,
                "error": str(exc),
                "trace": traceback.format_exc(),
                "fatal": fatal,
            })
            if fatal:
                _log("fatal CUDA error: exiting so the gateway can respawn a clean worker")
                return 1
    return 0


def _reply(obj: dict) -> None:
    _OUT.write(json.dumps(obj) + "\n")
    _OUT.flush()


if __name__ == "__main__":
    # Running as a script puts this file's directory first on sys.path, which can
    # shadow installed packages with same-named local modules. Drop it so the
    # isolated worker only ever imports its own venv's packages. Then keep library
    # prints off the JSON protocol stream.
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
    sys.stdout = sys.stderr
    raise SystemExit(main())
