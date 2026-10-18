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
    <- {"ok": true, "items": [{"path": "/tmp/xyz/0.wav", "sr": 24000}]}

    -> {"cmd": "languages"}   <- {"ok": true, "languages": ["en", "fr", ...]}
    -> {"cmd": "unload"}      <- {"ok": true}
    -> {"cmd": "shutdown"}    <- {"ok": true}   (then exit)
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
        wav = MODEL.generate(str(text), **gen_kwargs)
        path = out_dir / f"{idx}.wav"
        arr = wav.squeeze(0).detach().cpu().numpy() if hasattr(wav, "detach") else wav
        sf.write(str(path), arr, SR)
        items.append({"path": str(path), "sr": SR})
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
            _reply({"ok": False, "error": str(exc), "trace": traceback.format_exc()})
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
