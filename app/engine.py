"""TTS engine: Qwen3-TTS model management (hot-swap/residency) and inference.

The `qwen-tts` package and `torch` are imported lazily so the server (and the
tests) can start without a GPU or installed models.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Fixed sentences used to render a VoiceDesign reference clip (one per language),
# which is then cloned to keep the voice stable across generations.
_DESIGN_REF_TEXTS = {
    "english": "Hello, this is a short sample of my voice.",
    "french": "Bonjour, ceci est un court échantillon de ma voix.",
    "german": "Hallo, dies ist eine kurze Aufnahme meiner Stimme.",
    "spanish": "Hola, esta es una breve muestra de mi voz.",
    "italian": "Ciao, questo è un breve esempio della mia voce.",
    "portuguese": "Olá, esta é uma pequena amostra da minha voz.",
    "russian": "Привет, это короткий образец моего голоса.",
    "japanese": "こんにちは、これは私の声の短いサンプルです。",
    "korean": "안녕하세요, 이것은 제 목소리의 짧은 샘플입니다.",
    "chinese": "你好，这是我声音的简短样本。",
}


def _design_ref_text(language: Optional[str]) -> str:
    return _DESIGN_REF_TEXTS.get((language or "english").lower(), _DESIGN_REF_TEXTS["english"])


def design_ref_text(language: Optional[str]) -> str:
    """Public alias used by the API when creating a VoiceDesign avatar."""
    return _design_ref_text(language)


# Built-in Qwen3-TTS CustomVoice speakers, exposed even before the model is loaded
# so the conversation UI can always offer the default voices.
DEFAULT_SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]


@dataclass
class ModelHandle:
    family: str
    repo_id: str
    model: Any
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    default_speaker: str = ""


def _resolve_dtype(name: str) -> Any:  # pragma: no cover - requires torch
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get((name or "").lower(), torch.bfloat16)


def _resolve_device(name: str) -> str:  # pragma: no cover - requires torch
    import torch

    if name and name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TTSEngine:
    """Manages a bounded pool of Qwen3-TTS models (at most one per family)."""

    def __init__(self, settings):
        self.settings = settings
        self._handles: Dict[str, ModelHandle] = {}
        self._clone_prompts: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._backend = "native"

    # ------------------------------------------------------------------ info
    @property
    def backend(self) -> str:
        return self._backend

    def repo_for(self, family: str) -> str:
        return self.settings.models.registry.get(
            family, self.settings.models.registry[self.settings.models.default_family]
        )

    def available(self) -> List[Dict[str, Any]]:
        return [
            {"family": f, "repo_id": repo, "loaded": f in self._handles}
            for f, repo in self.settings.models.registry.items()
        ]

    def loaded_families(self) -> List[str]:
        return list(self._handles.keys())

    def active_family(self) -> Optional[str]:
        if not self._handles:
            return None
        return max(self._handles.values(), key=lambda h: h.last_used).family

    # -------------------------------------------------------------- lifecycle
    def max_resident(self) -> int:
        if self.settings.models.residency == "multi":
            return max(1, len(self.settings.models.resident) or 1)
        return 1

    def ensure(self, family: str) -> ModelHandle:
        with self._lock:
            if family in self._handles:
                self._handles[family].last_used = time.time()
                return self._handles[family]
            self._evict_for(family)
            handle = self._load(family)
            self._handles[family] = handle
            return handle

    def _evict_for(self, incoming: str) -> None:
        limit = self.max_resident()
        while len(self._handles) >= limit and incoming not in self._handles:
            victim = min(self._handles.values(), key=lambda h: h.last_used).family
            log.info("Evicting model '%s' (hot-swap)", victim)
            self._unload(victim)

    def unload(self, family: Optional[str] = None) -> None:
        with self._lock:
            for f in list(self._handles) if family is None else [family]:
                self._unload(f)

    def _unload(self, family: str) -> None:
        handle = self._handles.pop(family, None)
        self._clone_prompts.clear()
        if handle is None:
            return
        del handle.model
        gc.collect()
        try:  # pragma: no cover - no CUDA in CI
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------ load
    def _load(self, family: str) -> ModelHandle:  # pragma: no cover - GPU
        from qwen_tts import Qwen3TTSModel

        repo = self.repo_for(family)
        device = _resolve_device(self.settings.models.device)
        dtype = _resolve_dtype(self.settings.models.dtype)
        kwargs: Dict[str, Any] = {"device_map": device, "dtype": dtype}
        attn = self.settings.models.attn_implementation
        if attn and attn != "eager":
            kwargs["attn_implementation"] = attn
        log.info("Loading %s (%s) on %s [%s]", family, repo, device, attn)
        started = time.perf_counter()
        model = Qwen3TTSModel.from_pretrained(repo, **kwargs)
        log.info("Loaded %s in %.2fs", family, time.perf_counter() - started)
        return ModelHandle(family=family, repo_id=repo, model=model)

    # -------------------------------------------------------------- inference
    def synthesize_batch(self, family: str, items: List[Dict[str, Any]]) -> List[Tuple[np.ndarray, int]]:
        """Generate a batch of items of the same family. Returns [(wav, sr), ...]."""
        if not items:
            return []

        requested_family = family
        t_start = time.perf_counter()

        # VoiceDesign has no fixed speaker: the voice changes on every generation.
        # Render a reference clip once per (instruct, language) and clone it so the
        # voice stays identical until the voice definition changes.
        if family == "voice_design" and self.settings.models.stabilize_voice_design:
            prompt = self._design_clone_prompt(
                items[0].get("instruct") or "", items[0].get("language")
            )
            items = [{**it, "family": "base", "voice_clone_prompt": prompt} for it in items]
            family = "base"

        handle = self.ensure(family)
        gen = dict(self._generation_defaults())
        gen.update(items[0].get("generation") or {})

        t_gen = time.perf_counter()
        with self._lock:
            if family == "custom_voice":
                # A missing speaker would let the model pick a different voice per
                # generation; fall back to a stable default for consistency (this is
                # what caused the "different voice every sentence" in chat mode).
                speakers = [i.get("speaker") or "" for i in items]
                if any(not s for s in speakers):
                    default = self._default_speaker(handle)
                    speakers = [s or default for s in speakers]
                wavs, sr = handle.model.generate_custom_voice(
                    text=[i["text"] for i in items],
                    language=[i.get("language") or "Auto" for i in items],
                    speaker=speakers,
                    instruct=[i.get("instruct") or "" for i in items],
                    **gen,
                )
            elif family == "voice_design":
                wavs, sr = handle.model.generate_voice_design(
                    text=[i["text"] for i in items],
                    language=[i.get("language") or "Auto" for i in items],
                    instruct=[i.get("instruct") or "" for i in items],
                    **gen,
                )
            elif family == "base":
                prompt = items[0].get("voice_clone_prompt")
                if prompt is None:
                    ref_audio = items[0].get("ref_audio")
                    if not ref_audio:
                        raise ValueError("base model requires ref_audio or voice_clone_prompt")
                    prompt = self._cached_clone_prompt(
                        handle,
                        ref_audio,
                        items[0].get("ref_text"),
                        bool(items[0].get("x_vector_only_mode")),
                    )
                wavs, sr = handle.model.generate_voice_clone(
                    text=[i["text"] for i in items],
                    language=[i.get("language") or "Auto" for i in items],
                    voice_clone_prompt=prompt,
                    **gen,
                )
            else:
                raise ValueError(f"Unknown model family: {family}")

        results = [(np.asarray(w, dtype=np.float32), int(sr)) for w in wavs]
        gen_s = time.perf_counter() - t_gen
        total_s = time.perf_counter() - t_start
        audio_s = sum(len(w) / float(s) for w, s in results) if results else 0.0
        chars = sum(len(i.get("text") or "") for i in items)
        log.info(
            "TTS family=%s->%s n=%d chars=%d audio=%.2fs gen=%.2fs total=%.2fs "
            "rtf=%.2f%s",
            requested_family, family, len(items), chars, audio_s, gen_s, total_s,
            (gen_s / audio_s if audio_s else 0.0),
            "" if requested_family == family else " (incl. stabilize)",
        )
        return results

    def synthesize_one(self, item: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        return self.synthesize_batch(item["family"], [item])[0]

    def _default_speaker(self, handle: ModelHandle) -> str:
        """Stable default speaker for a loaded custom_voice model."""
        if handle.default_speaker:
            return handle.default_speaker
        try:
            speakers = list(handle.model.get_supported_speakers() or [])
        except Exception:
            speakers = []
        handle.default_speaker = speakers[0] if speakers else ""
        return handle.default_speaker

    def design_reference(self, instruct: str, language) -> Tuple[np.ndarray, int]:
        """Generate a reference clip from a VoiceDesign description."""
        handle = self.ensure("voice_design")
        ref_text = _design_ref_text(language)
        gen = self._generation_defaults()
        started = time.perf_counter()
        with self._lock:
            wavs, sr = handle.model.generate_voice_design(
                text=ref_text, language=language or "Auto", instruct=instruct, **gen
            )
        log.info("Design reference rendered in %.2fs", time.perf_counter() - started)
        return np.asarray(wavs[0], dtype=np.float32), int(sr)

    def clear_clone_prompts(self) -> None:
        with self._lock:
            self._clone_prompts.clear()

    def _cached_clone_prompt(self, handle: ModelHandle, ref_audio: str, ref_text, x_vector_only: bool) -> Any:
        """Build (and cache) a clone prompt from a reference audio path/payload."""
        key = f"ref:{self.clone_prompt_key(str(ref_audio), ref_text)}\x00{int(bool(x_vector_only))}"
        with self._lock:
            if key in self._clone_prompts:
                return self._clone_prompts[key]
            started = time.perf_counter()
            items = handle.model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=bool(x_vector_only),
            )
            log.info("Built clone prompt (x_vector=%s) in %.2fs", bool(x_vector_only), time.perf_counter() - started)
            if len(self._clone_prompts) >= 8:
                self._clone_prompts.clear()
            self._clone_prompts[key] = items
            return items

    def _design_clone_prompt(self, instruct: str, language) -> Any:
        """Cached voice-clone prompt built once from a VoiceDesign reference clip.

        VoiceDesign has no fixed speaker, so a raw VoiceDesign generation gives a
        different voice each time. We render a short reference clip for the current
        description, build a clone prompt with the Base model, and reuse it. The
        cache key includes the description, so editing the voice definition produces
        a new prompt (and thus a new voice), while keeping it stable otherwise.
        """
        key = f"vd:{instruct}\x00{language or ''}"
        with self._lock:
            if key in self._clone_prompts:
                log.info("Stabilized VoiceDesign voice (lang=%s): cache hit", language)
                return self._clone_prompts[key]

            started = time.perf_counter()
            design = self.ensure("voice_design")
            ref_text = _design_ref_text(language)
            gen = self._generation_defaults()
            wavs, sr = design.model.generate_voice_design(
                text=ref_text, language=language or "Auto", instruct=instruct, **gen
            )
            ref_wav = np.asarray(wavs[0], dtype=np.float32)

            base = self.ensure("base")
            items = base.model.create_voice_clone_prompt(
                ref_audio=(ref_wav, int(sr)), ref_text=ref_text
            )
            log.info(
                "Stabilized VoiceDesign voice (lang=%s) in %.2fs (model loads included)",
                language, time.perf_counter() - started,
            )

            if len(self._clone_prompts) >= 4:  # keep a small bounded cache
                self._clone_prompts.clear()
            self._clone_prompts[key] = items
            return items

    def _generation_defaults(self) -> Dict[str, Any]:
        g = self.settings.generation
        return {
            "temperature": g.temperature,
            "top_k": g.top_k,
            "top_p": g.top_p,
            "repetition_penalty": g.repetition_penalty,
            "max_new_tokens": g.max_new_tokens,
            "subtalker_temperature": g.subtalker_temperature,
            "subtalker_top_k": g.subtalker_top_k,
            "subtalker_top_p": g.subtalker_top_p,
        }

    # --------------------------------------------------------------- prompts
    def clone_prompt_key(self, ref_audio: str, ref_text: Optional[str]) -> str:
        digest = hashlib.sha1()
        digest.update(str(ref_audio).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(ref_text or "").encode("utf-8"))
        return digest.hexdigest()

    # ----------------------------------------------------------------- voices
    def speakers(self, family: str = "custom_voice") -> List[str]:
        # Only report voices for an already-loaded model (do not trigger a load),
        # but fall back to the known built-in list for custom_voice.
        with self._lock:
            handle = self._handles.get(family)
            if handle is None:
                return list(DEFAULT_SPEAKERS) if family == "custom_voice" else []
            try:
                return list(handle.model.get_supported_speakers() or []) or list(DEFAULT_SPEAKERS)
            except Exception:
                return list(DEFAULT_SPEAKERS)

    def languages(self, family: str = "custom_voice") -> List[str]:
        with self._lock:
            handle = self._handles.get(family)
            if handle is None:
                return []
            try:
                return list(handle.model.get_supported_languages() or [])
            except Exception:
                return []
