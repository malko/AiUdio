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
        model = Qwen3TTSModel.from_pretrained(repo, **kwargs)
        return ModelHandle(family=family, repo_id=repo, model=model)

    # -------------------------------------------------------------- inference
    def synthesize_batch(self, family: str, items: List[Dict[str, Any]]) -> List[Tuple[np.ndarray, int]]:
        """Generate a batch of items of the same family. Returns [(wav, sr), ...]."""
        if not items:
            return []
        handle = self.ensure(family)
        gen = dict(self._generation_defaults())
        gen.update(items[0].get("generation") or {})

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
                wavs, sr = handle.model.generate_voice_clone(
                    text=[i["text"] for i in items],
                    language=[i.get("language") or "Auto" for i in items],
                    ref_audio=[i.get("ref_audio") for i in items],
                    ref_text=[i.get("ref_text") for i in items],
                    x_vector_only_mode=[bool(i.get("x_vector_only_mode")) for i in items],
                    **gen,
                )
            else:
                raise ValueError(f"Unknown model family: {family}")

        return [(np.asarray(w, dtype=np.float32), int(sr)) for w in wavs]

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
        # Only report voices for an already-loaded model (do not trigger a load).
        with self._lock:
            handle = self._handles.get(family)
            if handle is None:
                return []
            try:
                return list(handle.model.get_supported_speakers() or [])
            except Exception:
                return []

    def languages(self, family: str = "custom_voice") -> List[str]:
        with self._lock:
            handle = self._handles.get(family)
            if handle is None:
                return []
            try:
                return list(handle.model.get_supported_languages() or [])
            except Exception:
                return []
