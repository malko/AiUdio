"""API request/response schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

FAMILIES = (
    "custom_voice", "voice_design", "base",
    "chatterbox", "chatterbox_turbo", "chatterbox_nano",
)


class GenerationParams(BaseModel):
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    max_new_tokens: Optional[int] = None
    subtalker_temperature: Optional[float] = None
    subtalker_top_k: Optional[int] = None
    subtalker_top_p: Optional[float] = None
    # Chatterbox-specific knobs.
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None
    min_p: Optional[float] = None

    def to_kwargs(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class TTSRequest(BaseModel):
    """Unified request for /api/tts and /v1/audio/speech.

    `text`/`input` and `family`/`mode`/`task_type` accept aliases to stay
    compatible with the OpenAI and vLLM-Omni vocabularies.
    """

    text: Optional[Union[str, List[str]]] = None
    input: Optional[Union[str, List[str]]] = None
    family: Optional[str] = None
    mode: Optional[str] = None
    task_type: Optional[str] = None
    language: Optional[Union[str, List[str]]] = None
    speaker: Optional[str] = None
    voice: Optional[str] = None
    instruct: Optional[str] = None
    instructions: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    avatar: Optional[str] = None
    # Per-request clone engine ("qwen3" | "chatterbox"); overrides the admin
    # default. `engine` is accepted as an alias.
    clone_backend: Optional[str] = None
    engine: Optional[str] = None
    x_vector_only_mode: bool = False
    stream: bool = False
    format: Optional[str] = None
    response_format: Optional[str] = None
    speed: Optional[float] = None
    chunking: Optional[bool] = None
    max_chars: Optional[int] = None
    generation: GenerationParams = Field(default_factory=GenerationParams)

    def requested_family(self) -> Optional[str]:
        """Normalize the explicitly requested family (None when unspecified)."""
        value = self.family or self.mode or self.task_type
        if value is None:
            return None
        value = str(value).strip().lower().replace("-", "_")
        aliases = {"customvoice": "custom_voice", "voicedesign": "voice_design", "clone": "base"}
        return aliases.get(value, value) or None

    def resolve_family(self, default: str) -> str:
        value = self.requested_family()
        if value is None:
            return default
        return value if value in FAMILIES else default

    def resolve_texts(self) -> List[str]:
        raw = self.text if self.text is not None else self.input
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw]
        return [str(x) for x in raw]

    def resolve_speaker(self) -> Optional[str]:
        return self.speaker or self.voice

    def resolve_clone_backend(self) -> Optional[str]:
        """Normalize the per-request clone engine (None when unspecified)."""
        value = self.clone_backend or self.engine
        if value is None:
            return None
        value = str(value).strip().lower().replace("-", "_")
        aliases = {
            "qwen": "qwen3", "qwen3_tts": "qwen3", "qwen3tts": "qwen3",
            "qwen3_base": "qwen3", "base": "qwen3",
            "cb": "chatterbox", "chatterbox_multilingual": "chatterbox",
        }
        return aliases.get(value, value)

    def resolve_instruct(self) -> Optional[str]:
        return self.instruct if self.instruct is not None else self.instructions

    def resolve_format(self, default: str) -> str:
        return (self.format or self.response_format or default).lower()


class SynthItem(BaseModel):
    """Single unit sent to the engine."""

    family: str
    text: str
    language: Optional[str] = None
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    x_vector_only_mode: bool = False
    generation: Dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    family: str
    repo_id: str
    loaded: bool


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    engine: str
    active_family: Optional[str] = None
    loaded: List[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str
    content: str
