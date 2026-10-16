"""API request/response schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

FAMILIES = ("custom_voice", "voice_design", "base")


class GenerationParams(BaseModel):
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    max_new_tokens: Optional[int] = None
    subtalker_temperature: Optional[float] = None
    subtalker_top_k: Optional[int] = None
    subtalker_top_p: Optional[float] = None

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
    x_vector_only_mode: bool = False
    stream: bool = False
    format: Optional[str] = None
    response_format: Optional[str] = None
    speed: Optional[float] = None
    chunking: Optional[bool] = None
    max_chars: Optional[int] = None
    generation: GenerationParams = Field(default_factory=GenerationParams)

    def resolve_family(self, default: str) -> str:
        value = self.family or self.mode or self.task_type or default
        value = str(value).strip().lower().replace("-", "_")
        aliases = {"customvoice": "custom_voice", "voicedesign": "voice_design", "clone": "base"}
        value = aliases.get(value, value)
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
