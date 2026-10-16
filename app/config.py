"""Configuration loading/saving (config.yaml + environment overrides)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "config.yaml"))


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8004
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    log_level: str = "info"


class PathsConfig(BaseModel):
    output: str = "outputs"
    voices: str = "voices"
    reference_audio: str = "reference_audio"
    logs: str = "logs"


class ModelsConfig(BaseModel):
    registry: Dict[str, str] = Field(
        default_factory=lambda: {
            "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        }
    )
    default_family: str = "custom_voice"
    residency: str = "single"  # single | multi
    resident: List[str] = Field(default_factory=lambda: ["custom_voice"])
    device: str = "auto"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    hf_home: str = "/app/hf_cache"


class GenerationConfig(BaseModel):
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    max_new_tokens: int = 2048
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    batch_size: int = 4
    batch_window_ms: int = 25


class ChunkingConfig(BaseModel):
    enabled: bool = True
    max_chars: int = 240
    silence_ms: int = 120


class AudioConfig(BaseModel):
    format: str = "wav"
    sample_rate: int = 24000
    # Reference-audio security. Defaults are restrictive to avoid SSRF (remote
    # URLs) and local file disclosure (path traversal) from API clients.
    allow_remote_media: bool = False
    allow_local_ref_paths: bool = False


class STTConfig(BaseModel):
    enabled: bool = True
    base_url: str = "http://stt:8080"
    transcribe_path: str = "/v1/audio/transcriptions"
    realtime_path: str = "/v1/audio/transcriptions/realtime"
    diarize_path: str = "/v1/audio/diarizations"
    api_key: str = ""
    # On-demand model management (shared volume with the STT service).
    # The model index is vendored with the app (see app/data/).
    model_index: str = str(Path(__file__).resolve().parent / "data" / "nemo-speech-model-index.json")
    models_dir: str = "/app/hf_cache/nemo-speech/models"
    hf_base_url: str = "https://huggingface.co"


class LLMConfig(BaseModel):
    base_url: str = "http://host.docker.internal:8080/v1"
    api_key: str = ""
    model: str = ""
    system_prompt: str = (
        "You are a helpful voice assistant. Answer concisely, in complete "
        "sentences, in the language of the user."
    )
    temperature: float = 0.7
    max_tokens: int = -1
    history_turns: int = 12


class VLLMConfig(BaseModel):
    enabled: bool = False
    base_url: str = "http://tts-vllm:8091"


class UIConfig(BaseModel):
    title: str = "AiUdio Server"
    show_stt_tab: bool = True
    show_chat_tab: bool = True


class AuthConfig(BaseModel):
    """API and UI access control.

    - ``enabled``: turns on global access control (except /health).
    - ``username`` / ``password``: HTTP Basic auth (web UI).
    - ``api_keys``: machine keys accepted via ``Authorization: Bearer`` or the
      ``X-API-Key`` header.
    - ``exempt_paths``: paths always reachable without authentication.
    """

    enabled: bool = False
    username: str = "admin"
    password: str = ""
    api_keys: List[str] = Field(default_factory=list)
    exempt_paths: List[str] = Field(default_factory=lambda: ["/health", "/ready"])


class Settings(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)


def _apply_env(settings: Settings) -> Settings:
    """Environment variable overrides (12-factor, for Docker)."""
    if v := os.environ.get("HF_HOME"):
        settings.models.hf_home = v
    if v := os.environ.get("LLM_BASE_URL"):
        settings.llm.base_url = v
    if v := os.environ.get("LLM_API_KEY"):
        settings.llm.api_key = v
    if v := os.environ.get("LLM_MODEL"):
        settings.llm.model = v
    if v := os.environ.get("STT_BASE_URL"):
        settings.stt.base_url = v
    if v := os.environ.get("STT_API_KEY"):
        settings.stt.api_key = v
    if v := os.environ.get("STT_MODEL_INDEX"):
        settings.stt.model_index = v
    if v := os.environ.get("AUTH_ENABLED"):
        settings.auth.enabled = v.lower() in {"1", "true", "yes", "on"}
    if v := os.environ.get("AUTH_USERNAME"):
        settings.auth.username = v
    if v := os.environ.get("AUTH_PASSWORD"):
        settings.auth.password = v
    if v := os.environ.get("AUTH_API_KEYS"):
        settings.auth.api_keys = [k.strip() for k in v.split(",") if k.strip()]
    for family, env in (
        ("custom_voice", "TTS_CUSTOM_VOICE"),
        ("voice_design", "TTS_VOICE_DESIGN"),
        ("base", "TTS_BASE"),
    ):
        if v := os.environ.get(env):
            settings.models.registry[family] = v
    return settings


class ConfigStore:
    """Thread-safe access to the configuration persisted in config.yaml."""

    def __init__(self, path: Path = DEFAULT_CONFIG_PATH):
        self.path = Path(path)
        self.settings = self.load()

    def load(self) -> Settings:
        if self.path.exists():
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            settings = Settings.model_validate(raw)
        else:
            settings = Settings()
        settings = _apply_env(settings)
        if settings.models.hf_home:
            os.environ.setdefault("HF_HOME", settings.models.hf_home)
        return settings

    def reload(self) -> Settings:
        self.settings = self.load()
        return self.settings

    def save(self, settings: Optional[Settings] = None) -> None:
        if settings is not None:
            self.settings = settings
        self.path.write_text(
            yaml.safe_dump(
                self.settings.model_dump(), allow_unicode=True, sort_keys=False
            ),
            encoding="utf-8",
        )

    def update(self, **sections) -> Settings:
        data = self.settings.model_dump()
        for key, value in sections.items():
            if value is not None:
                data[key] = value
        self.save(Settings.model_validate(data))
        return self.settings


_store: Optional[ConfigStore] = None


def get_store() -> ConfigStore:
    global _store
    if _store is None:
        _store = ConfigStore()
    return _store


def get_settings() -> Settings:
    return get_store().settings
