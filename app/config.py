"""Configuration loading/saving (config.yaml + environment overrides)."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from .apikeys import ApiKey, from_legacy
from .chunking import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS

log = logging.getLogger(__name__)

# Writable runtime config. Defaults to a state path so a bare run never writes
# the repository's config.yaml (which is used read-only as CONFIG_DEFAULTS in
# Docker). Override with CONFIG_PATH (docker-compose sets /app/state/config.yaml).
DEFAULT_CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "state/config.yaml"))


def _defaults_path() -> Optional[Path]:
    """Read-only defaults file (env ``CONFIG_DEFAULTS``), if configured."""
    value = os.environ.get("CONFIG_DEFAULTS")
    return Path(value) if value else None


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8004
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    log_level: str = "info"


class PathsConfig(BaseModel):
    voices: str = "voices"
    reference_audio: str = "reference_audio"


class ModelVariant(BaseModel):
    """A selectable model version for a family (e.g. custom_voice 0.6B / 1.7B)."""

    id: str
    family: str
    repo: str
    label: str = ""
    # Inference backend implementing the family ("qwen3" or "chatterbox").
    backend: str = "qwen3"


def _default_variants() -> List[ModelVariant]:
    return [
        ModelVariant(id="custom_voice:1.7B", family="custom_voice",
                     repo="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", label="1.7B"),
        ModelVariant(id="custom_voice:0.6B", family="custom_voice",
                     repo="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", label="0.6B"),
        ModelVariant(id="voice_design:1.7B", family="voice_design",
                     repo="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", label="1.7B"),
        ModelVariant(id="base:1.7B", family="base",
                     repo="Qwen/Qwen3-TTS-12Hz-1.7B-Base", label="1.7B"),
        ModelVariant(id="base:0.6B", family="base",
                     repo="Qwen/Qwen3-TTS-12Hz-0.6B-Base", label="0.6B"),
        # Chatterbox (Resemble AI) — isolated backend (see docs/backends.md).
        ModelVariant(id="chatterbox:v3", family="chatterbox",
                     repo="ResembleAI/chatterbox", label="v3", backend="chatterbox"),
        ModelVariant(id="chatterbox:v2", family="chatterbox",
                     repo="ResembleAI/chatterbox", label="v2", backend="chatterbox"),
        ModelVariant(id="chatterbox:zh-cmn", family="chatterbox",
                     repo="ResembleAI/Chatterbox-Multilingual-zh-cmn", label="zh-cmn", backend="chatterbox"),
        ModelVariant(id="chatterbox:es-mx-latam", family="chatterbox",
                     repo="ResembleAI/Chatterbox-Multilingual-es-mx-latam", label="es-mx-latam", backend="chatterbox"),
        ModelVariant(id="chatterbox:es-es", family="chatterbox",
                     repo="ResembleAI/Chatterbox-Multilingual-es-es", label="es-es", backend="chatterbox"),
        ModelVariant(id="chatterbox:pt-br", family="chatterbox",
                     repo="ResembleAI/Chatterbox-Multilingual-pt-br", label="pt-br", backend="chatterbox"),
        ModelVariant(id="chatterbox:pt-pt", family="chatterbox",
                     repo="ResembleAI/Chatterbox-Multilingual-pt-pt", label="pt-pt", backend="chatterbox"),
        ModelVariant(id="chatterbox:hi", family="chatterbox",
                     repo="ResembleAI/Chatterbox-Multilingual-hi", label="hi", backend="chatterbox"),
        ModelVariant(id="chatterbox_turbo:turbo", family="chatterbox_turbo",
                     repo="ResembleAI/chatterbox-turbo", label="Turbo", backend="chatterbox"),
        ModelVariant(id="chatterbox_nano:nano", family="chatterbox_nano",
                     repo="ResembleAI/chatterbox-nano", label="Nano", backend="chatterbox"),
    ]


# Families served by the isolated Chatterbox backend (see app/chatterbox_backend.py).
CHATTERBOX_FAMILIES = ("chatterbox", "chatterbox_turbo", "chatterbox_nano")


class ChatterboxConfig(BaseModel):
    """Settings for the isolated Chatterbox backend (own venv/process)."""

    # Python interpreter of the isolated environment that has chatterbox-tts
    # installed. Empty -> auto-detect (/opt/cb/venv/bin/python).
    python: str = ""
    # Which t3 checkpoint the multilingual family loads ("v3" or "v2").
    t3_model: str = "v3"
    # PerTh output watermarking. Off by default for local/self-hosted instances.
    watermark: bool = False
    # Language used when the request asks for "Auto" (Chatterbox has no auto mode).
    default_language: str = "English"


class ModelsConfig(BaseModel):
    registry: Dict[str, str] = Field(
        default_factory=lambda: {
            "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "chatterbox": "ResembleAI/chatterbox",
            "chatterbox_turbo": "ResembleAI/chatterbox-turbo",
            "chatterbox_nano": "ResembleAI/chatterbox-nano",
        }
    )
    # Selectable versions offered in the UI (per family). The chosen variant sets
    # `registry[family]`, so switching only changes which repo the family loads.
    variants: List[ModelVariant] = Field(default_factory=_default_variants)
    default_family: str = "custom_voice"
    # Global model size applied to every family that offers it (e.g. "0.6B" or
    # "1.7B"). Families without a matching variant keep their current repo.
    default_size: str = "1.7B"
    residency: str = "single"  # single | multi
    resident: List[str] = Field(default_factory=lambda: ["custom_voice"])
    # Global engine used to synthesize cloned avatars ("qwen3" or "chatterbox").
    clone_backend: str = "qwen3"
    # Deprecated: VoiceDesign avatars were replaced by fingerprint + clone.
    stabilize_voice_design: bool = False
    chatterbox: ChatterboxConfig = Field(default_factory=ChatterboxConfig)
    device: str = "auto"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    hf_home: str = "/app/hf_cache"

    def backend_for(self, family: str) -> str:
        """Inference backend implementing a family (name-prefix convention)."""
        if family in CHATTERBOX_FAMILIES:
            return "chatterbox"
        return "qwen3"


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
    # Chatterbox-specific knobs (ignored by Qwen3-TTS).
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    min_p: float = 0.05


class ChunkingConfig(BaseModel):
    enabled: bool = True
    max_chars: int = 240
    silence_ms: int = 120

    @field_validator("max_chars")
    @classmethod
    def _clamp_max_chars(cls, value: int) -> int:
        """Keep the stored chunk size in range (see `chunking.MAX_CHUNK_CHARS`)."""
        return min(MAX_CHUNK_CHARS, max(MIN_CHUNK_CHARS, int(value)))


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
    # Input language hint. Empty = "auto" (the model's default: detection for the
    # multilingual models, fixed for the language-specific ones). It only changes
    # the decoding of *prompt-conditioned* models (the streaming Nemotron ones);
    # Parakeet TDT v3 accepts it but ignores it. A language *name* is accepted and
    # normalized to its ISO-639-1 code (`French` → `fr`).
    language: str = ""
    # Codes offered in the UI selector (e.g. ["fr", "en"]); empty keeps the free
    # text field. Purely a client convenience: the gateway accepts any code.
    languages: List[str] = Field(default_factory=list)
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
    # OpenAI-compatible speech route. vLLM-Omni also exposes a dedicated
    # streaming endpoint (`/v1/audio/speech/stream`) — use it if the standard one
    # buffers the whole audio before answering.
    path: str = "/v1/audio/speech"
    # Which family the *service* serves (it loads a single checkpoint):
    # `custom_voice` for the preset speakers ("-CustomVoice" repos) or `base` for
    # voice cloning from a reference clip ("-Base" repos, `task_type=Base`).
    # Requests of the other family stay on the local engine.
    family: str = "custom_voice"
    # Read-only view of the voice bank *as the service sees it* (e.g.
    # `/svc/voices` when the gateway's `voices/` directory is mounted into the
    # container). When set, reference clips are sent as `file://` URIs — the
    # service reloads them automatically when the file changes — instead of being
    # inlined as base64 in every request.
    ref_root: str = ""
    # Idle hibernation: seconds without a vLLM request before the service is asked
    # to sleep (0 = never). It stays up and reachable, but gives its VRAM back
    # (needs `--enable-sleep-mode` on the service); the next request wakes it first.
    sleep_after: float = 0.0
    sleep_level: int = 1  # 1 = weights offloaded to host RAM (fast wake), 2 = freed
    # Launch profile applied when the gateway (re)creates the container itself:
    # `--max-model-len`, `--enforce-eager` and the per-stage VRAM budget. An empty
    # {stage: fraction} mapping lets the gateway compute it from the checkpoint and
    # the card (`GET /api/vllm/budget`); explicit values always win, and leaving a
    # stage out mixes both ("auto" only where it is not set).
    max_model_len: int = 2048
    enforce_eager: bool = True
    stage_memory: Dict[str, float] = Field(default_factory=dict)
    # Checkpoint the gateway keeps the service on ("" = what the compose started
    # it with). Setting it makes the UI's model picker the source of truth.
    model: str = ""
    # What to do when the service is selected for a family but unreachable/not
    # ready: `false` (default) fails the request with an explicit 503, so a silent
    # switch to the local models cannot hide a broken service; set it to `true` to
    # keep serving from the local engine as a fallback.
    fallback_to_local: bool = False

    @field_validator("family")
    @classmethod
    def _normalize_family(cls, value: str) -> str:
        normalized = str(value or "").strip().lower().replace("-", "_")
        aliases = {"customvoice": "custom_voice", "clone": "base", "cloning": "base"}
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in ("custom_voice", "base") else "custom_voice"


class DockerConfig(BaseModel):
    """Pilot the vLLM-Omni container from the API/UI.

    Off by default: it needs the Docker socket mounted in the gateway container,
    which is root-equivalent power over the host (see docs/security.md).
    """

    control: bool = False
    socket: str = "/var/run/docker.sock"
    container: str = "tts-vllm"
    start_timeout: float = 300.0  # vLLM needs ~1 min to load its two stages


class UIConfig(BaseModel):
    title: str = "AiUdio Server"
    show_stt_tab: bool = True
    show_chat_tab: bool = True


class StartupConfig(BaseModel):
    """What to preload when the server starts."""

    stt_model: str = ""   # STT model filename or alias to activate at startup
    voice: str = ""       # TTS family (custom_voice/voice_design/base) or "avatar:<id>"
    tts_family: str = ""  # TTS family to preload (keeps it warm for low latency)
    warmup: bool = False  # run a tiny synthesis after preload (absorbs CUDA init)


class AuthConfig(BaseModel):
    """API and UI access control.

    - ``enabled``: turns on global access control (except ``exempt_paths``).
    - ``username`` / ``password``: HTTP Basic auth (web UI / admin).
    - ``api_keys``: machine keys accepted via ``Authorization: Bearer`` or the
      ``X-API-Key`` header. Declared here (client keys, hashed, persisted); the
      admin also mints short-lived **session** keys at runtime (in memory only).
    - ``session_ttl_minutes``: lifetime of an admin session key.
    - ``exempt_paths``: paths always reachable without authentication.
    """

    enabled: bool = False
    username: str = "admin"
    password: str = ""
    api_keys: List[ApiKey] = Field(default_factory=list)
    session_ttl_minutes: int = 480
    exempt_paths: List[str] = Field(default_factory=lambda: ["/health", "/ready"])

    @field_validator("api_keys", mode="before")
    @classmethod
    def _coerce_legacy_keys(cls, value):
        # Accept a legacy list of plaintext strings and hash them on load.
        if not value:
            return []
        out = []
        for item in value:
            if isinstance(item, str):
                out.append(from_legacy(item))
            else:
                out.append(item)
        return out


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
    docker: DockerConfig = Field(default_factory=DockerConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    startup: StartupConfig = Field(default_factory=StartupConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)


def _merge_sections(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``patch`` into ``base`` (nested dicts are merged, not replaced).

    Values set to ``None`` are ignored, so a partial section such as
    ``{"models": {"registry": {"base": repo}}}`` only touches that entry instead of
    dropping the other families.
    """
    out = dict(base)
    for key, value in patch.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_sections(out[key], value)
        else:
            out[key] = value
    return out


def _apply_env(settings: Settings) -> Settings:
    """Environment variable overrides (12-factor, for Docker)."""
    # docker-compose sets HF_TOKEN="" when unset; an empty token makes
    # huggingface_hub send an invalid "Authorization: Bearer " header.
    for _var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if _var in os.environ and not os.environ[_var].strip():
            del os.environ[_var]
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
    if v := os.environ.get("STT_LANGUAGE"):
        settings.stt.language = v.strip()
    if v := os.environ.get("STT_LANGUAGES"):
        settings.stt.languages = [item.strip() for item in v.split(",") if item.strip()]
    if v := os.environ.get("AUTH_ENABLED"):
        settings.auth.enabled = v.lower() in {"1", "true", "yes", "on"}
    if v := os.environ.get("AUTH_USERNAME"):
        settings.auth.username = v
    if v := os.environ.get("AUTH_PASSWORD"):
        settings.auth.password = v
    for family, env in (
        ("custom_voice", "TTS_CUSTOM_VOICE"),
        ("voice_design", "TTS_VOICE_DESIGN"),
        ("base", "TTS_BASE"),
    ):
        if v := os.environ.get(env):
            settings.models.registry[family] = v
    if v := os.environ.get("TTS_CLONE_BACKEND"):
        settings.models.clone_backend = v.strip().lower()
    # Keep a single model in VRAM by default (local installs); opt into keeping
    # several families resident (chat + clone) when the GPU has room for it.
    if v := os.environ.get("TTS_RESIDENCY"):
        residency = v.strip().lower()
        if residency in {"single", "multi"}:
            settings.models.residency = residency
        else:
            log.warning("Ignoring TTS_RESIDENCY=%r (expected 'single' or 'multi')", v)
    if v := os.environ.get("TTS_CHATTERBOX_PYTHON"):
        settings.models.chatterbox.python = v
    # Container control (see docs/security.md): enabling it implies the gateway
    # can reach the Docker socket, which is a privileged decision.
    if v := os.environ.get("DOCKER_CONTROL"):
        settings.docker.control = v.strip().lower() in {"1", "true", "yes", "on"}
    if v := os.environ.get("DOCKER_CONTAINER"):
        settings.docker.container = v.strip()
    if v := os.environ.get("DOCKER_SOCKET"):
        settings.docker.socket = v.strip()
    if v := os.environ.get("VLLM_REF_ROOT"):
        settings.vllm.ref_root = v.strip()
    return settings


class ConfigStore:
    """Thread-safe access to the writable configuration (``CONFIG_PATH``).

    In Docker, ``config.yaml`` is shipped read-only as **defaults**
    (``CONFIG_DEFAULTS``) and the runtime state lives in a writable volume
    (``CONFIG_PATH``). On first run the defaults are copied to the state file;
    afterwards only the state file is read/written, so the repository file is
    never clobbered by the service.
    """

    def __init__(self, path: Path = DEFAULT_CONFIG_PATH):
        self.path = Path(path)
        self._seed()
        self.settings = self.load()

    def _seed(self) -> None:
        """Copy defaults to the writable config the first time it is missing."""
        defaults = _defaults_path()
        if self.path.exists() or defaults is None or not defaults.exists():
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(defaults, self.path)
            log.info("Seeded %s from defaults %s", self.path, defaults)
        except OSError:
            log.warning("Could not seed %s from %s", self.path, defaults, exc_info=True)

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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.safe_dump(
                self.settings.model_dump(mode="json"), allow_unicode=True, sort_keys=False
            ),
            encoding="utf-8",
        )

    def update(self, **sections) -> Settings:
        """Merge partial sections recursively (nested dicts are merged, not replaced)."""
        data = _merge_sections(self.settings.model_dump(), sections)
        from pydantic import ValidationError

        try:
            self.save(Settings.model_validate(data))
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
        return self.settings


_store: Optional[ConfigStore] = None


def get_store() -> ConfigStore:
    global _store
    if _store is None:
        _store = ConfigStore()
    return _store


def get_settings() -> Settings:
    return get_store().settings
