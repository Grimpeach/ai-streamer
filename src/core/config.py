"""Конфигурация приложения (Pydantic v2 + pydantic-settings).

Все значения читаются из окружения и файла ``.env`` с вложенным разделителем
``__``, например::

    TWITCH__CHANNEL=my_channel
    LLM__MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ
    PTT__HOTKEY=f13

Бюджет VRAM (см. .cursorrules п.3 и specs/architecture.md) валидируется на старте:
сумма запрошенной памяти всех локальных моделей не должна превышать
``vram.total_budget_gb`` (по умолчанию 15 ГБ из 16 ГБ RTX 4080).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EventBusSettings(BaseModel):
    """Параметры шины событий."""

    max_queue_size: int = 512
    #: Тайм-аут обработчика; ``None`` — без ограничения.
    handler_timeout_s: float | None = None
    #: Сбрасывать ли очередь чата при прерывании по PTT.
    drop_chat_on_preempt: bool = True


class TwitchSettings(BaseModel):
    """Чтение чата Twitch. Стример отвечает голосом, писать в чат не нужно."""

    channel: str = ""
    bot_nick: str = "ai_streamer_bot"
    #: OAuth-токен со scope ``chat:read``. Префикс ``oauth:`` необязателен.
    access_token: SecretStr = SecretStr("")
    client_id: SecretStr = SecretStr("")
    client_secret: SecretStr = SecretStr("")

    #: Игнорировать сообщения короче N символов и от этих авторов.
    min_message_length: int = 2
    ignored_authors: tuple[str, ...] = ("nightbot", "streamelements", "moobot", "fossabot")
    #: Не озвучивать команды ботов вида ``!drop``.
    ignore_commands: bool = True
    #: Длинные простыни обрезаются, иначе TTS читает их минуту.
    max_message_length: int = 300
    #: Время жизни сообщения чата: устаревшие реплики не озвучиваются.
    chat_ttl_s: float = 45.0

    # --- надёжность соединения ---
    #: Интервал ping/pong websocket'а (передаётся в twitchio).
    heartbeat_s: float = 30.0
    reconnect_initial_delay_s: float = 1.0
    reconnect_max_delay_s: float = 60.0
    #: Соединение, продержавшееся столько секунд, считается стабильным —
    #: после него backoff сбрасывается на минимум.
    stable_connection_s: float = 30.0
    watchdog_interval_s: float = 30.0
    #: Twitch присылает PING примерно раз в 5 минут. Полная тишина дольше —
    #: признак «мёртвого» сокета, который всё ещё выглядит открытым.
    stale_after_s: float = 600.0

    @field_validator("access_token", mode="before")
    @classmethod
    def _strip_oauth_prefix(cls, value: object) -> object:
        """Принимает токен и как ``oauth:xxx``, и как ``xxx``."""
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str):
            return value.strip().removeprefix("oauth:")
        return value

    @field_validator("channel", mode="before")
    @classmethod
    def _normalize_channel(cls, value: object) -> object:
        """Приводит ``#Channel`` к виду ``channel``."""
        if isinstance(value, str):
            return value.strip().lstrip("#").lower()
        return value


class PTTSettings(BaseModel):
    """Push-to-Talk хоста — абсолютный приоритет."""

    hotkey: str = "f13"
    #: Устройство ввода sounddevice; ``None`` — системное по умолчанию.
    input_device: int | str | None = None
    #: Whisper работает с 16 кГц моно — пересэмплирование только добавит задержку.
    sample_rate: int = 16_000
    channels: int = 1
    #: Размер аудио-блока PortAudio: 512 сэмплов ≈ 32 мс при 16 кГц.
    blocksize: int = 512
    #: Сколько звука до нажатия клавиши добавить в начало реплики. Открытие
    #: устройства и реакция человека дают задержку, из-за которой первый слог
    #: обрезается; pre-roll из постоянно открытого потока это компенсирует.
    preroll_ms: int = 300
    #: Ниже этой длительности запись считается случайным нажатием.
    min_utterance_s: float = 0.3
    max_utterance_s: float = 60.0


class STTSettings(BaseModel):
    """Faster-Whisper для расшифровки реплик хоста (~1 ГБ VRAM)."""

    model: str = "large-v3-turbo"
    device: Literal["cuda", "cpu"] = "cuda"
    compute_type: str = "float16"
    language: str = "ru"
    #: beam_size=1 (greedy) — самый быстрый вариант; реплики хоста короткие,
    #: и выигрыш в точности от beam search не окупает лишних сотен миллисекунд.
    beam_size: int = 1
    #: Отрезать тишину силами Silero VAD внутри faster-whisper.
    vad_filter: bool = True
    #: Whisper склонен «дописывать» текст по контексту и зацикливаться на
    #: коротких фразах — для PTT контекст только вредит.
    condition_on_previous_text: bool = False
    #: Выше этого порога считаем, что речи не было: на тишине Whisper выдаёт
    #: галлюцинации вида «Продолжение следует...».
    no_speech_threshold: float = 0.6
    #: Прогреть модель на тишине при старте: первый инференс на CUDA
    #: компилирует кернелы и иначе съел бы 2-3 секунды у живой реплики.
    warmup: bool = True
    #: Куда скачивать веса; ``None`` — кэш HuggingFace по умолчанию.
    download_root: Path | None = None
    vram_gb: float = 1.0
    #: Silero VAD (ONNX) — почти бесплатен, но входит в общий лимит.
    vad_vram_gb: float = 0.1


class LLMSettings(BaseModel):
    """Qwen 2.5 14B через OpenAI-совместимый сервер (vLLM / llama.cpp)."""

    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: SecretStr = SecretStr("local")
    model: str = "Qwen/Qwen2.5-14B-Instruct-AWQ"
    max_tokens: int = 220
    temperature: float = 0.8
    top_p: float = 0.9
    #: Минимальная длина фразы, отдаваемой в TTS: короче — речь звучит рвано.
    sentence_min_chars: int = 48
    #: Тайм-аут первого токена: превышение считается сбоем инференса.
    first_token_timeout_s: float = 5.0
    request_timeout_s: float = 60.0
    #: Qwen 2.5 14B в AWQ/Q4 — около 8.5 ГБ вместе с KV-кэшем.
    vram_gb: float = 8.5


class TTSSettings(BaseModel):
    """Стриминговый синтез речи (CosyVoice / Kokoro)."""

    engine: Literal["kokoro", "cosyvoice", "mock"] = "kokoro"
    voice: str = "ru_female_1"
    sample_rate: int = 24_000
    speed: float = 1.0
    output_device: int | str | None = None
    #: Kokoro укладывается в ~2.5 ГБ; для CosyVoice поднимите до 4.0.
    vram_gb: float = 2.5


class MemorySettings(BaseModel):
    """Redis — краткосрочный контекст, Qdrant — долгосрочная память."""

    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_namespace: str = "ai-streamer"
    #: Сколько последних реплик держать в горячем контексте.
    short_term_window: int = 20
    short_term_ttl_s: int = 3 * 60 * 60

    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "stream_memory"
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768
    top_k: int = 5
    vram_gb: float = 0.4


class VRAMSettings(BaseModel):
    """Бюджет видеопамяти RTX 4080 16GB."""

    total_budget_gb: float = 15.0
    #: Резерв под VTube Studio / OBS (см. architecture.md).
    reserved_gb: float = 2.0


class FlatEnvAliasSource(PydanticBaseSettingsSource):
    """Поддержка «плоских» имён переменных окружения.

    Полное имя вложенной настройки — ``TWITCH__ACCESS_TOKEN``, но в ``.env``
    удобнее писать привычные короткие: ``TWITCH_TOKEN``. Этот источник
    переводит короткие имена в структуру настроек; вложенный вариант, если
    задан, имеет приоритет.
    """

    ALIASES: ClassVar[dict[str, tuple[str, str]]] = {
        "TWITCH_CHANNEL": ("twitch", "channel"),
        "TWITCH_TOKEN": ("twitch", "access_token"),
        "TWITCH_ACCESS_TOKEN": ("twitch", "access_token"),
        "TWITCH_BOT_NICK": ("twitch", "bot_nick"),
        "TWITCH_CLIENT_ID": ("twitch", "client_id"),
        "TWITCH_CLIENT_SECRET": ("twitch", "client_secret"),
        "PTT_HOTKEY": ("ptt", "hotkey"),
        "STT_MODEL": ("stt", "model"),
        "LLM_BASE_URL": ("llm", "base_url"),
        "LLM_MODEL": ("llm", "model"),
        "REDIS_URL": ("memory", "redis_url"),
        "QDRANT_URL": ("memory", "qdrant_url"),
    }

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        """Не используется: значения собираются целиком в :meth:`__call__`."""
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        env = self._collect_env()
        values: dict[str, dict[str, str]] = {}
        for flat_name, (section, field_name) in self.ALIASES.items():
            value = env.get(flat_name)
            if value:
                values.setdefault(section, {})[field_name] = value
        return dict(values)

    def _collect_env(self) -> dict[str, str]:
        """Читает .env, затем перекрывает его реальным окружением процесса."""
        env: dict[str, str] = {}
        env_file = self.config.get("env_file")
        paths = env_file if isinstance(env_file, (list, tuple)) else [env_file]
        for path in paths:
            if not path or not Path(path).is_file():
                continue
            loaded = dotenv_values(path, encoding=self.config.get("env_file_encoding") or "utf-8")
            env.update({key.upper(): value for key, value in loaded.items() if value})
        env.update({key.upper(): value for key, value in os.environ.items() if value})
        return env


class Settings(BaseSettings):
    """Корневая конфигурация приложения."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Добавляет плоские алиасы с наименьшим приоритетом."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            FlatEnvAliasSource(settings_cls),
            file_secret_settings,
        )

    app_name: str = "ai-streamer"
    log_level: str = "INFO"
    log_file: Path | None = PROJECT_ROOT / "logs" / "ai-streamer.log"
    #: Режим без внешних зависимостей: моки Twitch/LLM/TTS для проверки шины.
    demo_mode: bool = False

    event_bus: EventBusSettings = Field(default_factory=EventBusSettings)
    twitch: TwitchSettings = Field(default_factory=TwitchSettings)
    ptt: PTTSettings = Field(default_factory=PTTSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    vram: VRAMSettings = Field(default_factory=VRAMSettings)

    @property
    def vram_breakdown_gb(self) -> dict[str, float]:
        """Разбивка запрошенной видеопамяти по потребителям."""
        return {
            "llm": self.llm.vram_gb,
            "tts": self.tts.vram_gb,
            "stt": self.stt.vram_gb,
            "vad": self.stt.vad_vram_gb,
            "embedder": self.memory.vram_gb,
            "reserved(vtube/obs)": self.vram.reserved_gb,
        }

    @property
    def vram_requested_gb(self) -> float:
        """Суммарный запрос VRAM всеми локальными моделями плюс резерв."""
        return round(sum(self.vram_breakdown_gb.values()), 2)

    @model_validator(mode="after")
    def _check_vram_budget(self) -> Self:
        """Не даёт стартовать с конфигурацией, которая гарантированно уронит CUDA."""
        if self.vram_requested_gb > self.vram.total_budget_gb:
            breakdown = ", ".join(f"{name}={gb}" for name, gb in self.vram_breakdown_gb.items())
            raise ValueError(
                f"Бюджет VRAM превышен: запрошено {self.vram_requested_gb} ГБ "
                f"при лимите {self.vram.total_budget_gb} ГБ ({breakdown}). "
                "Уменьшите модели или поднимите VRAM__TOTAL_BUDGET_GB."
            )
        return self


def load_settings(**overrides: object) -> Settings:
    """Читает конфигурацию из окружения/.env с возможностью переопределения."""
    return Settings(**overrides)  # type: ignore[arg-type]
