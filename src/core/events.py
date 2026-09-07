"""Модели событий шины (Pydantic v2).

Приоритеты соответствуют specs/architecture.md:

* ``EventPriority.INTERRUPT`` (1) — Push-to-Talk хоста, мгновенно прерывает LLM/TTS;
* ``EventPriority.HIGH``      (2) — донаты и подписки Twitch;
* ``EventPriority.NORMAL``    (3) — обычные сообщения чата;
* ``EventPriority.LOW``       (4) — фоновая активность (idle-реплики, телеметрия).
"""

from __future__ import annotations

import time
import uuid
from enum import IntEnum, StrEnum
from typing import Any, Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EventPriority(IntEnum):
    """Приоритет доставки. Меньшее значение — выше приоритет."""

    INTERRUPT = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class EventType(StrEnum):
    """Полный перечень событий системы."""

    # --- Push-to-Talk хоста (Priority 1) ---
    PTT_PRESSED = "ptt.pressed"
    PTT_RELEASED = "ptt.released"
    HOST_SPEECH = "ptt.host_speech"

    # --- Twitch ---
    TWITCH_DONATION = "twitch.donation"
    TWITCH_SUBSCRIPTION = "twitch.subscription"
    TWITCH_RAID = "twitch.raid"
    TWITCH_CHAT = "twitch.chat"

    # --- LLM ---
    LLM_REQUEST = "llm.request"
    LLM_CHUNK = "llm.chunk"
    LLM_COMPLETED = "llm.completed"

    # --- TTS ---
    TTS_REQUEST = "tts.request"
    TTS_STARTED = "tts.started"
    TTS_CHUNK = "tts.chunk"
    TTS_FINISHED = "tts.finished"

    # --- Система ---
    PREEMPT = "system.preempt"
    SHUTDOWN = "system.shutdown"


#: Приоритет по умолчанию для каждого типа события.
DEFAULT_PRIORITY: dict[EventType, EventPriority] = {
    EventType.PTT_PRESSED: EventPriority.INTERRUPT,
    EventType.PTT_RELEASED: EventPriority.INTERRUPT,
    EventType.HOST_SPEECH: EventPriority.INTERRUPT,
    EventType.PREEMPT: EventPriority.INTERRUPT,
    EventType.SHUTDOWN: EventPriority.INTERRUPT,
    EventType.TWITCH_DONATION: EventPriority.HIGH,
    EventType.TWITCH_SUBSCRIPTION: EventPriority.HIGH,
    EventType.TWITCH_RAID: EventPriority.HIGH,
    EventType.TWITCH_CHAT: EventPriority.NORMAL,
    EventType.LLM_REQUEST: EventPriority.NORMAL,
    EventType.LLM_CHUNK: EventPriority.NORMAL,
    EventType.LLM_COMPLETED: EventPriority.NORMAL,
    EventType.TTS_REQUEST: EventPriority.NORMAL,
    EventType.TTS_STARTED: EventPriority.NORMAL,
    EventType.TTS_CHUNK: EventPriority.NORMAL,
    EventType.TTS_FINISHED: EventPriority.NORMAL,
}

#: События, которые сбрасывают текущую генерацию LLM/TTS.
PREEMPTIVE_TYPES: frozenset[EventType] = frozenset(
    {EventType.PTT_PRESSED, EventType.HOST_SPEECH, EventType.PREEMPT, EventType.SHUTDOWN}
)


class EventPayload(BaseModel):
    """Базовый класс полезной нагрузки события."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EmptyPayload(EventPayload):
    """Нагрузка для событий-сигналов (нажатие PTT, shutdown)."""


class ChatMessage(EventPayload):
    """Обычное сообщение чата Twitch."""

    author: str
    text: str
    is_moderator: bool = False
    is_subscriber: bool = False
    channel: str = ""


class Donation(EventPayload):
    """Донат или награда за поинты канала."""

    author: str
    amount: float
    currency: str = "RUB"
    message: str = ""


class TwitchSubscription(EventPayload):
    """Новая подписка, продление или подарочная подписка."""

    author: str
    tier: str = "1000"
    months: int = 1
    message: str = ""
    gifted_by: str | None = None


class Raid(EventPayload):
    """Рейд с другого канала."""

    author: str
    viewers: int


class HostSpeech(EventPayload):
    """Расшифрованная реплика хоста (результат STT после отпускания PTT)."""

    text: str
    duration_s: float = 0.0
    confidence: float | None = None


class TextRequest(EventPayload):
    """Запрос на генерацию: в LLM (промпт) или в TTS (готовая реплика)."""

    text: str
    #: Кто спровоцировал реплику — для промпта и логов.
    trigger: str = ""
    #: Реплику нельзя прерывать (например, короткая реакция на донат).
    critical: bool = False


class LlmChunk(EventPayload):
    """Фрагмент потоковой генерации LLM."""

    text: str
    index: int = 0
    is_final: bool = False


class AudioChunk(EventPayload):
    """Фрагмент синтезированного аудио (PCM 16-bit)."""

    pcm: bytes
    sample_rate: int = 24_000
    index: int = 0
    is_final: bool = False


class PreemptReason(EventPayload):
    """Причина принудительного сброса генерации."""

    reason: str = "ptt"
    source: str = ""


PayloadT = TypeVar("PayloadT", bound=EventPayload)


class Event(BaseModel, Generic[PayloadT]):
    """Единица обмена на шине событий."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    type: EventType
    payload: PayloadT = Field(default_factory=EmptyPayload)  # type: ignore[assignment]
    priority: EventPriority = EventPriority.NORMAL
    source: str = "unknown"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    #: Монотонное время создания — используется для age/TTL и метрик задержки.
    created_at: float = Field(default_factory=time.monotonic)
    #: Событие отменяет активные прерываемые обработчики.
    preemptive: bool = False
    #: Время жизни в секундах: просроченные события отбрасываются диспетчером.
    ttl_s: float | None = None
    #: Идентификатор цепочки chat -> llm -> tts для трассировки.
    correlation_id: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _apply_type_defaults(self) -> Self:
        """Подставляет приоритет и флаг preemptive исходя из типа события."""
        if "priority" not in self.model_fields_set:
            object.__setattr__(self, "priority", DEFAULT_PRIORITY.get(self.type, EventPriority.NORMAL))
        if "preemptive" not in self.model_fields_set:
            object.__setattr__(self, "preemptive", self.type in PREEMPTIVE_TYPES)
        if self.correlation_id is None:
            object.__setattr__(self, "correlation_id", self.id)
        return self

    @property
    def age_s(self) -> float:
        """Сколько секунд событие живёт с момента создания."""
        return time.monotonic() - self.created_at

    @property
    def is_stale(self) -> bool:
        """``True``, если событие просрочено по TTL."""
        return self.ttl_s is not None and self.age_s > self.ttl_s

    def describe(self) -> str:
        """Короткое человекочитаемое представление для логов."""
        return f"{self.type}#{self.id}(P{int(self.priority)}, from={self.source})"


AnyEvent = Event[EventPayload]
