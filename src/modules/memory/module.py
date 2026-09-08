"""Модуль памяти: Redis (окно диалога) + Qdrant (факты о зрителях).

Перед ответом LLM запрашивает :meth:`context_for` — релевантные факты по нику
и теме. Запись идёт с шины независимо от генерации: даже прерванный ответ
не мешает запомнить реплику зрителя.
"""

from __future__ import annotations

from src.core.config import MemorySettings
from src.core.event_bus import EventBus
from src.core.events import (
    AnyEvent,
    ChatMessage,
    Donation,
    EventType,
    HostSpeech,
    Raid,
    TextRequest,
    TwitchSubscription,
)
from src.modules.base import Module
from src.modules.memory.qdrant_store import LongTermMemory, MemoryHit
from src.modules.memory.redis_store import ShortTermMemory, Turn

_INBOUND = (
    EventType.HOST_SPEECH,
    EventType.TWITCH_CHAT,
    EventType.TWITCH_DONATION,
    EventType.TWITCH_SUBSCRIPTION,
    EventType.TWITCH_RAID,
)

_HOST_AUTHOR = "host"


class MemoryModule(Module):
    """Подключает хранилища и отдаёт контекст для промпта LLM."""

    name = "memory"

    def __init__(
        self,
        bus: EventBus,
        settings: MemorySettings,
        *,
        session: str = "local",
        short: ShortTermMemory | None = None,
        long: LongTermMemory | None = None,
    ) -> None:
        super().__init__(bus)
        self.settings = settings
        self.session = session or "local"
        self.short = short or ShortTermMemory(settings)
        self.long = long or LongTermMemory(settings)
        self.redis_ok = False
        self.qdrant_ok = False

    async def start(self) -> None:
        """Открывает Redis/Qdrant, прогревает эмбеддер, подписывается на шину."""
        try:
            await self.short.connect()
            self.redis_ok = True
        except Exception:
            self.log.exception("Redis недоступен — краткосрочная память выключена")

        try:
            await self.long.connect()
            self.qdrant_ok = True
            # Прогрев весов, чтобы первый чат не ждал загрузки e5.
            await self.long.recall("warmup", limit=1)
        except Exception:
            self.log.exception("Qdrant недоступен — векторная память выключена")

        self.bus.subscribe(
            self._on_inbound,
            _INBOUND,
            name="memory.ingest",
            preemptible=False,
        )
        self.bus.subscribe(
            self._on_reply,
            EventType.LLM_COMPLETED,
            name="memory.reply",
            preemptible=False,
        )
        self._started = True
        self.log.info(
            "Память готова (redis={}, qdrant={}, session='{}')",
            self.redis_ok,
            self.qdrant_ok,
            self.session,
        )

    async def stop(self) -> None:
        """Закрывает клиенты хранилищ."""
        self._started = False
        try:
            await self.short.close()
        except Exception:
            self.log.debug("Redis close failed")
        try:
            await self.long.close()
        except Exception:
            self.log.debug("Qdrant close failed")
        self.redis_ok = False
        self.qdrant_ok = False

    async def context_for(self, event: AnyEvent) -> str:
        """Собирает блок памяти для промпта: факты о нике и по теме реплики."""
        author, text, _kind = _extract(event)
        if not text:
            return ""
        hits: list[MemoryHit] = []
        if self.qdrant_ok:
            try:
                hits = await self.long.recall(text, author=author)
            except Exception:
                self.log.exception("Не удалось прочитать Qdrant")
        if not hits:
            return ""
        return _format_hits(hits, author)

    async def _on_inbound(self, event: AnyEvent) -> None:
        """Пишет реплику зрителя/хоста в Redis и, если есть смысл, в Qdrant."""
        author, text, kind = _extract(event)
        if not text:
            return
        role: str = "host" if kind == "host" else "viewer"
        await self._append_short(Turn(role=role, text=text, author=author))  # type: ignore[arg-type]
        if self.qdrant_ok and len(text) >= self.settings.min_remember_chars:
            try:
                await self.long.remember(text, author=author, kind=kind)
            except Exception:
                self.log.exception("Не удалось записать факт в Qdrant")

    async def _on_reply(self, event: AnyEvent) -> None:
        """Пишет законченный ответ стримерши только в краткосрочное окно."""
        payload = event.payload
        if not isinstance(payload, TextRequest):
            return
        text = payload.text.strip()
        if not text:
            return
        await self._append_short(Turn(role="streamer", text=text, author=""))

    async def _append_short(self, turn: Turn) -> None:
        if not self.redis_ok:
            return
        try:
            await self.short.append(self.session, turn)
        except Exception:
            self.log.exception("Не удалось записать реплику в Redis")


def _extract(event: AnyEvent) -> tuple[str, str, str]:
    """Ник, текст и тип факта. Пустой текст — событие нужно пропустить."""
    payload = event.payload
    if isinstance(payload, HostSpeech):
        return _HOST_AUTHOR, payload.text.strip(), "host"
    if isinstance(payload, ChatMessage):
        return payload.author, payload.text.strip(), "chat"
    if isinstance(payload, Donation):
        text = payload.message.strip() or f"donation {payload.amount:.0f} {payload.currency}"
        return payload.author, text, "donation"
    if isinstance(payload, TwitchSubscription):
        return payload.author, payload.message.strip() or "subscription", "subscription"
    if isinstance(payload, Raid):
        return payload.author, f"raid {payload.viewers} viewers", "raid"
    if isinstance(payload, TextRequest):
        return payload.trigger, payload.text.strip(), "request"
    return "", "", "unknown"


def _format_hits(hits: list[MemoryHit], author: str) -> str:
    """Короткий блок для пользовательского сообщения LLM."""
    lines = ["[Memory]"]
    if author and author != _HOST_AUTHOR:
        lines.append(f"About viewer {author}:")
    for hit in hits[:5]:
        who = hit.author or "stream"
        snippet = hit.text.replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:217] + "..."
        lines.append(f"- {who}: {snippet}")
    return "\n".join(lines)
