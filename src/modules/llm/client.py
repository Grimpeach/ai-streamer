"""Потоковый клиент LLM.

Работает с любым OpenAI-совместимым сервером (vLLM, llama.cpp server, TGI)
через асинхронный ``openai.AsyncOpenAI``. Генерация выполняется внутри
``preemption_scope``: при нажатии PTT шина снимает задачу обработчика
(``CancelledError``) и помечает токен — HTTP-стрим закрывается, слот KV-кэша
на сервере освобождается. Оборванный ответ в историю не пишется.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from typing import Any

from src.core.config import LLMSettings
from src.core.event_bus import EventBus, Preempted, PreemptionToken
from src.core.events import (
    AnyEvent,
    ChatMessage,
    Donation,
    Event,
    EventType,
    HostSpeech,
    LlmChunk,
    Raid,
    TextRequest,
    TwitchSubscription,
)
from src.core.text import SentenceBuffer
from src.modules.base import Module
from src.modules.llm.context import DialogueContext
from src.modules.llm.persona import render_system_prompt
from src.modules.memory.module import MemoryModule

#: Входящие триггеры реплики. ``LLM_REQUEST`` оставлен для тестов и ручного ввода.
_INBOUND = (
    EventType.HOST_SPEECH,
    EventType.TWITCH_CHAT,
    EventType.TWITCH_DONATION,
    EventType.TWITCH_SUBSCRIPTION,
    EventType.TWITCH_RAID,
    EventType.LLM_REQUEST,
)


class LLMModule(Module):
    """Слушает чат (P3), хоста (P1) и алерты (P2); стримит ответ пофрагментно."""

    name = "llm"

    def __init__(
        self,
        bus: EventBus,
        settings: LLMSettings,
        memory: MemoryModule | None = None,
    ) -> None:
        super().__init__(bus)
        self.settings = settings
        self.memory = memory
        self.context = DialogueContext(
            system_prompt=_system_prompt(settings),
            max_messages=settings.history_turns,
        )
        self._client: Any = None

    async def start(self) -> None:
        """Создаёт OpenAI-клиент и подписывается на входящие реплики."""
        if self._client is None:
            self._client = self._make_client()
        self._subscribe()
        self._started = True
        self.log.info("LLM-клиент готов: {} @ {}", self.settings.model, self.settings.base_url)

    def _make_client(self) -> Any:
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            base_url=self.settings.base_url,
            api_key=self.settings.api_key.get_secret_value(),
            timeout=self.settings.request_timeout_s,
        )

    def _subscribe(self) -> None:
        self.bus.subscribe(
            self._on_inbound,
            _INBOUND,
            name="llm.generate",
            preemptible=True,
            exclusive=True,
        )

    async def stop(self) -> None:
        """Закрывает HTTP-клиент."""
        client = self._client
        self._client = None
        self._started = False
        await _close_async(client)

    # ------------------------------------------------------------------ handlers

    async def _on_inbound(self, event: AnyEvent) -> None:
        """Генерирует ответ, нарезает его на фразы и отдаёт в TTS.

        Нарезка живёт здесь, а не в TTS-модуле: только здесь поток токенов
        гарантированно упорядочен. Сборка фраз из отдельных событий ``LLM_CHUNK``
        была бы гонкой — обработчики событий выполняются параллельно.
        """
        prompt = _user_prompt(event)
        if prompt is None:
            self.log.warning("LLM пропустил событие без текста: {}", event.describe())
            return

        user_text, trigger, critical = prompt
        request = TextRequest(text=user_text, trigger=trigger, critical=critical)
        self.context.add_user(user_text)
        messages = self.context.messages()
        memory_block = await self._memory_block(event)
        if memory_block:
            messages = _with_memory(messages, memory_block)

        buffer = SentenceBuffer(self.settings.sentence_min_chars)
        parts: list[str] = []
        index = 0
        try:
            with self.bus.preemption_scope(f"llm:{event.id}") as token:
                async for chunk in self.stream(messages, token):
                    parts.append(chunk)
                    await self.bus.publish(
                        Event(
                            type=EventType.LLM_CHUNK,
                            source=self.name,
                            correlation_id=event.correlation_id,
                            payload=LlmChunk(text=chunk, index=index),
                        )
                    )
                    index += 1
                    for sentence in buffer.push(chunk):
                        await self._speak(sentence, event, request)

                rest = buffer.flush()
                if rest:
                    await self._speak(rest, event, request)
        except (asyncio.CancelledError, Preempted):
            buffer.clear()
            self.log.debug("Генерация LLM прервана, незавершённый ответ в историю не записан")
            raise

        text = "".join(parts).strip()
        if text:
            self.context.add_assistant(text)
        await self.bus.publish(
            Event(
                type=EventType.LLM_COMPLETED,
                source=self.name,
                correlation_id=event.correlation_id,
                payload=TextRequest(text=text, trigger=trigger, critical=critical),
            )
        )

    async def _memory_block(self, event: AnyEvent) -> str:
        """Достаёт релевантный контекст из Redis/Qdrant; сбой памяти не роняет ответ."""
        if self.memory is None:
            return ""
        try:
            return await self.memory.context_for(event)
        except Exception:
            self.log.exception("Память не ответила, генерирую без неё")
            return ""

    async def _speak(self, sentence: str, origin: AnyEvent, request: TextRequest) -> None:
        """Отправляет готовый кусок текста в TTS."""
        await self.bus.publish(
            Event(
                type=EventType.TTS_REQUEST,
                source=self.name,
                correlation_id=origin.correlation_id,
                payload=TextRequest(text=sentence, trigger=request.trigger, critical=request.critical),
            )
        )

    # -------------------------------------------------------------------- stream

    async def stream(
        self,
        messages: list[dict[str, str]],
        token: PreemptionToken,
    ) -> AsyncIterator[str]:
        """Отдаёт токены ответа по мере генерации, прерываясь по ``token``."""
        if self._client is None:
            raise RuntimeError("LLM-клиент не запущен")

        token.raise_if_cancelled()
        stream = await self._client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            stream=True,
        )
        try:
            iterator = stream.__aiter__()
            first_timeout = self.settings.first_token_timeout_s
            while True:
                token.raise_if_cancelled()
                try:
                    chunk = await self._next_chunk(iterator, first_timeout)
                except StopAsyncIteration:
                    break
                first_timeout = None
                delta = _choice_delta(chunk)
                if delta:
                    yield delta
        finally:
            await _close_async(stream)

    async def _next_chunk(self, iterator: AsyncIterator[Any], timeout_s: float | None) -> Any:
        """Читает следующий чанк; первый токен ограничен ``first_token_timeout_s``."""
        if timeout_s is None or timeout_s <= 0:
            return await iterator.__anext__()
        try:
            return await asyncio.wait_for(iterator.__anext__(), timeout=timeout_s)
        except TimeoutError:
            self.log.error("LLM не отдал первый токен за {:.1f} с", timeout_s)
            raise


class MockLLMModule(LLMModule):
    """Псевдо-генерация с задержкой: показывает, что PTT реально прерывает LLM."""

    name = "llm-mock"

    _REPLY_WORDS = (
        "Well,",
        "look,",
        "I",
        "am",
        "thinking",
        "about",
        "this",
        "question",
        "and",
        "I",
        "shall",
        "answer",
        "at",
        "some",
        "length",
        "because",
        "the",
        "host",
        "likes",
        "to",
        "talk.",
    )

    def __init__(self, bus: EventBus, settings: LLMSettings, *, token_delay_s: float = 0.12) -> None:
        super().__init__(bus, settings)
        self.token_delay_s = token_delay_s

    async def start(self) -> None:
        """Подписывается на входящие события без HTTP-клиента."""
        self._subscribe()
        self._started = True
        self.log.info("Mock-LLM готов (задержка токена {:.0f} мс)", self.token_delay_s * 1000)

    async def stop(self) -> None:
        """Освобождать нечего."""
        self._started = False

    async def stream(
        self,
        messages: list[dict[str, str]],
        token: PreemptionToken,
    ) -> AsyncIterator[str]:
        """Выдаёт слова с паузами, честно проверяя токен прерывания."""
        _ = messages
        count = random.randint(8, len(self._REPLY_WORDS))
        for word in self._REPLY_WORDS[:count]:
            token.raise_if_cancelled()
            await asyncio.sleep(self.token_delay_s)
            yield word + " "


def _system_prompt(settings: LLMSettings) -> str:
    custom = settings.system_prompt.strip()
    if custom:
        return render_system_prompt(settings.character_name, custom)
    return render_system_prompt(settings.character_name)


def _with_memory(messages: list[dict[str, str]], block: str) -> list[dict[str, str]]:
    """Добавляет блок памяти к последней user-реплике, не портя историю."""
    enriched = [dict(item) for item in messages]
    for index in range(len(enriched) - 1, -1, -1):
        if enriched[index]["role"] == "user":
            enriched[index]["content"] = f"{enriched[index]['content']}\n\n{block}"
            break
    return enriched


def _user_prompt(event: AnyEvent) -> tuple[str, str, bool] | None:
    """Собирает реплику пользователя, триггер и флаг critical. ``None`` — пропуск."""
    payload = event.payload
    if isinstance(payload, HostSpeech):
        text = payload.text.strip()
        return (f"Host says: {text}", str(event.type), False) if text else None
    if isinstance(payload, ChatMessage):
        text = payload.text.strip()
        if not text:
            return None
        return (f"Viewer {payload.author} in chat: {text}", str(event.type), False)
    if isinstance(payload, Donation):
        line = f"{payload.author} donated {payload.amount:.0f} {payload.currency}"
        message = payload.message.strip()
        if message:
            line += f": {message}"
        return (f"{line}. Thank them with royal grace, briefly.", str(event.type), False)
    if isinstance(payload, TwitchSubscription):
        return (f"{payload.author} subscribed. Acknowledge their generosity.", str(event.type), False)
    if isinstance(payload, Raid):
        return (
            f"{payload.author} raided with {payload.viewers} viewers. Greet them.",
            str(event.type),
            False,
        )
    if isinstance(payload, TextRequest):
        text = payload.text.strip()
        return (text, payload.trigger, payload.critical) if text else None
    return None


def _choice_delta(chunk: object) -> str | None:
    """Достаёт текст из чанка Chat Completions. Пустые keep-alive чанки игнорируются."""
    choices = getattr(chunk, "choices", None)
    if not choices:
        return None
    delta = getattr(choices[0], "delta", None)
    content = getattr(delta, "content", None) if delta is not None else None
    if isinstance(content, str) and content:
        return content
    return None


async def _close_async(resource: object | None) -> None:
    """Закрывает клиент или стрим, не завися от имени метода в версии SDK."""
    if resource is None:
        return
    for name in ("aclose", "close"):
        method = getattr(resource, name, None)
        if method is None:
            continue
        result = method()
        if asyncio.iscoroutine(result):
            await result
        return
