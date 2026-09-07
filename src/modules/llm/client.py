"""Потоковый клиент LLM.

Работает с любым OpenAI-совместимым сервером (vLLM, llama.cpp server, TGI).
Генерация выполняется внутри ``preemption_scope``: при нажатии PTT шина и снимает
задачу обработчика, и помечает токен — HTTP-стрим закрывается, а сервер получает
разрыв соединения, освобождая слот KV-кэша.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator

from src.core.config import LLMSettings
from src.core.event_bus import EventBus, PreemptionToken
from src.core.events import AnyEvent, Event, EventType, LlmChunk, TextRequest
from src.core.text import SentenceBuffer
from src.modules.base import Module


class LLMModule(Module):
    """Слушает ``LLM_REQUEST`` и стримит ответ пофрагментно."""

    name = "llm"

    def __init__(self, bus: EventBus, settings: LLMSettings) -> None:
        super().__init__(bus)
        self.settings = settings
        self._client: object | None = None

    async def start(self) -> None:
        """Создаёт HTTP-клиент и подписывается на запросы генерации."""
        import httpx

        self._client = httpx.AsyncClient(
            base_url=self.settings.base_url,
            timeout=httpx.Timeout(self.settings.request_timeout_s, connect=5.0),
            headers={"Authorization": f"Bearer {self.settings.api_key.get_secret_value()}"},
        )
        self.bus.subscribe(
            self._on_request,
            EventType.LLM_REQUEST,
            name="llm.generate",
            preemptible=True,
            exclusive=True,
        )
        self._started = True
        self.log.info("LLM-клиент готов: {} @ {}", self.settings.model, self.settings.base_url)

    async def stop(self) -> None:
        """Закрывает HTTP-клиент."""
        client = self._client
        self._client = None
        self._started = False
        if client is not None:
            await client.aclose()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ handlers

    async def _on_request(self, event: AnyEvent) -> None:
        """Генерирует ответ, нарезает его на фразы и отдаёт в TTS.

        Нарезка живёт здесь, а не в TTS-модуле: только здесь поток токенов
        гарантированно упорядочен. Сборка фраз из отдельных событий ``LLM_CHUNK``
        была бы гонкой — обработчики событий выполняются параллельно.
        """
        request = event.payload
        if not isinstance(request, TextRequest):
            self.log.warning("LLM_REQUEST без TextRequest: {}", event.describe())
            return

        buffer = SentenceBuffer(self.settings.sentence_min_chars)
        parts: list[str] = []
        index = 0
        with self.bus.preemption_scope(f"llm:{event.id}") as token:
            async for chunk in self.stream(request.text, token):
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
                sentence = buffer.push(chunk)
                if sentence:
                    await self._speak(sentence, event, request)

            rest = buffer.flush()
            if rest:
                await self._speak(rest, event, request)

        text = "".join(parts).strip()
        await self.bus.publish(
            Event(
                type=EventType.LLM_COMPLETED,
                source=self.name,
                correlation_id=event.correlation_id,
                payload=TextRequest(text=text, trigger=request.trigger, critical=request.critical),
            )
        )

    async def _speak(self, sentence: str, origin: AnyEvent, request: TextRequest) -> None:
        """Отправляет готовую фразу в TTS."""
        await self.bus.publish(
            Event(
                type=EventType.TTS_REQUEST,
                source=self.name,
                correlation_id=origin.correlation_id,
                payload=TextRequest(text=sentence, trigger=request.trigger, critical=request.critical),
            )
        )

    # -------------------------------------------------------------------- stream

    async def stream(self, prompt: str, token: PreemptionToken) -> AsyncIterator[str]:
        """Отдаёт токены ответа по мере генерации, прерываясь по ``token``."""
        import httpx

        if self._client is None:
            raise RuntimeError("LLM-клиент не запущен")
        client: httpx.AsyncClient = self._client  # type: ignore[assignment]

        body = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "stream": True,
        }
        async with client.stream("POST", "/chat/completions", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                token.raise_if_cancelled()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if not data or data == "[DONE]":
                    continue
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta


class MockLLMModule(LLMModule):
    """Псевдо-генерация с задержкой: показывает, что PTT реально прерывает LLM."""

    name = "llm-mock"

    _REPLY_WORDS = (
        "Так,",
        "смотрите,",
        "я",
        "сейчас",
        "думаю",
        "над",
        "этим",
        "вопросом",
        "и",
        "отвечаю",
        "довольно",
        "подробно",
        "потому",
        "что",
        "стример",
        "любит",
        "поговорить.",
    )

    def __init__(self, bus: EventBus, settings: LLMSettings, *, token_delay_s: float = 0.12) -> None:
        super().__init__(bus, settings)
        self.token_delay_s = token_delay_s

    async def start(self) -> None:
        """Подписывается на запросы без поднятия HTTP-клиента."""
        self.bus.subscribe(
            self._on_request,
            EventType.LLM_REQUEST,
            name="llm.generate",
            preemptible=True,
            exclusive=True,
        )
        self._started = True
        self.log.info("Mock-LLM готов (задержка токена {:.0f} мс)", self.token_delay_s * 1000)

    async def stop(self) -> None:
        """Освобождать нечего."""
        self._started = False

    async def stream(self, prompt: str, token: PreemptionToken) -> AsyncIterator[str]:
        """Выдаёт слова с паузами, честно проверяя токен прерывания."""
        count = random.randint(8, len(self._REPLY_WORDS))
        for word in self._REPLY_WORDS[:count]:
            token.raise_if_cancelled()
            await asyncio.sleep(self.token_delay_s)
            yield word + " "
