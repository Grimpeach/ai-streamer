"""Сквозные проверки конвейера: чат → LLM → TTS и прерывание по PTT."""

from __future__ import annotations

import asyncio

import pytest

from src.core.config import Settings, load_settings
from src.core.event_bus import EventBus
from src.core.events import AnyEvent, ChatMessage, Event, EventType, TextRequest
from src.modules.llm.client import MockLLMModule
from src.modules.ptt.hotkey import MockPushToTalkModule
from src.modules.tts.streamer import MockTTSModule


@pytest.fixture
def settings() -> Settings:
    """Конфигурация демо-режима без внешних сервисов."""
    return load_settings(demo_mode=True)


def _chat(text: str = "привет") -> AnyEvent:
    return Event(type=EventType.TWITCH_CHAT, payload=ChatMessage(author="viewer", text=text))


async def test_chat_reaches_tts(settings: Settings) -> None:
    """Сообщение чата доходит до синтеза речи через LLM без отдельного роутера."""
    spoken: list[str] = []
    first_phrase = asyncio.Event()
    bus = EventBus()

    async def on_started(event: AnyEvent) -> None:
        assert isinstance(event.payload, TextRequest)
        spoken.append(event.payload.text)
        first_phrase.set()

    bus.subscribe(on_started, EventType.TTS_STARTED, name="probe", preemptible=False)

    llm = MockLLMModule(bus, settings.llm, token_delay_s=0.01)
    tts = MockTTSModule(bus, settings.tts, chunk_duration_s=0.01)

    async with bus:
        await llm.start()
        await tts.start()
        try:
            await bus.publish(_chat())
            await asyncio.wait_for(first_phrase.wait(), timeout=5)
        finally:
            await llm.stop()
            await tts.stop()

    # Mock-LLM отдаёт «Так,» первым словом — запятая сразу публикует кусок в TTS.
    assert spoken[0] == "Well,"


async def test_ptt_silences_tts_immediately(settings: Settings) -> None:
    """После нажатия PTT очередь TTS сбрасывается, текущий синтез снимается."""
    chunks = 0
    first_chunk = asyncio.Event()
    bus = EventBus()

    async def on_chunk(_event: AnyEvent) -> None:
        nonlocal chunks
        chunks += 1
        first_chunk.set()

    bus.subscribe(on_chunk, EventType.TTS_CHUNK, name="probe", preemptible=False)

    llm = MockLLMModule(bus, settings.llm, token_delay_s=0.15)
    tts = MockTTSModule(bus, settings.tts, chunk_duration_s=0.02)
    ptt = MockPushToTalkModule(bus, settings.ptt)

    async with bus:
        await llm.start()
        await tts.start()
        await ptt.start()
        try:
            await bus.publish(_chat())
            await asyncio.wait_for(first_chunk.wait(), timeout=5)

            await ptt.say("я сам отвечу")
            assert tts.pending_sentences == 0

            after_reset = chunks
            await asyncio.sleep(0.08)
            assert chunks == after_reset, "TTS продолжил говорить текущую фразу после PTT"
        finally:
            await llm.stop()
            await tts.stop()
            await ptt.stop()
