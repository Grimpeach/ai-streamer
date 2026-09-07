"""Проверки шины событий: порядок приоритетов и прерывание по PTT."""

from __future__ import annotations

import asyncio

import pytest

from src.core.event_bus import EventBus, Preempted
from src.core.events import ChatMessage, Event, EventPriority, EventType, HostSpeech

pytestmark = pytest.mark.asyncio


async def test_priority_order_beats_arrival_order() -> None:
    """Донат и PTT обгоняют ранее поставленный чат."""
    seen: list[EventType] = []
    # drop_on_preempt отключён: здесь проверяется только порядок, без чистки очереди.
    bus = EventBus(drop_on_preempt=())

    async def collect(event: Event) -> None:
        seen.append(event.type)

    bus.subscribe(collect, name="collector", preemptible=False)

    # Диспетчер ещё не запущен — все события успевают попасть в очередь.
    bus.publish_nowait(Event(type=EventType.TWITCH_CHAT, payload=ChatMessage(author="a", text="раз")))
    bus.publish_nowait(Event(type=EventType.TWITCH_CHAT, payload=ChatMessage(author="b", text="два")))
    bus.publish_nowait(Event(type=EventType.TWITCH_DONATION, priority=EventPriority.HIGH))
    bus.publish_nowait(Event(type=EventType.HOST_SPEECH, payload=HostSpeech(text="я говорю")))

    async with bus:
        await asyncio.wait_for(bus.wait_idle(), timeout=2)

    assert seen[0] is EventType.HOST_SPEECH
    assert seen[1] is EventType.TWITCH_DONATION
    assert seen[2:] == [EventType.TWITCH_CHAT, EventType.TWITCH_CHAT]


async def test_ptt_cancels_running_generation() -> None:
    """Долгий обработчик снимается нажатием PTT, флаг завершения не выставляется."""
    finished = asyncio.Event()
    started = asyncio.Event()
    bus = EventBus()

    async def long_generation(_event: Event) -> None:
        started.set()
        await asyncio.sleep(5)
        finished.set()

    bus.subscribe(long_generation, EventType.LLM_REQUEST, name="llm", preemptible=True)

    async with bus:
        await bus.publish(Event(type=EventType.LLM_REQUEST))
        await asyncio.wait_for(started.wait(), timeout=1)
        assert bus.active_handlers == 1

        await bus.publish(Event(type=EventType.PTT_PRESSED))
        await asyncio.wait_for(bus.wait_idle(), timeout=2)

    assert not finished.is_set()
    assert bus.stats.cancelled_handlers == 1
    assert bus.stats.preemptions == 1


async def test_preemption_token_stops_cooperative_loop() -> None:
    """Токен прерывания останавливает потоковую генерацию между фрагментами."""
    chunks: list[int] = []
    bus = EventBus()

    async def streaming_handler(_event: Event) -> None:
        with bus.preemption_scope("llm") as token:
            try:
                for index in range(100):
                    token.raise_if_cancelled()
                    await asyncio.sleep(0.01)
                    chunks.append(index)
            except Preempted:
                chunks.append(-1)

    # Не preemptible: задача не снимается, генерация обязана остановиться сама.
    bus.subscribe(streaming_handler, EventType.LLM_REQUEST, name="llm", preemptible=False)

    async with bus:
        await bus.publish(Event(type=EventType.LLM_REQUEST))
        await asyncio.sleep(0.1)
        bus.preempt(reason="ptt")
        await asyncio.wait_for(bus.wait_idle(), timeout=2)

    assert chunks[-1] == -1
    assert len(chunks) < 100


async def test_preempt_purges_low_priority_backlog() -> None:
    """Накопленный чат выбрасывается, донат в очереди сохраняется."""
    bus = EventBus()

    for i in range(3):
        bus.publish_nowait(Event(type=EventType.TWITCH_CHAT, payload=ChatMessage(author="a", text=str(i))))
    bus.publish_nowait(Event(type=EventType.TWITCH_DONATION))

    bus.preempt(reason="ptt")

    remaining = [event.type for event in bus.pending_events()]
    assert remaining == [EventType.TWITCH_DONATION]
    assert bus.stats.dropped_preempt == 3


async def test_stale_events_are_dropped() -> None:
    """Просроченный по TTL чат не доходит до обработчика."""
    seen: list[str] = []
    bus = EventBus()

    async def collect(event: Event) -> None:
        seen.append(event.id)

    bus.subscribe(collect, EventType.TWITCH_CHAT, name="chat", preemptible=False)

    stale = Event(type=EventType.TWITCH_CHAT, ttl_s=0.01, payload=ChatMessage(author="a", text="старое"))
    await asyncio.sleep(0.05)

    async with bus:
        await bus.publish(stale)
        await asyncio.wait_for(bus.wait_idle(), timeout=2)

    assert seen == []
    assert bus.stats.dropped_stale == 1


async def test_exclusive_subscription_defers_while_busy() -> None:
    """Exclusive-обработчик выполняется последовательно, событие не теряется."""
    concurrent = 0
    peak = 0
    calls = 0
    bus = EventBus()

    async def handler(_event: Event) -> None:
        nonlocal concurrent, peak, calls
        calls += 1
        concurrent += 1
        peak = max(peak, concurrent)
        await asyncio.sleep(0.1)
        concurrent -= 1

    bus.subscribe(handler, EventType.LLM_REQUEST, name="llm", preemptible=False, exclusive=True)

    async with bus:
        await bus.publish(Event(type=EventType.LLM_REQUEST))
        await asyncio.sleep(0.02)
        await bus.publish(Event(type=EventType.LLM_REQUEST))
        await asyncio.wait_for(bus.wait_idle(), timeout=2)

    assert calls == 2
    assert peak == 1


async def test_deferred_events_are_dropped_by_priority_overflow() -> None:
    """Очередь ожидания exclusive-обработчика короткая: чат вытесняет себя же."""
    handled: list[str] = []
    bus = EventBus()

    async def handler(event: Event) -> None:
        assert isinstance(event.payload, ChatMessage)
        handled.append(event.payload.text)
        await asyncio.sleep(0.05)

    bus.subscribe(
        handler, EventType.TWITCH_CHAT, name="chat", preemptible=False, exclusive=True, max_deferred=1
    )

    async with bus:
        for i in range(5):
            await bus.publish(Event(type=EventType.TWITCH_CHAT, payload=ChatMessage(author="a", text=str(i))))
        await asyncio.wait_for(bus.wait_idle(), timeout=3)

    assert handled[0] == "0"
    assert len(handled) < 5
    assert bus.stats.dropped_overflow > 0
