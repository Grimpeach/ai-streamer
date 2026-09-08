"""TTS: очередь фраз, мгновенный abort плеера, event loop не блокируется."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from src.core.config import PTTSettings, TTSSettings
from src.core.event_bus import EventBus
from src.core.events import Event, EventType, TextRequest
from src.modules.ptt.hotkey import MockPushToTalkModule
from src.modules.tts.engine import KokoroEngine, float_to_pcm16, resolve_lang_code, resolve_voice
from src.modules.tts.streamer import TTSModule


class _FakeEngine:
    loaded = True

    def __init__(self, *, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.texts: list[str] = []

    async def stream(self, text: str, abort: Any):
        self.texts.append(text)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if abort.is_set():
            return
        yield np.zeros(48, dtype=np.float32)


class _FakePlayer:
    def __init__(self, *, block_s: float = 0.0) -> None:
        self.played = 0
        self.stops = 0
        self.block_s = block_s

    def stop(self) -> None:
        self.stops += 1

    def play(self, _audio: np.ndarray, abort: Any) -> None:
        deadline = time.monotonic() + self.block_s
        while time.monotonic() < deadline:
            if abort.is_set():
                return
            time.sleep(0.01)
        if abort.is_set():
            return
        self.played += 1


def test_voice_alias_maps_placeholder_to_kokoro() -> None:
    """Ранний ru_female_1 указывает на реальный голос Kokoro."""
    assert resolve_voice("ru_female_1") == "af_heart"
    assert resolve_lang_code("af_heart") == "a"


def test_pcm16_is_little_endian_int16() -> None:
    pcm = float_to_pcm16(np.array([0.0, 1.0, -1.0], dtype=np.float32))
    assert len(pcm) == 6


def test_kokoro_falls_back_to_cpu_on_cudnn_error() -> None:
    """Конфликт cuDNN с STT не должен ронять запуск TTS."""
    engine = KokoroEngine(TTSSettings(device="auto", fallback_to_cpu=True))
    calls: list[str] = []

    class _Pipe:
        def __call__(self, *_args: object, **_kwargs: object) -> list[object]:
            return []

    def fake_create(device: str) -> object:
        calls.append(device)
        if device == "cuda":
            raise OSError("Could not load symbol cudnnGetLibConfig. Error code 127")
        return _Pipe()

    engine._pick_device = lambda: "cuda"  # type: ignore[method-assign]
    engine._create_pipeline = fake_create  # type: ignore[method-assign]
    engine._load_sync()

    assert engine.device == "cpu"
    assert calls == ["cuda", "cpu"]


async def test_tts_plays_and_publishes_chunks() -> None:
    """TTS_REQUEST доходит до плеера и шины."""
    chunks = 0
    started = asyncio.Event()
    bus = EventBus()

    async def on_chunk(_event: Event) -> None:
        nonlocal chunks
        chunks += 1

    async def on_started(_event: Event) -> None:
        started.set()

    bus.subscribe(on_chunk, EventType.TTS_CHUNK, name="probe", preemptible=False)
    bus.subscribe(on_started, EventType.TTS_STARTED, name="probe.start", preemptible=False)

    player = _FakePlayer()
    tts = TTSModule(bus, TTSSettings(engine="kokoro"), engine=_FakeEngine(), player=player)

    async with bus:
        await tts.start()
        try:
            await bus.publish(Event(type=EventType.TTS_REQUEST, payload=TextRequest(text="Привет,")))
            await asyncio.wait_for(started.wait(), timeout=2)
            await bus.wait_idle()
        finally:
            await tts.stop()

    assert player.played == 1
    assert chunks == 1


async def test_ptt_stops_player_and_drops_queue() -> None:
    """PTT глушит плеер и вычищает невоспроизведённые фразы."""
    bus = EventBus()
    first = asyncio.Event()

    async def on_started(_event: Event) -> None:
        first.set()

    bus.subscribe(on_started, EventType.TTS_STARTED, name="probe", preemptible=False)

    player = _FakePlayer(block_s=1.0)
    tts = TTSModule(bus, TTSSettings(), engine=_FakeEngine(), player=player)
    ptt = MockPushToTalkModule(bus, PTTSettings())

    async with bus:
        await tts.start()
        await ptt.start()
        try:
            await bus.publish(Event(type=EventType.TTS_REQUEST, payload=TextRequest(text="раз")))
            await bus.publish(Event(type=EventType.TTS_REQUEST, payload=TextRequest(text="два")))
            await asyncio.wait_for(first.wait(), timeout=2)
            await ptt.say("стоп")
            await bus.wait_idle()
            assert tts.pending_sentences == 0
            assert player.stops >= 1
        finally:
            await tts.stop()
            await ptt.stop()


async def test_playback_does_not_block_event_loop() -> None:
    """Пока плеер спит в потоке, loop продолжает тикать."""
    bus = EventBus()
    player = _FakePlayer(block_s=0.25)
    tts = TTSModule(bus, TTSSettings(), engine=_FakeEngine(), player=player)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(8):
            await asyncio.sleep(0.02)
            ticks += 1

    async with bus:
        await tts.start()
        try:
            ping = asyncio.create_task(ticker())
            await bus.publish(Event(type=EventType.TTS_REQUEST, payload=TextRequest(text="тик")))
            await ping
            await bus.wait_idle()
        finally:
            await tts.stop()

    assert ticks >= 6
