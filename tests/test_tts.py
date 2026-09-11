"""TTS: очередь фраз, мгновенный abort плеера, event loop не блокируется."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from src.core.config import PTTSettings, RVCSettings, TTSSettings
from src.core.event_bus import EventBus
from src.core.events import Event, EventType, TextRequest
from src.modules.ptt.hotkey import MockPushToTalkModule
from src.modules.tts.engine import KokoroEngine, float_to_pcm16, resolve_lang_code, resolve_voice
from src.modules.tts.rvc import RVCConverter
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
        self.last_audio: np.ndarray | None = None

    def stop(self) -> None:
        self.stops += 1

    def play(self, audio: np.ndarray, abort: Any) -> None:
        deadline = time.monotonic() + self.block_s
        while time.monotonic() < deadline:
            if abort.is_set():
                return
            time.sleep(0.01)
        if abort.is_set():
            return
        self.last_audio = np.asarray(audio)
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


def test_player_stop_does_not_abort_or_close_stream() -> None:
    """PTT глушит буфер, но не вызывает abort/close — иначе микрофон зависает на Windows."""
    from src.modules.tts.player import SoundPlayer

    class _Stream:
        abort_calls = 0
        stop_calls = 0
        close_calls = 0
        active = True

        def abort(self) -> None:
            self.abort_calls += 1

        def stop(self) -> None:
            self.stop_calls += 1

        def close(self) -> None:
            self.close_calls += 1

    player = SoundPlayer()
    stream = _Stream()
    player._stream = stream
    player.stop()
    assert stream.abort_calls == 0
    assert stream.close_calls == 0
    player.close()
    assert stream.close_calls == 1


async def test_kokoro_waits_for_producer_and_closes_generator_on_abort() -> None:
    """По PTT генератор Kokoro закрывается, поток инференса не остаётся висеть на GPU."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    closed = threading.Event()
    abort = threading.Event()

    def pipeline(*_args: object, **_kwargs: object):
        try:
            yield SimpleNamespace(audio=np.ones(8, dtype=np.float32))
            while not abort.is_set():
                time.sleep(0.01)
            yield SimpleNamespace(audio=np.ones(8, dtype=np.float32))
        finally:
            closed.set()

    engine = KokoroEngine(TTSSettings())
    engine._pipeline = pipeline
    engine._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-test")
    chunks = 0
    try:
        async for _audio in engine.stream("hello", abort):
            chunks += 1
            abort.set()
        assert chunks == 1
        assert closed.wait(timeout=2)
    finally:
        engine._executor.shutdown(wait=True)


class _FakeRVC:
    """Заглушка конвертера: помечает чанк и умеет тормозить, как живой инференс."""

    def __init__(self, *, delay_s: float = 0.0, mark: float = 0.5) -> None:
        self.delay_s = delay_s
        self.mark = mark
        self.calls = 0
        self.started = asyncio.Event()

    async def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        abort: Any,
        epoch: int,
        current_epoch: Any,
    ) -> np.ndarray | None:
        _ = sample_rate
        self.calls += 1
        self.started.set()
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if abort.is_set() or epoch != current_epoch():
            return None
        out = np.asarray(audio, dtype=np.float32).copy()
        if out.size:
            out[0] = self.mark
        return out


async def test_rvc_runs_before_playback() -> None:
    """Чанк Kokoro проходит через RVC до sounddevice."""
    player = _FakePlayer()
    rvc = _FakeRVC(mark=0.75)
    bus = EventBus()
    tts = TTSModule(bus, TTSSettings(engine="kokoro"), engine=_FakeEngine(), player=player)

    async with bus:
        await tts.start()
        tts._rvc = rvc
        try:
            await bus.publish(Event(type=EventType.TTS_REQUEST, payload=TextRequest(text="hi")))
            await bus.wait_idle()
        finally:
            await tts.stop()

    assert rvc.calls == 1
    assert player.played == 1
    assert player.last_audio is not None
    assert player.last_audio.reshape(-1)[0] == pytest.approx(0.75)


async def test_rvc_chunk_dropped_on_preempt() -> None:
    """PTT во время RVC не должен отдать чанк в плеер."""
    player = _FakePlayer()
    rvc = _FakeRVC(delay_s=0.25)
    bus = EventBus()
    tts = TTSModule(bus, TTSSettings(), engine=_FakeEngine(), player=player)
    ptt = MockPushToTalkModule(bus, PTTSettings())

    async with bus:
        await tts.start()
        await ptt.start()
        tts._rvc = rvc
        try:
            await bus.publish(Event(type=EventType.TTS_REQUEST, payload=TextRequest(text="hi")))
            await asyncio.wait_for(rvc.started.wait(), timeout=2)
            await ptt.say("стоп")
            await bus.wait_idle()
        finally:
            await tts.stop()
            await ptt.stop()

    assert player.played == 0
    assert tts.pending_sentences == 0


async def test_rvc_converter_discards_result_after_preempt() -> None:
    """Результат синхронного инференса не возвращается, если за время расчёта сменился epoch."""
    converter = RVCConverter(RVCSettings())
    converter._infer = object()
    converter._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-rvc")
    abort = threading.Event()
    epoch = {"n": 1}

    def slow_sync(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        _ = audio, sample_rate
        time.sleep(0.05)
        abort.set()
        epoch["n"] += 1
        return np.ones(8, dtype=np.float32)

    converter.convert_sync = slow_sync  # type: ignore[method-assign]
    try:
        result = await converter.convert(
            np.zeros(8, dtype=np.float32),
            24_000,
            abort=abort,
            epoch=1,
            current_epoch=lambda: epoch["n"],
        )
    finally:
        converter._executor.shutdown(wait=True)

    assert result is None
