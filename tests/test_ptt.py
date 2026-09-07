"""Проверки Push-to-Talk: мгновенное прерывание, приоритет 1, отсутствие блокировок.

Микрофон и Whisper подменяются заглушками, поэтому тесты не требуют ни звуковой
карты, ни GPU, ни весов модели. При этом асинхронные обёртки (``to_thread`` и
пул потоков STT) — настоящие: именно они проверяются на неблокирующесть.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from src.core.config import PTTSettings, STTSettings
from src.core.event_bus import EventBus
from src.core.events import AnyEvent, EventPriority, EventType, HostSpeech
from src.modules.ptt.hotkey import PushToTalkModule
from src.modules.ptt.recorder import AudioRecorder, RecordingResult
from src.modules.ptt.stt import Transcript, WhisperTranscriber

pytestmark = pytest.mark.asyncio


class FakeRecorder(AudioRecorder):
    """Микрофон-заглушка: отдаёт заданное аудио без обращения к устройству."""

    def __init__(self, settings: PTTSettings, *, duration_s: float = 1.5) -> None:
        super().__init__(settings)
        self.duration_s = duration_s
        self.captures = 0
        self.opened = False

    async def open(self) -> None:
        self.opened = True

    async def close(self) -> None:
        self.opened = False

    def start_capture(self) -> None:
        self.captures += 1

    def stop_capture(self) -> RecordingResult | None:
        samples = int(self.duration_s * self.settings.sample_rate)
        return RecordingResult(
            audio=np.zeros(samples, dtype="float32"),
            sample_rate=self.settings.sample_rate,
            duration_s=self.duration_s,
        )


class FakeTranscriber(WhisperTranscriber):
    """STT-заглушка: подменён только синхронный инференс.

    Асинхронная обёртка и пул потоков — из настоящего кода, поэтому тесты
    действительно проверяют, что блокирующая работа уходит с event loop.
    """

    def __init__(self, settings: STTSettings, *, text: str = "я сам отвечу", delay_s: float = 0.0):
        super().__init__(settings)
        self.text = text
        self.delay_s = delay_s
        self.calls = 0
        self.thread_names: list[str] = []

    def _load_sync(self) -> None:
        self._model = object()  # веса не нужны

    def _transcribe_sync(self, audio: object, sample_rate: int) -> Transcript:
        self.calls += 1
        self.thread_names.append(threading.current_thread().name)
        if self.delay_s:
            time.sleep(self.delay_s)  # блокирующая работа, как в настоящем Whisper
        return Transcript(text=self.text, audio_s=1.5, latency_s=self.delay_s, avg_logprob=-0.2)


def make_ptt(
    bus: EventBus,
    *,
    text: str = "я сам отвечу",
    delay_s: float = 0.0,
    duration_s: float = 1.5,
    ptt_overrides: dict[str, object] | None = None,
) -> tuple[PushToTalkModule, FakeRecorder, FakeTranscriber]:
    """Собирает модуль PTT на заглушках."""
    settings = PTTSettings(**(ptt_overrides or {}))  # type: ignore[arg-type]
    recorder = FakeRecorder(settings, duration_s=duration_s)
    transcriber = FakeTranscriber(STTSettings(warmup=False), text=text, delay_s=delay_s)
    module = PushToTalkModule(
        bus, settings, recorder=recorder, transcriber=transcriber
    )
    return module, recorder, transcriber


async def start_without_hook(module: PushToTalkModule) -> None:
    """Запускает модуль, пропуская глобальный хук (библиотеки keyboard нет в CI)."""
    module._loop = asyncio.get_running_loop()
    await module._transcriber.load()
    await module._recorder.open()
    module._started = True


class Collector:
    """Складывает события шины."""

    def __init__(self, bus: EventBus) -> None:
        self.events: list[AnyEvent] = []
        bus.subscribe(self._collect, name="collector", preemptible=False)

    async def _collect(self, event: AnyEvent) -> None:
        self.events.append(event)

    def of_type(self, event_type: EventType) -> list[AnyEvent]:
        return [event for event in self.events if event.type is event_type]


# ------------------------------------------------------------------- прерывание


async def test_press_preempts_generation_immediately() -> None:
    """Нажатие PTT снимает работающую генерацию LLM/TTS."""
    bus = EventBus()
    finished = asyncio.Event()
    started = asyncio.Event()

    async def long_generation(_event: AnyEvent) -> None:
        started.set()
        await asyncio.sleep(5)
        finished.set()

    bus.subscribe(long_generation, EventType.LLM_REQUEST, name="llm", preemptible=True)
    module, _, _ = make_ptt(bus)

    async with bus:
        await start_without_hook(module)
        try:
            await bus.emit(EventType.LLM_REQUEST, source="test")
            await asyncio.wait_for(started.wait(), timeout=2)

            module._on_key_down()
            await asyncio.wait_for(bus.wait_idle(), timeout=2)
        finally:
            await module.stop()

    assert not finished.is_set(), "генерация не была прервана"
    assert bus.stats.cancelled_handlers >= 1


async def test_press_preempts_exactly_once() -> None:
    """Нажатие даёт одно прерывание, а не два.

    Событие PTT_PRESSED публикуется с preemptive=False, потому что preempt уже
    вызван напрямую: второй вызов сбил бы токены только что запущенных
    обработчиков (например, сброса очереди TTS).
    """
    bus = EventBus()
    module, _, _ = make_ptt(bus)

    async with bus:
        await start_without_hook(module)
        try:
            module._on_key_down()
            await asyncio.wait_for(bus.wait_idle(), timeout=2)
        finally:
            await module.stop()

    assert bus.stats.preemptions == 1


async def test_key_autorepeat_is_ignored() -> None:
    """Пока клавишу держат, keyboard сыплет событиями нажатия."""
    bus = EventBus()
    module, recorder, _ = make_ptt(bus)

    async with bus:
        await start_without_hook(module)
        try:
            for _ in range(5):
                module._on_key_down()
            await asyncio.wait_for(bus.wait_idle(), timeout=2)
        finally:
            await module.stop()

    assert recorder.captures == 1
    assert bus.stats.preemptions == 1


# -------------------------------------------------------------- публикация текста


async def test_host_speech_published_with_priority_1() -> None:
    """Расшифрованная реплика уходит на шину с наивысшим приоритетом."""
    bus = EventBus()
    collector = Collector(bus)
    module, _, transcriber = make_ptt(bus, text="стоп, я объясню сам")

    async with bus:
        await start_without_hook(module)
        try:
            module._on_key_down()
            module._on_key_up()
            for _ in range(200):
                if collector.of_type(EventType.HOST_SPEECH):
                    break
                await asyncio.sleep(0.01)
        finally:
            await module.stop()

    speech = collector.of_type(EventType.HOST_SPEECH)
    assert len(speech) == 1
    event = speech[0]
    assert event.priority is EventPriority.INTERRUPT
    assert int(event.priority) == 1
    assert isinstance(event.payload, HostSpeech)
    assert event.payload.text == "стоп, я объясню сам"
    assert event.payload.confidence and event.payload.confidence > 0
    assert transcriber.calls == 1


async def test_short_press_skips_transcription() -> None:
    """Случайный тап прерывает генерацию, но расшифровывать нечего."""
    bus = EventBus()
    collector = Collector(bus)
    module, _, transcriber = make_ptt(bus, duration_s=0.05)

    async with bus:
        await start_without_hook(module)
        try:
            module._on_key_down()
            module._on_key_up()
            await asyncio.sleep(0.15)
            await asyncio.wait_for(bus.wait_idle(), timeout=2)
        finally:
            await module.stop()

    assert transcriber.calls == 0
    assert collector.of_type(EventType.HOST_SPEECH) == []
    assert bus.stats.preemptions == 1, "прерывание должно случиться даже на коротком нажатии"


async def test_empty_transcript_is_not_published() -> None:
    """Если Whisper ничего не распознал, событие не публикуется."""
    bus = EventBus()
    collector = Collector(bus)
    module, _, _ = make_ptt(bus, text="   ")

    async with bus:
        await start_without_hook(module)
        try:
            module._on_key_down()
            module._on_key_up()
            await asyncio.sleep(0.2)
            await asyncio.wait_for(bus.wait_idle(), timeout=2)
        finally:
            await module.stop()

    assert collector.of_type(EventType.HOST_SPEECH) == []


async def test_stale_transcript_is_dropped() -> None:
    """Реплику, которую хост уже перебил новым нажатием, публиковать нельзя."""
    bus = EventBus()
    collector = Collector(bus)
    module, _, _ = make_ptt(bus, text="первая реплика", delay_s=0.3)

    async with bus:
        await start_without_hook(module)
        try:
            module._on_key_down()
            module._on_key_up()
            await asyncio.sleep(0.05)  # расшифровка первой реплики уже идёт
            module._on_key_down()  # хост передумал и заговорил снова
            await asyncio.sleep(0.5)
            await asyncio.wait_for(bus.wait_idle(), timeout=3)
        finally:
            await module.stop()

    assert collector.of_type(EventType.HOST_SPEECH) == [], "устаревшая расшифровка попала на шину"


# ------------------------------------------------------------ неблокирующесть


async def test_transcription_does_not_block_event_loop() -> None:
    """Пока Whisper считает, event loop продолжает обрабатывать события."""
    bus = EventBus()
    module, _, transcriber = make_ptt(bus, delay_s=0.4)

    heartbeats = 0

    async def heartbeat() -> None:
        nonlocal heartbeats
        while True:
            await asyncio.sleep(0.01)
            heartbeats += 1

    async with bus:
        await start_without_hook(module)
        pulse = asyncio.create_task(heartbeat())
        try:
            module._on_key_down()
            module._on_key_up()
            await asyncio.sleep(0.5)
        finally:
            pulse.cancel()
            await module.stop()

    assert transcriber.calls == 1
    # 0.4с блокирующей работы при такте 10мс — без offload было бы почти ноль.
    assert heartbeats > 20, f"event loop простаивал: тактов {heartbeats}"
    assert all(name.startswith("stt") for name in transcriber.thread_names), (
        f"инференс выполнялся не в потоке STT: {transcriber.thread_names}"
    )


async def test_chat_keeps_flowing_during_transcription() -> None:
    """Чат обрабатывается, пока идёт расшифровка реплики хоста."""
    bus = EventBus()
    collector = Collector(bus)
    module, _, _ = make_ptt(bus, delay_s=0.3)

    async with bus:
        await start_without_hook(module)
        try:
            module._on_key_down()
            module._on_key_up()
            await asyncio.sleep(0.05)
            for _ in range(5):
                await bus.emit(EventType.TWITCH_CHAT, source="test")
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.4)
            await asyncio.wait_for(bus.wait_idle(), timeout=3)
        finally:
            await module.stop()

    assert len(collector.of_type(EventType.TWITCH_CHAT)) == 5


# ------------------------------------------------------------------- pre-roll


async def test_preroll_included_in_capture() -> None:
    """Pre-roll подклеивает звук, записанный до нажатия клавиши."""
    settings = PTTSettings(sample_rate=16_000, blocksize=512, preroll_ms=320)
    recorder = AudioRecorder(settings)
    block = np.ones((settings.blocksize, 1), dtype="float32")

    # Десять блоков «до нажатия» уходят в кольцевой буфер.
    for _ in range(10):
        recorder._on_audio(block, settings.blocksize, None, None)

    recorder.start_capture()
    recorder._on_audio(block, settings.blocksize, None, None)
    result = recorder.stop_capture()

    assert result is not None
    # 320 мс pre-roll = 10 блоков по 32 мс, плюс один блок после нажатия.
    expected_blocks = 10 + 1
    assert result.audio.shape[0] == expected_blocks * settings.blocksize
    assert result.preroll_s == pytest.approx(0.32)


async def test_ring_buffer_is_bounded() -> None:
    """Кольцевой буфер не растёт бесконечно, пока клавишу не нажимали."""
    settings = PTTSettings(sample_rate=16_000, blocksize=512, preroll_ms=100, max_utterance_s=1.0)
    recorder = AudioRecorder(settings)
    block = np.zeros((settings.blocksize, 1), dtype="float32")

    for _ in range(500):
        recorder._on_audio(block, settings.blocksize, None, None)

    assert len(recorder._ring) <= recorder._ring_capacity


async def test_long_utterance_is_truncated_keeping_the_end() -> None:
    """Затянувшаяся реплика обрезается, последние слова сохраняются."""
    settings = PTTSettings(sample_rate=16_000, blocksize=512, preroll_ms=0, max_utterance_s=0.2)
    recorder = AudioRecorder(settings)

    recorder.start_capture()
    for index in range(20):
        block = np.full((settings.blocksize, 1), float(index), dtype="float32")
        recorder._on_audio(block, settings.blocksize, None, None)
    result = recorder.stop_capture()

    assert result is not None
    assert result.truncated is True
    assert result.audio.shape[0] == int(0.2 * settings.sample_rate)
    # Конец записи сохранён: последний блок — с самым большим номером.
    assert result.audio[-1] == pytest.approx(19.0)


# ---------------------------------------------------------------- настройки / потоки


async def test_stt_model_comes_from_settings() -> None:
    """Боевой модуль должен использовать STT__MODEL, а не хардкод."""
    bus = EventBus()
    stt = STTSettings(model="Systran/faster-whisper-tiny", warmup=False)
    module = PushToTalkModule(bus, PTTSettings(), stt)
    assert module.stt_settings.model == "Systran/faster-whisper-tiny"


async def test_press_from_keyboard_thread_preempts() -> None:
    """Реальный хук keyboard живёт в другом потоке — preempt должен дойти."""
    bus = EventBus()
    finished = asyncio.Event()
    started = asyncio.Event()

    async def long_generation(_event: AnyEvent) -> None:
        started.set()
        await asyncio.sleep(5)
        finished.set()

    bus.subscribe(long_generation, EventType.LLM_REQUEST, name="llm", preemptible=True)
    module, _, _ = make_ptt(bus)

    async with bus:
        await start_without_hook(module)
        try:
            await bus.emit(EventType.LLM_REQUEST, source="test")
            await asyncio.wait_for(started.wait(), timeout=2)
            thread = threading.Thread(target=module._on_key_down, name="keyboard-sim")
            thread.start()
            thread.join(timeout=2)
            await asyncio.wait_for(bus.wait_idle(), timeout=2)
        finally:
            await module.stop()

    assert not finished.is_set()
    assert bus.stats.cancelled_handlers >= 1
    assert bus.stats.preemptions == 1


async def test_silent_tail_does_not_drop_spoken_words() -> None:
    """Тихий хвост реплики не должен выкидывать уже сказанный текст."""
    from types import SimpleNamespace

    transcriber = WhisperTranscriber(STTSettings(warmup=False, no_speech_threshold=0.6))
    transcriber._model = SimpleNamespace(
        transcribe=lambda *_args, **_kwargs: (
            [
                SimpleNamespace(text="привет чат", no_speech_prob=0.1, avg_logprob=-0.2),
                SimpleNamespace(text="Продолжение следует", no_speech_prob=0.95, avg_logprob=-1.0),
            ],
            SimpleNamespace(language="ru"),
        )
    )
    result = transcriber._transcribe_sync([0.0] * 16_000, 16_000)
    assert result.text == "привет чат"


async def test_all_silent_segments_yield_empty_transcript() -> None:
    """Полная тишина не публикуется как речь."""
    from types import SimpleNamespace

    transcriber = WhisperTranscriber(STTSettings(warmup=False, no_speech_threshold=0.6))
    transcriber._model = SimpleNamespace(
        transcribe=lambda *_args, **_kwargs: (
            [SimpleNamespace(text="ммм", no_speech_prob=0.9, avg_logprob=-1.2)],
            SimpleNamespace(language="ru"),
        )
    )
    result = transcriber._transcribe_sync([0.0] * 16_000, 16_000)
    assert result.is_empty
