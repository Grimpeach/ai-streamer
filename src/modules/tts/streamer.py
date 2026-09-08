"""Потоковый TTS: озвучивает фразы, пришедшие от LLM.

Порядок речи критичен, поэтому фразы не обрабатываются параллельными
обработчиками шины, а складываются во внутреннюю очередь с единственным
worker-ом. Нажатие PTT сбрасывает очередь, снимает текущий синтез и рвёт
аудиопоток — хост перебивает стримера без «догоняющего» хвоста фразы.

Инференс Kokoro и PortAudio живут в рабочих потоках: event loop только
координирует очередь и прерывание.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from src.core.config import TTSSettings
from src.core.event_bus import EventBus, Preempted, PreemptionToken
from src.core.events import AnyEvent, AudioChunk, Event, EventType, TextRequest
from src.core.tasks import cancel_task, consume_exception, join_task
from src.modules.base import Module
from src.modules.tts.engine import KokoroEngine, float_to_pcm16
from src.modules.tts.player import SoundPlayer

__all__ = ["MockTTSModule", "TTSModule"]


class TTSModule(Module):
    """Синтезирует речь, играет её в звуковую карту и стримит PCM на шину."""

    name = "tts"

    #: Больше фраз в очереди озвучивать бессмысленно — стример отстанет от чата.
    max_pending_sentences = 8

    def __init__(
        self,
        bus: EventBus,
        settings: TTSSettings,
        *,
        engine: Any | None = None,
        player: Any | None = None,
    ) -> None:
        super().__init__(bus)
        self.settings = settings
        self._engine = engine
        self._player = player
        self._pending: asyncio.Queue[AnyEvent] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._current: asyncio.Task[None] | None = None
        self._abort = threading.Event()

    async def start(self) -> None:
        """Загружает модель синтеза, поднимает worker и подписки."""
        await self.load_engine()
        self.bus.subscribe(self._on_speak, EventType.TTS_REQUEST, name="tts.enqueue", preemptible=False)
        # Сброс не должен сниматься по PTT — иначе очередь фраз «догонит» хоста.
        self.bus.subscribe(self._on_preempt, EventType.PTT_PRESSED, name="tts.reset", preemptible=False)
        self._worker = asyncio.create_task(self._run_worker(), name="tts-worker")
        self._started = True
        self.log.info("TTS готов: {} / голос '{}'", self.settings.engine, self.settings.voice)

    async def stop(self) -> None:
        """Гасит worker, освобождает VRAM синтезатора и выходной поток."""
        self.flush()
        await cancel_task(self._worker, log=self.log)
        self._worker = None
        if self._player is not None:
            self._player.stop()
        engine = self._engine
        self._engine = None
        unload = getattr(engine, "unload", None)
        if unload is not None:
            result = unload()
            if asyncio.iscoroutine(result):
                await result
        self._started = False

    async def load_engine(self) -> None:
        """Загрузка Kokoro на CUDA/CPU и открытие выходного устройства."""
        if self.settings.engine == "mock":
            self._engine = self._engine or "mock"
            return
        if self._engine is None:
            engine = KokoroEngine(self.settings)
            await engine.load()
            self._engine = engine
        elif hasattr(self._engine, "load") and not getattr(self._engine, "loaded", True):
            await self._engine.load()
        if self._player is None:
            self._player = SoundPlayer(
                sample_rate=self.settings.sample_rate,
                device=self.settings.output_device,
            )

    # ------------------------------------------------------------------ handlers

    async def _on_speak(self, event: AnyEvent) -> None:
        """Ставит фразу в очередь озвучивания (не ждёт синтеза)."""
        if not isinstance(event.payload, TextRequest):
            self.log.warning("TTS_REQUEST без TextRequest: {}", event.describe())
            return
        if self._pending.qsize() >= self.max_pending_sentences:
            dropped = self._pending.get_nowait()
            self.log.warning("Очередь TTS переполнена, пропускаю {}", dropped.describe())
        self._pending.put_nowait(event)

    async def _on_preempt(self, _event: AnyEvent) -> None:
        """Мгновенно замолкает по нажатию PTT."""
        dropped = self.flush()
        self.log.debug("TTS сброшен по PTT (снято фраз: {})", dropped)

    @property
    def pending_sentences(self) -> int:
        """Сколько фраз ждёт озвучивания."""
        return self._pending.qsize()

    def flush(self) -> int:
        """Снимает текущий синтез/воспроизведение и очищает очередь."""
        self._abort.set()
        if self._player is not None:
            self._player.stop()
        count = 0
        if self._current is not None and not self._current.done():
            self._current.cancel()
            count += 1
        while not self._pending.empty():
            self._pending.get_nowait()
            count += 1
        return count

    # -------------------------------------------------------------------- worker

    async def _run_worker(self) -> None:
        """Последовательно озвучивает фразы из очереди."""
        while True:
            event = await self._pending.get()
            if event.is_stale:
                self.log.debug("Фраза просрочена, не озвучиваю: {}", event.describe())
                continue

            speak = asyncio.create_task(self._speak_one(event), name=f"tts-speak:{event.id}")
            self._current = speak
            try:
                # join_task не поднимает отмену вложенной задачи: прерывание
                # фразы по PTT — это норма, а вот отмену самого worker-а
                # (остановка модуля) нужно пропустить наружу.
                await join_task(speak)
            except asyncio.CancelledError:
                speak.cancel()
                raise
            finally:
                self._current = None
            consume_exception(speak, log=self.log)

    async def _speak_one(self, event: AnyEvent) -> None:
        """Синтезирует, играет и публикует аудио одной фразы."""
        request = event.payload
        assert isinstance(request, TextRequest)
        self._abort.clear()

        await self.bus.publish(
            Event(
                type=EventType.TTS_STARTED,
                source=self.name,
                correlation_id=event.correlation_id,
                payload=TextRequest(text=request.text, trigger=request.trigger),
            )
        )
        index = 0
        try:
            with self.bus.preemption_scope(f"tts:{event.id}") as token:
                async for pcm in self.synthesize(request.text, token):
                    token.raise_if_cancelled()
                    if self._abort.is_set():
                        break
                    await self.bus.publish(
                        Event(
                            type=EventType.TTS_CHUNK,
                            source=self.name,
                            correlation_id=event.correlation_id,
                            payload=AudioChunk(
                                pcm=pcm,
                                sample_rate=self.settings.sample_rate,
                                index=index,
                            ),
                        )
                    )
                    index += 1
        except (asyncio.CancelledError, Preempted):
            self.log.debug("Озвучка прервана: {}", event.describe())
            raise
        finally:
            await self.bus.publish(
                Event(
                    type=EventType.TTS_FINISHED,
                    source=self.name,
                    correlation_id=event.correlation_id,
                )
            )

    async def synthesize(self, text: str, token: PreemptionToken) -> AsyncIterator[bytes]:
        """Синтез фразы: чанки PCM 16-bit + воспроизведение каждого чанка."""
        engine = self._engine
        if engine is None or not hasattr(engine, "stream"):
            raise RuntimeError("Движок TTS не загружен")

        async for samples in engine.stream(text, self._abort):
            token.raise_if_cancelled()
            if self._abort.is_set():
                return
            audio = np.asarray(samples, dtype=np.float32)
            if self._player is not None:
                await asyncio.to_thread(self._player.play, audio, self._abort)
                if self._abort.is_set():
                    return
            yield float_to_pcm16(audio)


class MockTTSModule(TTSModule):
    """Имитация синтеза: спит вместо генерации звука, но уважает прерывание."""

    name = "tts-mock"

    def __init__(self, bus: EventBus, settings: TTSSettings, *, chunk_duration_s: float = 0.25) -> None:
        super().__init__(bus, settings)
        self.chunk_duration_s = chunk_duration_s

    async def load_engine(self) -> None:
        """Модель и звуковая карта не нужны."""
        self._engine = "mock"
        self._player = None

    async def synthesize(self, text: str, token: PreemptionToken) -> AsyncIterator[bytes]:
        """Отдаёт тишину порциями, как будто произносит текст в реальном времени."""
        samples_per_chunk = int(self.settings.sample_rate * self.chunk_duration_s)
        chunks = max(1, round(len(text) / 12))
        for _ in range(chunks):
            token.raise_if_cancelled()
            if self._abort.is_set():
                return
            await asyncio.sleep(self.chunk_duration_s)
            yield b"\x00\x00" * samples_per_chunk
