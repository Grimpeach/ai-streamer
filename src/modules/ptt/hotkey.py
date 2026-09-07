"""Push-to-Talk хоста: глобальный бинд клавиши, запись и расшифровка.

Порядок действий при нажатии продуман под требование «мгновенно заткнуть
генерацию» (.cursorrules п.4):

1. Колбэк ``keyboard`` срабатывает в своём потоке и первым делом просит event
   loop вызвать :meth:`EventBus.preempt` — раньше любой работы с устройствами.
2. Захват звука — это установка флага в уже открытом потоке микрофона, поэтому
   hook-поток не делает ничего долгого.
3. После отпускания расшифровка уходит в поток STT, а результат публикуется как
   ``HOST_SPEECH`` с приоритетом 1.

Ни одна тяжёлая операция не выполняется в event loop: открытие микрофона,
загрузка весов Whisper и инференс уходят в отдельные потоки.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.core.config import PTTSettings, STTSettings
from src.core.event_bus import EventBus
from src.core.events import Event, EventPriority, EventType, HostSpeech, PreemptReason
from src.core.tasks import join_task
from src.modules.base import Module
from src.modules.ptt.recorder import AudioRecorder, RecordingResult
from src.modules.ptt.stt import Transcript, WhisperTranscriber


class PushToTalkModule(Module):
    """Слушает клавишу PTT, пишет микрофон и публикует реплику хоста."""

    name = "ptt"

    def __init__(
        self,
        bus: EventBus,
        settings: PTTSettings,
        stt_settings: STTSettings | None = None,
        *,
        recorder: AudioRecorder | None = None,
        transcriber: WhisperTranscriber | None = None,
    ) -> None:
        super().__init__(bus)
        self.settings = settings
        self.stt_settings = stt_settings or STTSettings()
        self._recorder = recorder if recorder is not None else AudioRecorder(settings)
        self._transcriber = (
            transcriber if transcriber is not None else WhisperTranscriber(self.stt_settings)
        )
        self._hooks: list[Any] = []
        self._keyboard: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._held = False
        #: Номер реплики: инкрементируется в hook-потоке на каждое нажатие и
        #: позволяет отбросить расшифровку, которую хост уже перебил.
        self._utterance = 0
        #: Одна модель на GPU — одна расшифровка за раз.
        self._stt_lock = asyncio.Lock()
        self._pressed_at = 0.0
        self._inflight: set[asyncio.Task[None]] = set()
        self.utterances_published = 0

    @property
    def holding(self) -> bool:
        """Зажата ли клавиша PTT прямо сейчас."""
        return self._held

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Загружает STT, открывает микрофон и ставит глобальный хук."""
        self._loop = asyncio.get_running_loop()
        await self._transcriber.load()
        try:
            await self._recorder.open()
            self._install_hook()
        except Exception:
            # Не оставляем модель в VRAM и открытый микрофон, если хук не встал.
            self._remove_hook()
            await self._recorder.close()
            await self._transcriber.unload()
            raise
        self._started = True
        self.log.info(
            "PTT активен: клавиша '{}' прерывает генерацию мгновенно, STT '{}'",
            self.settings.hotkey,
            self.stt_settings.model,
        )

    async def stop(self) -> None:
        """Снимает хук, закрывает микрофон и выгружает модель."""
        self._started = False
        self._remove_hook()
        self._held = False
        self._utterance += 1  # любые незавершённые расшифровки станут устаревшими
        self._recorder.abort()
        pending = list(self._inflight)
        for task in pending:
            # Инференс Whisper в пуле потоков отменой задачи не остановить:
            # дожидаемся конца и отбрасываем результат по номеру реплики.
            await join_task(task)
        self._inflight.clear()
        await self._recorder.close()
        await self._transcriber.unload()

    def _install_hook(self) -> None:
        """Ставит глобальный бинд на клавишу из ``PTT__HOTKEY``."""
        import keyboard

        try:
            self._hooks = [
                keyboard.on_press_key(self.settings.hotkey, self._on_key_down, suppress=False),
                keyboard.on_release_key(self.settings.hotkey, self._on_key_up, suppress=False),
            ]
        except Exception as exc:
            raise RuntimeError(
                f"Не удалось повесить глобальный хук на '{self.settings.hotkey}': {exc}. "
                "Проверьте имя клавиши (PTT__HOTKEY) и запустите терминал от администратора — "
                "библиотека keyboard требует прав на перехват ввода."
            ) from exc
        self._keyboard = keyboard

    def _remove_hook(self) -> None:
        keyboard = self._keyboard
        self._keyboard = None
        if keyboard is None:
            return
        for hook in self._hooks:
            # Снимаем именно свои хуки: unhook_all() убил бы чужие бинды.
            try:
                keyboard.unhook(hook)
            except Exception:
                self.log.debug("Хук PTT уже снят")
        self._hooks = []

    # ------------------------------------------------- колбэки hook-потока keyboard

    def _on_key_down(self, _event: Any = None) -> None:
        """Клавиша нажата. Выполняется в потоке библиотеки ``keyboard``."""
        if self._held:
            return  # автоповтор клавиши, пока её держат
        self._held = True
        self._pressed_at = time.monotonic()
        # Счётчик реплик меняется только здесь — все колбэки keyboard живут в
        # одном потоке, поэтому гонки с чтением на отпускании нет.
        self._utterance += 1
        self.preempt_now()
        self._recorder.start_capture()

    def _on_key_up(self, _event: Any = None) -> None:
        """Клавиша отпущена. Выполняется в потоке библиотеки ``keyboard``."""
        if not self._held:
            return
        self._held = False
        utterance = self._utterance
        try:
            recording = self._recorder.stop_capture()
        except Exception:
            self.log.exception("Ошибка остановки записи PTT")
            return

        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._on_release_in_loop, recording, utterance)

    def _on_release_in_loop(self, recording: RecordingResult | None, utterance: int) -> None:
        """Публикует отпускание и ставит расшифровку в event loop."""
        self.bus.publish_nowait(Event(type=EventType.PTT_RELEASED, source=self.name))
        if recording is None:
            return
        task = asyncio.create_task(self._process(recording, utterance), name=f"ptt-stt:{utterance}")
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    def preempt_now(self) -> None:
        """Просит шину немедленно прервать генерацию LLM/TTS.

        Вызывается из hook-потока, поэтому сама отмена планируется в event loop:
        ``preempt`` работает с asyncio-задачами и токенами, а их нельзя трогать
        из чужого потока.
        """
        loop = self._loop
        if loop is None:
            self.log.warning("PTT нажат, но модуль не запущен")
            return
        loop.call_soon_threadsafe(self._preempt_in_loop)

    def _preempt_in_loop(self) -> None:
        """Отмена генерации и уведомление подписчиков. Выполняется в event loop."""
        cancelled = self.bus.preempt(reason="ptt", source=self.name)
        # Событие помечено preemptive=False намеренно: отмена уже выполнена
        # строкой выше, а повторный preempt сбил бы токены обработчиков,
        # которые только что запустились на это же нажатие (например, сброс TTS).
        self.bus.publish_nowait(
            Event(
                type=EventType.PTT_PRESSED,
                source=self.name,
                priority=EventPriority.INTERRUPT,
                preemptive=False,
                payload=PreemptReason(reason="ptt", source=self.name),
            )
        )
        self.log.debug("PTT нажат: снято обработчиков {}", cancelled)

    # -------------------------------------------------------- расшифровка и публикация

    async def _process(self, recording: RecordingResult, utterance: int) -> None:
        """Расшифровывает реплику и публикует её с приоритетом 1."""
        if recording.duration_s < self.settings.min_utterance_s:
            # Прерывание уже случилось — это главное; расшифровывать нечего.
            self.log.debug("Короткое нажатие PTT ({:.2f}с), расшифровка не нужна", recording.duration_s)
            return
        if recording.truncated:
            self.log.warning("Реплика длиннее {}с — обрезана", self.settings.max_utterance_s)

        async with self._stt_lock:
            if utterance != self._utterance:
                self.log.debug("Реплика #{} устарела: хост нажал PTT снова", utterance)
                return
            try:
                transcript = await self._transcriber.transcribe(
                    recording.audio, recording.sample_rate
                )
            except Exception:
                self.log.exception("Не удалось расшифровать реплику хоста")
                return

        if utterance != self._utterance:
            self.log.debug("Расшифровка #{} устарела, не публикую", utterance)
            return
        if transcript.is_empty:
            self.log.info("Речь не распознана ({:.2f}с аудио)", recording.duration_s)
            return

        await self._publish(transcript, recording)

    async def _publish(self, transcript: Transcript, recording: RecordingResult) -> None:
        """Публикует ``HOST_SPEECH`` — Priority 1, выше донатов и чата."""
        self.utterances_published += 1
        self.log.info(
            "Хост: «{}» (аудио {:.1f}с, STT {:.0f}мс, от нажатия {:.0f}мс)",
            transcript.text,
            recording.duration_s,
            transcript.latency_s * 1000,
            (time.monotonic() - self._pressed_at) * 1000,
        )
        await self.bus.publish(
            Event(
                type=EventType.HOST_SPEECH,
                source=self.name,
                priority=EventPriority.INTERRUPT,
                payload=HostSpeech(
                    text=transcript.text,
                    duration_s=recording.duration_s,
                    confidence=transcript.confidence,
                ),
            )
        )


class MockPushToTalkModule(PushToTalkModule):
    """PTT без микрофона, хука и Whisper — для демо и отладки прерывания."""

    name = "ptt-mock"

    def __init__(self, bus: EventBus, settings: PTTSettings) -> None:
        super().__init__(
            bus,
            settings,
            recorder=_NullRecorder(settings),
            transcriber=_NullTranscriber(STTSettings()),
        )

    async def start(self) -> None:
        """Ничего не перехватывает: нажатия эмулируются вызовом :meth:`say`."""
        self._loop = asyncio.get_running_loop()
        self._started = True
        self.log.info("Mock-PTT готов (клавиша не перехватывается)")

    async def stop(self) -> None:
        """Освобождать нечего."""
        self._started = False

    async def say(self, text: str, *, duration_s: float = 1.0) -> None:
        """Полный цикл нажатия: мгновенное прерывание и реплика хоста."""
        self._utterance += 1
        self._pressed_at = time.monotonic()
        self._preempt_in_loop()
        await self.bus.publish(
            Event(
                type=EventType.HOST_SPEECH,
                source=self.name,
                priority=EventPriority.INTERRUPT,
                payload=HostSpeech(text=text, duration_s=duration_s),
            )
        )
        self.utterances_published += 1


class _NullRecorder(AudioRecorder):
    """Заглушка микрофона для демо-режима."""

    async def open(self) -> None:
        """Устройство не открывается."""

    async def close(self) -> None:
        """Закрывать нечего."""

    def start_capture(self) -> None:
        """Захват не ведётся."""

    def stop_capture(self) -> RecordingResult | None:
        """Аудио никогда не появляется."""
        return None


class _NullTranscriber(WhisperTranscriber):
    """Заглушка STT для демо-режима."""

    async def load(self) -> None:
        """Модель не загружается."""

    async def unload(self) -> None:
        """Выгружать нечего."""
