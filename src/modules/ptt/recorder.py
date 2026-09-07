"""Запись микрофона хоста через sounddevice.

Поток PortAudio открывается один раз при старте модуля и живёт всё время
стрима. Причины именно такой схемы:

* открытие устройства занимает десятки миллисекунд — если делать это на нажатие
  клавиши, первый слог реплики теряется;
* постоянный поток позволяет держать **pre-roll**: кольцевой буфер последних
  ``preroll_ms`` миллисекунд, который подклеивается в начало реплики и
  компенсирует задержку реакции человека на собственное нажатие;
* нажатие и отпускание превращаются в установку флага, то есть hook-поток
  клавиатуры не делает ничего долгого.

Аудио-колбэк живёт в реальном времени: в нём нет логирования, аллокаций сверх
копии кадра и ожиданий — только запись в кольцевой буфер.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import threading
from dataclasses import dataclass
from typing import Any

from src.core.config import PTTSettings
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RecordingResult:
    """Результат одной сессии PTT."""

    #: Моно-аудио float32 в диапазоне [-1, 1] (``numpy.ndarray``).
    audio: Any
    sample_rate: int
    duration_s: float
    #: Реплика упёрлась в ``max_utterance_s`` и была обрезана.
    truncated: bool = False
    #: Сколько секунд звука добавлено из pre-roll буфера.
    preroll_s: float = 0.0
    #: Сообщения PortAudio о переполнении/недогрузе за время реплики.
    glitches: int = 0


class AudioRecorder:
    """Постоянно открытый поток микрофона с pre-roll и захватом по флагу."""

    def __init__(self, settings: PTTSettings) -> None:
        self.settings = settings
        self._stream: Any | None = None
        # Кольцевой буфер: хранит последние (max_utterance + preroll) секунд.
        self._ring: list[Any] = []
        self._ring_capacity = self._compute_capacity()
        self._ring_start = 0
        # Индекс блока, с которого началась текущая реплика; None — не пишем.
        self._capture_from: int | None = None
        self._blocks_written = 0
        self._glitches = 0
        # Лок удерживается микросекунды: только append/срез списка блоков.
        self._lock = threading.Lock()

    def _compute_capacity(self) -> int:
        window_s = self.settings.max_utterance_s + self.settings.preroll_ms / 1000
        blocks = math.ceil(window_s * self.settings.sample_rate / self.settings.blocksize)
        return max(blocks + 4, 8)

    @property
    def is_open(self) -> bool:
        """Открыт ли поток микрофона."""
        return self._stream is not None

    @property
    def capturing(self) -> bool:
        """Идёт ли запись реплики прямо сейчас."""
        return self._capture_from is not None

    # ------------------------------------------------------------------ lifecycle

    async def open(self) -> None:
        """Открывает поток микрофона. Блокирующий вызов уходит в отдельный поток."""
        if self._stream is not None:
            return
        await asyncio.to_thread(self._open_sync)

    def _open_sync(self) -> None:
        import sounddevice as sd  # локальный импорт: аудио-стек нужен не всем режимам

        try:
            stream = sd.InputStream(
                samplerate=self.settings.sample_rate,
                channels=self.settings.channels,
                dtype="float32",
                device=self.settings.input_device,
                blocksize=self.settings.blocksize,
                callback=self._on_audio,
            )
            stream.start()
        except Exception as exc:
            device = self.settings.input_device
            raise RuntimeError(
                f"Не удалось открыть микрофон (device={device!r}, "
                f"{self.settings.sample_rate} Гц): {exc}. "
                "Список устройств: python -c \"import sounddevice; print(sounddevice.query_devices())\""
            ) from exc
        self._stream = stream
        logger.info(
            "Микрофон открыт: {} Гц, блок {} сэмплов, pre-roll {} мс",
            self.settings.sample_rate,
            self.settings.blocksize,
            self.settings.preroll_ms,
        )

    async def close(self) -> None:
        """Закрывает поток микрофона."""
        stream = self._stream
        self._stream = None
        self._capture_from = None
        if stream is None:
            return
        await asyncio.to_thread(self._close_sync, stream)
        with self._lock:
            self._ring.clear()
        logger.debug("Микрофон закрыт")

    @staticmethod
    def _close_sync(stream: Any) -> None:
        # Закрытие устройства не должно ронять остановку приложения.
        with contextlib.suppress(Exception):
            stream.stop()
        with contextlib.suppress(Exception):
            stream.close()

    # -------------------------------------------------------------------- capture

    def start_capture(self) -> None:
        """Начинает захват реплики. Вызывается из hook-потока клавиатуры."""
        with self._lock:
            self._glitches = 0
            # Отступаем назад на pre-roll: звук уже лежит в кольцевом буфере.
            preroll_blocks = math.ceil(
                self.settings.preroll_ms / 1000 * self.settings.sample_rate / self.settings.blocksize
            )
            self._capture_from = max(self._ring_start, self._blocks_written - preroll_blocks)

    def stop_capture(self) -> RecordingResult | None:
        """Завершает захват и возвращает склеенное аудио."""
        import numpy as np

        with self._lock:
            capture_from = self._capture_from
            self._capture_from = None
            if capture_from is None:
                return None
            offset = max(0, capture_from - self._ring_start)
            blocks = self._ring[offset:]
            glitches = self._glitches

        if not blocks:
            logger.warning("PTT отпущен, но звук с микрофона не пришёл")
            return None

        audio = np.concatenate(blocks, axis=0).reshape(-1)
        sample_rate = self.settings.sample_rate
        max_samples = int(self.settings.max_utterance_s * sample_rate)
        truncated = audio.shape[0] > max_samples
        if truncated:
            # Оставляем конец: последние слова важнее начала затянувшейся речи.
            audio = audio[-max_samples:]

        preroll_s = min(self.settings.preroll_ms / 1000, audio.shape[0] / sample_rate)
        if glitches:
            logger.warning("PortAudio сообщил о {} сбоях за реплику", glitches)
        return RecordingResult(
            audio=audio,
            sample_rate=sample_rate,
            duration_s=audio.shape[0] / sample_rate,
            truncated=truncated,
            preroll_s=preroll_s,
            glitches=glitches,
        )

    def abort(self) -> None:
        """Отменяет захват, не возвращая аудио."""
        with self._lock:
            self._capture_from = None

    # ------------------------------------------------------ колбэк реального времени

    def _on_audio(self, indata: Any, _frames: int, _time: Any, status: Any) -> None:
        """Вызывается потоком PortAudio. Никакого IO и ожиданий здесь быть не должно."""
        block = indata.copy()
        with self._lock:
            if status:
                self._glitches += 1
            self._ring.append(block)
            self._blocks_written += 1
            overflow = len(self._ring) - self._ring_capacity
            if overflow > 0:
                del self._ring[:overflow]
                self._ring_start += overflow
