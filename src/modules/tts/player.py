"""Воспроизведение PCM через постоянный sounddevice.OutputStream.

``sd.play()`` и ``stream.abort()`` с чужого потока на Windows рвут не только
выход, но и входной поток микрофона: PortAudio «отходит» около десяти секунд,
и следующая запись PTT выглядит зависшей. Поэтому поток вывода открывается
один раз и живёт до ``close()``. PTT только обнуляет буфер — колбэк пишет тишину,
устройство не закрывается.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any

import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)


class SoundPlayer:
    """Блокирующий плеер для рабочего потока. Event loop вызывает только ``stop()``."""

    def __init__(self, *, sample_rate: int = 24_000, device: int | str | None = None) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._lock = threading.Lock()
        self._stream: Any | None = None
        self._buffer = np.zeros((0, 1), dtype=np.float32)
        self._index = 0
        self._epoch = 0
        self._cut = threading.Event()
        self._chunk_done = threading.Event()
        self._chunk_done.set()

    def start(self) -> None:
        """Открывает выходной поток. Идемпотентно."""
        self._ensure_started()

    def stop(self) -> None:
        """Мгновенно глушит текущий звук, не трогая PortAudio-поток и микрофон."""
        self._epoch += 1
        self._cut.set()
        with self._lock:
            self._buffer = np.zeros((0, 1), dtype=np.float32)
            self._index = 0
        self._chunk_done.set()

    def close(self) -> None:
        """Останавливает устройство. Только при выгрузке модуля, не по PTT."""
        self.stop()
        with self._lock:
            stream = self._stream
            self._stream = None
        if stream is None:
            return
        with contextlib.suppress(Exception):
            stream.stop()
        with contextlib.suppress(Exception):
            stream.close()

    def play(self, audio: np.ndarray, abort: threading.Event, *, sample_rate: int | None = None) -> None:
        """Играет моно float32 до конца либо до ``abort``. Вызывать из пула потоков."""
        _ = sample_rate
        if abort.is_set():
            return
        samples = np.asarray(audio, dtype=np.float32).reshape(-1, 1)
        if samples.size == 0:
            return
        epoch = self._epoch
        self._cut.clear()
        if abort.is_set() or self._epoch != epoch:
            return
        self._ensure_started()
        with self._lock:
            self._buffer = samples
            self._index = 0
        self._chunk_done.clear()
        if abort.is_set() or self._epoch != epoch:
            self._chunk_done.set()
            return
        while not abort.is_set() and self._epoch == epoch:
            if self._chunk_done.wait(timeout=0.05):
                break

    def _ensure_started(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=512,
            callback=self._callback,
        )
        stream.start()
        with self._lock:
            if self._stream is not None:
                with contextlib.suppress(Exception):
                    stream.stop()
                with contextlib.suppress(Exception):
                    stream.close()
                return
            self._stream = stream
        logger.debug("Выходной аудиопоток открыт ({} Гц)", self.sample_rate)

    def _callback(self, outdata: np.ndarray, frames: int, _time: object, _status: object) -> None:
        """Колбэк PortAudio: только копия PCM или тишина, без abort/stop."""
        if self._cut.is_set():
            outdata.fill(0)
            self._chunk_done.set()
            return
        with self._lock:
            remaining = self._buffer.shape[0] - self._index
            if remaining <= 0:
                outdata.fill(0)
                self._chunk_done.set()
                return
            take = min(frames, remaining)
            outdata[:take] = self._buffer[self._index : self._index + take]
            if take < frames:
                outdata[take:].fill(0)
            self._index += take
            if self._index >= self._buffer.shape[0]:
                self._chunk_done.set()
