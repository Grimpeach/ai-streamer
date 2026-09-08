"""Воспроизведение PCM через sounddevice с мгновенным abort по PTT.

``sd.play()`` нельзя корректно прервать из другого потока быстрее, чем
закончится буфер. ``OutputStream.abort()`` рвёт вывод сразу — это и есть
абсолютный приоритет хоста.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any

import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)


class SoundPlayer:
    """Блокирующий плеер для рабочего потока. Event loop его не вызывает напрямую."""

    def __init__(self, *, sample_rate: int = 24_000, device: int | str | None = None) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._lock = threading.Lock()
        self._stream: Any | None = None

    def stop(self) -> None:
        """Мгновенно глушит текущий поток. Безопасно вызывать из event loop."""
        with self._lock:
            stream = self._stream
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.abort()
        with contextlib.suppress(Exception):
            import sounddevice as sd

            sd.stop()

    def play(self, audio: np.ndarray, abort: threading.Event, *, sample_rate: int | None = None) -> None:
        """Играет моно float32 до конца либо до ``abort``. Вызывать из пула потоков."""
        import sounddevice as sd

        if abort.is_set():
            return
        samples = np.asarray(audio, dtype=np.float32).reshape(-1, 1)
        if samples.size == 0:
            return
        rate = sample_rate or self.sample_rate
        index = 0
        done = threading.Event()

        def callback(outdata: np.ndarray, frames: int, _time: object, _status: object) -> None:
            nonlocal index
            if abort.is_set():
                outdata[:] = 0
                raise sd.CallbackAbort
            remaining = samples.shape[0] - index
            if remaining <= 0:
                outdata[:] = 0
                raise sd.CallbackStop
            take = min(frames, remaining)
            outdata[:take] = samples[index : index + take]
            if take < frames:
                outdata[take:] = 0
                index += take
                raise sd.CallbackStop
            index += take

        stream = sd.OutputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=callback,
            finished_callback=done.set,
        )
        with self._lock:
            self._stream = stream
        try:
            with stream:
                while stream.active and not abort.is_set():
                    if done.wait(timeout=0.05):
                        break
        except sd.CallbackAbort:
            pass
        finally:
            with contextlib.suppress(Exception):
                if stream.active:
                    stream.abort()
            with self._lock:
                if self._stream is stream:
                    self._stream = None
