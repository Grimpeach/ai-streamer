"""Распознавание речи хоста через faster-whisper.

``WhisperModel`` синхронный и держит модель на GPU, поэтому:

* загрузка весов и инференс выполняются в отдельном потоке — event loop не
  замирает на секунды, пока идёт расшифровка;
* поток ровно один (``ThreadPoolExecutor(max_workers=1)``): модель не
  потокобезопасна, а параллельные запросы всё равно упёрлись бы в одну GPU;
* модель прогревается на тишине при старте — первый инференс на CUDA
  компилирует кернелы и иначе украл бы 2-3 секунды у живой реплики хоста.
"""

from __future__ import annotations

import asyncio
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from src.core.config import STTSettings
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class Transcript:
    """Результат расшифровки одной реплики."""

    text: str
    language: str = ""
    #: Длительность распознанного аудио, секунды.
    audio_s: float = 0.0
    #: Время инференса, секунды.
    latency_s: float = 0.0
    #: Средняя логарифмическая вероятность токенов (грубый прокси уверенности).
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0

    @property
    def confidence(self) -> float:
        """Оценка уверенности в диапазоне (0, 1]."""
        return round(math.exp(self.avg_logprob), 3) if self.avg_logprob else 0.0

    @property
    def is_empty(self) -> bool:
        """Речь не распознана."""
        return not self.text.strip()


class WhisperTranscriber:
    """Асинхронная обёртка над ``faster_whisper.WhisperModel``."""

    def __init__(self, settings: STTSettings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._executor: ThreadPoolExecutor | None = None

    @property
    def loaded(self) -> bool:
        """Загружена ли модель."""
        return self._model is not None

    # ------------------------------------------------------------------ lifecycle

    async def load(self) -> None:
        """Загружает модель в VRAM и прогревает её."""
        if self._model is not None:
            return
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        logger.info(
            "Загружаю STT '{}' на {} ({}), первый запуск скачивает веса",
            self.settings.model,
            self.settings.device,
            self.settings.compute_type,
        )
        started = time.monotonic()
        await self._run(self._load_sync)
        logger.info("STT готов за {:.1f}с (~{} ГБ VRAM)", time.monotonic() - started, self.settings.vram_gb)

        if self.settings.warmup:
            await self._warmup()

    async def unload(self) -> None:
        """Освобождает VRAM и останавливает рабочий поток."""
        self._model = None
        executor = self._executor
        self._executor = None
        if executor is not None:
            await asyncio.to_thread(executor.shutdown, True)

    async def _warmup(self) -> None:
        """Гоняет модель по секунде тишины, чтобы прогреть CUDA-кернелы."""
        import numpy as np

        started = time.monotonic()
        silence = np.zeros(16_000, dtype="float32")
        await self.transcribe(silence, 16_000)
        logger.info("STT прогрет за {:.1f}с", time.monotonic() - started)

    # ------------------------------------------------------------------ inference

    async def transcribe(self, audio: Any, sample_rate: int) -> Transcript:
        """Расшифровывает моно-аудио float32, не блокируя event loop."""
        if self._model is None:
            raise RuntimeError("Модель STT не загружена: вызовите load()")
        return await self._run(self._transcribe_sync, audio, sample_rate)

    async def _run(self, func: Any, *args: Any) -> Any:
        """Выполняет блокирующий вызов в выделенном потоке STT."""
        loop = asyncio.get_running_loop()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        return await loop.run_in_executor(self._executor, func, *args)

    # --------------------------------------------------- синхронная часть (в потоке)

    def _load_sync(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.settings.model,
            device=self.settings.device,
            compute_type=self.settings.compute_type,
            download_root=str(self.settings.download_root) if self.settings.download_root else None,
        )

    def _transcribe_sync(self, audio: Any, sample_rate: int) -> Transcript:
        """Синхронный инференс. Выполняется только в потоке STT."""
        if sample_rate != 16_000:
            # Whisper всегда работает на 16 кГц; пересэмплирование внутри
            # faster-whisper обошлось бы лишней задержкой на каждой реплике.
            logger.warning("STT ожидает 16 кГц, получено {} Гц", sample_rate)

        started = time.monotonic()
        segments, info = self._model.transcribe(  # type: ignore[union-attr]
            audio,
            language=self.settings.language,
            beam_size=self.settings.beam_size,
            vad_filter=self.settings.vad_filter,
            condition_on_previous_text=self.settings.condition_on_previous_text,
            no_speech_threshold=self.settings.no_speech_threshold,
        )

        # segments — генератор: работа выполняется здесь, внутри рабочего потока.
        # Тихий хвост реплики (высокая no_speech_prob) не должен выкидывать
        # уже сказанные слова: отбрасываем только сегменты без речи.
        parts: list[str] = []
        logprobs: list[float] = []
        speech_probs: list[float] = []
        for segment in segments:
            no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
            if no_speech_prob > self.settings.no_speech_threshold:
                logger.debug(
                    "Сегмент STT отброшен как тишина (no_speech={:.2f}): {!r}",
                    no_speech_prob,
                    getattr(segment, "text", ""),
                )
                continue
            piece = (segment.text or "").strip()
            if not piece:
                continue
            parts.append(piece)
            logprobs.append(float(getattr(segment, "avg_logprob", 0.0) or 0.0))
            speech_probs.append(no_speech_prob)

        text = " ".join(parts)
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0
        no_speech = min(speech_probs) if speech_probs else 1.0

        return Transcript(
            text=text,
            language=getattr(info, "language", "") or self.settings.language,
            audio_s=len(audio) / sample_rate,
            latency_s=time.monotonic() - started,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech,
        )
