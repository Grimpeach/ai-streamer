"""Движок Kokoro: синтез в отдельном потоке, чтобы не блокировать event loop.

Модель ~82M и не потокобезопасна, поэтому пул ровно из одного потока — как у STT.
Официальный Kokoro-82M не имеет русской озвучки: голос ``ru_female_1`` из настроек
алиасится на ``af_heart``. Язык пайплайна берётся из первой буквы голоса
(``af_*`` → ``a``), либо из ``TTS__LANG_CODE``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from src.core.config import TTSSettings
from src.core.cuda import ensure_windows_cuda_dlls, is_missing_cuda_runtime
from src.core.logger import get_logger

logger = get_logger(__name__)

#: Плейсхолдеры из ранних спеков → реальные голоса hexgrad/Kokoro-82M.
VOICE_ALIASES: dict[str, str] = {
    "ru_female_1": "af_heart",
    "ru_female": "af_heart",
    "ru_male_1": "am_michael",
}

KOKORO_SAMPLE_RATE = 24_000


def resolve_voice(name: str) -> str:
    """Переводит удобное имя голоса в идентификатор Kokoro."""
    key = name.strip()
    return VOICE_ALIASES.get(key, key)


def resolve_lang_code(voice: str, explicit: str = "") -> str:
    """Язык KPipeline: явный код или первая буква голоса (``af_heart`` → ``a``)."""
    if explicit.strip():
        return explicit.strip().lower()
    prefix = voice[:1].lower() if voice else "a"
    return prefix if prefix.isalpha() else "a"


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """float32 [-1, 1] → little-endian PCM 16-bit."""
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    clipped = np.clip(data, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def tensor_to_float32(audio: Any) -> np.ndarray:
    """Снимает тензор Kokoro/torch в numpy float32."""
    if audio is None:
        return np.zeros(0, dtype=np.float32)
    detach = getattr(audio, "detach", None)
    if detach is not None:
        audio = detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class KokoroEngine:
    """Обёртка над ``kokoro.KPipeline`` с инференсом в ThreadPoolExecutor."""

    def __init__(self, settings: TTSSettings) -> None:
        self.settings = settings
        self.voice = resolve_voice(settings.voice)
        self.lang_code = resolve_lang_code(self.voice, settings.lang_code)
        self._pipeline: Any | None = None
        self._executor: ThreadPoolExecutor | None = None
        self.device = "cpu"

    @property
    def loaded(self) -> bool:
        """Загружен ли пайплайн."""
        return self._pipeline is not None

    async def load(self) -> None:
        """Грузит веса Kokoro в рабочем потоке и прогревает синтез."""
        if self._pipeline is not None:
            return
        ensure_windows_cuda_dlls()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-kokoro")
        logger.info("Загружаю Kokoro (голос '{}', lang={})", self.voice, self.lang_code)
        started = time.monotonic()
        await self._run(self._load_sync)
        logger.info(
            "Kokoro готов за {:.1f}с ({} / голос '{}', ~{} ГБ VRAM)",
            time.monotonic() - started,
            self.device,
            self.voice,
            self.settings.vram_gb if self.device == "cuda" else 0,
        )

    async def unload(self) -> None:
        """Снимает модель и останавливает рабочий поток."""
        self._pipeline = None
        executor = self._executor
        self._executor = None
        if executor is not None:
            await asyncio.to_thread(executor.shutdown, True)

    async def synthesize(self, text: str, abort: asyncio.Event | Any) -> list[np.ndarray]:
        """Синтезирует фразу целиком. Для стрима используйте :meth:`stream`."""
        chunks: list[np.ndarray] = []
        async for chunk in self.stream(text, abort):
            chunks.append(chunk)
        return chunks

    async def stream(self, text: str, abort: Any) -> AsyncIterator[np.ndarray]:
        """Отдаёт numpy-чанки по мере генерации Kokoro, не блокируя loop."""
        if self._pipeline is None or self._executor is None:
            raise RuntimeError("Kokoro не загружен: вызовите load()")
        cleaned = text.strip()
        if not cleaned:
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray | Exception | None] = asyncio.Queue()

        def produce() -> None:
            try:
                for result in self._pipeline(cleaned, voice=self.voice, speed=self.settings.speed):
                    if _is_set(abort):
                        break
                    audio = tensor_to_float32(_result_audio(result))
                    if audio.size:
                        loop.call_soon_threadsafe(queue.put_nowait, audio)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
                return
            loop.call_soon_threadsafe(queue.put_nowait, None)

        future = self._executor.submit(produce)
        try:
            while True:
                if _is_set(abort):
                    return
                item = await queue.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            future.cancel()

    def _load_sync(self) -> None:
        device = self._pick_device()
        if self.voice != self.settings.voice:
            logger.info("Голос '{}' → Kokoro '{}'", self.settings.voice, self.voice)
        try:
            self._pipeline = self._create_pipeline(device)
            self._warmup_pipeline()
            self.device = device
        except Exception as exc:
            if not (
                device == "cuda"
                and self.settings.fallback_to_cpu
                and (is_missing_cuda_runtime(exc) or _looks_like_cudnn_loader_error(exc))
            ):
                raise
            logger.warning("Kokoro CUDA недоступен ({}). Синтез на CPU.", exc)
            self._pipeline = self._create_pipeline("cpu")
            self._warmup_pipeline()
            self.device = "cpu"

    def _create_pipeline(self, device: str) -> Any:
        from kokoro import KPipeline

        if device == "cuda":
            import torch

            # LSTM Kokoro ходит в cuDNN. Чужой cudnn64_9.dll (CTranslate2) без
            # cudnnGetLibConfig роняет процесс — нативные ядра PyTorch безопаснее.
            torch.backends.cudnn.enabled = False
        return KPipeline(lang_code=self.lang_code, repo_id="hexgrad/Kokoro-82M", device=device)

    def _warmup_pipeline(self) -> None:
        if self._pipeline is None:
            return
        list(self._pipeline("Hi.", voice=self.voice, speed=self.settings.speed))

    def _pick_device(self) -> str:
        requested = (self.settings.device or "auto").lower()
        if requested == "cpu":
            return "cpu"
        try:
            import torch

            if requested == "cuda":
                if not torch.cuda.is_available():
                    logger.warning("TTS просил CUDA, её нет — CPU")
                    return "cpu"
                return "cuda"
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    async def _run(self, func: Any, *args: Any) -> Any:
        if self._executor is None:
            raise RuntimeError("Путь TTS не создан")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func, *args)


def _result_audio(result: Any) -> Any:
    audio = getattr(result, "audio", None)
    if audio is not None:
        return audio
    if isinstance(result, (tuple, list)) and len(result) >= 3:
        return result[2]
    return None


def _is_set(abort: Any) -> bool:
    is_set = getattr(abort, "is_set", None)
    if is_set is None:
        return bool(abort)
    return bool(is_set())


def _looks_like_cudnn_loader_error(exc: BaseException) -> bool:
    """Нативный загрузчик cuDNN иногда кладёт текст в OSError без слова 'cudnn'."""
    text = str(exc).lower()
    return "error code 127" in text or "could not load symbol" in text
