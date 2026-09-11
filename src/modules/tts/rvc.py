"""Опциональный RVC: конверсия чанков Kokoro в голос персонажа.

``rvc_python.infer.RVCInference`` синхронный и держит модель на GPU, поэтому
загрузка и инференс идут в ``ThreadPoolExecutor(max_workers=1)``. Event loop
только ждёт результат и проверяет abort/epoch: прерванный чанк в плеер не идёт.
"""

from __future__ import annotations

import asyncio
import math
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from src.core.config import RVCSettings
from src.core.cuda import ensure_windows_cuda_dlls
from src.core.hf_hub import ensure_windows_hf_cache
from src.core.logger import get_logger

logger = get_logger(__name__)


class RVCConverter:
    """Обёртка над ``RVCInference``: один инстанс на жизнь процесса."""

    def __init__(self, settings: RVCSettings) -> None:
        self.settings = settings
        self._infer: Any | None = None
        self._executor: ThreadPoolExecutor | None = None

    @property
    def loaded(self) -> bool:
        """Загружена ли модель."""
        return self._infer is not None

    async def load(self) -> None:
        """Грузит .pth/.index в память. Вызывать один раз при старте TTS."""
        if self._infer is not None:
            return
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-rvc")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_sync)

    async def unload(self) -> None:
        """Снимает модель и останавливает рабочий поток."""
        executor = self._executor
        infer = self._infer
        self._infer = None
        self._executor = None
        if infer is not None and executor is not None:
            unload_model = getattr(infer, "unload_model", None)
            if unload_model is not None:
                await asyncio.get_running_loop().run_in_executor(executor, unload_model)
        if executor is not None:
            await asyncio.to_thread(executor.shutdown, True)

    async def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        abort: Any,
        epoch: int,
        current_epoch: Any,
    ) -> np.ndarray | None:
        """Конвертирует чанк вне event loop. ``None`` — отмена, в плеер не играть."""
        if _is_set(abort) or epoch != current_epoch():
            return None
        if self._infer is None or self._executor is None:
            raise RuntimeError("RVC не загружен: вызовите load()")
        loop = asyncio.get_running_loop()
        converted = await loop.run_in_executor(
            self._executor,
            self.convert_sync,
            audio,
            sample_rate,
        )
        if _is_set(abort) or epoch != current_epoch():
            return None
        return converted

    def convert_sync(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Синхронный инференс. Только из рабочего потока RVC."""
        import soundfile as sf

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return samples
        if self._infer is None:
            raise RuntimeError("RVC не загружен")

        with tempfile.TemporaryDirectory(prefix="rvc-") as tmp:
            src = Path(tmp) / "in.wav"
            dst = Path(tmp) / "out.wav"
            sf.write(src, samples, sample_rate)
            self._infer.infer_file(str(src), str(dst))
            converted, out_sr = sf.read(str(dst), dtype="float32", always_2d=False)

        mono = np.asarray(converted, dtype=np.float32).reshape(-1)
        if out_sr and out_sr != sample_rate and mono.size:
            mono = _resample(mono, int(out_sr), sample_rate)
        return mono

    def _load_sync(self) -> None:
        ensure_windows_hf_cache()
        ensure_windows_cuda_dlls()
        from rvc_python.infer import RVCInference

        model_path = self.settings.model_path
        index_path = self.settings.index_path
        if not model_path.is_file():
            raise FileNotFoundError(f"RVC-модель не найдена: {model_path}")
        index = str(index_path) if index_path and Path(index_path).is_file() else ""
        logger.info(
            "Загружаю RVC '{}' (device={}, pitch={}, f0={})",
            model_path.name,
            self.settings.device,
            self.settings.pitch,
            self.settings.f0_method,
        )
        infer = RVCInference(
            device=self.settings.device,
            model_path=str(model_path),
            index_path=index,
            version="v2",
        )
        infer.set_params(
            f0method=self.settings.f0_method,
            f0up_key=int(self.settings.pitch),
            resample_sr=0,
            index_rate=0.75 if index else 0.0,
        )
        self._infer = infer
        logger.info("RVC готов")


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Меняет частоту дискретизации без лишней зависимости от librosa."""
    from scipy.signal import resample_poly

    gcd = math.gcd(src_sr, dst_sr)
    return np.asarray(resample_poly(audio, dst_sr // gcd, src_sr // gcd), dtype=np.float32)


def _is_set(abort: Any) -> bool:
    is_set = getattr(abort, "is_set", None)
    if is_set is None:
        return bool(abort)
    return bool(is_set())
