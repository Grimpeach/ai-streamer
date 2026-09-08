"""Поиск CUDA-библиотек на Windows без полного CUDA Toolkit.

CTranslate2 (faster-whisper) грузит ``cublas64_12.dll`` через системный
``LoadLibrary``. Драйвер NVIDIA этого файла не содержит, а toolkit часто не
установлен. Зато тот же DLL уже лежит в ``torch/lib`` — его просто нет в
каталогах поиска процесса.

``os.add_dll_directory`` (Python 3.8+) регистрирует путь для последующих
загрузок. Этого мало для cuDNN: пакет ``ctranslate2`` кладёт рядом со своим
``.pyd`` собственный ``cudnn64_9.dll`` без символа ``cudnnGetLibConfig``.
Windows не подгружает второй DLL с тем же именем, и LSTM Kokoro (PyTorch)
падает с Error code 127. Поэтому DLL из ``torch/lib`` ещё и
``LoadLibrary``-ятся явно, до импорта CTranslate2.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from src.core.logger import get_logger

logger = get_logger(__name__)

_CUDA_DLL_MARKERS = ("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll")
#: Порядок важен: зонтик ``cudnn64_9.dll`` тянет graph/ops/cnn.
_PRELOAD_FIRST = (
    "cudart64_12.dll",
    "cublasLt64_12.dll",
    "cublas64_12.dll",
    "nvrtc64_120_0.dll",
    "nvrtc-builtins64_124.dll",
    "nvJitLink_120_0.dll",
    "cufft64_11.dll",
    "cusparse64_12.dll",
    "cudnn_graph64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn64_9.dll",
)
_applied: list[str] = []
_preloaded: list[str] = []


def cuda_dll_directories() -> list[Path]:
    """Каталоги, где лежат CUDA 12 DLL: ``torch/lib`` и пакеты ``nvidia-*``."""
    roots: list[Path] = []
    for raw in (sys.prefix, *sys.path):
        if raw:
            roots.append(Path(raw))

    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        torch_lib = root / "torch" / "lib"
        if _looks_like_cuda_dir(torch_lib) and torch_lib.resolve() not in seen:
            found.append(torch_lib)
            seen.add(torch_lib.resolve())

        nvidia = root / "nvidia"
        if not nvidia.is_dir():
            continue
        for package in nvidia.iterdir():
            for child in ("bin", "lib", Path("lib") / "x64"):
                candidate = package / child
                if candidate.is_dir() and candidate.resolve() not in seen:
                    found.append(candidate)
                    seen.add(candidate.resolve())
    return found


def _looks_like_cuda_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / name).is_file() for name in _CUDA_DLL_MARKERS)


def ensure_windows_cuda_dlls() -> list[str]:
    """Добавляет CUDA DLL в поиск процесса и предзагружает cuDNN из torch.

    На не-Windows ничего не делает. Вызывать до импорта CTranslate2 / Kokoro.
    """
    global _applied
    if os.name != "nt":
        return []
    if _applied:
        return list(_applied)

    directories = cuda_dll_directories()
    added: list[str] = []
    for directory in directories:
        path = str(directory)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(path)
        added.append(path)

    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
        _preload_torch_cuda_dlls(directories)
        logger.info("CUDA DLL зарегистрированы: {}", added)
    else:
        logger.warning(
            "cublas64_12.dll не найден ни в torch/lib, ни в nvidia-*. "
            "Установите пакет torch с CUDA 12 или CUDA Toolkit 12."
        )
    _applied = added
    return list(added)


def _preload_torch_cuda_dlls(directories: list[Path]) -> None:
    """Грузит DLL из torch/lib в процесс, пока CTranslate2 не подсунул свой cuDNN."""
    global _preloaded
    loaded: list[str] = []
    seen: set[str] = set()
    for directory in directories:
        if directory.name == "lib" and directory.parent.name == "torch":
            for name in _PRELOAD_FIRST:
                path = directory / name
                if _try_win_dll(path, seen, loaded):
                    continue
            for path in sorted(directory.glob("cudnn*.dll")):
                _try_win_dll(path, seen, loaded)
    _preloaded = loaded
    if loaded:
        logger.debug("CUDA DLL предзагружены: {}", [Path(item).name for item in loaded])


def _try_win_dll(path: Path, seen: set[str], loaded: list[str]) -> bool:
    """Пытается LoadLibrary по полному пути. ``True``, если файла нет (пропуск)."""
    if not path.is_file():
        return False
    key = str(path.resolve()).lower()
    if key in seen:
        return True
    seen.add(key)
    try:
        ctypes.WinDLL(str(path), winmode=0)
        loaded.append(str(path))
    except OSError:
        logger.debug("Не удалось предзагрузить {}", path)
    return True


def is_missing_cuda_runtime(exc: BaseException) -> bool:
    """Ошибка загрузки cuBLAS/cuDNN, а не логическая ошибка модели."""
    text = str(exc).lower()
    if "cudnngetlibconfig" in text.replace(" ", ""):
        return True
    if "cudnn" in text and "127" in text:
        return True
    libraries = ("cublas", "cudnn", "cudart", "cublaslt")
    if not any(name in text for name in libraries):
        return False
    return "not found" in text or "cannot be loaded" in text or "error while loading" in text
