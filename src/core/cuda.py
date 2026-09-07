"""Поиск CUDA-библиотек на Windows без полного CUDA Toolkit.

CTranslate2 (faster-whisper) грузит ``cublas64_12.dll`` через системный
``LoadLibrary``. Драйвер NVIDIA этого файла не содержит, а toolkit часто не
установлен. Зато тот же DLL уже лежит в ``torch/lib`` — его просто нет в
каталогах поиска процесса.

``os.add_dll_directory`` (Python 3.8+) регистрирует путь для последующих
загрузок. Делать это нужно до первого ``model.encode()``: загрузка весов
Whisper на CUDA проходит без cuBLAS, а инференс — нет. Поэтому прогрев на
тишине (VAD отбрасывает кадр) маскирует проблему до первой живой реплики.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.core.logger import get_logger

logger = get_logger(__name__)

_CUDA_DLL_MARKERS = ("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll")
_applied: list[str] = []


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
    """Добавляет CUDA DLL в поиск процесса. На не-Windows ничего не делает."""
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
        logger.info("CUDA DLL зарегистрированы: {}", added)
    else:
        logger.warning(
            "cublas64_12.dll не найден ни в torch/lib, ни в nvidia-*. "
            "Установите пакет torch с CUDA 12 или CUDA Toolkit 12."
        )
    _applied = added
    return list(added)


def is_missing_cuda_runtime(exc: BaseException) -> bool:
    """Ошибка загрузки cuBLAS/cuDNN, а не логическая ошибка модели."""
    text = str(exc).lower()
    libraries = ("cublas", "cudnn", "cudart", "cublaslt")
    if not any(name in text for name in libraries):
        return False
    return "not found" in text or "cannot be loaded" in text or "error while loading" in text
