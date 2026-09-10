"""Кэш Hugging Face на Windows без symlink'ов.

Hub по умолчанию связывает ``snapshots/`` с ``blobs/`` через ``os.symlink``.
Без Developer Mode и прав администратора это падает с WinError 1314, даже
после предупреждения «symlinks not supported»: проверка и реальная ссылка
идут по разным путям. Тогда STT/TTS/эмбеддер не поднимаются.

``HF_HUB_DISABLE_SYMLINKS=1`` заставляет копировать файлы. Вызывать до
первого ``import huggingface_hub`` — константа читается при импорте пакета.
"""

from __future__ import annotations

import os
import sys


def ensure_windows_hf_cache() -> bool:
    """На Windows отключает symlink'и в кэше Hub. На других ОС ничего не делает."""
    if os.name != "nt":
        return False
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        constants.HF_HUB_DISABLE_SYMLINKS = True
        constants.HF_HUB_DISABLE_SYMLINKS_WARNING = True
    download = sys.modules.get("huggingface_hub.file_download")
    cache = getattr(download, "_are_symlinks_supported_in_dir", None) if download is not None else None
    if isinstance(cache, dict):
        cache.clear()
    return True


ensure_windows_hf_cache()
