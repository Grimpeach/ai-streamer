"""Windows: кэш Hugging Face без symlink'ов (WinError 1314)."""

from __future__ import annotations

import os
import sys

from src.core.hf_hub import ensure_windows_hf_cache


def test_windows_hf_cache_disables_symlinks(monkeypatch) -> None:
    """На Windows Hub копирует веса, а не делает symlink без прав администратора."""
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising=False)

    fake_constants = type("C", (), {"HF_HUB_DISABLE_SYMLINKS": False, "HF_HUB_DISABLE_SYMLINKS_WARNING": False})()
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", fake_constants)
    fake_download = type("D", (), {"_are_symlinks_supported_in_dir": {"x": True}})()
    monkeypatch.setitem(sys.modules, "huggingface_hub.file_download", fake_download)

    assert ensure_windows_hf_cache() is True
    assert os.environ["HF_HUB_DISABLE_SYMLINKS"] == "1"
    assert fake_constants.HF_HUB_DISABLE_SYMLINKS is True
    assert fake_download._are_symlinks_supported_in_dir == {}
