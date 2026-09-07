"""Настройка логирования на loguru с brace-форматированием.

Модули получают логгер через :func:`get_logger`, чтобы в записи попадало имя
компонента, а вся конфигурация оставалась в одном месте.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

from loguru import logger as _root_logger

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{extra[component]: <22}</cyan> | <level>{message}</level>"
)
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[component]} | {message}"

_configured = False


def setup_logging(
    level: str = "INFO",
    *,
    log_file: Path | str | None = None,
    diagnose: bool = False,
) -> None:
    """Конфигурирует sink-и loguru. Повторные вызовы переинициализируют вывод."""
    global _configured

    # Консоль Windows по умолчанию в cp1251 — без этого русский лог превращается в кашу.
    if hasattr(sys.stderr, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

    _root_logger.remove()
    _root_logger.configure(extra={"component": "app"})
    _root_logger.add(
        sys.stderr,
        level=level.upper(),
        format=_CONSOLE_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=diagnose,
        enqueue=True,
    )
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        _root_logger.add(
            path,
            level=level.upper(),
            format=_FILE_FORMAT,
            rotation="20 MB",
            retention="7 days",
            encoding="utf-8",
            enqueue=True,
        )
    _configured = True


def get_logger(name: str) -> Any:
    """Возвращает логгер, привязанный к компоненту (обычно ``__name__``)."""
    if not _configured:
        setup_logging()
    component = name.removeprefix("src.").removeprefix("modules.")
    return _root_logger.bind(component=component)
