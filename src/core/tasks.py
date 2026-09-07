"""Аккуратная отмена asyncio-задач.

Наивная конструкция::

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

выглядит безобидно, но проглатывает отмену *вызывающей* задачи, а не только
отменяемой: если во время ``await task`` отменяют нас самих, ``suppress``
поглотит и это, и вызывающий код продолжит работу как ни в чём не бывало.
Для супервизора соединения или worker-а TTS это означает вечный цикл и
зависание при остановке приложения.

``asyncio.wait`` такой подмены не делает: отмена вложенной задачи не поднимается
наружу, а отмена вызывающего кода — поднимается.
"""

from __future__ import annotations

import asyncio
from typing import Any

Task = asyncio.Task[Any]


async def cancel_task(task: Task | None, *, log: Any | None = None) -> None:
    """Отменяет задачу и дожидается её завершения.

    Не поглощает отмену вызывающего кода. Исключение упавшей задачи забирается,
    чтобы asyncio не ругался на «exception was never retrieved».
    """
    if task is None:
        return
    if not task.done():
        task.cancel()
    await asyncio.wait({task})
    consume_exception(task, log=log)


async def join_task(task: Task) -> None:
    """Ждёт завершения задачи, не подменяя отмену вызывающего кода."""
    await asyncio.wait({task})


def consume_exception(task: Task, *, log: Any | None = None) -> BaseException | None:
    """Забирает исключение завершённой задачи и при необходимости логирует его."""
    if not task.done() or task.cancelled():
        return None
    error = task.exception()
    if error is not None and log is not None:
        log.opt(exception=error).error("Задача '{}' завершилась ошибкой", task.get_name())
    return error
