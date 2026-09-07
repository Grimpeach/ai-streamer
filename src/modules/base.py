"""Базовый контракт модуля: единый lifecycle для всех компонентов."""

from __future__ import annotations

import abc

from src.core.event_bus import EventBus
from src.core.logger import get_logger


class Module(abc.ABC):
    """Компонент системы с жизненным циклом ``start`` / ``stop``.

    Наследники подписываются на шину в :meth:`start` и обязаны освободить
    ресурсы (модели, сокеты, аудио-стримы) в :meth:`stop`.
    """

    #: Человекочитаемое имя для логов и метрик.
    name: str = "module"

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.log = get_logger(f"{type(self).__module__}")
        self._started = False

    @property
    def started(self) -> bool:
        """Запущен ли модуль."""
        return self._started

    @abc.abstractmethod
    async def start(self) -> None:
        """Инициализирует ресурсы и подписки."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Освобождает ресурсы. Должен быть идемпотентным."""

    async def __aenter__(self) -> Module:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.stop()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, started={self._started})"
