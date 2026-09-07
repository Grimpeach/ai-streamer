"""Функциональные модули: источники событий и потребители генерации.

Каждый модуль наследует :class:`src.modules.base.Module`, общается с остальной
системой только через :class:`src.core.event_bus.EventBus` и не импортирует
другие модули напрямую.
"""

from src.modules.base import Module

__all__ = ["Module"]
