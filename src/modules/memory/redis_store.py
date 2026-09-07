"""Краткосрочная память на Redis: скользящее окно последних реплик.

Хранится как список ``{namespace}:dialogue:{session}`` — вставка слева,
обрезка до ``short_term_window``. Промпт-билдер читает это окно на каждом запросе,
поэтому операции должны быть дешёвыми и никогда не блокировать event loop.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

from src.core.config import MemorySettings
from src.core.logger import get_logger

logger = get_logger(__name__)

Role = Literal["host", "viewer", "streamer"]


@dataclass(slots=True)
class Turn:
    """Одна реплика диалога."""

    role: Role
    text: str
    author: str = ""
    created_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.time()


class ShortTermMemory:
    """Асинхронная обёртка над Redis для горячего контекста."""

    def __init__(self, settings: MemorySettings) -> None:
        self.settings = settings
        self._redis: Any | None = None

    async def connect(self) -> None:
        """Открывает пул соединений и проверяет доступность Redis."""
        from redis.asyncio import Redis

        self._redis = Redis.from_url(self.settings.redis_url, decode_responses=True)
        await self._redis.ping()
        logger.info("Redis подключён: {}", self.settings.redis_url)

    async def close(self) -> None:
        """Закрывает пул соединений."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    def _key(self, session: str) -> str:
        return f"{self.settings.redis_namespace}:dialogue:{session}"

    async def append(self, session: str, turn: Turn) -> None:
        """Добавляет реплику и обрезает окно до заданной длины."""
        redis = self._require()
        key = self._key(session)
        pipe = redis.pipeline()
        pipe.lpush(key, json.dumps(asdict(turn), ensure_ascii=False))
        pipe.ltrim(key, 0, self.settings.short_term_window - 1)
        pipe.expire(key, self.settings.short_term_ttl_s)
        await pipe.execute()

    async def recent(self, session: str, limit: int | None = None) -> list[Turn]:
        """Возвращает последние реплики в хронологическом порядке."""
        redis = self._require()
        count = limit or self.settings.short_term_window
        raw = await redis.lrange(self._key(session), 0, count - 1)
        return [Turn(**json.loads(item)) for item in reversed(raw)]

    async def clear(self, session: str) -> None:
        """Стирает контекст сессии."""
        await self._require().delete(self._key(session))

    def _require(self) -> Any:
        if self._redis is None:
            raise RuntimeError("ShortTermMemory не подключена: вызовите connect()")
        return self._redis
