"""Долгосрочная память на Qdrant: факты о зрителях и событиях стрима.

Эмбеддер держится на GPU и учитывается в бюджете VRAM
(``memory.vram_gb``). Модель грузится при ``connect()`` модуля памяти,
чтобы первый ответ в чат не ждал загрузки весов.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

from src.core.config import MemorySettings
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class MemoryHit:
    """Найденный фрагмент долгосрочной памяти."""

    text: str
    score: float
    author: str = ""
    kind: str = "fact"
    created_at: float = 0.0


class LongTermMemory:
    """Векторное хранилище фактов с семантическим поиском."""

    def __init__(self, settings: MemorySettings) -> None:
        self.settings = settings
        self._client: Any | None = None
        self._embedder: Any | None = None

    async def connect(self) -> None:
        """Подключается к Qdrant и создаёт коллекцию при необходимости."""
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = AsyncQdrantClient(url=self.settings.qdrant_url)
        collections = await self._client.get_collections()
        if self.settings.qdrant_collection not in {c.name for c in collections.collections}:
            await self._client.create_collection(
                collection_name=self.settings.qdrant_collection,
                vectors_config=VectorParams(size=self.settings.embedding_dim, distance=Distance.COSINE),
            )
            logger.info("Создана коллекция Qdrant '{}'", self.settings.qdrant_collection)
        await self._ensure_author_index()
        logger.info("Qdrant подключён: {}", self.settings.qdrant_url)

    async def close(self) -> None:
        """Закрывает клиент и выгружает эмбеддер из VRAM."""
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._embedder = None

    def _load_embedder(self) -> Any:
        """Лениво загружает модель эмбеддингов на CUDA, с запасным вариантом на CPU."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            device = "cpu"
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                pass
            self._embedder = SentenceTransformer(self.settings.embedding_model, device=device)
            logger.info("Эмбеддер загружен: {} ({})", self.settings.embedding_model, device)
        return self._embedder

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        """Считает эмбеддинги (блокирующая операция — вызывать через to_thread)."""
        model = self._load_embedder()
        prepared = [_e5_prefix(text, is_query=is_query, model=self.settings.embedding_model) for text in texts]
        vectors = model.encode(prepared, normalize_embeddings=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]

    async def remember(self, text: str, *, author: str = "", kind: str = "fact") -> str:
        """Сохраняет факт и возвращает его идентификатор."""
        from qdrant_client.models import PointStruct

        client = self._require()
        vector = (await asyncio.to_thread(self.embed, [text], is_query=False))[0]
        point_id = str(uuid.uuid4())
        await client.upsert(
            collection_name=self.settings.qdrant_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"text": text, "author": author, "kind": kind, "created_at": time.time()},
                )
            ],
        )
        return point_id

    async def recall(
        self,
        query: str,
        *,
        author: str = "",
        limit: int | None = None,
    ) -> list[MemoryHit]:
        """Ищет релевантные факты по смыслу запроса и, если есть, по нику."""
        client = self._require()
        vector = (await asyncio.to_thread(self.embed, [query], is_query=True))[0]
        top = limit or self.settings.top_k
        hits = await self._query(client, vector, top, author=author)
        if author and not hits:
            hits = await self._query(client, vector, top, author="")
        min_score = self.settings.min_score
        return [hit for hit in hits if hit.score >= min_score or min_score <= 0]

    async def _ensure_author_index(self) -> None:
        """Индекс по нику, чтобы фильтр recall не сканировал всю коллекцию."""
        from qdrant_client.models import PayloadSchemaType

        if self._client is None:
            return
        try:
            await self._client.create_payload_index(
                collection_name=self.settings.qdrant_collection,
                field_name="author",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            logger.debug("Индекс Qdrant author уже есть или не создался")

    async def _query(
        self,
        client: Any,
        vector: list[float],
        limit: int,
        *,
        author: str,
    ) -> list[MemoryHit]:
        kwargs: dict[str, Any] = {
            "collection_name": self.settings.qdrant_collection,
            "query": vector,
            "limit": limit,
            "with_payload": True,
        }
        if author:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            kwargs["query_filter"] = Filter(
                must=[FieldCondition(key="author", match=MatchValue(value=author))]
            )
        result = await client.query_points(**kwargs)
        hits: list[MemoryHit] = []
        for point in result.points:
            payload = point.payload or {}
            hits.append(
                MemoryHit(
                    text=str(payload.get("text", "")),
                    score=float(point.score),
                    author=str(payload.get("author", "")),
                    kind=str(payload.get("kind", "fact")),
                    created_at=float(payload.get("created_at", 0.0)),
                )
            )
        return hits

    def _require(self) -> Any:
        if self._client is None:
            raise RuntimeError("LongTermMemory не подключена: вызовите connect()")
        return self._client


def _e5_prefix(text: str, *, is_query: bool, model: str) -> str:
    """multilingual-e5 ожидает префиксы query:/passage:."""
    if "e5" not in model.lower():
        return text
    tag = "query: " if is_query else "passage: "
    return tag + text
