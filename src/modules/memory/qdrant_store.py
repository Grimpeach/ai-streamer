"""Долгосрочная память на Qdrant: факты о зрителях и событиях стрима.

Эмбеддер держится на GPU и учитывается в бюджете VRAM
(``memory.vram_gb``), поэтому модель загружается лениво — только при первом
обращении к семантическому поиску.
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
        logger.info("Qdrant подключён: {}", self.settings.qdrant_url)

    async def close(self) -> None:
        """Закрывает клиент и выгружает эмбеддер из VRAM."""
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._embedder = None

    def _load_embedder(self) -> Any:
        """Лениво загружает модель эмбеддингов на CUDA."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.settings.embedding_model, device="cuda")
            logger.info("Эмбеддер загружен: {}", self.settings.embedding_model)
        return self._embedder

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Считает эмбеддинги (блокирующая операция — вызывать через to_thread)."""
        model = self._load_embedder()
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]

    async def remember(self, text: str, *, author: str = "", kind: str = "fact") -> str:
        """Сохраняет факт и возвращает его идентификатор."""
        from qdrant_client.models import PointStruct

        client = self._require()
        vector = (await asyncio.to_thread(self.embed, [text]))[0]
        point_id = uuid.uuid4().hex
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

    async def recall(self, query: str, *, limit: int | None = None) -> list[MemoryHit]:
        """Ищет релевантные факты по смыслу запроса."""
        client = self._require()
        vector = (await asyncio.to_thread(self.embed, [query]))[0]
        result = await client.query_points(
            collection_name=self.settings.qdrant_collection,
            query=vector,
            limit=limit or self.settings.top_k,
            with_payload=True,
        )
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
