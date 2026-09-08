"""Память: запись, поиск по нику, инъекция фактов в промпт LLM."""

from __future__ import annotations

from types import SimpleNamespace

from src.core.config import LLMSettings, MemorySettings
from src.core.event_bus import EventBus
from src.core.events import ChatMessage, Event, EventType
from src.modules.llm.client import LLMModule
from src.modules.memory.module import MemoryModule
from src.modules.memory.qdrant_store import MemoryHit
from src.modules.memory.redis_store import Turn


class _FakeShort:
    def __init__(self) -> None:
        self.turns: list[Turn] = []

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def append(self, _session: str, turn: Turn) -> None:
        self.turns.append(turn)

    async def recent(self, _session: str, limit: int | None = None) -> list[Turn]:
        return list(self.turns[-(limit or 20) :])


class _FakeLong:
    def __init__(self, hits: list[MemoryHit] | None = None) -> None:
        self.hits = hits or []
        self.stored: list[tuple[str, str, str]] = []

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def remember(self, text: str, *, author: str = "", kind: str = "fact") -> str:
        self.stored.append((author, text, kind))
        return "fake-id"

    async def recall(self, query: str, *, author: str = "", limit: int | None = None) -> list[MemoryHit]:
        if query == "warmup":
            return []
        found = [hit for hit in self.hits if not author or hit.author == author]
        if not found:
            found = list(self.hits)
        return found[: limit or 5]


class _FakeStream:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.closed = False
        self._index = 0

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self._index >= len(self.tokens):
            raise StopAsyncIteration
        token = self.tokens[self._index]
        self._index += 1
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=token))])

    async def close(self) -> None:
        self.closed = True


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> _FakeStream:
        self.calls.append(kwargs)
        return _FakeStream(["Ок."])


class _FakeClient:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)

    async def close(self) -> None:
        return None


async def test_memory_stores_chat_and_recalls_by_nick() -> None:
    """Реплика зрителя попадает в Redis и Qdrant; поиск по нику её находит."""
    bus = EventBus()
    short = _FakeShort()
    long = _FakeLong()
    memory = MemoryModule(bus, MemorySettings(min_remember_chars=3), session="test", short=short, long=long)

    async with bus:
        await memory.start()
        try:
            memory.redis_ok = True
            memory.qdrant_ok = True
            await bus.publish(
                Event(type=EventType.TWITCH_CHAT, payload=ChatMessage(author="alice", text="я с Урала"))
            )
            await bus.wait_idle()
        finally:
            await memory.stop()

    assert any(turn.author == "alice" for turn in short.turns)
    assert long.stored and long.stored[0][0] == "alice"


async def test_llm_prompt_includes_memory_about_viewer() -> None:
    """Перед генерацией в user-сообщение попадает блок памяти по нику."""
    bus = EventBus()
    hits = [MemoryHit(text="любит донатить золото", score=0.9, author="alice", kind="chat")]
    memory = MemoryModule(
        bus,
        MemorySettings(),
        session="test",
        short=_FakeShort(),
        long=_FakeLong(hits),
    )
    llm = LLMModule(bus, LLMSettings(sentence_min_chars=1, first_token_timeout_s=2.0), memory=memory)
    client = _FakeClient()
    llm._client = client

    async with bus:
        await memory.start()
        memory.redis_ok = True
        memory.qdrant_ok = True
        await llm.start()
        try:
            await bus.publish(
                Event(type=EventType.TWITCH_CHAT, payload=ChatMessage(author="alice", text="привет"))
            )
            await bus.wait_idle()
        finally:
            await llm.stop()
            await memory.stop()

    assert client.completions.calls
    messages = client.completions.calls[0]["messages"]
    assert isinstance(messages, list)
    user = next(m["content"] for m in messages if m["role"] == "user")
    assert "[Memory]" in user
    assert "alice" in user
    assert "золото" in user
    # История диалога не должна содержать дамп памяти.
    stored = llm.context.messages()[-1]["content"]
    assert "[Memory]" not in stored
