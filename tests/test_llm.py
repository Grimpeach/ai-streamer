"""LLM: стриминг, паузы для TTS, история диалога, отмена без записи ответа."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.core.config import LLMSettings
from src.core.event_bus import EventBus
from src.core.events import ChatMessage, Event, EventType, HostSpeech, TextRequest
from src.core.text import SentenceBuffer
from src.modules.llm.client import LLMModule
from src.modules.llm.context import DialogueContext
from src.modules.llm.persona import render_system_prompt


class _FakeStream:
    """Асинхронный поток чанков в формате OpenAI Chat Completions."""

    def __init__(self, tokens: list[str], *, delay_s: float = 0.0, hang_after: int | None = None) -> None:
        self.tokens = tokens
        self.delay_s = delay_s
        self.hang_after = hang_after
        self.closed = False
        self._index = 0

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self.hang_after is not None and self._index >= self.hang_after:
            await asyncio.sleep(60)
        if self._index >= len(self.tokens):
            raise StopAsyncIteration
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        token = self.tokens[self._index]
        self._index += 1
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=token))])

    async def close(self) -> None:
        self.closed = True


class _FakeCompletions:
    def __init__(self, tokens: list[str], **stream_kw: float | int | None) -> None:
        self.tokens = tokens
        self.stream_kw = stream_kw
        self.calls: list[dict[str, object]] = []
        self.last_stream: _FakeStream | None = None

    async def create(self, **kwargs: object) -> _FakeStream:
        self.calls.append(kwargs)
        self.last_stream = _FakeStream(self.tokens, **self.stream_kw)  # type: ignore[arg-type]
        return self.last_stream


class _FakeClient:
    def __init__(self, tokens: list[str], **stream_kw: float | int | None) -> None:
        self.completions = _FakeCompletions(tokens, **stream_kw)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _settings(**overrides: object) -> LLMSettings:
    return LLMSettings(sentence_min_chars=1, first_token_timeout_s=2.0, **overrides)  # type: ignore[arg-type]


def _chat(text: str = "привет") -> Event[ChatMessage]:
    return Event(type=EventType.TWITCH_CHAT, payload=ChatMessage(author="viewer", text=text))


def _host(text: str = "я сам") -> Event[HostSpeech]:
    return Event(type=EventType.HOST_SPEECH, payload=HostSpeech(text=text, duration_s=1.0))


async def _start(bus: EventBus, module: LLMModule) -> None:
    await bus.start()
    await module.start()


async def _stop(bus: EventBus, module: LLMModule) -> None:
    await module.stop()
    await bus.stop()


def test_sentence_buffer_emits_on_logical_pauses() -> None:
    """Запятая и точка отдают куски, хвост без паузы остаётся до flush."""
    buffer = SentenceBuffer(min_chars=1)

    assert buffer.push("Привет, ") == ["Привет,"]
    assert buffer.push("как дела?") == ["как дела?"]
    assert buffer.push(" хвост") == []
    assert buffer.flush() == "хвост"


def test_sentence_buffer_clear_discards_remainder() -> None:
    """При отмене хвост не должен уйти в TTS."""
    buffer = SentenceBuffer()
    buffer.push("незаконченная фраза")
    buffer.clear()
    assert buffer.flush() is None


def test_sentence_buffer_respects_min_chars() -> None:
    """Короткая пауза не эмитится, пока кусок короче порога."""
    buffer = SentenceBuffer(min_chars=10)
    assert buffer.push("Так, ") == []
    assert buffer.push("смотрите.") == ["Так, смотрите."]


def test_system_prompt_substitutes_character_name() -> None:
    """Имя из настроек подставляется в шаблон; в коде персоны нет."""
    prompt = render_system_prompt("Ada", "You are {name}. Speak English.")
    assert prompt == "You are Ada. Speak English."
    assert render_system_prompt("Ada", "") == "You are Ada."


def test_dialogue_context_keeps_system_first() -> None:
    """В API-сообщениях system всегда первый, окно режет старые реплики."""
    ctx = DialogueContext(system_prompt="ты стримерша", max_messages=2)
    ctx.add_user("раз")
    ctx.add_assistant("два")
    ctx.add_user("три")
    messages = ctx.messages()
    assert messages[0] == {"role": "system", "content": "ты стримерша"}
    assert [m["content"] for m in messages[1:]] == ["два", "три"]


async def test_stream_is_requested_and_pauses_go_to_tts() -> None:
    """create() вызывается со stream=True; паузы публикуются как TTS_REQUEST."""
    spoken: list[str] = []
    bus = EventBus()
    client = _FakeClient(["Привет, ", "чат!"])
    module = LLMModule(bus, _settings())
    module._client = client

    async def on_tts(event: Event[TextRequest]) -> None:
        assert isinstance(event.payload, TextRequest)
        spoken.append(event.payload.text)

    bus.subscribe(on_tts, EventType.TTS_REQUEST, name="probe", preemptible=False)

    await _start(bus, module)
    try:
        await bus.publish(_chat("привет"))
        await bus.wait_idle()
    finally:
        await _stop(bus, module)

    assert client.completions.calls, "LLM не обратился к API"
    call = client.completions.calls[0]
    assert call["stream"] is True
    assert call["model"] == module.settings.model
    assert call["max_tokens"] == module.settings.max_tokens
    assert spoken == ["Привет,", "чат!"]
    assert client.completions.last_stream is not None
    assert client.completions.last_stream.closed is True


async def test_completed_reply_is_stored_in_history() -> None:
    """Законченный ответ попадает в контекст вместе с репликой зрителя."""
    bus = EventBus()
    client = _FakeClient(["Ок, ", "поняла."])
    module = LLMModule(bus, _settings())
    module._client = client

    await _start(bus, module)
    try:
        await bus.publish(_chat("как дела"))
        await bus.wait_idle()
    finally:
        await _stop(bus, module)

    assert module.context.last_assistant() == "Ок, поняла."
    roles = [m["role"] for m in module.context.messages()]
    assert roles[0] == "system"
    assert roles[1:] == ["user", "assistant"]
    user = module.context.messages()[1]["content"]
    assert "viewer" in user
    assert "как дела" in user


async def test_second_turn_sends_prior_assistant() -> None:
    """Следующий запрос к модели содержит предыдущий ответ."""
    bus = EventBus()
    client = _FakeClient(["Да."])
    module = LLMModule(bus, _settings())
    module._client = client

    await _start(bus, module)
    try:
        await bus.publish(_chat("раз"))
        await bus.wait_idle()
        client.completions.tokens = ["Два."]
        await bus.publish(_chat("два"))
        await bus.wait_idle()
    finally:
        await _stop(bus, module)

    second = client.completions.calls[1]
    messages = second["messages"]
    assert isinstance(messages, list)
    assert any(m.get("role") == "assistant" and "Да." in m.get("content", "") for m in messages)


async def test_preempt_does_not_store_partial_assistant() -> None:
    """Отмена во время генерации не записывает оборванный ответ в историю."""
    spoken: list[str] = []
    first_clause = asyncio.Event()
    bus = EventBus()
    client = _FakeClient(["Привет, ", "это длинный хвост"], delay_s=0.02, hang_after=1)
    module = LLMModule(bus, _settings())
    module._client = client

    async def on_tts(event: Event[TextRequest]) -> None:
        assert isinstance(event.payload, TextRequest)
        spoken.append(event.payload.text)
        first_clause.set()

    bus.subscribe(on_tts, EventType.TTS_REQUEST, name="probe", preemptible=False)

    await _start(bus, module)
    try:
        await bus.publish(_chat("привет"))
        await asyncio.wait_for(first_clause.wait(), timeout=2)
        bus.preempt(reason="ptt", source="test")
        await bus.wait_idle()
    finally:
        await _stop(bus, module)

    assert spoken == ["Привет,"]
    assert module.context.last_assistant() is None
    user_turns = [m for m in module.context.messages() if m["role"] == "user"]
    assert len(user_turns) == 1


async def test_host_speech_has_priority_over_chat_in_prompt() -> None:
    """Реплика хоста маркируется отдельно от чата."""
    bus = EventBus()
    client = _FakeClient(["Слушаю."])
    module = LLMModule(bus, _settings())
    module._client = client

    await _start(bus, module)
    try:
        await bus.publish(_host("проверь железо"))
        await bus.wait_idle()
    finally:
        await _stop(bus, module)

    user = module.context.messages()[1]["content"]
    assert user.startswith("Host says:")
    assert "проверь железо" in user
    assert "Respond in English." in user
    assert module.context.last_assistant() == "Слушаю."
