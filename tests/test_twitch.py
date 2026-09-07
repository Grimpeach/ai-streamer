"""Проверки Twitch-модуля: разбор сообщений, фильтры и надёжность соединения.

Транспорт подменяется фейковым клиентом, поэтому тесты не ходят в сеть и не
требуют токена.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.core.config import TwitchSettings
from src.core.event_bus import EventBus
from src.core.events import AnyEvent, ChatMessage, Donation, EventPriority, EventType, Raid
from src.modules.twitch.client import IncomingMessage, TwitchChatModule, unescape_tag

pytestmark = pytest.mark.asyncio


def make_settings(**overrides: object) -> TwitchSettings:
    """Настройки с мгновенным backoff, чтобы тесты не спали секундами."""
    base: dict[str, object] = {
        "channel": "olegexec",
        "access_token": "oauth:testtoken",
        "reconnect_initial_delay_s": 0.01,
        "reconnect_max_delay_s": 0.02,
        "stable_connection_s": 0.05,
        "watchdog_interval_s": 0.01,
        "stale_after_s": 0.05,
    }
    base.update(overrides)
    return TwitchSettings(**base)  # type: ignore[arg-type]


class Collector:
    """Складывает события шины для проверок."""

    def __init__(self, bus: EventBus) -> None:
        self.events: list[AnyEvent] = []
        bus.subscribe(self._collect, name="collector", preemptible=False)

    async def _collect(self, event: AnyEvent) -> None:
        self.events.append(event)


class FakeClient:
    """Транспорт-заглушка: сценарий поведения задаётся списком действий."""

    def __init__(self, script: list[str], *, on_start: object = None) -> None:
        self.script = script
        self.on_start = on_start
        self.starts = 0
        self.closed = 0
        self._closing = asyncio.Event()

    async def start(self) -> None:
        self.starts += 1
        action = self.script.pop(0) if self.script else "hang"
        if action == "auth-error":
            raise FakeAuthError("Invalid or unauthorized Access Token passed.")
        if action == "network-error":
            raise ConnectionResetError("сеть отвалилась")
        if action == "close":
            return  # соединение закрылось само
        if callable(self.on_start):
            await self.on_start(self)  # type: ignore[misc]
        await self._closing.wait()

    async def close(self) -> None:
        self.closed += 1
        self._closing.set()


class FakeAuthError(Exception):
    """Имя класса важно: модуль распознаёт ошибки авторизации по нему."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


FakeAuthError.__name__ = "AuthenticationError"


# --------------------------------------------------------------- разбор сообщений


async def test_chat_message_published_with_priority_3() -> None:
    """Сообщение чата уходит на шину с приоритетом 3 и TTL."""
    bus = EventBus()
    collector = Collector(bus)
    module = TwitchChatModule(bus, make_settings())

    async with bus:
        await module.on_message(
            IncomingMessage(author="Kotik", text="привет, стример!", is_subscriber=True)
        )
        await asyncio.wait_for(bus.wait_idle(), timeout=2)

    event = collector.events[0]
    assert event.type is EventType.TWITCH_CHAT
    assert event.priority is EventPriority.NORMAL
    assert int(event.priority) == 3
    assert event.ttl_s == module.settings.chat_ttl_s
    assert isinstance(event.payload, ChatMessage)
    assert event.payload.author == "Kotik"
    assert event.payload.text == "привет, стример!"
    assert event.payload.channel == "olegexec"
    assert event.payload.is_subscriber is True


async def test_twitchio_message_is_normalized() -> None:
    """Объект twitchio приводится к транспортно-независимому виду."""
    bus = EventBus()
    collector = Collector(bus)
    module = TwitchChatModule(bus, make_settings())

    message = SimpleNamespace(
        content="  какая   у тебя  видеокарта?  ",
        author=SimpleNamespace(display_name="Pepega", name="pepega", is_mod=True, is_subscriber=0),
        channel=SimpleNamespace(name="olegexec"),
        tags={},
        echo=False,
    )

    async with bus:
        await module.handle_privmsg(message)
        await asyncio.wait_for(bus.wait_idle(), timeout=2)

    payload = collector.events[0].payload
    assert isinstance(payload, ChatMessage)
    assert payload.author == "Pepega"
    assert payload.text == "какая у тебя видеокарта?"  # пробелы схлопнуты
    assert payload.is_moderator is True
    assert payload.is_subscriber is False


async def test_cheer_becomes_donation_priority_2() -> None:
    """Сообщение с битами — это донат (Priority 2), а не обычная реплика."""
    bus = EventBus()
    collector = Collector(bus)
    module = TwitchChatModule(bus, make_settings())

    async with bus:
        await module.handle_privmsg(
            SimpleNamespace(
                content="Cheer100 держи на кофе",
                author=SimpleNamespace(display_name="Щедрый", name="generous"),
                channel=SimpleNamespace(name="olegexec"),
                tags={"bits": "100"},
                echo=False,
            )
        )
        await asyncio.wait_for(bus.wait_idle(), timeout=2)

    event = collector.events[0]
    assert event.type is EventType.TWITCH_DONATION
    assert int(event.priority) == 2
    assert isinstance(event.payload, Donation)
    assert event.payload.amount == 100
    assert event.payload.currency == "bits"


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        (IncomingMessage(author="Nightbot", text="реклама канала"), "бот из ignored_authors"),
        (IncomingMessage(author="viewer", text="!drop"), "команда бота"),
        (IncomingMessage(author="viewer", text="."), "слишком короткое"),
        (IncomingMessage(author="viewer", text="   "), "пустое"),
    ],
)
async def test_noise_is_filtered(message: IncomingMessage, reason: str) -> None:
    """Шум не доходит до шины."""
    bus = EventBus()
    collector = Collector(bus)
    module = TwitchChatModule(bus, make_settings())

    async with bus:
        await module.on_message(message)
        await asyncio.wait_for(bus.wait_idle(), timeout=2)

    assert collector.events == [], f"не отфильтровано: {reason}"


async def test_long_message_is_truncated() -> None:
    """Простыня обрезается, чтобы TTS не читал её минуту."""
    bus = EventBus()
    collector = Collector(bus)
    module = TwitchChatModule(bus, make_settings(max_message_length=20))

    async with bus:
        await module.on_message(IncomingMessage(author="viewer", text="а" * 100))
        await asyncio.wait_for(bus.wait_idle(), timeout=2)

    payload = collector.events[0].payload
    assert isinstance(payload, ChatMessage)
    assert payload.text == "а" * 20 + "…"


async def test_me_action_is_unwrapped() -> None:
    """Команда /me приходит в IRC-обёртке, её нужно снять."""
    bus = EventBus()
    collector = Collector(bus)
    module = TwitchChatModule(bus, make_settings())

    async with bus:
        await module.on_message(IncomingMessage(author="viewer", text="\x01ACTION машет рукой\x01"))
        await asyncio.wait_for(bus.wait_idle(), timeout=2)

    payload = collector.events[0].payload
    assert isinstance(payload, ChatMessage)
    assert payload.text == "машет рукой"


async def test_usernotice_raid_and_subscription() -> None:
    """Подписки и рейды разбираются из тегов USERNOTICE."""
    bus = EventBus()
    collector = Collector(bus)
    module = TwitchChatModule(bus, make_settings())

    async with bus:
        await module.handle_usernotice(
            {
                "msg-id": "resub",
                "display-name": "Верный\\sЗритель",
                "msg-param-sub-plan": "2000",
                "msg-param-cumulative-months": "7",
                "message": "седьмой\\sмесяц!",
            }
        )
        await module.handle_usernotice(
            {"msg-id": "raid", "display-name": "Сосед", "msg-param-viewerCount": "42"}
        )
        await asyncio.wait_for(bus.wait_idle(), timeout=2)

    sub, raid = collector.events
    assert sub.type is EventType.TWITCH_SUBSCRIPTION
    assert int(sub.priority) == 2
    assert sub.payload.author == "Верный Зритель"  # IRC-экранирование снято
    assert sub.payload.months == 7
    assert raid.type is EventType.TWITCH_RAID
    assert isinstance(raid.payload, Raid)
    assert raid.payload.viewers == 42


async def test_unescape_tag() -> None:
    """Значения IRCv3-тегов приходят экранированными."""
    assert unescape_tag(r"привет\sмир") == "привет мир"
    assert unescape_tag(r"a\:b") == "a;b"


# ------------------------------------------------------------------- соединение


async def test_missing_credentials_fail_fast() -> None:
    """Без канала и токена модуль не запускается."""
    bus = EventBus()
    module = TwitchChatModule(bus, TwitchSettings())

    with pytest.raises(RuntimeError, match="TWITCH_CHANNEL"):
        await module.start()


async def test_oauth_prefix_is_stripped() -> None:
    """twitchio ждёт токен без префикса oauth:."""
    settings = make_settings(access_token="oauth:abc123")
    assert settings.access_token.get_secret_value() == "abc123"


async def test_reconnects_after_connection_drop() -> None:
    """Закрытие соединения приводит к пересозданию клиента."""
    bus = EventBus()
    clients: list[FakeClient] = []

    def factory() -> FakeClient:
        client = FakeClient(script=["close"])
        clients.append(client)
        return client

    module = TwitchChatModule(bus, make_settings(), client_factory=factory)

    async with bus:
        await module.start()
        try:
            # Ждём, пока супервизор переподключится хотя бы трижды.
            for _ in range(200):
                if len(clients) >= 3:
                    break
                await asyncio.sleep(0.01)
        finally:
            await module.stop()

    assert len(clients) >= 3, "супервизор не переподключился"
    assert module.reconnects >= 2


async def test_network_error_does_not_kill_supervisor() -> None:
    """Сетевая ошибка — не фатальна: попытки продолжаются."""
    bus = EventBus()
    clients: list[FakeClient] = []

    def factory() -> FakeClient:
        client = FakeClient(script=["network-error"])
        clients.append(client)
        return client

    module = TwitchChatModule(bus, make_settings(), client_factory=factory)

    async with bus:
        await module.start()
        try:
            for _ in range(200):
                if len(clients) >= 3:
                    break
                await asyncio.sleep(0.01)
        finally:
            await module.stop()

    assert len(clients) >= 3


async def test_auth_error_stops_retrying() -> None:
    """Невалидный токен не ретраится: иначе Twitch забанит за спам попыток."""
    bus = EventBus()
    clients: list[FakeClient] = []

    def factory() -> FakeClient:
        client = FakeClient(script=["auth-error"])
        clients.append(client)
        return client

    module = TwitchChatModule(bus, make_settings(), client_factory=factory)

    async with bus:
        await module.start()
        await asyncio.sleep(0.2)
        try:
            assert len(clients) == 1, "после ошибки авторизации попытки должны прекратиться"
            assert module.started is False
        finally:
            await module.stop()


async def test_ready_state_and_stable_connection() -> None:
    """После event_ready модуль считается подключённым."""
    bus = EventBus()

    async def on_start(client: FakeClient) -> None:
        await module.on_ready("ai_streamer_bot")

    module = TwitchChatModule(
        bus,
        make_settings(stale_after_s=999.0),
        client_factory=lambda: FakeClient(script=["hang"], on_start=on_start),
    )

    async with bus:
        await module.start()
        try:
            assert await module.wait_until_ready(timeout_s=2)
            assert module.connected is True
        finally:
            await module.stop()

    assert module.connected is False


async def test_watchdog_reconnects_on_silence() -> None:
    """Тишина дольше stale_after_s означает мёртвый сокет — нужен реконнект."""
    bus = EventBus()
    clients: list[FakeClient] = []

    async def on_start(client: FakeClient) -> None:
        await module.on_ready("ai_streamer_bot")

    def factory() -> FakeClient:
        client = FakeClient(script=["hang"], on_start=on_start)
        clients.append(client)
        return client

    # Соединение живое с точки зрения сокета, но данные не приходят.
    module = TwitchChatModule(bus, make_settings(stale_after_s=0.05), client_factory=factory)

    async with bus:
        await module.start()
        try:
            for _ in range(200):
                if len(clients) >= 2:
                    break
                await asyncio.sleep(0.01)
        finally:
            await module.stop()

    assert len(clients) >= 2, "watchdog не разорвал зависшее соединение"
    assert clients[0].closed >= 1
