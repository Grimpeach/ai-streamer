"""Клиент чата Twitch: читает сообщения и публикует их на шину событий.

**Только чтение.** ИИ-стример отвечает голосом через TTS, поэтому клиент никогда
не пишет в чат: нет ни команд, ни ответов, ни отправки сообщений. Для подключения
достаточно токена со scope ``chat:read``.

Надёжность соединения выстроена в два слоя:

1. Внутренний слой ``twitchio`` сам переподключается при обрыве websocket'а и
   обрабатывает команду ``RECONNECT`` от Twitch.
2. Супервизор этого модуля перезапускает клиент целиком, когда ``Client.start()``
   возвращает управление (соединение закрыто) или падает с ошибкой. Без него
   любое закрытие соединения оставило бы стримера навсегда без чата.

Экспоненциальный backoff с джиттером не даёт долбить Twitch при длительной
недоступности, а watchdog ловит «мёртвые» сокеты, которые выглядят открытыми.
Единственная ошибка, при которой повтор бессмысленен, — невалидный токен.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.core.config import TwitchSettings
from src.core.event_bus import EventBus
from src.core.events import (
    ChatMessage,
    Donation,
    Event,
    EventPriority,
    EventType,
    Raid,
    TwitchSubscription,
)
from src.core.tasks import cancel_task
from src.modules.base import Module

_WHITESPACE = re.compile(r"\s+")
#: Обёртка команды ``/me`` в IRC.
_ACTION = re.compile(r"^\x01ACTION (.*)\x01$")
#: Экранирование значений IRCv3-тегов (см. спецификацию message-tags).
_TAG_ESCAPES = ((r"\s", " "), (r"\:", ";"), (r"\r", ""), (r"\n", ""), (r"\\", "\\"))

_client_class: type[Any] | None = None


@dataclass(slots=True)
class IncomingMessage:
    """Нормализованное сообщение чата, независимое от транспорта."""

    author: str
    text: str
    channel: str = ""
    is_moderator: bool = False
    is_subscriber: bool = False
    #: Биты в сообщении: чир — это донат, а не обычная реплика.
    bits: int = 0


def unescape_tag(value: str) -> str:
    """Разэкранирует значение IRCv3-тега (``\\s`` → пробел и т.п.)."""
    for escaped, plain in _TAG_ESCAPES:
        value = value.replace(escaped, plain)
    return value.strip()


class TwitchChatModule(Module):
    """Читает чат Twitch и публикует события: чат — Priority 3, алерты — Priority 2."""

    name = "twitch"

    def __init__(
        self,
        bus: EventBus,
        settings: TwitchSettings,
        *,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(bus)
        self.settings = settings
        #: Подмена транспорта в тестах.
        self._client_factory = client_factory
        self._client: Any | None = None
        self._supervisor: asyncio.Task[None] | None = None
        self._watchdog: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._connected_at = 0.0
        self._last_data_at = 0.0
        self.reconnects = 0
        self.messages_received = 0

    @property
    def connected(self) -> bool:
        """Подключён ли клиент к IRC прямо сейчас."""
        return self._ready.is_set()

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Запускает супервизор соединения.

        Не ждёт подключения: недоступный Twitch не должен задерживать старт
        PTT и остальных локальных модулей.
        """
        if not self.settings.channel or not self.settings.access_token.get_secret_value():
            raise RuntimeError(
                "Не заданы TWITCH_CHANNEL / TWITCH_TOKEN (см. .env.example). "
                "Для отладки шины запустите: python main.py --demo"
            )
        self._supervisor = asyncio.create_task(self._supervise(), name="twitch-supervisor")
        self._started = True
        self.log.info("Подключаюсь к чату #{} (режим только чтение)", self.settings.channel)

    async def stop(self) -> None:
        """Останавливает супервизор и закрывает соединение."""
        self._started = False
        await self._cancel(self._supervisor)
        self._supervisor = None
        await self._cancel(self._watchdog)
        self._watchdog = None
        await self._close_client()
        self._ready.clear()

    async def wait_until_ready(self, timeout_s: float | None = None) -> bool:
        """Ждёт подключения к чату. ``False`` — не успел за ``timeout_s``."""
        try:
            await asyncio.wait_for(self._ready.wait(), timeout_s)
        except TimeoutError:
            return False
        return True

    # ----------------------------------------------------------------- supervisor

    async def _supervise(self) -> None:
        """Держит соединение живым, пересоздавая клиент при любом обрыве."""
        attempt = 0
        while True:
            client = self._create_client()
            self._client = client
            self._connected_at = 0.0
            self._last_data_at = time.monotonic()
            self._watchdog = asyncio.create_task(self._watch(client), name="twitch-watchdog")

            try:
                await client.start()
                reason = "соединение закрыто"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._is_auth_error(exc):
                    self.log.error(
                        "Twitch отверг токен ({}). Обновите TWITCH_TOKEN в .env — "
                        "нужен scope chat:read. Повторные попытки бессмысленны.",
                        type(exc).__name__,
                    )
                    self._started = False
                    return
                reason = f"{type(exc).__name__}: {exc}"
            finally:
                uptime = time.monotonic() - self._connected_at if self._connected_at else 0.0
                self._ready.clear()
                await self._cancel(self._watchdog)
                self._watchdog = None
                await self._close_client()

            # Стабильное соединение обнуляет backoff, иначе редкие ночные обрывы
            # накапливались бы и к утру давали минутные паузы.
            if uptime >= self.settings.stable_connection_s:
                attempt = 0
            delay = self._backoff_delay(attempt)
            attempt += 1
            self.reconnects += 1
            self.log.warning(
                "Twitch: {} (продержалось {:.0f}с). Переподключение через {:.1f}с, попытка {}",
                reason,
                uptime,
                delay,
                attempt,
            )
            await asyncio.sleep(delay)

    def _backoff_delay(self, attempt: int) -> float:
        """Экспоненциальный backoff с джиттером.

        Показатель степени ограничен: за долгую недоступность Twitch счётчик
        попыток успевает дорасти до тысяч, и ``2**attempt`` переполняет float,
        обрушивая супервизор — то есть чат пропадал бы навсегда именно там,
        где переподключение нужнее всего.
        """
        exponent = min(attempt, 16)
        capped = min(
            self.settings.reconnect_initial_delay_s * 2**exponent,
            self.settings.reconnect_max_delay_s,
        )
        return capped * random.uniform(0.5, 1.0)

    async def _watch(self, client: Any) -> None:
        """Рвёт соединение, если от Twitch давно не приходило вообще ничего."""
        interval = max(1.0, self.settings.watchdog_interval_s)
        while True:
            await asyncio.sleep(interval)
            idle = time.monotonic() - self._last_data_at
            if idle < self.settings.stale_after_s:
                continue
            self.log.warning("От Twitch нет данных {:.0f}с — принудительный реконнект", idle)
            with contextlib.suppress(Exception):
                await client.close()
            return

    @staticmethod
    def _is_auth_error(exc: BaseException) -> bool:
        """Отличает невалидный токен от сетевых сбоев.

        Сверяемся по именам классов, а не по isinstance: twitchio импортируется
        лениво, и тесты подменяют транспорт своими исключениями.
        """
        names = {cls.__name__ for cls in type(exc).__mro__}
        return bool(names & {"AuthenticationError", "Unauthorized", "NoToken"})

    def _create_client(self) -> Any:
        """Собирает twitchio-клиент (или подменённый в тестах транспорт)."""
        if self._client_factory is not None:
            return self._client_factory()
        client_class = _read_only_client_class()
        return client_class(
            token=self.settings.access_token.get_secret_value(),
            channel=self.settings.channel,
            heartbeat=self.settings.heartbeat_s,
            module=self,
        )

    async def _close_client(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        with contextlib.suppress(Exception):
            await client.close()

    async def _cancel(self, task: asyncio.Task[None] | None) -> None:
        await cancel_task(task, log=self.log)

    # ------------------------------------------------- колбэки транспорта twitchio

    async def on_ready(self, nick: str) -> None:
        """Соединение установлено и канал присоединён."""
        self._connected_at = time.monotonic()
        self.touch()
        self._ready.set()
        self.log.info("Читаю чат #{} под ником '{}'", self.settings.channel, nick)

    def touch(self) -> None:
        """Отмечает любую активность соединения, включая PING от Twitch."""
        self._last_data_at = time.monotonic()

    # ------------------------------------------------------------------ обработка

    def should_ignore(self, message: IncomingMessage) -> bool:
        """Фильтр шума: боты, команды, слишком короткие сообщения."""
        if message.author.lower() in {author.lower() for author in self.settings.ignored_authors}:
            return True
        text = message.text.strip()
        if self.settings.ignore_commands and text.startswith("!"):
            return True
        return len(text) < self.settings.min_message_length

    def clean_text(self, text: str) -> str:
        """Схлопывает пробелы, разворачивает ``/me`` и обрезает простыни."""
        text = _WHITESPACE.sub(" ", text or "").strip()
        action = _ACTION.match(text)
        if action:
            text = action.group(1).strip()
        limit = self.settings.max_message_length
        if len(text) > limit:
            text = text[:limit].rstrip() + "…"
        return text

    async def on_message(self, message: IncomingMessage) -> None:
        """Публикует сообщение чата: Priority 3, а чир с битами — Priority 2."""
        self.messages_received += 1
        if self.should_ignore(message):
            self.log.debug("Отфильтровано сообщение от {}", message.author)
            return

        text = self.clean_text(message.text)
        if not text:
            return

        if message.bits > 0:
            await self.on_donation(message.author, float(message.bits), text, currency="bits")
            return

        await self.bus.publish(
            Event(
                type=EventType.TWITCH_CHAT,
                source=self.name,
                # Priority 3 по specs/architecture.md — задано явно, чтобы
                # требование было видно в коде, а не только в таблице дефолтов.
                priority=EventPriority.NORMAL,
                ttl_s=self.settings.chat_ttl_s,
                payload=ChatMessage(
                    author=message.author,
                    text=text,
                    channel=message.channel or self.settings.channel,
                    is_moderator=message.is_moderator,
                    is_subscriber=message.is_subscriber,
                ),
            )
        )

    async def on_donation(
        self, author: str, amount: float, message: str = "", currency: str = "RUB"
    ) -> None:
        """Публикует донат или чир как событие Priority 2."""
        await self.bus.publish(
            Event(
                type=EventType.TWITCH_DONATION,
                source=self.name,
                priority=EventPriority.HIGH,
                payload=Donation(author=author, amount=amount, currency=currency, message=message),
            )
        )

    async def on_subscription(
        self,
        author: str,
        tier: str = "1000",
        months: int = 1,
        message: str = "",
        gifted_by: str | None = None,
    ) -> None:
        """Публикует подписку как событие Priority 2."""
        await self.bus.publish(
            Event(
                type=EventType.TWITCH_SUBSCRIPTION,
                source=self.name,
                priority=EventPriority.HIGH,
                payload=TwitchSubscription(
                    author=author, tier=tier, months=months, message=message, gifted_by=gifted_by
                ),
            )
        )

    async def on_raid(self, author: str, viewers: int) -> None:
        """Публикует рейд как событие Priority 2."""
        await self.bus.publish(
            Event(
                type=EventType.TWITCH_RAID,
                source=self.name,
                priority=EventPriority.HIGH,
                payload=Raid(author=author, viewers=viewers),
            )
        )

    # ---------------------------------------------------- нормализация из twitchio

    async def handle_privmsg(self, message: Any) -> None:
        """Переводит ``twitchio.Message`` в транспортно-независимый вид."""
        author = getattr(message, "author", None)
        tags = getattr(message, "tags", None) or {}
        bits = tags.get("bits") or 0
        await self.on_message(
            IncomingMessage(
                author=(
                    getattr(author, "display_name", None) or getattr(author, "name", None) or "anon"
                ),
                text=getattr(message, "content", "") or "",
                channel=getattr(getattr(message, "channel", None), "name", "") or "",
                is_moderator=bool(getattr(author, "is_mod", False)),
                is_subscriber=bool(getattr(author, "is_subscriber", False)),
                bits=int(bits),
            )
        )

    async def handle_usernotice(self, tags: dict[str, str]) -> None:
        """Подписки и рейды приходят тем же IRC-соединением через USERNOTICE."""
        kind = tags.get("msg-id", "")
        author = unescape_tag(tags.get("display-name") or tags.get("login") or "anon")
        tier = tags.get("msg-param-sub-plan") or "1000"
        text = self.clean_text(unescape_tag(tags.get("message", "")))

        if kind in {"sub", "resub"}:
            months = _as_int(tags.get("msg-param-cumulative-months"), default=1)
            await self.on_subscription(author, tier=tier, months=months, message=text)
        elif kind in {"subgift", "anonsubgift"}:
            recipient = unescape_tag(tags.get("msg-param-recipient-display-name") or "зритель")
            await self.on_subscription(recipient, tier=tier, gifted_by=author)
        elif kind == "raid":
            viewers = _as_int(tags.get("msg-param-viewerCount"), default=0)
            await self.on_raid(author, viewers)
        else:
            self.log.debug("USERNOTICE '{}' пропущен", kind)


def _as_int(value: str | None, *, default: int) -> int:
    """Числовые IRC-теги приходят строками и иногда пустыми."""
    try:
        return int(value) if value else default
    except ValueError:
        return default


def _read_only_client_class() -> type[Any]:
    """Лениво собирает подкласс ``twitchio.Client``.

    Импорт отложен: в демо-режиме и в тестах шины twitchio не нужен.
    """
    global _client_class
    if _client_class is not None:
        return _client_class

    import twitchio

    class _ReadOnlyChatClient(twitchio.Client):  # type: ignore[misc]
        """twitchio-клиент без отправки сообщений: только чтение чата."""

        def __init__(
            self, *, token: str, channel: str, heartbeat: float, module: TwitchChatModule
        ) -> None:
            super().__init__(token=token, initial_channels=[channel], heartbeat=heartbeat)
            self._module = module

        async def event_ready(self) -> None:
            await self._module.on_ready(self.nick or self._module.settings.bot_nick)

        async def event_raw_data(self, data: str) -> None:
            # Любая строка от Twitch, включая PING, — признак живого соединения.
            self._module.touch()

        async def event_message(self, message: twitchio.Message) -> None:
            if message.echo:
                return
            await self._module.handle_privmsg(message)

        async def event_raw_usernotice(self, channel: Any, tags: dict[str, str]) -> None:
            await self._module.handle_usernotice(tags)

        async def event_channel_join_failure(self, channel: str) -> None:
            self._module.log.warning("Не удалось зайти на канал #{}, переподключаюсь", channel)
            await self.close()

        async def event_token_expired(self) -> None:
            # Статический токен из .env обновить нечем: сообщаем и уходим в реконнект.
            self._module.log.error("Токен Twitch истёк — обновите TWITCH_TOKEN в .env")
            return None

        async def event_error(self, error: Exception, data: str | None = None) -> None:
            self._module.log.opt(exception=error).error("Ошибка twitchio")

    _client_class = _ReadOnlyChatClient
    return _client_class


class MockTwitchModule(TwitchChatModule):
    """Генератор синтетического трафика чата для отладки приоритетов шины."""

    name = "twitch-mock"

    _PHRASES = (
        "привет, как настроение?",
        "какая у тебя видеокарта?",
        "расскажи что-нибудь про нейросети",
        "го катку",
        "почему стрим лагает?",
    )
    _AUTHORS = ("viewer_1", "pepega_god", "kotik", "mod_anna", "random_dude")

    def __init__(self, bus: EventBus, settings: TwitchSettings, *, interval_s: float = 1.5) -> None:
        super().__init__(bus, settings)
        self.interval_s = interval_s
        self._feed: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Запускает фоновую генерацию сообщений."""
        self._feed = asyncio.create_task(self._produce(), name="twitch-mock-feed")
        self._ready.set()
        self._started = True
        self.log.info("Mock-чат запущен (интервал {:.1f}с)", self.interval_s)

    async def stop(self) -> None:
        """Останавливает генератор."""
        await self._cancel(self._feed)
        self._feed = None
        self._ready.clear()
        self._started = False

    async def _produce(self) -> None:
        while True:
            await asyncio.sleep(self.interval_s)
            roll = random.random()
            if roll < 0.1:
                await self.on_donation(
                    random.choice(self._AUTHORS),
                    round(random.uniform(50, 500), 2),
                    "спасибо за стрим!",
                )
            elif roll < 0.2:
                await self.on_subscription(random.choice(self._AUTHORS))
            else:
                await self.on_message(
                    IncomingMessage(
                        author=random.choice(self._AUTHORS), text=random.choice(self._PHRASES)
                    )
                )
