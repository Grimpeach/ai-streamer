"""Асинхронная шина событий с приоритетами и preemption по Push-to-Talk.

Свойства, важные для low-latency стрима:

1. **Приоритеты.** Очередь — бинарная куча по ``(priority, seq)``, поэтому событие
   PTT обгоняет накопленный чат независимо от размера очереди.
2. **Мгновенное прерывание.** ``publish()`` для preemptive-события отменяет активные
   прерываемые обработчики *до* постановки события в очередь, не дожидаясь диспетчера.
3. **Кооперативная отмена.** Долгие генерации (LLM, TTS) берут ``PreemptionToken`` и
   могут корректно освободить ресурсы вместо жёсткого обрыва.
4. **Сброс устаревшего.** При прерывании из очереди вычищаются события низкого
   приоритета, а события с истёкшим ``ttl_s`` отбрасываются диспетчером.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

from src.core.events import AnyEvent, Event, EventPriority, EventType, PreemptReason
from src.core.logger import get_logger

logger = get_logger(__name__)

Handler = Callable[[AnyEvent], Awaitable[None]]
T = TypeVar("T")


class Preempted(RuntimeError):
    """Генерация прервана более приоритетным событием (обычно PTT)."""

    def __init__(self, reason: str = "preempted") -> None:
        super().__init__(reason)
        self.reason = reason


class PreemptionToken:
    """Токен кооперативной отмены для одной генерации LLM/TTS.

    Выдаётся через :meth:`EventBus.preemption_scope`. Долгий цикл обязан
    периодически проверять :attr:`cancelled` или оборачивать поток в :meth:`guard`.
    """

    __slots__ = ("_event", "_generation", "_name", "reason")

    def __init__(self, generation: int, name: str = "") -> None:
        self._event = asyncio.Event()
        self._generation = generation
        self._name = name
        self.reason: str | None = None

    @property
    def generation(self) -> int:
        """Номер генерации шины, в которой создан токен."""
        return self._generation

    @property
    def cancelled(self) -> bool:
        """``True``, если генерацию нужно прекратить."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Бросает :class:`Preempted`, если генерация уже прервана."""
        if self.cancelled:
            raise Preempted(self.reason or "preempted")

    async def wait(self) -> str:
        """Ждёт прерывания и возвращает его причину."""
        await self._event.wait()
        return self.reason or "preempted"

    async def guard(self, source: AsyncIterator[T]) -> AsyncIterator[T]:
        """Прокидывает поток наружу, обрывая его при прерывании."""
        self.raise_if_cancelled()
        async for item in source:
            self.raise_if_cancelled()
            yield item

    def trigger(self, reason: str) -> None:
        """Помечает токен прерванным (вызывается шиной)."""
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    def __repr__(self) -> str:
        state = "cancelled" if self.cancelled else "active"
        return f"PreemptionToken({self._name or 'anon'}, gen={self._generation}, {state})"


@dataclass(slots=True)
class Subscription:
    """Подписка обработчика на типы событий."""

    name: str
    handler: Handler
    event_types: frozenset[EventType] | None = None
    #: Обработчик может быть снят по PTT (LLM/TTS — да, логирование — нет).
    preemptible: bool = True
    #: Не запускать новый обработчик, пока предыдущий не завершился.
    exclusive: bool = False
    #: Сколько событий отложить, пока exclusive-обработчик занят.
    max_deferred: int = 2
    busy: bool = field(default=False, repr=False)
    #: Куча ``(priority, seq, event)`` отложенных событий.
    deferred: list[tuple[int, int, AnyEvent]] = field(default_factory=list, repr=False)

    def matches(self, event: AnyEvent) -> bool:
        """Подходит ли событие под фильтр подписки."""
        return self.event_types is None or event.type in self.event_types


@dataclass(slots=True)
class BusStats:
    """Счётчики шины для телеметрии и тестов."""

    published: int = 0
    delivered: int = 0
    completed: int = 0
    failed: int = 0
    preemptions: int = 0
    dropped_preempt: int = 0
    dropped_stale: int = 0
    dropped_overflow: int = 0
    cancelled_handlers: int = 0

    def as_dict(self) -> dict[str, int]:
        """Снимок счётчиков."""
        return {
            "published": self.published,
            "delivered": self.delivered,
            "completed": self.completed,
            "failed": self.failed,
            "preemptions": self.preemptions,
            "dropped_preempt": self.dropped_preempt,
            "dropped_stale": self.dropped_stale,
            "dropped_overflow": self.dropped_overflow,
            "cancelled_handlers": self.cancelled_handlers,
        }


@dataclass(order=True, slots=True)
class _QueueItem:
    priority: int
    seq: int
    event: AnyEvent = field(compare=False)


class _PriorityQueue:
    """Куча событий с одним потребителем и возможностью выборочной чистки."""

    def __init__(self) -> None:
        self._heap: list[_QueueItem] = []
        self._seq = itertools.count()
        self._not_empty = asyncio.Event()

    def __len__(self) -> int:
        return len(self._heap)

    def push(self, event: AnyEvent) -> None:
        heapq.heappush(self._heap, _QueueItem(int(event.priority), next(self._seq), event))
        self._not_empty.set()

    async def pop(self) -> AnyEvent:
        while not self._heap:
            self._not_empty.clear()
            await self._not_empty.wait()
        item = heapq.heappop(self._heap)
        if not self._heap:
            self._not_empty.clear()
        return item.event

    def purge(self, predicate: Callable[[AnyEvent], bool]) -> list[AnyEvent]:
        """Удаляет события, для которых ``predicate`` истинно."""
        removed = [item.event for item in self._heap if predicate(item.event)]
        if removed:
            self._heap = [item for item in self._heap if not predicate(item.event)]
            heapq.heapify(self._heap)
            if not self._heap:
                self._not_empty.clear()
        return removed

    def evict_lowest(self) -> AnyEvent | None:
        """Выбрасывает наименее ценное событие: низший приоритет, самое старое.

        При равном приоритете жертвуем старым событием — свежая реплика чата
        интереснее зрителям, чем та, что уже уехала из виду.
        """
        if not self._heap:
            return None
        victim = max(self._heap, key=lambda item: (item.priority, -item.seq))
        self._heap.remove(victim)
        heapq.heapify(self._heap)
        if not self._heap:
            self._not_empty.clear()
        return victim.event

    def snapshot(self) -> list[AnyEvent]:
        """Текущее содержимое очереди в порядке приоритета."""
        return [item.event for item in sorted(self._heap)]


class EventBus:
    """Центральный диспетчер событий приложения."""

    def __init__(
        self,
        *,
        max_queue_size: int = 512,
        drop_on_preempt: Iterable[EventPriority] = (EventPriority.NORMAL, EventPriority.LOW),
        handler_timeout_s: float | None = None,
    ) -> None:
        self._queue = _PriorityQueue()
        self._subscriptions: list[Subscription] = []
        self._active: dict[asyncio.Task[None], Subscription] = {}
        self._tokens: set[PreemptionToken] = set()
        self._generation = 0
        self._max_queue_size = max_queue_size
        self._drop_on_preempt = frozenset(drop_on_preempt)
        self._handler_timeout_s = handler_timeout_s
        self._dispatcher: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._defer_seq = itertools.count()
        self._idle = asyncio.Event()
        self._idle.set()
        self._lock = threading.Lock()
        self.stats = BusStats()

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Запускает диспетчер. Идемпотентно."""
        if self._dispatcher is not None and not self._dispatcher.done():
            return
        self._loop = asyncio.get_running_loop()
        self._dispatcher = asyncio.create_task(self._dispatch_loop(), name="event-bus-dispatcher")
        logger.info("Event bus запущена (max_queue_size={})", self._max_queue_size)

    async def stop(self, *, drain_timeout_s: float = 2.0) -> None:
        """Останавливает диспетчер и дожидается завершения обработчиков."""
        if self._dispatcher is None:
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.wait_idle(), timeout=drain_timeout_s)

        self._dispatcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._dispatcher
        self._dispatcher = None

        for task in list(self._active):
            task.cancel()
        if self._active:
            await asyncio.gather(*self._active, return_exceptions=True)
        self._active.clear()
        for subscription in self._subscriptions:
            subscription.deferred.clear()
            subscription.busy = False
        for token in list(self._tokens):
            token.trigger("shutdown")
        logger.info("Event bus остановлена: {}", self.stats.as_dict())

    async def __aenter__(self) -> EventBus:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.stop()

    # --------------------------------------------------------------- subscriptions

    def subscribe(
        self,
        handler: Handler,
        event_types: Iterable[EventType] | EventType | None = None,
        *,
        name: str | None = None,
        preemptible: bool = True,
        exclusive: bool = False,
        max_deferred: int = 2,
    ) -> Subscription:
        """Регистрирует обработчик. ``event_types=None`` — подписка на всё."""
        if isinstance(event_types, EventType):
            types: frozenset[EventType] | None = frozenset({event_types})
        elif event_types is None:
            types = None
        else:
            types = frozenset(event_types)

        subscription = Subscription(
            name=name or getattr(handler, "__qualname__", repr(handler)),
            handler=handler,
            event_types=types,
            preemptible=preemptible,
            exclusive=exclusive,
            max_deferred=max_deferred,
        )
        self._subscriptions.append(subscription)
        logger.debug(
            "Подписка '{}' на {} (preemptible={})",
            subscription.name,
            "*" if types is None else sorted(types),
            preemptible,
        )
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        """Снимает ранее зарегистрированную подписку."""
        with contextlib.suppress(ValueError):
            self._subscriptions.remove(subscription)

    # -------------------------------------------------------------------- publish

    async def publish(self, event: AnyEvent) -> None:
        """Публикует событие. Не блокирует: обработка идёт в диспетчере."""
        self.publish_nowait(event)

    def publish_nowait(self, event: AnyEvent) -> None:
        """Синхронная публикация из корутины или колбэка в этом же loop."""
        if event.preemptive:
            reason = event.payload.reason if isinstance(event.payload, PreemptReason) else str(event.type)
            self.preempt(reason=reason, source=event.source)

        if len(self._queue) >= self._max_queue_size:
            victim = self._queue.evict_lowest()
            if victim is not None and victim.priority <= event.priority:
                # Очередь забита равным или более важным — отбрасываем новое событие.
                self._queue.push(victim)
                self.stats.dropped_overflow += 1
                logger.warning("Очередь переполнена, событие отброшено: {}", event.describe())
                return
            self.stats.dropped_overflow += 1
            if victim is not None:
                logger.warning("Очередь переполнена, вытеснено: {}", victim.describe())

        self._queue.push(event)
        self.stats.published += 1
        self._idle.clear()
        logger.debug("→ publish {}", event.describe())

    def publish_threadsafe(self, event: AnyEvent) -> None:
        """Публикация из чужого потока (хук клавиатуры PTT, аудио-колбэк)."""
        loop = self._loop
        if loop is None:
            raise RuntimeError("Event bus не запущена: publish_threadsafe недоступен")
        loop.call_soon_threadsafe(self.publish_nowait, event)

    async def emit(
        self,
        event_type: EventType,
        payload: Any = None,
        *,
        source: str = "unknown",
        priority: EventPriority | None = None,
        correlation_id: str | None = None,
        ttl_s: float | None = None,
    ) -> AnyEvent:
        """Удобная обёртка: собирает :class:`Event` и публикует его."""
        fields: dict[str, Any] = {"type": event_type, "source": source}
        if payload is not None:
            fields["payload"] = payload
        if priority is not None:
            fields["priority"] = priority
        if correlation_id is not None:
            fields["correlation_id"] = correlation_id
        if ttl_s is not None:
            fields["ttl_s"] = ttl_s
        event: AnyEvent = Event(**fields)
        await self.publish(event)
        return event

    # ------------------------------------------------------------------ preemption

    @property
    def generation(self) -> int:
        """Номер текущей генерации: инкрементируется при каждом прерывании."""
        return self._generation

    def preempt(self, *, reason: str = "ptt", source: str = "") -> int:
        """Прерывает активные генерации. Возвращает число снятых обработчиков."""
        with self._lock:
            self._generation += 1
        self.stats.preemptions += 1

        for token in list(self._tokens):
            token.trigger(reason)

        current = asyncio.current_task()
        cancelled = 0
        for task, subscription in list(self._active.items()):
            if subscription.preemptible and task is not current and not task.done():
                task.cancel()
                cancelled += 1
        self.stats.cancelled_handlers += cancelled

        dropped = len(self._queue.purge(lambda ev: ev.priority in self._drop_on_preempt))
        # Отложенные реплики после прерывания уже неактуальны — хост сменил тему.
        for subscription in self._subscriptions:
            if subscription.preemptible and subscription.deferred:
                dropped += len(subscription.deferred)
                subscription.deferred.clear()
        self.stats.dropped_preempt += dropped

        if cancelled or dropped:
            logger.info(
                "PREEMPT ({} from '{}'): снято обработчиков={}, вычищено событий={}, gen={}",
                reason,
                source or "system",
                cancelled,
                dropped,
                self._generation,
            )
        return cancelled

    @contextlib.contextmanager
    def preemption_scope(self, name: str = "") -> Iterator[PreemptionToken]:
        """Контекст для долгой генерации: выдаёт токен и снимает его в конце."""
        token = PreemptionToken(self._generation, name)
        self._tokens.add(token)
        try:
            yield token
        finally:
            self._tokens.discard(token)

    # -------------------------------------------------------------------- dispatch

    async def _dispatch_loop(self) -> None:
        while True:
            event = await self._queue.pop()
            if event.is_stale:
                self.stats.dropped_stale += 1
                logger.debug("Событие просрочено, отброшено: {}", event.describe())
                self._update_idle()
                continue
            self._deliver(event)

    def _deliver(self, event: AnyEvent) -> None:
        """Раздаёт событие подписчикам, не дожидаясь их завершения."""
        matched = [sub for sub in self._subscriptions if sub.matches(event)]
        if not matched:
            logger.debug("Нет подписчиков на {}", event.describe())
            self._update_idle()
            return

        for subscription in matched:
            self._dispatch_to(subscription, event)

    def _dispatch_to(self, subscription: Subscription, event: AnyEvent) -> None:
        """Запускает обработчик либо откладывает событие, если тот занят."""
        if subscription.exclusive and subscription.busy:
            self._defer(subscription, event)
            return
        subscription.busy = True
        task = asyncio.create_task(
            self._run_handler(subscription, event),
            name=f"{subscription.name}:{event.type}",
        )
        self._active[task] = subscription
        self.stats.delivered += 1
        task.add_done_callback(self._on_handler_done)

    def _defer(self, subscription: Subscription, event: AnyEvent) -> None:
        """Откладывает событие для занятого exclusive-обработчика.

        Очередь ожидания намеренно короткая: реагировать на донат через минуту
        хуже, чем не реагировать вовсе, поэтому лишнее вытесняется по приоритету.
        """
        heapq.heappush(subscription.deferred, (int(event.priority), next(self._defer_seq), event))
        while len(subscription.deferred) > subscription.max_deferred:
            victim = max(subscription.deferred, key=lambda item: (item[0], -item[1]))
            subscription.deferred.remove(victim)
            heapq.heapify(subscription.deferred)
            self.stats.dropped_overflow += 1
            logger.debug("'{}' перегружен, вытеснено {}", subscription.name, victim[2].describe())
        logger.debug("'{}' занят, отложено {}", subscription.name, event.describe())

    def _pump_deferred(self, subscription: Subscription) -> None:
        """Отдаёт обработчику самое приоритетное из отложенных событий."""
        while subscription.deferred and not subscription.busy:
            _, _, event = heapq.heappop(subscription.deferred)
            if event.is_stale:
                self.stats.dropped_stale += 1
                logger.debug("Отложенное событие просрочено: {}", event.describe())
                continue
            self._dispatch_to(subscription, event)

    async def _run_handler(self, subscription: Subscription, event: AnyEvent) -> None:
        try:
            if self._handler_timeout_s is not None:
                await asyncio.wait_for(subscription.handler(event), timeout=self._handler_timeout_s)
            else:
                await subscription.handler(event)
            self.stats.completed += 1
        except asyncio.CancelledError:
            logger.debug("'{}' отменён на {}", subscription.name, event.describe())
            raise
        except Preempted as exc:
            logger.debug("'{}' прерван ({}) на {}", subscription.name, exc.reason, event.describe())
        except Exception:
            self.stats.failed += 1
            logger.exception("Ошибка в обработчике '{}' на {}", subscription.name, event.describe())
        finally:
            subscription.busy = False

    def _on_handler_done(self, task: asyncio.Task[None]) -> None:
        subscription = self._active.pop(task, None)
        if subscription is not None:
            self._pump_deferred(subscription)
        self._update_idle()

    def _update_idle(self) -> None:
        if self._active or len(self._queue):
            return
        if any(sub.deferred for sub in self._subscriptions):
            return
        self._idle.set()

    # ----------------------------------------------------------------- inspection

    async def wait_idle(self) -> None:
        """Ждёт, пока очередь пуста и все обработчики завершились."""
        self._update_idle()
        await self._idle.wait()

    @property
    def queue_size(self) -> int:
        """Число событий, ожидающих доставки."""
        return len(self._queue)

    @property
    def active_handlers(self) -> int:
        """Число выполняющихся обработчиков."""
        return len(self._active)

    def pending_events(self) -> list[AnyEvent]:
        """Снимок очереди в порядке приоритета (для отладки)."""
        return self._queue.snapshot()

    def __repr__(self) -> str:
        return (
            f"EventBus(queue={self.queue_size}, active={self.active_handlers}, "
            f"gen={self._generation}, subs={len(self._subscriptions)})"
        )
