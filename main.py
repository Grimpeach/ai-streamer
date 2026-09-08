"""Точка входа AI Streamer.

Режимы запуска::

    python main.py --demo            # шина + моки: видно приоритеты и прерывание по PTT
    python main.py --twitch-only     # чтение живого чата Twitch с выводом в консоль
    python main.py --ptt-only        # живой PTT + STT без LLM/TTS (проверка прерывания)
    python main.py --llm-only        # Twitch + PTT + LLM, куски TTS_REQUEST в консоль (без синтеза)
    python main.py                   # полный стример: Twitch + PTT + STT + LLM + TTS + Memory

Демо-режим не требует ни GPU, ни внешних сервисов: он поднимает шину событий,
подсовывает синтетический чат, донаты и нажатия Push-to-Talk и показывает, что
реплика хоста мгновенно снимает текущую генерацию LLM и TTS.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
from dataclasses import dataclass

from pydantic import ValidationError

from src.core.config import Settings, load_settings
from src.core.cuda import ensure_windows_cuda_dlls
from src.core.event_bus import EventBus
from src.core.events import (
    AnyEvent,
    AudioChunk,
    ChatMessage,
    Donation,
    EventType,
    HostSpeech,
    Raid,
    TextRequest,
    TwitchSubscription,
)
from src.core.logger import get_logger, setup_logging
from src.modules.base import Module
from src.modules.llm.client import LLMModule, MockLLMModule
from src.modules.memory.module import MemoryModule
from src.modules.ptt.hotkey import MockPushToTalkModule, PushToTalkModule
from src.modules.tts.streamer import MockTTSModule, TTSModule
from src.modules.twitch.client import MockTwitchModule, TwitchChatModule

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CliArgs:
    """Аргументы запуска. Всегда содержит отладочные флаги модулей.

    ``argparse.Namespace`` легко собрать без ``twitch_only``/``ptt_only``/
    ``llm_only``, если флаг забыли зарегистрировать. Этот тип задаёт значения
    по умолчанию, поэтому ``amain`` не падает с AttributeError до старта модулей.
    """

    demo: bool = False
    twitch_only: bool = False
    ptt_only: bool = False
    llm_only: bool = False
    duration: float = 20.0
    log_level: str = "INFO"

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> CliArgs:
        """Собирает аргументы даже из неполного Namespace."""
        return cls(
            demo=bool(getattr(namespace, "demo", False)),
            twitch_only=bool(getattr(namespace, "twitch_only", False)),
            ptt_only=bool(getattr(namespace, "ptt_only", False)),
            llm_only=bool(getattr(namespace, "llm_only", False)),
            duration=float(getattr(namespace, "duration", 20.0)),
            log_level=str(getattr(namespace, "log_level", "INFO")),
        )


class ConsoleAudioSink:
    """Вместо звуковой карты пишет в лог, сколько аудио «проиграно»."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.chunks = 0

    def register(self) -> None:
        """Подписывается на аудио-поток TTS (снимается по PTT вместе с речью)."""
        self.bus.subscribe(self._on_chunk, EventType.TTS_CHUNK, name="audio.play", preemptible=True)
        self.bus.subscribe(self._on_started, EventType.TTS_STARTED, name="audio.log", preemptible=False)

    async def _on_chunk(self, event: AnyEvent) -> None:
        if isinstance(event.payload, AudioChunk):
            self.chunks += 1

    async def _on_started(self, event: AnyEvent) -> None:
        if isinstance(event.payload, TextRequest):
            log.info("🔊 Говорит стример: «{}»", event.payload.text)


def build_modules(bus: EventBus, settings: Settings) -> list[Module]:
    """Собирает набор модулей под выбранный режим."""
    if settings.demo_mode:
        return [
            MockTwitchModule(bus, settings.twitch, interval_s=1.2),
            MockPushToTalkModule(bus, settings.ptt),
            MockLLMModule(bus, settings.llm),
            MockTTSModule(bus, settings.tts),
        ]
    memory = MemoryModule(
        bus,
        settings.memory,
        session=settings.twitch.channel or "local",
    )
    return [
        TwitchChatModule(bus, settings.twitch),
        PushToTalkModule(bus, settings.ptt, settings.stt),
        memory,
        LLMModule(bus, settings.llm, memory=memory),
        TTSModule(bus, settings.tts),
    ]


class ChatEcho:
    """Печатает входящий чат в консоль — проверка Twitch-модуля без LLM и TTS."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.count = 0

    def register(self) -> None:
        """Подписывается на чат и алерты."""
        self.bus.subscribe(
            self._on_event,
            (
                EventType.TWITCH_CHAT,
                EventType.TWITCH_DONATION,
                EventType.TWITCH_SUBSCRIPTION,
                EventType.TWITCH_RAID,
            ),
            name="chat.echo",
            preemptible=False,
        )

    async def _on_event(self, event: AnyEvent) -> None:
        self.count += 1
        payload = event.payload
        if isinstance(payload, ChatMessage):
            badge = "MOD " if payload.is_moderator else ("SUB " if payload.is_subscriber else "")
            log.info("[P{}] {}{}: {}", int(event.priority), badge, payload.author, payload.text)
        elif isinstance(payload, Donation):
            log.info(
                "[P{}] 💰 {} — {:.0f} {}: {}",
                int(event.priority),
                payload.author,
                payload.amount,
                payload.currency,
                payload.message,
            )
        elif isinstance(payload, TwitchSubscription):
            gift = f" (подарил {payload.gifted_by})" if payload.gifted_by else ""
            log.info("[P{}] ⭐ подписка: {}{}", int(event.priority), payload.author, gift)
        elif isinstance(payload, Raid):
            log.info("[P{}] 🚀 рейд от {}: {} зрителей", int(event.priority), payload.author, payload.viewers)


async def run_twitch_only(bus: EventBus, settings: Settings, stop: asyncio.Event) -> int:
    """Читает живой чат и печатает его в консоль. Ctrl+C для остановки."""
    echo = ChatEcho(bus)
    echo.register()

    twitch = TwitchChatModule(bus, settings.twitch)
    try:
        await twitch.start()
    except RuntimeError as exc:
        log.error("{}", exc)
        return 78

    if await twitch.wait_until_ready(timeout_s=20):
        log.info("Подключение установлено. Пишите в чат — сообщения появятся здесь.")
    else:
        log.warning("Подключение за 20с не установлено, супервизор продолжает попытки")

    try:
        await stop.wait()
    finally:
        await twitch.stop()
        log.info("Получено сообщений: {}, переподключений: {}", echo.count, twitch.reconnects)
    return 0


class HostEcho:
    """Печатает реплики хоста — проверка PTT без LLM и TTS."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.presses = 0
        self.utterances = 0

    def register(self) -> None:
        """Подписывается на нажатия и расшифрованный текст."""
        self.bus.subscribe(self._on_press, EventType.PTT_PRESSED, name="ptt.echo", preemptible=False)
        self.bus.subscribe(self._on_speech, EventType.HOST_SPEECH, name="host.echo", preemptible=False)

    async def _on_press(self, _event: AnyEvent) -> None:
        self.presses += 1
        log.info("[P1] PTT нажат — генерация LLM/TTS прервана")

    async def _on_speech(self, event: AnyEvent) -> None:
        self.utterances += 1
        payload = event.payload
        if isinstance(payload, HostSpeech):
            log.info("[P{}] Хост: «{}»", int(event.priority), payload.text)


class TtsRequestEcho:
    """Печатает куски, которые LLM отдал бы в TTS — видно нарезку потока."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.clauses = 0
        self.replies = 0
        self._index = 0
        self._correlation: str | None = None

    def register(self) -> None:
        """Лог не прерывается по PTT: иначе не видно, на каком куске оборвалось."""
        self.bus.subscribe(self._on_clause, EventType.TTS_REQUEST, name="tts.echo", preemptible=False)
        self.bus.subscribe(self._on_done, EventType.LLM_COMPLETED, name="llm.done", preemptible=False)
        self.bus.subscribe(self._on_preempt, EventType.PTT_PRESSED, name="tts.echo.preempt", preemptible=False)

    async def _on_clause(self, event: AnyEvent) -> None:
        payload = event.payload
        if not isinstance(payload, TextRequest):
            return
        if event.correlation_id != self._correlation:
            self._correlation = event.correlation_id
            self._index = 0
        self._index += 1
        self.clauses += 1
        log.info("🔊 TTS_REQUEST [{}] «{}»", self._index, payload.text)

    async def _on_done(self, _event: AnyEvent) -> None:
        self.replies += 1
        log.info("—— реплика завершена ({} фраз) ——", self._index)
        self._index = 0
        self._correlation = None

    async def _on_preempt(self, _event: AnyEvent) -> None:
        if self._index:
            log.info("—— генерация прервана после {} фраз ——", self._index)
        self._index = 0
        self._correlation = None


async def run_llm_only(bus: EventBus, settings: Settings, stop: asyncio.Event) -> int:
    """Живые Twitch, PTT и LLM. Синтез не поднимается — фразы печатаются в консоль."""
    chat = ChatEcho(bus)
    chat.register()
    host = HostEcho(bus)
    host.register()
    tts_echo = TtsRequestEcho(bus)
    tts_echo.register()

    modules: list[Module] = [
        TwitchChatModule(bus, settings.twitch),
        PushToTalkModule(bus, settings.ptt, settings.stt),
        LLMModule(bus, settings.llm),
    ]
    started: list[Module] = []
    exit_code = 0

    try:
        for module in modules:
            try:
                await module.start()
                started.append(module)
            except RuntimeError as exc:
                log.error("{}", exc)
                return 78 if isinstance(module, TwitchChatModule) else 1
            except Exception as exc:
                log.error("Модуль '{}' не поднялся: {}", module.name, exc)
                return 1

        twitch = next(m for m in started if isinstance(m, TwitchChatModule))
        if await twitch.wait_until_ready(timeout_s=20):
            log.info("Twitch подключён. Пишите в чат или говорите, удерживая '{}'.", settings.ptt.hotkey)
        else:
            log.warning("Twitch за 20с не подключился, супервизор продолжает попытки")
        log.info(
            "LLM: {} @ {} — ниже будут куски TTS_REQUEST по мере генерации.",
            settings.llm.model,
            settings.llm.base_url,
        )
        await stop.wait()
    except KeyboardInterrupt:
        log.info("Прерывание с клавиатуры")
    finally:
        for module in started:
            with contextlib.suppress(Exception):
                await module.stop()
        log.info(
            "Чат: {}, реплик хоста: {}, фраз TTS_REQUEST: {}, законченных ответов: {}",
            chat.count,
            host.utterances,
            tts_echo.clauses,
            tts_echo.replies,
        )
    return exit_code


async def run_ptt_only(bus: EventBus, settings: Settings, stop: asyncio.Event) -> int:
    """Живой микрофон и Whisper. Ctrl+C для остановки."""
    echo = HostEcho(bus)
    echo.register()

    ptt = PushToTalkModule(bus, settings.ptt, settings.stt)
    try:
        await ptt.start()
    except Exception as exc:
        log.error("PTT не поднялся: {}", exc)
        return 1

    log.info(
        "Говорите, удерживая '{}'. Отпускание запускает STT '{}' ({}).",
        settings.ptt.hotkey,
        settings.stt.model,
        ptt.stt_device,
    )
    try:
        await stop.wait()
    finally:
        await ptt.stop()
        log.info("Нажатий: {}, распознано реплик: {}", echo.presses, echo.utterances)
    return 0


async def run_demo(bus: EventBus, modules: list[Module], duration_s: float) -> None:
    """Сценарий проверки приоритетов: чат работает, PTT его перебивает."""
    ptt = next(m for m in modules if isinstance(m, MockPushToTalkModule))

    log.info("=== ДЕМО: {} с. Хост вмешается на 5-й и 11-й секунде ===", duration_s)
    await asyncio.sleep(min(5.0, duration_s))
    log.info("--- PTT нажат: генерация должна оборваться немедленно ---")
    await ptt.say("Секунду, я сам отвечу на этот вопрос.")

    if duration_s > 11:
        await asyncio.sleep(6.0)
        log.info("--- PTT нажат повторно ---")
        await ptt.say("И ещё добавлю: железо у нас RTX 4080.")
        await asyncio.sleep(duration_s - 11)
    else:
        await asyncio.sleep(max(0.0, duration_s - 5))

    log.info("=== Метрики шины: {} ===", bus.stats.as_dict())


async def amain(args: CliArgs) -> int:
    """Поднимает шину и модули, ждёт остановки."""
    try:
        settings = load_settings(demo_mode=args.demo, log_level=args.log_level.upper())
    except ValidationError as exc:
        setup_logging(args.log_level.upper())
        for error in exc.errors():
            log.error("Конфигурация: {}", error["msg"])
        return 78  # EX_CONFIG

    setup_logging(settings.log_level, log_file=settings.log_file)
    ensure_windows_cuda_dlls()

    log.info("{} v0 запускается (demo={})", settings.app_name, settings.demo_mode)
    log.info(
        "Бюджет VRAM: запрошено {} / {} ГБ — {}",
        settings.vram_requested_gb,
        settings.vram.total_budget_gb,
        settings.vram_breakdown_gb,
    )

    bus = EventBus(
        max_queue_size=settings.event_bus.max_queue_size,
        handler_timeout_s=settings.event_bus.handler_timeout_s,
    )

    stop = asyncio.Event()
    _install_signal_handlers(stop)

    if args.twitch_only:
        await bus.start()
        try:
            return await run_twitch_only(bus, settings, stop)
        finally:
            await bus.stop()

    if args.ptt_only:
        await bus.start()
        try:
            return await run_ptt_only(bus, settings, stop)
        finally:
            await bus.stop()

    if args.llm_only:
        await bus.start()
        try:
            return await run_llm_only(bus, settings, stop)
        finally:
            await bus.stop()

    sink = ConsoleAudioSink(bus)
    sink.register()

    modules = build_modules(bus, settings)
    started: list[Module] = []
    exit_code = 0

    await bus.start()
    try:
        for module in modules:
            try:
                await module.start()
                started.append(module)
            except NotImplementedError as exc:
                log.error("Модуль '{}' ещё не реализован: {}", module.name, exc)
                log.error("Запустите демо-режим: python main.py --demo")
                return 2
            except Exception as exc:
                log.error("Модуль '{}' не поднялся: {}", module.name, exc)
                return 1

        if settings.demo_mode:
            demo = asyncio.create_task(run_demo(bus, started, args.duration), name="demo")
            done, _ = await asyncio.wait(
                {demo, asyncio.create_task(stop.wait(), name="stop")},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    task.result()
        else:
            log.info("Готов. Ctrl+C для остановки. Полный конвейер: Twitch + PTT + STT + LLM + TTS + Memory.")
            await stop.wait()
    except KeyboardInterrupt:
        log.info("Прерывание с клавиатуры")
    finally:
        # Порядок остановки = порядок запуска: сначала источники событий
        # (Twitch, PTT), потом потребители (LLM, TTS), иначе шина продолжает
        # наполняться, пока конвейер уже разобран.
        for module in started:
            with contextlib.suppress(Exception):
                await module.stop()
        await bus.stop()
        log.info("Аудио-фрагментов проиграно: {}", sink.chunks)

    return exit_code


def _install_signal_handlers(stop: asyncio.Event) -> None:
    """Ставит обработчики SIGINT/SIGTERM там, где loop это поддерживает."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, AttributeError, ValueError):
            loop.add_signal_handler(sig, stop.set)


def _use_fast_event_loop() -> None:
    """Включает winloop/uvloop, если они установлены (ниже задержка таймеров)."""
    with contextlib.suppress(ImportError):
        import sys

        if sys.platform == "win32":
            import winloop

            winloop.install()
        else:
            import uvloop

            uvloop.install()


def build_parser() -> argparse.ArgumentParser:
    """Собирает парсер CLI. Флаги отладки регистрируются здесь одним блоком."""
    parser = argparse.ArgumentParser(prog="ai-streamer", description="Локальный ИИ-стример для Twitch")
    parser.add_argument("--demo", action="store_true", dest="demo", help="моки вместо Twitch/LLM/TTS")
    debug = parser.add_argument_group("отладка модулей")
    debug.add_argument(
        "--twitch-only",
        action="store_true",
        dest="twitch_only",
        default=False,
        help="только чтение живого чата с выводом в консоль (без LLM/TTS/PTT)",
    )
    debug.add_argument(
        "--ptt-only",
        action="store_true",
        dest="ptt_only",
        default=False,
        help="живой PTT + STT с выводом в консоль (без Twitch/LLM/TTS)",
    )
    debug.add_argument(
        "--llm-only",
        action="store_true",
        dest="llm_only",
        default=False,
        help="живой Twitch + PTT + LLM; куски TTS_REQUEST в консоль (без синтеза речи)",
    )
    parser.add_argument("--duration", type=float, default=20.0, dest="duration", help="длительность демо, с")
    parser.add_argument("--log-level", default="INFO", dest="log_level", help="DEBUG/INFO/WARNING/ERROR")
    return parser


def parse_args(argv: list[str] | None = None) -> CliArgs:
    """Разбирает аргументы командной строки."""
    return CliArgs.from_namespace(build_parser().parse_args(argv))


def main() -> int:
    """Синхронная обёртка для консольного запуска."""
    args = parse_args()
    _use_fast_event_loop()
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
