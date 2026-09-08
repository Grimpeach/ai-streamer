"""Парсер CLI: отладочные флаги --twitch-only / --ptt-only / --llm-only всегда на месте."""

from __future__ import annotations

import argparse

from main import CliArgs, TtsRequestEcho, build_parser, parse_args
from src.core.event_bus import EventBus
from src.core.events import Event, EventType, TextRequest


def test_default_namespace_has_debug_flags() -> None:
    """Без аргументов все режимы выключены, атрибуты существуют."""
    args = parse_args([])
    assert args.twitch_only is False
    assert args.ptt_only is False
    assert args.llm_only is False
    assert args.demo is False


def test_ptt_only_does_not_require_twitch_flag() -> None:
    """--ptt-only больше не падает с AttributeError: twitch_only."""
    args = parse_args(["--ptt-only"])
    assert args.ptt_only is True
    assert args.twitch_only is False
    assert args.llm_only is False


def test_twitch_only_flag() -> None:
    """--twitch-only включает только чтение чата."""
    args = parse_args(["--twitch-only"])
    assert args.twitch_only is True
    assert args.ptt_only is False
    assert args.llm_only is False


def test_llm_only_flag() -> None:
    """--llm-only поднимает Twitch+PTT+LLM без синтеза."""
    args = parse_args(["--llm-only"])
    assert args.llm_only is True
    assert args.twitch_only is False
    assert args.ptt_only is False


def test_incomplete_namespace_does_not_crash() -> None:
    """Даже Namespace без dest-ов даёт безопасные значения по умолчанию."""
    args = CliArgs.from_namespace(argparse.Namespace(demo=True, log_level="DEBUG"))
    assert args.demo is True
    assert args.twitch_only is False
    assert args.ptt_only is False
    assert args.llm_only is False
    assert args.log_level == "DEBUG"


def test_parser_registers_debug_dest_names() -> None:
    """Регистрация флагов не должна снова разъехаться при правке help-текста."""
    dests = {action.dest for action in build_parser()._actions}
    assert "twitch_only" in dests
    assert "ptt_only" in dests
    assert "llm_only" in dests


def test_help_mentions_llm_only() -> None:
    """--help показывает режим проверки Фазы 4."""
    text = build_parser().format_help()
    assert "--llm-only" in text
    assert "TTS_REQUEST" in text


async def test_tts_request_echo_captures_clauses() -> None:
    """Зонд --llm-only считает и нумерует куски, не поднимая TTS."""
    bus = EventBus()
    echo = TtsRequestEcho(bus)
    echo.register()
    async with bus:
        await bus.publish(Event(type=EventType.TTS_REQUEST, payload=TextRequest(text="Привет,")))
        await bus.publish(Event(type=EventType.TTS_REQUEST, payload=TextRequest(text="чат!")))
        await bus.publish(Event(type=EventType.LLM_COMPLETED, payload=TextRequest(text="Привет, чат!")))
        await bus.wait_idle()

    assert echo.clauses == 2
    assert echo.replies == 1
