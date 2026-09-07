"""Парсер CLI: --ptt-only и --twitch-only всегда присутствуют в результате."""

from __future__ import annotations

import argparse

from main import CliArgs, build_parser, parse_args


def test_default_namespace_has_debug_flags() -> None:
    """Без аргументов оба режима выключены, атрибуты существуют."""
    args = parse_args([])
    assert args.twitch_only is False
    assert args.ptt_only is False
    assert args.demo is False


def test_ptt_only_does_not_require_twitch_flag() -> None:
    """--ptt-only больше не падает с AttributeError: twitch_only."""
    args = parse_args(["--ptt-only"])
    assert args.ptt_only is True
    assert args.twitch_only is False


def test_twitch_only_flag() -> None:
    """--twitch-only включает только чтение чата."""
    args = parse_args(["--twitch-only"])
    assert args.twitch_only is True
    assert args.ptt_only is False


def test_incomplete_namespace_does_not_crash() -> None:
    """Даже Namespace без dest-ов даёт безопасные значения по умолчанию."""
    args = CliArgs.from_namespace(argparse.Namespace(demo=True, log_level="DEBUG"))
    assert args.demo is True
    assert args.twitch_only is False
    assert args.ptt_only is False
    assert args.log_level == "DEBUG"


def test_parser_registers_both_dest_names() -> None:
    """Регистрация флагов не должна снова разъехаться при правке help-текста."""
    dests = {action.dest for action in build_parser()._actions}
    assert "twitch_only" in dests
    assert "ptt_only" in dests
