"""Проверки конфигурации: плоские имена переменных, нормализация, бюджет VRAM.

Все тесты работают на синтетических значениях и с отключённым ``.env``
(``_env_file=None``), поэтому реальные секреты проекта не читаются.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.config import Settings, TwitchSettings, load_settings


def test_flat_env_names_are_understood(monkeypatch: pytest.MonkeyPatch) -> None:
    """`.env` может использовать короткие имена вместо TWITCH__ACCESS_TOKEN."""
    monkeypatch.setenv("TWITCH_CHANNEL", "some_channel")
    monkeypatch.setenv("TWITCH_TOKEN", "oauth:flat-token")

    settings = load_settings(_env_file=None)

    assert settings.twitch.channel == "some_channel"
    assert settings.twitch.access_token.get_secret_value() == "flat-token"


def test_nested_name_wins_over_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Полное имя настройки приоритетнее короткого алиаса."""
    monkeypatch.setenv("TWITCH_TOKEN", "flat")
    monkeypatch.setenv("TWITCH__ACCESS_TOKEN", "nested")

    settings = load_settings(_env_file=None)

    assert settings.twitch.access_token.get_secret_value() == "nested"


def test_other_sections_have_flat_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Короткие имена работают не только для Twitch."""
    monkeypatch.setenv("REDIS_URL", "redis://example:6379/1")
    monkeypatch.setenv("QDRANT_URL", "http://example:6333")
    monkeypatch.setenv("PTT_HOTKEY", "f8")
    monkeypatch.setenv("STT_MODEL", "tiny")
    monkeypatch.setenv("STT_TASK", "translate")
    monkeypatch.setenv("LLM_MAX_TOKENS", "64")
    monkeypatch.setenv("LLM_CHARACTER_NAME", "Ada")
    monkeypatch.setenv("LLM_SYSTEM_PROMPT", "You are {name}.")
    monkeypatch.setenv("TTS_VOICE", "af_bella")

    settings = load_settings(_env_file=None)

    assert settings.memory.redis_url == "redis://example:6379/1"
    assert settings.memory.qdrant_url == "http://example:6333"
    assert settings.ptt.hotkey == "f8"
    assert settings.stt.model == "tiny"
    assert settings.stt.task == "translate"
    assert settings.llm.max_tokens == 64
    assert settings.llm.character_name == "Ada"
    assert settings.llm.system_prompt == "You are {name}."
    assert settings.tts.voice == "af_bella"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("oauth:abc", "abc"), ("abc", "abc"), ("  oauth:abc  ", "abc")],
)
def test_token_accepts_both_forms(raw: str, expected: str) -> None:
    """Токен принимается и с префиксом oauth:, и без него."""
    assert TwitchSettings(access_token=raw).access_token.get_secret_value() == expected


@pytest.mark.parametrize(("raw", "expected"), [("#Channel", "channel"), (" Chan ", "chan")])
def test_channel_is_normalized(raw: str, expected: str) -> None:
    """Имя канала приводится к нижнему регистру без решётки."""
    assert TwitchSettings(channel=raw).channel == expected


def test_secrets_are_not_printed_in_repr() -> None:
    """Токен не должен утечь в лог через repr настроек."""
    settings = TwitchSettings(access_token="super-secret-token")

    assert "super-secret-token" not in repr(settings)


def test_vram_budget_is_enforced() -> None:
    """Конфигурация, превышающая лимит VRAM, не проходит валидацию."""
    with pytest.raises(ValidationError, match="Бюджет VRAM превышен"):
        Settings(_env_file=None, llm={"vram_gb": 14.0})  # type: ignore[arg-type]


def test_tts_output_device_comes_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """TTS__OUTPUT_DEVICE попадает в SoundPlayer: индекс, имя или пусто."""
    monkeypatch.setenv("TTS__OUTPUT_DEVICE", "7")
    settings = load_settings(_env_file=None)
    assert settings.tts.output_device == 7

    monkeypatch.setenv("TTS__OUTPUT_DEVICE", "CABLE Input")
    settings = load_settings(_env_file=None)
    assert settings.tts.output_device == "CABLE Input"

    monkeypatch.setenv("TTS__OUTPUT_DEVICE", "")
    settings = load_settings(_env_file=None)
    assert settings.tts.output_device is None


def test_rvc_settings_come_from_nested_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """TTS__RVC__* читается вложенной секцией, пути резолвятся от корня репо."""
    monkeypatch.setenv("TTS__RVC__ENABLED", "true")
    monkeypatch.setenv("TTS__RVC__PITCH", "2")
    monkeypatch.setenv("TTS__RVC__DEVICE", "cuda:0")
    monkeypatch.setenv("TTS__RVC__F0_METHOD", "pm")
    monkeypatch.setenv("TTS__RVC__MODEL_PATH", "models/rvc/egirl.pth")

    settings = load_settings(_env_file=None)

    assert settings.tts.rvc.enabled is True
    assert settings.tts.rvc.pitch == 2
    assert settings.tts.rvc.device == "cuda:0"
    assert settings.tts.rvc.f0_method == "pm"
    assert settings.tts.rvc.model_path.as_posix().endswith("models/rvc/egirl.pth")


def test_vram_breakdown_sums_to_requested() -> None:
    """Разбивка по потребителям совпадает с суммарным запросом."""
    settings = load_settings(_env_file=None)

    assert sum(settings.vram_breakdown_gb.values()) == pytest.approx(settings.vram_requested_gb)
    assert settings.vram_requested_gb <= settings.vram.total_budget_gb


def test_llm_persona_multiline_prompt_from_env_file(tmp_path: Path) -> None:
    """Многострочный LLM__SYSTEM_PROMPT из файла .env доходит до настроек целиком."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        'LLM__CHARACTER_NAME=Ada\n'
        'LLM__SYSTEM_PROMPT="You are {name}.\nSpeak English."\n',
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_file)

    assert settings.llm.character_name == "Ada"
    assert settings.llm.system_prompt == "You are {name}.\nSpeak English."
