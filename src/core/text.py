"""Текстовые утилиты конвейера генерации."""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?…]+[\s\"')\]]*$")


class SentenceBuffer:
    """Накапливает токены LLM и отдаёт куски, пригодные для синтеза.

    Компромисс между задержкой и качеством: фраза уходит в TTS, как только она
    достаточно длинная и завершена знаком препинания. Слишком короткие куски
    звучат рвано, слишком длинные — задерживают первый звук.
    """

    def __init__(self, min_chars: int = 48) -> None:
        self.min_chars = min_chars
        self._buffer: list[str] = []

    @property
    def pending(self) -> str:
        """Текст, ещё не отданный в синтез."""
        return "".join(self._buffer)

    def push(self, text: str) -> str | None:
        """Добавляет фрагмент; возвращает готовую фразу либо ``None``."""
        self._buffer.append(text)
        current = self.pending
        if len(current.strip()) >= self.min_chars and _SENTENCE_END.search(current.rstrip()):
            self._buffer.clear()
            return current.strip()
        return None

    def flush(self) -> str | None:
        """Отдаёт остаток буфера (в конце генерации)."""
        current = self.pending.strip()
        self._buffer.clear()
        return current or None

    def clear(self) -> None:
        """Выбрасывает накопленное — например, при прерывании по PTT."""
        self._buffer.clear()
