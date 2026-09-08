"""Текстовые утилиты конвейера генерации."""

from __future__ import annotations

import re

#: Логическая пауза для TTS: точка, запятая, восклицательный/вопросительный, многоточие.
_PAUSE = re.compile(r"[,.!?…]+")


class SentenceBuffer:
    """Накапливает токены LLM и отдаёт куски, пригодные для синтеза.

    Нарезка живёт у источника токенов: только там поток упорядочен. Как только
    в буфере появляется пауза (``,`` ``.`` ``!`` ``?``), готовый кусок уходит
    в TTS — так первый звук не ждёт конца всей реплики.
    """

    def __init__(self, min_chars: int = 1) -> None:
        self.min_chars = min_chars
        self._buf = ""

    @property
    def pending(self) -> str:
        """Текст, ещё не отданный в синтез."""
        return self._buf

    def push(self, text: str) -> list[str]:
        """Добавляет фрагмент и возвращает все законченные на паузе куски."""
        if text:
            self._buf += text
        ready: list[str] = []
        search_from = 0
        while True:
            match = _PAUSE.search(self._buf, search_from)
            if match is None:
                break
            clause = self._buf[: match.end()].strip()
            if len(clause) < self.min_chars:
                # Короткая пауза («Так,») ещё не тянет на фразу — ищем следующую.
                search_from = match.end()
                continue
            ready.append(clause)
            self._buf = self._buf[match.end() :].lstrip()
            search_from = 0
        return ready

    def flush(self) -> str | None:
        """Отдаёт остаток буфера (в конце успешной генерации)."""
        current = self._buf.strip()
        self._buf = ""
        return current or None

    def clear(self) -> None:
        """Выбрасывает накопленное — при прерывании по PTT хвост не озвучиваем."""
        self._buf = ""
