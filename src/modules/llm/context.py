"""Краткосрочное окно диалога в памяти процесса.

Redis/Qdrant дополняют это окно через MemoryModule: факты о зрителях
подмешиваются в текущий промпт, а не дублируются здесь.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(slots=True, frozen=True)
class ChatTurn:
    """Одна реплика в формате Chat Completions."""

    role: Role
    content: str


class DialogueContext:
    """Окно последних реплик. Системный промпт хранится отдельно и всегда первый."""

    def __init__(self, *, system_prompt: str, max_messages: int = 20) -> None:
        if max_messages < 2:
            raise ValueError("max_messages должен быть >= 2 (вопрос + ответ)")
        self.system_prompt = system_prompt
        self._turns: deque[ChatTurn] = deque(maxlen=max_messages)

    def __len__(self) -> int:
        return len(self._turns)

    def add_user(self, text: str) -> None:
        """Добавляет реплику зрителя или хоста."""
        cleaned = text.strip()
        if cleaned:
            self._turns.append(ChatTurn("user", cleaned))

    def add_assistant(self, text: str) -> None:
        """Добавляет законченный ответ стримерши. Оборванный ответ сюда не кладут."""
        cleaned = text.strip()
        if cleaned:
            self._turns.append(ChatTurn("assistant", cleaned))

    def messages(self) -> list[dict[str, str]]:
        """Сообщения для API: system + история в хронологическом порядке."""
        packed: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        packed.extend({"role": turn.role, "content": turn.content} for turn in self._turns)
        return packed

    def last_assistant(self) -> str | None:
        """Текст последнего ответа или ``None``."""
        for turn in reversed(self._turns):
            if turn.role == "assistant":
                return turn.content
        return None

    def clear(self) -> None:
        """Стирает историю, системный промпт остаётся."""
        self._turns.clear()
