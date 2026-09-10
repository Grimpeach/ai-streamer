"""Сборка системного промпта: шаблон и имя приходят из настроек (.env)."""

from __future__ import annotations


def render_system_prompt(name: str, template: str) -> str:
    """Подставляет ``{name}`` в шаблон. Пустой шаблон — минимальная заглушка."""
    who = name.strip()
    body = template.strip()
    if not body:
        return f"You are {who}." if who else ""
    return body.replace("{name}", who) if who else body
