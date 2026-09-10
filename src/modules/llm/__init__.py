"""Клиент инференса LLM (Qwen 2.5 14B через OpenAI-совместимый сервер)."""

from src.modules.llm.client import LLMModule, MockLLMModule
from src.modules.llm.context import DialogueContext
from src.modules.llm.persona import render_system_prompt

__all__ = [
    "DialogueContext",
    "LLMModule",
    "MockLLMModule",
    "render_system_prompt",
]
