"""Клиент инференса LLM (Qwen 2.5 14B через OpenAI-совместимый сервер)."""

from src.modules.llm.client import LLMModule, MockLLMModule
from src.modules.llm.context import DialogueContext
from src.modules.llm.persona import DEFAULT_SYSTEM_PROMPT, render_system_prompt

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DialogueContext",
    "LLMModule",
    "MockLLMModule",
    "render_system_prompt",
]
