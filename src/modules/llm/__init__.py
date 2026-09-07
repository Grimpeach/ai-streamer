"""Клиент инференса LLM (Qwen 2.5 14B через OpenAI-совместимый сервер)."""

from src.modules.llm.client import LLMModule, MockLLMModule

__all__ = ["LLMModule", "MockLLMModule"]
