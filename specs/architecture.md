# Архитектура ИИ-стримера

## Модули и Распределение памяти (RTX 4080 16GB)

- LLM: Qwen 2.5 14B (vLLM / llama.cpp) ~ 8-9 GB VRAM. Модуль — [llm.md](llm.md).
- TTS: Kokoro-82M (streaming, sounddevice) ~ 0.5–2.5 GB VRAM. Модуль — [tts.md](tts.md).
- STT: Faster-Whisper large-v3 (PTT, translate→English) ~ 3 GB VRAM
- Memory: Redis + Qdrant (e5-base ~0.4 GB). Модуль — [memory.md](memory.md).
- VTube Studio / OBS: ~ 2-3 GB VRAM

## Шина событий (Event Bus) и Приоритеты

1. Priority 1 (Interrupt): Push-to-Talk (Микрофон хоста). Мгновенно прерывает TTS и LLM.
2. Priority 2: Twitch Донаты и Подписки.
3. Priority 3: Twitch Чат (обычные сообщения).

