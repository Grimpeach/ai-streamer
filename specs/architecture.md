# Архитектура ИИ-стримера

## Модули и Распределение памяти (RTX 4080 16GB)

- LLM: Qwen 2.5 14B (vLLM / llama.cpp) ~ 8-9 GB VRAM
- TTS: CosyVoice / Kokoro-TTS (Streaming) ~ 3-4 GB VRAM
- STT: Faster-Whisper (Push-to-Talk) ~ 1 GB VRAM
- VTube Studio / OBS: ~ 2-3 GB VRAM

## Шина событий (Event Bus) и Приоритеты

1. Priority 1 (Interrupt): Push-to-Talk (Микрофон хоста). Мгновенно прерывает TTS и LLM.
2. Priority 2: Twitch Донаты и Подписки.
3. Priority 3: Twitch Чат (обычные сообщения).

