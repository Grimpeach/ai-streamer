# Спецификация: TTS (Фаза 5)

Реализация: `src/modules/tts/streamer.py`, движок — `engine.py` (Kokoro),
воспроизведение — `player.py` (sounddevice). Настройки — `TTSSettings`
(`TTS__ENGINE`, `TTS__VOICE`, `TTS__LANG_CODE`).

## 1. Движок Kokoro

`kokoro.KPipeline` (~82M, 24 кГц). Инференс в `ThreadPoolExecutor(max_workers=1)`:
модель не потокобезопасна, event loop не должен ждать GPU.

На Windows CTranslate2 (STT) грузит свой `cudnn64_9.dll` раньше PyTorch.
У того DLL нет символа `cudnnGetLibConfig`, и LSTM Kokoro падает
(`Error code 127`). Поэтому `ensure_windows_cuda_dlls()` явно
`LoadLibrary` cuDNN из `torch/lib` до STT, а при создании пайплайна
`torch.backends.cudnn.enabled = False`. Если CUDA всё равно не встаёт —
синтез на CPU (`TTS__`/`fallback_to_cpu`).

| Параметр | Откуда |
|---|---|
| `engine` | `TTS__ENGINE` (`kokoro` / `mock`) |
| `voice` | `TTS__VOICE` (по умолчанию `af_heart`) |
| `lang_code` | `TTS__LANG_CODE` или первая буква голоса (`af_*` → `a`) |
| `device` | `TTS__DEVICE` (`auto` / `cuda` / `cpu`) |

Официальный Kokoro-82M не имеет русской озвучки. Имя `ru_female_1` из ранних
спеков алиасится на `af_heart`. LLM отвечает по-английски — это совпадает
с голосом.

## 2. Воспроизведение

Каждый чанк Kokoro сразу играется через `sounddevice.OutputStream`.
Публикация `tts.chunk` (PCM 16-bit) остаётся для логов и оверлеев.

## 3. Прерывание

Подписка `tts.reset` на `ptt.pressed` с `preemptible=False`:

1. Взводится `threading.Event` abort.
2. `OutputStream.abort()` / `sd.stop()` — звук рвётся сразу, не доигрывая буфер.
3. Отменяется текущая asyncio-задача озвучки.
4. Очередь невоспроизведённых `TTS_REQUEST` очищается.

## 4. Проверка

* `tests/test_tts.py` — очередь, abort плеера, event loop не блокируется.
* `tests/test_pipeline.py` — чат → LLM → mock-TTS и тишина после PTT.

Живая проверка полного конвейера:

```powershell
docker compose up -d
python main.py
```
