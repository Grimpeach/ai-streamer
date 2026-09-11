# Спецификация: TTS (Фаза 5)

Реализация: `src/modules/tts/streamer.py`, движок — `engine.py` (Kokoro),
RVC — `rvc.py`, воспроизведение — `player.py` (sounddevice). Настройки —
`TTSSettings` / `RVCSettings` (`TTS__ENGINE`, `TTS__RVC__ENABLED`).

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
| `output_device` | `TTS__OUTPUT_DEVICE` (индекс/имя PortAudio, пусто = default) |
| `rvc.enabled` | `TTS__RVC__ENABLED` (по умолчанию выкл.) |
| `rvc.model_path` | `TTS__RVC__MODEL_PATH` (`.pth`) |
| `rvc.index_path` | `TTS__RVC__INDEX_PATH` (`.index`) |
| `rvc.pitch` | `TTS__RVC__PITCH` (полутона, `f0up_key`) |
| `rvc.device` | `TTS__RVC__DEVICE` (по умолчанию `cuda:0`) |
| `rvc.f0_method` | `TTS__RVC__F0_METHOD` (`rmvpe` / `pm`) |

Официальный Kokoro-82M не имеет русской озвучки. Имя `ru_female_1` из ранних
спеков алиасится на `af_heart`. LLM отвечает по-английски — это совпадает
с голосом. RVC поверх Kokoro даёт тембр персонажа.

## 2. Воспроизведение

Выходной `sounddevice.OutputStream` открывается один раз и живёт до остановки
модуля. Устройство задаётся `TTS__OUTPUT_DEVICE` и передаётся в `SoundPlayer(device=...)`.
Каждый чанк Kokoro (после опционального RVC) кладётся в PCM-буфер
колбэка; публикация `tts.chunk` (PCM 16-bit) остаётся для логов и оверлеев.

Если `TTS__RVC__ENABLED=true`, при старте TTS один раз загружается
`rvc_python.infer.RVCInference` (`.pth` + `.index`) и держится в памяти.
Чанк Kokoro (numpy, 24 кГц) конвертируется синхронным инференсом в отдельном
потоке (`ThreadPoolExecutor` / не event loop). `f0_method` — `rmvpe` или `pm`.
Частота для плеера снова 24 кГц.

## 3. Прерывание

Подписка `tts.reset` на `ptt.pressed` с `preemptible=False`:

1. Взводится `threading.Event` abort и счётчик epoch — текущая фраза не
   воскресает после `abort.clear()`.
2. Очередь невоспроизведённых `TTS_REQUEST` очищается.
3. Колбэк плеера пишет тишину. **Не** вызываются `OutputStream.abort()` и
   `sd.stop()`: на Windows они из чужого потока стопорят и микрофон на ~10 с.
4. Отменяется asyncio-задача озвучки (`CancelledError`).
5. Генератор Kokoro закрывается (`iterator.close()`), worker дожидается
   потока инференса — GPU/CPU свободны до следующего STT. `Future.cancel()`
   уже идущий CUDA-кернел не останавливает.
6. Если RVC ещё считает чанк, результат после abort/смены epoch **отбрасывается**
   и в очередь sounddevice не попадает.

## 4. Проверка

* `tests/test_tts.py` — очередь, abort плеера, RVC до плеера, отброс чанка RVC по PTT.
* `tests/test_pipeline.py` — чат → LLM → mock-TTS и тишина после PTT.

Живая проверка полного конвейера:

```powershell
docker compose up -d
python main.py
```
