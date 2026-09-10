# Спецификация: LLM (Фаза 4)

Реализация: `src/modules/llm/client.py`, история — `context.py`,
подстановка `{name}` — `persona.py`. Настройки — `LLMSettings`
(`LLM__BASE_URL`, `LLM__MODEL`, `LLM__CHARACTER_NAME`, `LLM__SYSTEM_PROMPT`).

## 1. Клиент

Асинхронный `openai.AsyncOpenAI` к локальному OpenAI-совместимому серверу
(vLLM / llama.cpp). `chat.completions.create(..., stream=True)` — строго
потоковый режим: ждать полный ответ перед TTS нельзя.

| Параметр | Откуда |
|---|---|
| `base_url` | `LLM__BASE_URL` (по умолчанию `http://127.0.0.1:8000/v1`) |
| `model` | `LLM__MODEL` |
| `max_tokens` | `LLM__MAX_TOKENS` (плоский алиас той же настройки) |
| `api_key` | `LLM__API_KEY` (для локального сервера достаточно заглушки) |
| `character_name` | `LLM__CHARACTER_NAME` |
| `system_prompt` | `LLM__SYSTEM_PROMPT` (многострочный, в двойных кавычках) |

## 2. Входящие события

Подписка `llm.generate`: `preemptible=True`, `exclusive=True`.

| Событие | Приоритет | Роль |
|---|---|---|
| `ptt.host_speech` | 1 | Реплика хоста — главный собеседник |
| `twitch.donation` / `subscription` / `raid` | 2 | Короткая благодарность |
| `twitch.chat` | 3 | Ответ зрителю |
| `llm.request` | как у события | Ручной/тестовый промпт |

## 3. Контекст и персона

`DialogueContext` — скользящее окно реплик в памяти процесса (не Redis).
Имя и системный промпт **не хранятся в коде**: их читают из `.env`
(`LLM__CHARACTER_NAME`, `LLM__SYSTEM_PROMPT`; плоские алиасы
`LLM_CHARACTER_NAME` / `LLM_SYSTEM_PROMPT`). В шаблоне `{name}` заменяется
на имя. Файл `.env` в git не попадает; в `.env.example` только пустые ключи.
К каждой реплике хоста/чата дописывается `Respond in English.`: Qwen иначе
зеркалит язык входящего текста, даже при английском системном промпте.

- Реплика пользователя кладётся в историю **до** генерации.
- Ответ ассистента — **только после** успешного завершения стрима.
- При `CancelledError` / `Preempted` незавершённый ответ **не** пишется;
  хвост `SentenceBuffer` выбрасывается (`clear()`, не `flush()`).
  Уже опубликованные законченные куски для TTS не отзываются.

## 4. Паузы для TTS

Токены буферизуются в `SentenceBuffer`. Логическая пауза — `,` `.` `!` `?`
(и многоточие). Каждый готовый кусок публикуется как `tts.request`.
Остаток без паузы уходит в TTS только в конце **успешной** генерации.

## 5. Прерывание

PTT отменяет задачу `llm.generate` (`CancelledError`) и взводит
`PreemptionToken`. Стрим закрывается в `finally`, чтобы сервер освободил KV-кэш.
Обработчик обязан пробросить `CancelledError` наружу — глотать его нельзя.

## 6. Тесты

* `tests/test_llm.py` — стрим, паузы → `tts.request`, история, отмена без записи ответа.
* `tests/test_pipeline.py` — чат → LLM → TTS и сброс очереди TTS по PTT.

Перед генерацией `LLMModule` запрашивает `MemoryModule.context_for` и дописывает
факты о зрителе/теме в текущий user-промпт. Подробности — [memory.md](memory.md).

Живая проверка без синтеза речи (нужны токен Twitch, микрофон/STT и локальный LLM-сервер):

```powershell
python main.py --llm-only
```

В консоль печатаются входящий чат/PTT и каждый `TTS_REQUEST` — так видно, как поток нарезается на фразы. Модуль TTS не запускается.
