# Спецификация: память (Фаза 5)

Реализация: `src/modules/memory/module.py`, Redis — `redis_store.py`,
Qdrant — `qdrant_store.py`. Настройки — `MemorySettings`
(`MEMORY__REDIS_URL`, `MEMORY__QDRANT_URL`, плоские алиасы `REDIS_URL` /
`QDRANT_URL`).

Инфраструктура: `docker compose up -d` (Redis 6379, Qdrant 6333, только localhost).

## 1. Краткосрочная (Redis)

Список `{namespace}:dialogue:{session}` — скользящее окно реплик хоста,
зрителей и стримерши. TTL `short_term_ttl_s` (по умолчанию 3 часа).
Сессия = имя канала Twitch.

## 2. Долгосрочная (Qdrant)

Коллекция `stream_memory`, эмбеддер `intfloat/multilingual-e5-base` (~0.4 ГБ VRAM).
Эмбеддинги считаются в `asyncio.to_thread`. Поиск: сначала по нику зрителя,
если пусто — по теме реплики без фильтра.

В вектор не кладутся фразы короче `min_remember_chars` (шум чата).

## 3. Пайплайн до LLM

1. Событие чата/хоста приходит на шину.
2. `memory.ingest` (`preemptible=False`) пишет Redis/Qdrant.
3. `llm.generate` вызывает `MemoryModule.context_for(event)` и дописывает
   блок `[Memory]` к **текущему** user-сообщению. История `DialogueContext`
   хранит чистый текст без дампа фактов.
4. `llm.completed` → ответ стримерши только в Redis.

Если Redis или Qdrant недоступны, модуль логирует ошибку и работает в
деградированном режиме: стример не падает.

## 4. Проверка

* `tests/test_memory.py` — запись, recall по нику, инъекция в промпт LLM.
