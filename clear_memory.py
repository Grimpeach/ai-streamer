import redis
from qdrant_client import QdrantClient

# 1. Очистка Redis (краткосрочная память/очереди)
try:
    r = redis.Redis.from_url("redis://127.0.0.1:6379/0")
    r.flushdb()
    print("✅ Redis успешно очищен!")
except Exception as e:
    print(f"❌ Ошибка Redis: {e}")

# 2. Очистка Qdrant (долгосрочная векторная память)
try:
    q = QdrantClient(url="http://127.0.0.1:6333")
    collections = q.get_collections().collections
    if not collections:
        print("✅ В Qdrant пусто, удалять нечего.")
    for collection in collections:
        q.delete_collection(collection_name=collection.name)
        print(f"✅ Коллекция '{collection.name}' удалена из Qdrant.")
except Exception as e:
    print(f"❌ Ошибка Qdrant: {e}")