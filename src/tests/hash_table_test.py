from src.models.custom_hash_table import CustomHashTable

# Создаем экземпляр нашей таблицы с небольшой емкостью
vocab = CustomHashTable(capacity=10)

# Симулируем частотный словарь (Bag of Words) для Наивного Байеса
# Слово "spam" встречается 3 раза
words = ["spam", "ham", "spam", "spam", "eggs"]

# Пытаемся заполнить таблицу
for word in words:
    vocab.increment(word)

# Проверяем, что получилось
print("--- Результаты поиска ---")
print(f"Частота 'spam': {vocab.get('spam')} (Ожидаем: 3)")
print(f"Частота 'ham': {vocab.get('ham')} (Ожидаем: 1)")
print(f"Частота 'eggs': {vocab.get('eggs')} (Ожидаем: 1)")

print("\n--- Внутреннее состояние ---")
print(f"Размер таблицы (vocab.size): {vocab.size} (Ожидаем: 3 уникальных слова)")
print("Занятые ячейки в памяти:")
print([item for item in vocab.table if item is not None])