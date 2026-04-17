class CustomHashTable:
    def __init__(self, capacity=1024):
        self.capacity = capacity
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity

    def _hash(self, key):
        # hash_val = hash(key)
        hash_val = 5381
        for char in str(key):
            hash_val = ((hash_val << 5) + hash_val) + ord(char)
        return hash_val % self.capacity

    def increment(self, key):
        # Линейное пробирование для разрешения коллизий
        idx = self._hash(key)
        while self.keys[idx] is not None:
            if self.keys[idx] == key:
                self.values[idx] += 1
                return
            idx = (idx + 1) % self.capacity

        self.keys[idx] = key
        self.values[idx] = 1
        self.size += 1

        # Контроль коэффициента заполнения
        if self.size / self.capacity >= 0.7:
            self._resize()

    def get(self, key):
        idx = self._hash(key)
        while self.keys[idx] is not None:
            if self.keys[idx] == key:
                return self.values[idx]
            idx = (idx + 1) % self.capacity
        return 0

    def _resize(self):
        old_keys = self.keys
        old_values = self.values

        self.capacity *= 2
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.size = 0
        for key, val in zip(old_keys, old_values):
            if key is not None:
                # Вставка элементов в новую таблицу
                idx = self._hash(key)
                while self.keys[idx] is not None:
                    idx = (idx + 1) % self.capacity
                self.keys[idx] = key
                self.values[idx] = val
                self.size += 1
