import math
from custom_hash_table import CustomHashTable

class NaiveBayesClassifier:
    def __init__(self):
        self.class_counts = CustomHashTable()
        self.feature_counts = CustomHashTable()
        self.total_samples = 0
        self.classes = set()
        self.vocab_sizes = {}  # Число уникальных значений на признак для сглаживания Лапласа

    def fit(self, X, y):
        self.total_samples = len(y)
        num_features = len(X[0])

        for i in range(num_features):
            self.vocab_sizes[i] = set()

        for i in range(self.total_samples):
            cls = y[i]
            self.classes.add(cls)
            self.class_counts.increment(f"class_{cls}")

            for j in range(num_features):
                val = X[i][j]
                self.vocab_sizes[j].add(val)
                # key = f"feat_{j}_val_{val}_class_{cls}"
                key = (j, val, cls)
                self.feature_counts.increment(key)

        # Перевод множеств в размеры для O(1) доступа
        for i in range(num_features):
            self.vocab_sizes[i] = len(self.vocab_sizes[i])

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        best_prob = -float("inf")
        best_class = None

        for cls in self.classes:
            # Использование логарифмов для предотвращения арифметического переполнения (underflow)
            prob = math.log(self.class_counts.get(f"class_{cls}") / self.total_samples)

            for j in range(len(x)):
                val = x[j]
                # key = f"feat_{j}_val_{val}_class_{cls}"
                key = (j, val, cls)
                count = self.feature_counts.get(key)
                total_class_feat = self.class_counts.get(f"class_{cls}")
                vocab_size = self.vocab_sizes[j]

                # Аддитивное сглаживание Лапласа
                prob += math.log((count + 1) / (total_class_feat + vocab_size))

            if prob > best_prob:
                best_prob = prob
                best_class = cls

        return best_class
