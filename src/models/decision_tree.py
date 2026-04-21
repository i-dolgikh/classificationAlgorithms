class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Индекс признака для разбиения
        self.threshold = threshold  # Пороговое значение
        self.left = left  # Левое поддерево
        self.right = right  # Правое поддерево
        self.value = value  # Значение класса (для листового узла)


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples = len(y)
        num_features = len(X[0])

        # Проверка критериев останова
        if num_samples >= self.min_samples_split and depth < self.max_depth:
            best_split = self._get_best_split(X, y, num_samples, num_features)
            if best_split:
                left_subtree = self._build_tree(best_split["X_left"], best_split["y_left"], depth + 1)
                right_subtree = self._build_tree(best_split["X_right"], best_split["y_right"], depth + 1)
                return TreeNode(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree)

        if not y:  # На случай, если в ветку не попало ни одного объекта
            return None
        # Вычисление значения листа (мажоритарное голосование)
        leaf_value = max(set(y), key=y.count)
        return TreeNode(value=leaf_value)

    def _get_best_split(self, X, y, num_samples, num_features):
        best_split = {}
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            feature_values = [row[feature_index] for row in X]

            # Предварительная сортировка массива (O(N log N)) для быстрого перебора
            sorted_indices = sorted(range(num_samples), key=lambda i: feature_values[i])
            sorted_y = [y[i] for i in sorted_indices]
            sorted_features = [feature_values[i] for i in sorted_indices]

            # Перебор порогов за один линейный проход (O(N))
            for i in range(1, num_samples):
                if sorted_features[i] == sorted_features[i - 1]:
                    continue
                threshold = (sorted_features[i] + sorted_features[i - 1]) / 2

                y_left, y_right = sorted_y[:i], sorted_y[i:]
                curr_info_gain = self._gini_gain(y, y_left, y_right)

                if curr_info_gain > max_info_gain:
                    max_info_gain = curr_info_gain
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "X_left": [X[idx] for idx in sorted_indices[:i]],
                        "X_right": [X[idx] for idx in sorted_indices[i:]],
                        "y_left": y_left,
                        "y_right": y_right
                    }
        return best_split if max_info_gain > 0 else None

    def _gini_gain(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        return self._gini(parent) - (weight_l * self._gini(l_child) + weight_r * self._gini(r_child))

    def _gini(self, y):
        classes = set(y)
        gini = 1
        for cls in classes:
            p_cls = y.count(cls) / len(y)
            gini -= p_cls ** 2
        return gini

    def predict(self, X):
        return [self._predict_single(x, self.root) for x in X]

    def _predict_single(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._predict_single(x, tree.left)
        else:
            return self._predict_single(x, tree.right)
