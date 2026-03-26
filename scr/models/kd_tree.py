
class KDNode:
    def __init__(self, point, label, axis, left=None, right=None):
        self.point = point
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right


class CustomKDTree_KNN:
    def __init__(self, k=5):
        self.k = k
        self.root = None

    def fit(self, X, y):
        data = list(zip(X, y))
        self.root = self._build_tree(data, depth=0)

    def _build_tree(self, data, depth):
        if not data:
            return None

        k_dim = len(data[0][0])
        axis = depth % k_dim

        # Сортировка для поиска медианы (O(N log N) на уровне)
        data.sort(key=lambda x: x[0][axis])
        median_idx = len(data) // 2

        return KDNode(
            point=data[median_idx][0],
            label=data[median_idx][1],
            axis=axis,
            left=self._build_tree(data[:median_idx], depth + 1),
            right=self._build_tree(data[median_idx + 1:], depth + 1)
        )

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        best_k = [] # Очередь ближайших соседей [(distance, label),...]
        self._search(self.root, x, best_k)

        # Мажоритарное голосование
        labels = [item[1] for item in best_k]
        return max(set(labels), key=labels.count)
        """
        1. Таким голосованием варианты с одинаковым кол-вом меток будут выбираться случайно
        1.а. set ставит элементы по возрастанию из-за чего в таких случаях будет выйгрывать меньшее значение класса
        """

    def _search(self, node, x, best_k):
        if node is None:
            return

        dist = sum((node.point[i] - x[i]) ** 2 for i in range(len(x)))

        # Поддержание ровно k элементов
        if len(best_k) < self.k:
            best_k.append((dist, node.label))
            best_k.sort(key=lambda item: item[0])
        elif dist < best_k[-1][0]:
            best_k[-1] = (dist, node.label)
            best_k.sort(key=lambda item: item[0])

        axis = node.axis
        diff = x[axis] - node.point[axis]

        if diff < 0:
            close_branch = node.left
            far_branch = node.right
        else:
            close_branch = node.right
            far_branch = node.left

        self._search(close_branch, x, best_k)

        # Backtracking: проверяем, пересекает ли сфера гиперплоскость
        if len(best_k) < self.k or (diff ** 2) < best_k[-1][0]:
            self._search(far_branch, x, best_k)
