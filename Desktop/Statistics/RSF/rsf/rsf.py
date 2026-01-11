import math
import numpy as np


def _unique_deaths(times, events):
    return len(np.unique(times[events == 1]))


def nelson_aalen(times, events):
    order = np.argsort(times)
    times_sorted = times[order]
    events_sorted = events[order]

    event_times = np.unique(times_sorted[events_sorted == 1])
    if len(event_times) == 0:
        return np.array([]), np.array([])

    chf = []
    cumulative = 0.0
    for t in event_times:
        at_risk = np.sum(times_sorted >= t)
        deaths = np.sum((times_sorted == t) & (events_sorted == 1))
        if at_risk > 0:
            cumulative += deaths / at_risk
        chf.append(cumulative)
    return event_times, np.array(chf)


def logrank_statistic(times_left, events_left, times_right, events_right):
    event_times = np.unique(
        np.concatenate([times_left[events_left == 1], times_right[events_right == 1]])
    )
    if len(event_times) == 0:
        return 0.0

    numerator = 0.0
    variance = 0.0
    for t in event_times:
        n1 = np.sum(times_left >= t)
        n2 = np.sum(times_right >= t)
        n = n1 + n2
        if n == 0:
            continue
        o1 = np.sum((times_left == t) & (events_left == 1))
        o2 = np.sum((times_right == t) & (events_right == 1))
        o = o1 + o2
        if o == 0:
            continue
        e1 = (n1 / n) * o
        numerator += (o1 - e1)
        if n > 1:
            variance += (n1 * n2 * o * (n - o)) / (n * n * (n - 1))

    if variance == 0.0:
        return 0.0
    return abs(numerator) / math.sqrt(variance)


def _candidate_thresholds(values, max_candidates=10):
    uniq = np.unique(values)
    if len(uniq) <= 1:
        return []
    if len(uniq) <= max_candidates:
        return (uniq[:-1] + uniq[1:]) / 2.0
    quantiles = np.linspace(0.1, 0.9, max_candidates)
    return np.quantile(values, quantiles)


class SurvivalTree:
    def __init__(self, min_unique_deaths=1, mtry=None, max_depth=None, rng=None):
        self.min_unique_deaths = min_unique_deaths
        self.mtry = mtry
        self.max_depth = max_depth
        self.rng = rng or np.random.default_rng()
        self.root = None

    def fit(self, X, times, events):
        self.root = self._grow(X, times, events, depth=0)
        return self

    def _grow(self, X, times, events, depth):
        node = _TreeNode()
        node.times = times
        node.events = events
        node.event_times, node.chf = nelson_aalen(times, events)

        if _unique_deaths(times, events) <= self.min_unique_deaths:
            node.is_leaf = True
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            return node

        n_features = X.shape[1]
        mtry = self.mtry or int(math.sqrt(n_features))
        feature_indices = self.rng.choice(n_features, size=min(mtry, n_features), replace=False)

        best_feature = None
        best_threshold = None
        best_score = -1.0
        best_split = None

        for feature in feature_indices:
            values = X[:, feature]
            thresholds = _candidate_thresholds(values)
            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask):
                    continue
                times_left = times[left_mask]
                events_left = events[left_mask]
                times_right = times[right_mask]
                events_right = events[right_mask]
                if _unique_deaths(times_left, events_left) == 0 or _unique_deaths(times_right, events_right) == 0:
                    continue
                score = logrank_statistic(times_left, events_left, times_right, events_right)
                if score > best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
                    best_split = (left_mask, right_mask)

        if best_split is None:
            node.is_leaf = True
            return node

        node.is_leaf = False
        node.feature = best_feature
        node.threshold = best_threshold
        left_mask, right_mask = best_split
        node.left = self._grow(X[left_mask], times[left_mask], events[left_mask], depth + 1)
        node.right = self._grow(X[right_mask], times[right_mask], events[right_mask], depth + 1)
        return node

    def predict_chf(self, X, eval_times):
        chf_values = np.zeros((X.shape[0], len(eval_times)))
        for i, row in enumerate(X):
            node = self.root
            while not node.is_leaf:
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            chf_values[i] = _step_function(node.event_times, node.chf, eval_times)
        return chf_values


class RandomSurvivalForest:
    def __init__(self, n_trees=200, min_unique_deaths=1, mtry=None, max_depth=None, rng=None):
        self.n_trees = n_trees
        self.min_unique_deaths = min_unique_deaths
        self.mtry = mtry
        self.max_depth = max_depth
        self.rng = rng or np.random.default_rng()
        self.trees = []

    def fit(self, X, times, events):
        n_samples = X.shape[0]
        self.trees = []
        for _ in range(self.n_trees):
            sample_indices = self.rng.integers(0, n_samples, size=n_samples)
            tree = SurvivalTree(
                min_unique_deaths=self.min_unique_deaths,
                mtry=self.mtry,
                max_depth=self.max_depth,
                rng=self.rng,
            )
            tree.fit(X[sample_indices], times[sample_indices], events[sample_indices])
            self.trees.append(tree)
        return self

    def predict_chf(self, X, eval_times):
        if not self.trees:
            raise ValueError("Model is not fitted.")
        chf_sum = np.zeros((X.shape[0], len(eval_times)))
        for tree in self.trees:
            chf_sum += tree.predict_chf(X, eval_times)
        return chf_sum / len(self.trees)

    def predict_survival(self, X, eval_times):
        chf = self.predict_chf(X, eval_times)
        return np.exp(-chf)


def _step_function(x, y, eval_times):
    if len(x) == 0:
        return np.zeros_like(eval_times, dtype=float)
    indices = np.searchsorted(x, eval_times, side="right") - 1
    indices = np.clip(indices, 0, len(y) - 1)
    values = y[indices]
    values[eval_times < x[0]] = 0.0
    return values


class _TreeNode:
    __slots__ = [
        "is_leaf",
        "feature",
        "threshold",
        "left",
        "right",
        "times",
        "events",
        "event_times",
        "chf",
    ]

    def __init__(self):
        self.is_leaf = True
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.times = None
        self.events = None
        self.event_times = None
        self.chf = None
