import numpy as np


def _hamming_distance(row, data):
    return np.sum(row != data, axis=1)


def relief_rank(X, y, m=None, rng=None):
    """
    Relief algorithm for binary targets and discrete predictors.

    X: numpy array (n_samples, n_features) with discrete values.
    y: numpy array (n_samples,) with values 0/1.
    m: number of iterations; defaults to n_samples.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_samples, n_features = X.shape
    if m is None:
        m = n_samples

    weights = np.zeros(n_features, dtype=float)

    for _ in range(m):
        idx = rng.integers(0, n_samples)
        sample = X[idx]
        target = y[idx]

        same_mask = y == target
        diff_mask = y != target

        # Exclude self from nearest hit
        same_mask[idx] = False

        if not np.any(same_mask) or not np.any(diff_mask):
            continue

        distances_same = _hamming_distance(sample, X[same_mask])
        distances_diff = _hamming_distance(sample, X[diff_mask])

        hit = X[same_mask][np.argmin(distances_same)]
        miss = X[diff_mask][np.argmin(distances_diff)]

        diff_hit = (sample != hit).astype(float)
        diff_miss = (sample != miss).astype(float)

        weights -= diff_hit / m
        weights += diff_miss / m

    return weights
