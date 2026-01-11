import numpy as np


def survival_area(times, survival_probs):
    # NumPy 2.0 removed np.trapz; use trapezoid when available.
    if hasattr(np, "trapezoid"):
        return np.trapezoid(survival_probs, times)
    return np.trapz(survival_probs, times)


def concordance_index_by_area(areas, times, events):
    total = 0
    count = 0
    n = len(areas)
    for j in range(n):
        if events[j] != 1:
            continue
        for k in range(n):
            if k == j:
                continue
            if times[k] > times[j] or (events[k] == 0 and times[k] >= times[j]):
                count += 1
                if areas[j] < areas[k]:
                    total += 1
    if count == 0:
        return np.nan
    return total / count
