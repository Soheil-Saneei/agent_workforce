"""Small, dependency-free statistics helpers."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def bootstrap_ci(
    values: Sequence[float], *, resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float] | None:
    """Percentile bootstrap CI of the mean. ``values`` are per-item means (items are the
    resampling unit, so replications of an item stay together)."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    lo = means[int(alpha / 2 * resamples)]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return lo, hi


def gini(values: Sequence[float]) -> float | None:
    """Gini coefficient of non-negative values (0 = equal shares, -> 1 = one holds all)."""
    if not values or sum(values) == 0:
        return None
    ordered = sorted(values)
    n = len(ordered)
    weighted = sum((i + 1) * v for i, v in enumerate(ordered))
    return (2 * weighted) / (n * sum(ordered)) - (n + 1) / n


def phi(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    """Phi coefficient (Pearson correlation of two binary variables)."""
    n11 = sum(1 for a, b in pairs if a and b)
    n10 = sum(1 for a, b in pairs if a and not b)
    n01 = sum(1 for a, b in pairs if not a and b)
    n00 = sum(1 for a, b in pairs if not a and not b)
    denominator = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if denominator == 0:
        return None
    return (n11 * n00 - n10 * n01) / denominator


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
