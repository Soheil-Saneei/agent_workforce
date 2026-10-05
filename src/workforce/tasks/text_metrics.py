"""Answer metrics used by the official MuSiQue evaluation (SQuAD 2.0 normalization, EM, F1)."""

from __future__ import annotations

import collections
import re
import string
from collections.abc import Callable, Iterable, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCTUATION = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """Lower-case, strip punctuation and articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCTUATION)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _tokens(s: str) -> list[str]:
    return normalize_answer(s).split() if s else []


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = _tokens(prediction), _tokens(gold)
    if not pred or not ref:
        return float(pred == ref)
    same = sum((collections.Counter(pred) & collections.Counter(ref)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(ref)
    return 2 * precision * recall / (precision + recall)


def max_over_golds(
    metric: Callable[[str, str], float], prediction: str, golds: Iterable[str]
) -> float:
    return max(metric(prediction, gold) for gold in golds)


def set_prf(predicted: Sequence[int], gold: Sequence[int]) -> tuple[float, float, float]:
    """Precision, recall and F1 between two sets of ids (MuSiQue supporting-paragraph metric)."""
    pred, ref = set(predicted), set(gold)
    if not pred and not ref:
        return 1.0, 1.0, 1.0
    if not pred or not ref:
        return 0.0, 0.0, 0.0
    hits = len(pred & ref)
    precision, recall = hits / len(pred), hits / len(ref)
    f1 = 0.0 if hits == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1
