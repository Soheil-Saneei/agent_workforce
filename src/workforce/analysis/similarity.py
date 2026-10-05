"""Lexical similarity of agent texts: TF-IDF cosine.

This is a lexical proxy for "semantic similarity of solutions"; embedding-based similarity
is listed as deferred in the README.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import combinations

_TOKEN = re.compile(r"[a-z0-9]+")
_TAG = re.compile(r"</?(answer|confidence|evidence|critique)>", re.IGNORECASE)
STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have he her
    his i if in into is it its may might more most must no not of on or our she should so
    such than that the their them then there these they this those to was we were what when
    which who will with would you your""".split()  # noqa: SIM905
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(_TAG.sub(" ", text.lower())) if t not in STOPWORDS]


class TfidfModel:
    def __init__(self, corpus: Iterable[str]):
        documents = [set(tokenize(text)) for text in corpus]
        self.n_docs = len(documents)
        df: Counter[str] = Counter()
        for doc in documents:
            df.update(doc)
        self.idf = {t: math.log((1 + self.n_docs) / (1 + c)) + 1 for t, c in df.items()}

    def vector(self, text: str) -> dict[str, float]:
        counts = Counter(tokenize(text))
        default = math.log(1 + self.n_docs) + 1
        return {t: c * self.idf.get(t, default) for t, c in counts.items()}

    def cosine(self, a: str, b: str) -> float | None:
        va, vb = self.vector(a), self.vector(b)
        na = math.sqrt(sum(v * v for v in va.values()))
        nb = math.sqrt(sum(v * v for v in vb.values()))
        if na == 0 or nb == 0:
            return None
        return sum(v * vb.get(t, 0.0) for t, v in va.items()) / (na * nb)

    def mean_pairwise(self, texts: Sequence[str]) -> float | None:
        scores = [s for a, b in combinations(texts, 2) if (s := self.cosine(a, b)) is not None]
        return sum(scores) / len(scores) if scores else None
