"""Common benchmark interface."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass(frozen=True)
class Item:
    benchmark: str
    item_id: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"benchmark": self.benchmark, "item_id": self.item_id, "data": self.data}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Item:
        return cls(benchmark=d["benchmark"], item_id=d["item_id"], data=d["data"])


@dataclass(frozen=True)
class Score:
    correct: bool
    metrics: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"correct": self.correct, "metrics": self.metrics, "detail": self.detail}


class Task(ABC):
    """A benchmark: item loading, prompt rendering, answer parsing, equivalence and scoring.

    Answers are plain JSON values (str or int) so trial records stay self-contained.
    """

    name: ClassVar[str]
    asks_for_evidence: ClassVar[bool] = False

    def __init__(self, **options: Any):
        self.options = options

    @abstractmethod
    def load_items(self, n: int | None, seed: int) -> list[Item]:
        """Deterministically sample ``n`` items (all items when ``n`` is None)."""

    @abstractmethod
    def problem_prompt(self, item: Item) -> str:
        """The problem statement and any context the agents may use."""

    @abstractmethod
    def answer_format(self) -> str:
        """Instructions for what goes inside ``<answer>`` (and ``<evidence>`` if used)."""

    @abstractmethod
    def parse_answer(self, answer_text: str | None, full_text: str, item: Item) -> Any | None:
        """Extract the answer from the tagged answer (or the whole response as a fallback)."""

    @abstractmethod
    async def score(
        self, item: Item, answer: Any | None, evidence: Sequence[int] | None = None
    ) -> Score:
        """Score one answer with the benchmark's official or standard metric."""

    def answer_key(self, answer: Any, item: Item) -> Hashable:
        """Equivalence key used for voting: answers with equal keys are the same vote."""
        return answer

    async def equivalence_keys(
        self, item: Item, answers: Sequence[Any | None]
    ) -> list[Hashable | None]:
        return [None if a is None else self.answer_key(a, item) for a in answers]

    def display_answer(self, answer: Any | None) -> str:
        return "" if answer is None else str(answer)

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release resources (for example sandbox workers)."""


def sample_items(items: Sequence[Item], n: int | None, seed: int) -> list[Item]:
    """Seeded sample without replacement, returned in dataset order."""
    if n is None or n >= len(items):
        return list(items)
    picked = sorted(random.Random(seed).sample(range(len(items)), n))
    return [items[i] for i in picked]
