"""Synthetic arithmetic task: generated locally, so mock runs and tests need no download."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any

from workforce.tasks.base import Item, Score, Task
from workforce.tasks.registry import register_task

COMPUTE_PREFIX = "Compute: "


@register_task("toy_arithmetic")
class ToyArithmeticTask(Task):
    name = "toy_arithmetic"

    def load_items(self, n: int | None, seed: int) -> list[Item]:
        rng = random.Random(seed)
        items = []
        for index in range(n or 20):
            a, b, c = rng.randint(2, 40), rng.randint(2, 40), rng.randint(2, 9)
            op = rng.choice(["+", "-"])
            expression = f"({a} {op} {b}) * {c}"
            value = (a + b if op == "+" else a - b) * c
            items.append(
                Item(self.name, f"toy-{index}", {"expression": expression, "value": value})
            )
        return items

    def problem_prompt(self, item: Item) -> str:
        return f"{COMPUTE_PREFIX}{item.data['expression']}"

    def answer_format(self) -> str:
        return "After your reasoning, give the integer result:\n<answer>integer</answer>"

    def parse_answer(self, answer_text: str | None, full_text: str, item: Item) -> int | None:
        match = re.search(r"-?\d+", answer_text or "")
        return int(match.group()) if match else None

    async def score(
        self, item: Item, answer: Any | None, evidence: Sequence[int] | None = None
    ) -> Score:
        correct = answer is not None and int(answer) == item.data["value"]
        return Score(correct=correct, metrics={"accuracy": float(correct)})
