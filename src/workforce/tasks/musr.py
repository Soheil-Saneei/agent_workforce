"""MuSR: long-narrative soft reasoning (murder mysteries, object placements, team allocation).

Standard metric: multiple-choice accuracy. Items are sampled evenly across the three domains.
"""

from __future__ import annotations

import ast
import csv
import re
import sys
from collections.abc import Sequence
from typing import Any

from workforce.tasks.base import Item, Score, Task, sample_items
from workforce.tasks.download import RemoteFile, fetch
from workforce.tasks.registry import register_task
from workforce.tasks.text_metrics import normalize_answer

_REVISION = "7c365b439a222150f317764d4f16ae6c96d7d94a"
_BASE = f"https://huggingface.co/datasets/TAUR-Lab/MuSR/resolve/{_REVISION}"

DOMAIN_FILES = {
    "murder_mysteries": RemoteFile(
        f"{_BASE}/murder_mystery.csv",
        "ce19b6dc0b953f698e79528c72705804bc97e772a4303734808971f63ce233a7",
        "musr_murder_mystery.csv",
    ),
    "object_placements": RemoteFile(
        f"{_BASE}/object_placements.csv",
        "98cd17d2c9ea53664e274365e901c90dfcaa40d17547dfbd369f1cd26fd2a81c",
        "musr_object_placements.csv",
    ),
    "team_allocation": RemoteFile(
        f"{_BASE}/team_allocation.csv",
        "ddbef3bb61857a7cc3b73db62fac4605679e1b7ff4357d57fdfcf91f7caaa1f8",
        "musr_team_allocation.csv",
    ),
}


@register_task("musr")
class MuSRTask(Task):
    name = "musr"

    def load_items(self, n: int | None, seed: int) -> list[Item]:
        csv.field_size_limit(sys.maxsize)
        domains = self.options.get("domains", list(DOMAIN_FILES))
        per_domain: list[list[Item]] = []
        for domain in domains:
            with fetch(DOMAIN_FILES[domain]).open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            per_domain.append(
                [
                    Item(
                        benchmark=self.name,
                        item_id=f"{domain}-{index}",
                        data={
                            "domain": domain,
                            "narrative": row["narrative"],
                            "question": row["question"],
                            "choices": list(ast.literal_eval(row["choices"])),
                            "answer_index": int(row["answer_index"]),
                        },
                    )
                    for index, row in enumerate(rows)
                ]
            )
        if n is None:
            return [item for items in per_domain for item in items]
        # Even split across domains; the first domains absorb the remainder.
        quotas = [n // len(per_domain) + (i < n % len(per_domain)) for i in range(len(per_domain))]
        sampled = []
        for offset, (items, quota) in enumerate(zip(per_domain, quotas, strict=True)):
            sampled.extend(sample_items(items, quota, seed + offset))
        return sampled

    def problem_prompt(self, item: Item) -> str:
        choices = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(item.data["choices"]))
        return (
            f"{item.data['narrative']}\n\nQuestion: {item.data['question']}\n\nChoices:\n{choices}"
        )

    def answer_format(self) -> str:
        return (
            "After your reasoning, give the number of the single best choice:\n"
            "<answer>choice number</answer>"
        )

    def parse_answer(self, answer_text: str | None, full_text: str, item: Item) -> int | None:
        choices = item.data["choices"]
        if answer_text is None:
            match = re.findall(r"(?i)answer\s*(?:is|:)?\s*\(?(\d+)\)?", full_text)
            answer_text = match[-1] if match else None
        if not answer_text:
            return None
        # "2", "(2) Ana", and "choice 2" (models sometimes echo the placeholder's wording).
        number = re.match(r"\s*(?:(?:choice|option)\s*#?\s*)?\(?(\d+)\)?", answer_text, re.I)
        if number and 1 <= int(number.group(1)) <= len(choices):
            return int(number.group(1))
        normalized = normalize_answer(answer_text)
        for i, choice in enumerate(choices):
            if normalize_answer(choice) == normalized:
                return i + 1
        return None

    def display_answer(self, answer: Any | None) -> str:
        return "" if answer is None else f"choice {answer}"

    async def score(
        self, item: Item, answer: Any | None, evidence: Sequence[int] | None = None
    ) -> Score:
        correct = answer is not None and int(answer) == item.data["answer_index"] + 1
        return Score(correct=correct, metrics={"accuracy": float(correct)})
