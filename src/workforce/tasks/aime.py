"""AIME: competition math whose answers are integers from 0 to 999, scored by exact match.

Problems and answers are MathArena's curated copies of AIME 2025 and AIME 2026 (30 problems
per year; AIME I is problems 1-15, AIME II is 16-30; CC BY-NC-SA 4.0). MathArena publishes
them only as parquet, so they are fetched as JSON rows from the Hugging Face dataset viewer
API and pinned by checksum: if MathArena edits a set, the download fails loudly instead of
changing items silently. (The math-ai/aime26 JSONL mirror matches AIME 2026 exactly; the
math-ai/aime25 mirror has corrupted LaTeX escapes, so neither mirror is used.)

Both years predate the knowledge cutoff of current models, so contamination is possible; 2026
is the most recent complete set with integer answers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from workforce.tasks.base import Item, Score, Task, sample_items
from workforce.tasks.download import RemoteFile, fetch
from workforce.tasks.registry import register_task

_ROWS = (
    "https://datasets-server.huggingface.co/rows?dataset=MathArena%2Faime_{year}"
    "&config=default&split=train&offset=0&length=100"
)
YEAR_FILES = {
    2025: RemoteFile(
        _ROWS.format(year=2025),
        "98c5c90c82464441e48ccb6bf0af85464febf2cdd064d3d85eb6137b64536908",
        "matharena_aime_2025.json",
    ),
    2026: RemoteFile(
        _ROWS.format(year=2026),
        "3af7cbd02284c8b998a71f7b247be5916439314861cc908ed1ee786acd1b43f0",
        "matharena_aime_2026.json",
    ),
}

_INTEGER = re.compile(r"-?\d+")


def parse_integer_answer(answer_text: str | None, full_text: str) -> int | None:
    """The integer in the answer tag (the whole tag, or else its last integer); without a tag,
    the last ``\\boxed{...}`` or "answer is N" in the response."""
    if answer_text is None:
        boxed = re.findall(r"\\boxed\{\s*(-?\d+)\s*\}", full_text)
        stated = re.findall(r"(?i)answer\s*(?:is|:)\s*\$?\s*(-?\d+)\b", full_text)
        found = boxed or stated
        return int(found[-1]) if found else None
    text = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", answer_text).replace(",", "")
    text = text.strip().strip("$*").strip().rstrip(".").strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    numbers = _INTEGER.findall(text)
    return int(numbers[-1]) if numbers else None


@register_task("aime")
class AIMETask(Task):
    name = "aime"

    def load_items(self, n: int | None, seed: int) -> list[Item]:
        items = []
        for year in self.options.get("years", sorted(YEAR_FILES)):
            rows = json.loads(fetch(YEAR_FILES[year]).read_text())["rows"]
            for entry in sorted(rows, key=lambda r: r["row"]["problem_idx"]):
                row = entry["row"]
                index = int(row["problem_idx"])
                contest, number = ("I", index) if index <= 15 else ("II", index - 15)
                items.append(
                    Item(
                        benchmark=self.name,
                        item_id=f"aime{year}-{contest}-{number:02d}",
                        data={
                            "year": year,
                            "contest": contest,
                            "number": number,
                            "problem": row["problem"],
                            "answer": int(row["answer"]),
                        },
                    )
                )
        return sample_items(items, n, seed)

    def problem_prompt(self, item: Item) -> str:
        return (
            "Solve the following competition mathematics problem. Its answer is an integer "
            f"from 0 to 999.\n\nProblem: {item.data['problem']}"
        )

    def answer_format(self) -> str:
        return (
            "After your reasoning, give the final answer as a single integer from 0 to 999, "
            "digits only:\n<answer>integer</answer>"
        )

    def parse_answer(self, answer_text: str | None, full_text: str, item: Item) -> int | None:
        return parse_integer_answer(answer_text, full_text)

    async def score(
        self, item: Item, answer: Any | None, evidence: Sequence[int] | None = None
    ) -> Score:
        correct = answer is not None and int(answer) == item.data["answer"]
        return Score(correct=correct, metrics={"accuracy": float(correct)})
