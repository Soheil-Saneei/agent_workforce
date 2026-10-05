"""MuSiQue-Answerable (dev split): multi-hop QA over 20 paragraphs, 2-4 supporting.

Official metrics: answer EM / F1 against the answer and its aliases, plus supporting-paragraph
F1. ``correct`` is answer EM; F1 and support metrics are recorded alongside.
"""

from __future__ import annotations

import json
import re
from collections.abc import Hashable, Sequence
from typing import Any

from workforce.tasks.base import Item, Score, Task, sample_items
from workforce.tasks.download import RemoteFile, fetch
from workforce.tasks.registry import register_task
from workforce.tasks.text_metrics import (
    exact_match,
    max_over_golds,
    normalize_answer,
    set_prf,
    token_f1,
)

# Mirror of the official musique_ans_v1.0_dev.jsonl (2,417 items). The official release is on
# Google Drive; this Hugging Face copy is byte-identical to the dgslibisey/MuSiQue mirror.
DEV_FILE = RemoteFile(
    url=(
        "https://huggingface.co/datasets/bdsaglam/musique/resolve/"
        "22873a405dd809893b22ada0b499299fb612d2df/musique_ans_v1.0_dev.jsonl"
    ),
    sha256="15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b",
    filename="musique_ans_v1.0_dev.jsonl",
)


@register_task("musique")
class MuSiQueTask(Task):
    name = "musique"
    asks_for_evidence = True

    def load_items(self, n: int | None, seed: int) -> list[Item]:
        items = []
        with fetch(DEV_FILE).open() as handle:
            for line in handle:
                row = json.loads(line)
                items.append(
                    Item(
                        benchmark=self.name,
                        item_id=row["id"],
                        data={
                            "question": row["question"],
                            "paragraphs": [
                                {"title": p["title"], "text": p["paragraph_text"]}
                                for p in row["paragraphs"]
                            ],
                            "supporting": [
                                i for i, p in enumerate(row["paragraphs"]) if p["is_supporting"]
                            ],
                            "answer": row["answer"],
                            "aliases": row.get("answer_aliases", []),
                        },
                    )
                )
        return sample_items(items, n, seed)

    def problem_prompt(self, item: Item) -> str:
        paragraphs = "\n\n".join(
            f"[P{i + 1}] {p['title']}\n{p['text']}" for i, p in enumerate(item.data["paragraphs"])
        )
        return (
            "Answer the question using only the paragraphs below. The answer requires combining "
            "information from several paragraphs.\n\n"
            f"Paragraphs:\n\n{paragraphs}\n\nQuestion: {item.data['question']}"
        )

    def answer_format(self) -> str:
        return (
            "After your reasoning, list the numbers of the paragraphs your answer relies on, "
            "then give the answer as a short span (a name, date, number or short phrase, "
            "without explanation):\n"
            "<evidence>paragraph numbers, for example 3, 12</evidence>\n"
            "<answer>short answer</answer>"
        )

    def parse_answer(self, answer_text: str | None, full_text: str, item: Item) -> str | None:
        if answer_text is None:
            match = re.findall(r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+)$", full_text)
            answer_text = match[-1] if match else None
        if not answer_text:
            return None
        answer = answer_text.strip().splitlines()[0].strip().strip("*").strip()
        return answer or None

    def answer_key(self, answer: Any, item: Item) -> Hashable:
        return normalize_answer(str(answer))

    def evidence_ids(self, evidence: Sequence[int] | None) -> list[int] | None:
        """Displayed paragraph numbers are 1-based; dataset indices are 0-based."""
        return None if evidence is None else [e - 1 for e in evidence]

    async def score(
        self, item: Item, answer: Any | None, evidence: Sequence[int] | None = None
    ) -> Score:
        golds = [item.data["answer"], *item.data["aliases"]]
        metrics: dict[str, float] = {"em": 0.0, "f1": 0.0}
        if answer is not None:
            metrics["em"] = max_over_golds(exact_match, str(answer), golds)
            metrics["f1"] = max_over_golds(token_f1, str(answer), golds)
        cited = self.evidence_ids(evidence)
        if cited is not None:
            p, r, f = set_prf(cited, item.data["supporting"])
            metrics.update({"support_precision": p, "support_recall": r, "support_f1": f})
        return Score(correct=metrics["em"] == 1.0, metrics=metrics)
