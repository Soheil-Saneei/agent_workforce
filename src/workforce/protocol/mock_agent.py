"""Protocol-aware responder for the mock provider: a fake agent with tunable accuracy.

It answers in the protocol's tagged format, picks plausible answers for each benchmark's
prompt shape (choices, paragraph titles, arithmetic, code stubs) and, in discussion rounds,
sometimes adopts the most common peer answer, so offline runs exercise votes, ties and
convergence. In solo self-checks it sometimes re-solves the problem, so solo trajectories
change answers. Accuracy is set per model with ``params = { accuracy = 0.7 }``.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter

from workforce.models.base import ChatRequest, ModelSpec
from workforce.protocol.parsing import parse_output
from workforce.protocol.prompts import PEER_HEADING, SOLO_CHECK_HEADING
from workforce.tasks.toy import COMPUTE_PREFIX

_PEER_PREFIX = PEER_HEADING.split("{")[0]
_SOLO_PREFIX = SOLO_CHECK_HEADING.split("{")[0]
_VOCABULARY = [
    "consider",
    "the",
    "evidence",
    "first",
    "then",
    "check",
    "each",
    "step",
    "carefully",
    "because",
    "the",
    "second",
    "clue",
    "suggests",
    "another",
    "reading",
    "so",
    "we",
    "compare",
    "candidates",
    "and",
    "verify",
    "the",
    "constraint",
    "against",
    "the",
    "statement",
    "before",
    "deciding",
    "which",
    "option",
    "survives",
    "scrutiny",
    "overall",
]


def _rng(spec: ModelSpec, request: ChatRequest) -> random.Random:
    material = f"{spec.model}|{request.seed}|{request.messages[-1].content}"
    return random.Random(int(hashlib.sha256(material.encode()).hexdigest()[:16], 16))


def _filler(rng: random.Random) -> str:
    return " ".join(rng.choice(_VOCABULARY) for _ in range(rng.randint(15, 60)))


def _fresh_answer(problem: str, rng: random.Random, accuracy: float) -> tuple[str, str]:
    """Return (answer, extra tags) for an initial solution."""
    if "```python" in problem:
        signatures = re.findall(r"^(def\s+\w+\(.*?\).*?:)\s*$", problem, flags=re.MULTILINE)
        signature = signatures[-1] if signatures else "def solution(*args):"
        value = rng.choice(["None", "0", "[]", "True"])
        return f"\n```python\n{signature}\n    return {value}\n```\n", ""
    if "\nChoices:\n" in problem:
        numbers = re.findall(r"^(\d+)\. ", problem.split("\nChoices:\n", 1)[1], re.MULTILINE)
        return rng.choice(numbers or ["1"]), ""
    if COMPUTE_PREFIX in problem:
        expression = problem.split(COMPUTE_PREFIX, 1)[1].splitlines()[0]
        if re.fullmatch(r"[\d\s+\-*()]+", expression):
            value = eval(expression)  # digits and operators only
            if rng.random() >= accuracy:
                value += rng.choice([-2, -1, 1, 2])
            return str(value), ""
    titles = re.findall(r"^\[P(\d+)\] (.+)$", problem, flags=re.MULTILINE)
    if titles:
        number, title = rng.choice(titles)
        return title, f"<evidence>{number}</evidence>\n"
    return "unknown", ""


def default_responder(request: ChatRequest, spec: ModelSpec) -> str:
    rng = _rng(spec, request)
    accuracy = float(spec.params.get("accuracy", 0.7))
    problem = next(m.content for m in request.messages if m.role == "user")
    last = request.messages[-1].content
    confidence = f"<confidence>{rng.randint(30, 95)}</confidence>"
    previous = [m.content for m in request.messages if m.role == "assistant"]
    if not previous:
        answer, extra = _fresh_answer(problem, rng, accuracy)
        return f"{_filler(rng)}\n{extra}<answer>{answer}</answer>\n{confidence}"
    own = parse_output(previous[-1]).answer_text or "unknown"
    if last.startswith(_SOLO_PREFIX):
        answer, extra = _fresh_answer(problem, rng, accuracy) if rng.random() < 0.35 else (own, "")
        return (
            f"<critique>{_filler(rng)}</critique>\n{extra}<answer>{answer}</answer>\n{confidence}"
        )
    peers = (
        re.findall(r"<answer>(.*?)</answer>", last, flags=re.DOTALL) if _PEER_PREFIX in last else []
    )
    answer = own
    if peers and rng.random() < 0.5:
        answer = Counter(peers).most_common(1)[0][0].strip()
    return f"<critique>{_filler(rng)}</critique>\n<answer>{answer}</answer>\n{confidence}"
