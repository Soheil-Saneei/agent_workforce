"""Scorers on known examples (synthetic items in each benchmark's format; no downloads)."""

import asyncio

import pytest

from workforce.tasks import Item, create_task
from workforce.tasks.humaneval import extract_code, extract_probes
from workforce.tasks.text_metrics import exact_match, normalize_answer, set_prf, token_f1

# MuSiQue -----------------------------------------------------------------------------------
MUSIQUE_ITEM = Item(
    "musique",
    "2hop__1_2",
    {
        "question": "Who is the spouse of the Green performer?",
        "paragraphs": [{"title": f"T{i}", "text": f"text {i}"} for i in range(5)],
        "supporting": [1, 3],
        "answer": "Miquette Giraudy",
        "aliases": ["Miquette"],
    },
)


def test_squad_normalization_and_metrics():
    assert normalize_answer("The  Eiffel-Tower!") == "eiffeltower"
    assert exact_match("the Beatles", "Beatles") == 1.0
    assert token_f1("Steve Hillage band", "Steve Hillage") == pytest.approx(0.8)
    assert token_f1("", "x") == 0.0 and token_f1("", "") == 1.0
    assert set_prf([2, 4], [1, 4]) == (0.5, 0.5, 0.5)


def test_musique_scoring_uses_aliases_and_support():
    task = create_task("musique")
    score = asyncio.run(task.score(MUSIQUE_ITEM, "miquette", evidence=[2, 4]))
    assert score.correct and score.metrics["em"] == 1.0
    assert score.metrics["support_f1"] == 1.0  # displayed P2, P4 -> indices 1, 3
    partial = asyncio.run(task.score(MUSIQUE_ITEM, "Giraudy family", evidence=[1]))
    assert not partial.correct and partial.metrics["f1"] == pytest.approx(0.5)
    assert partial.metrics["support_recall"] == 0.0
    assert task.parse_answer("Miquette Giraudy.\nextra", "", MUSIQUE_ITEM) == "Miquette Giraudy."
    assert task.parse_answer(None, "...\nFinal answer: Steve", MUSIQUE_ITEM) == "Steve"
    assert task.answer_key("The Miquette  Giraudy", MUSIQUE_ITEM) == "miquette giraudy"
    assert "[P2] T1" in task.problem_prompt(MUSIQUE_ITEM)


# MuSR --------------------------------------------------------------------------------------
MUSR_ITEM = Item(
    "musr",
    "murder_mysteries-0",
    {
        "domain": "murder_mysteries",
        "narrative": "story",
        "question": "Who?",
        "choices": ["Mackenzie", "Ana"],
        "answer_index": 0,
    },
)


def test_musr_parsing_and_accuracy():
    task = create_task("musr")
    assert task.parse_answer("1", "", MUSR_ITEM) == 1
    assert task.parse_answer("(2) Ana", "", MUSR_ITEM) == 2
    assert task.parse_answer("ana", "", MUSR_ITEM) == 2
    assert task.parse_answer("7", "", MUSR_ITEM) is None
    assert task.parse_answer(None, "so the answer is 1", MUSR_ITEM) == 1
    assert asyncio.run(task.score(MUSR_ITEM, 1)).correct
    assert not asyncio.run(task.score(MUSR_ITEM, 2)).correct
    assert not asyncio.run(task.score(MUSR_ITEM, None)).correct
    assert "1. Mackenzie\n2. Ana" in task.problem_prompt(MUSR_ITEM)


# HumanEval ---------------------------------------------------------------------------------
PROMPT = '''from typing import List


def running_max(numbers: List[int]) -> List[int]:
    """ Running maximum of a list.
    >>> running_max([1, 3, 2])
    [1, 3, 3]
    >>> running_max([])
    []
    """
'''
TEST = """

def check(candidate):
    assert candidate([1, 3, 2]) == [1, 3, 3]
    assert candidate([]) == []
    assert candidate([5, 1, 7, 2]) == [5, 5, 7, 7]
"""
HE_ITEM = Item(
    "humaneval", "Synthetic/0", {"prompt": PROMPT, "entry_point": "running_max", "test": TEST}
)

CORRECT = """from typing import List

def running_max(numbers: List[int]) -> List[int]:
    out, best = [], None
    for n in numbers:
        best = n if best is None else max(best, n)
        out.append(best)
    return out
"""
CORRECT_ALT = """def running_max(numbers):
    import itertools
    return list(itertools.accumulate(numbers, max))
"""
WRONG = "def running_max(numbers):\n    return sorted(numbers)\n"


@pytest.fixture(scope="module")
def humaneval():
    return create_task("humaneval", timeout_s=5)


def score(task, code):
    return asyncio.run(task.score(HE_ITEM, code))


def test_humaneval_executes_hidden_tests(humaneval):
    assert score(humaneval, CORRECT).correct
    assert score(humaneval, CORRECT_ALT).correct
    body_only = "    return [max(numbers[: i + 1]) for i in range(len(numbers))]\n"
    assert score(humaneval, body_only).correct  # completion-style answers are accepted
    wrong = score(humaneval, WRONG)
    assert not wrong.correct and wrong.detail["status"] == "failed"
    assert not score(humaneval, "def running_max(numbers) return 1").correct
    assert not score(humaneval, None).correct


def test_humaneval_timeout_and_guard(humaneval):
    quick = create_task("humaneval", timeout_s=1)
    loop = score(quick, "def running_max(numbers):\n    while True:\n        pass\n")
    assert not loop.correct and loop.detail["status"] in ("timeout", "failed")
    destructive = (
        "import os\ndef running_max(numbers):\n    os.remove('anything')\n    return numbers\n"
    )
    result = score(humaneval, destructive)
    assert not result.correct and "NoneType" in result.detail["stderr_tail"]


def test_humaneval_votes_cluster_by_behaviour_on_docstring_examples(humaneval):
    keys = asyncio.run(
        humaneval.equivalence_keys(HE_ITEM, [CORRECT, CORRECT_ALT, WRONG, None, "def broken("])
    )
    assert keys[0] == keys[1]  # different code, same behaviour on the public examples
    assert keys[2] != keys[0]
    assert keys[3] is None
    assert keys[4] == "syntax_error"


def test_probe_and_code_extraction():
    assert extract_probes(PROMPT, "running_max") == ["running_max([1, 3, 2])", "running_max([])"]
    fenced = "Here:\n```python\ndef f():\n    return 1\n```\n"
    assert extract_code(fenced, "") == "def f():\n    return 1\n"
    assert extract_code(None, "text ```python\nx = 1\n``` more") == "x = 1\n"
    assert extract_code("```python\ndef f(): return 2", "") == "def f(): return 2"
