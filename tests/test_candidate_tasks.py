"""AIME and LiveCodeBench scorers, and abstention detection (synthetic items; no downloads)."""

import asyncio
import base64
import json
import os
import pickle
import zlib

import pytest

from workforce.tasks import Item, create_task
from workforce.tasks.aime import parse_integer_answer
from workforce.tasks.livecodebench import decode_tests, outputs_match

# AIME --------------------------------------------------------------------------------------
AIME_ITEM = Item(
    "aime",
    "aime2026-I-01",
    {"year": 2026, "contest": "I", "number": 1, "problem": "Find m + n.", "answer": 277},
)


@pytest.mark.parametrize(
    ("answer_text", "full_text", "expected"),
    [
        ("277", "", 277),
        (" 077 ", "", 77),
        ("$\\boxed{277}$", "", 277),
        ("277.", "", 277),
        ("m + n = 3 + 274 = 277", "", 277),
        ("no idea", "", None),
        (None, "so the answer is \\boxed{12} and \\boxed{277}", 277),
        (None, "The answer is 42.", 42),
        (None, "nothing here", None),
    ],
)
def test_aime_answer_parsing(answer_text, full_text, expected):
    assert parse_integer_answer(answer_text, full_text) == expected


def test_aime_scoring_and_prompt():
    task = create_task("aime")
    assert asyncio.run(task.score(AIME_ITEM, 277)).correct
    assert not asyncio.run(task.score(AIME_ITEM, 276)).correct
    assert not asyncio.run(task.score(AIME_ITEM, None)).correct
    assert "integer from 0 to 999" in task.problem_prompt(AIME_ITEM)
    assert task.parse_answer("277", "", AIME_ITEM) == 277
    assert task.is_abstention(None) and not task.is_abstention(0)


# Abstention --------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("answer", "abstained"),
    [
        ("not stated", True),
        ("Cannot be determined from the provided paragraphs", True),
        ("unknown from the provided paragraphs", True),
        ("Not specified in the paragraphs", True),
        ("none specified", True),
        ("N/A", True),
        (None, True),
        ("United States", False),
        ("Unknownville", False),
        ("None of the above", False),
        ("Nonesuch", False),
    ],
)
def test_musique_abstention_detection(answer, abstained):
    assert create_task("musique").is_abstention(answer) is abstained


# LiveCodeBench -----------------------------------------------------------------------------
def _encode(tests):
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()


SUM_TESTS = [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]
STDIN_ITEM = Item(
    "livecodebench",
    "atcoder-test_a",
    {
        "title": "Sum",
        "content": "Print A + B.",
        "platform": "atcoder",
        "difficulty": "medium",
        "contest_date": "2025-01-01T00:00:00",
        "starter_code": "",
        "func_name": None,
        "public_tests": SUM_TESTS,
        "hidden_tests": _encode(
            [
                {"input": "100 -5", "output": "95", "testtype": "stdin"},
                {"input": "7 8\n", "output": "15.0", "testtype": "stdin"},
            ]
        ),
    },
)
FUNCTIONAL_ITEM = Item(
    "livecodebench",
    "leetcode-1",
    {
        "title": "Pair",
        "content": "Return the pair sorted.",
        "platform": "leetcode",
        "difficulty": "medium",
        "contest_date": "2025-01-01T00:00:00",
        "starter_code": "class Solution:\n    def pair(self, a: int, b: List[int]) -> List[int]:\n",
        "func_name": "pair",
        "public_tests": [{"input": "2\n[1]", "output": "[1, 2]", "testtype": "functional"}],
        "hidden_tests": _encode(
            [{"input": "5\n[9]", "output": "[5, 9]", "testtype": "functional"}]
        ),
    },
)


@pytest.fixture
def lcb():
    return create_task("livecodebench", timeout_s=2.0, sandbox_workers=2)


def test_lcb_output_comparison_matches_the_official_grader():
    assert outputs_match("3\n", "3")
    assert outputs_match("  1 2 \n3\n\n", "1 2\n3")
    assert outputs_match("15", "15.0")  # equal as decimals
    assert outputs_match("0.50 1", "0.5 1")
    assert not outputs_match("1 2", "1 3")
    assert not outputs_match("1\n2", "1 2")
    assert not outputs_match("yes", "Yes")


def test_lcb_hidden_tests_decode_without_running_pickled_code():
    assert decode_tests(_encode(SUM_TESTS)) == SUM_TESTS
    assert decode_tests(json.dumps(SUM_TESTS)) == SUM_TESTS

    class Exploit:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    payload = base64.b64encode(zlib.compress(pickle.dumps(Exploit()))).decode()
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        decode_tests(payload)


def test_lcb_stdin_programs_run_as_scripts_on_every_test(lcb):
    good = "a, b = map(int, input().split())\nprint(a + b)\n"
    as_main = (
        "import sys\ndef main():\n    a, b = map(int, sys.stdin.buffer.read().split())\n"
        "    print(a + b)\n    exit()\nif __name__ == '__main__':\n    main()\n"
    )
    wrong_on_hidden = "a, b = map(int, input().split())\nprint(abs(a) + abs(b))\n"
    crash = "raise ValueError('boom')\n"
    slow = "while True:\n    pass\n"
    results = {
        name: asyncio.run(lcb.score(STDIN_ITEM, code))
        for name, code in [
            ("good", good),
            ("as_main", as_main),
            ("wrong_on_hidden", wrong_on_hidden),
            ("crash", crash),
            ("slow", slow),
        ]
    }
    assert results["good"].correct and results["good"].detail["tests"] == 3
    assert results["as_main"].correct
    assert results["wrong_on_hidden"].detail == {"status": "failed", "index": 1, "tests": 3}
    assert results["crash"].detail["status"] == "error"
    assert results["slow"].detail["status"] == "timeout"
    keys = asyncio.run(lcb.equivalence_keys(STDIN_ITEM, [good, wrong_on_hidden, crash, None]))
    assert keys[0] == keys[1]  # identical on the public example; the hidden tests differ
    assert keys[2] != keys[0] and keys[3] is None


def test_lcb_functional_solutions_are_called_as_methods(lcb):
    good = "class Solution:\n    def pair(self, a, b):\n        return tuple(sorted([a, *b]))\n"
    typed = "class Solution:\n    def pair(self, a: int, b: List[int]) -> List[int]:\n" + (
        "        return sorted(deque([a, *b]))\n"  # names from the judge's preamble
    )
    wrong = "class Solution:\n    def pair(self, a, b):\n        return [a, *b]\n"
    slow = "class Solution:\n    def pair(self, a, b):\n        while True:\n            pass\n"
    missing = "def pair(a, b):\n    return [a]\n"
    assert asyncio.run(lcb.score(FUNCTIONAL_ITEM, good)).correct
    assert asyncio.run(lcb.score(FUNCTIONAL_ITEM, typed)).correct
    assert asyncio.run(lcb.score(FUNCTIONAL_ITEM, wrong)).detail["status"] == "failed"
    assert asyncio.run(lcb.score(FUNCTIONAL_ITEM, slow)).detail["status"] == "timeout"
    assert asyncio.run(lcb.score(FUNCTIONAL_ITEM, missing)).detail["status"] == "error"
    keys = asyncio.run(lcb.equivalence_keys(FUNCTIONAL_ITEM, [good, typed, wrong]))
    assert keys[0] == keys[1] != keys[2]


def test_lcb_prompts_and_parsing(lcb):
    assert "standard input" in lcb.problem_prompt(STDIN_ITEM)
    prompt = lcb.problem_prompt(FUNCTIONAL_ITEM)
    assert "class Solution" in prompt and "def pair" in prompt
    text = "```python\nclass Solution:\n    def pair(self, a, b):\n        return []\n```"
    assert "def pair" in lcb.parse_answer(None, f"reasoning\n{text}", FUNCTIONAL_ITEM)
    assert lcb.parse_answer("", "no code", STDIN_ITEM) is None
