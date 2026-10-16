"""LiveCodeBench (release v6): competitive programming in Python, scored by the official tests.

Release v6 (``test6.jsonl``) is the newest official release: 175 problems from LeetCode and
AtCoder contests held January-April 2025, labelled easy / medium / hard. By default only medium
and hard problems are used. A solution passes (pass@1) when it passes every public and hidden
test, judged as LiveCodeBench's own grader judges them:

* AtCoder problems are programs reading standard input. Each test runs the program as a script
  in the sandbox; outputs match when their stripped lines are equal, or equal as decimals.
* LeetCode problems complete a ``class Solution`` method. All tests run in one process; each
  input line is a JSON argument and the result must equal the JSON expected value (a returned
  tuple counts as a list).

Each test has a time limit (``timeout_s``, default 6 s as in LiveCodeBench). Both modes prepend
the standard-library imports a judge environment provides. Hidden tests are stored zlib- and
pickle-encoded upstream; they are decoded with an unpickler that refuses every object except
plain strings, so a tampered file cannot run code. Voting clusters programs by their behaviour
on the public tests (the examples in the statement), never the hidden ones.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import pickle
import zlib
from collections.abc import Hashable, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from workforce.tasks.base import Item, Score, Task, sample_items
from workforce.tasks.download import RemoteFile, fetch
from workforce.tasks.humaneval import extract_code
from workforce.tasks.registry import register_task
from workforce.tasks.sandbox import Sandbox

DATA_FILE = RemoteFile(
    url=(
        "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/"
        "0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test6.jsonl"
    ),
    sha256="bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5",
    filename="livecodebench_v6_test6.jsonl",
)

PREAMBLE = """\
import bisect, collections, copy, functools, heapq, itertools, math, operator, random, re
import string, sys
from bisect import *
from collections import *
from functools import *
from heapq import *
from itertools import *
from math import *
from typing import *
from builtins import *
"""

MARKER = "@@WORKFORCE_LCB@@"

_FUNCTIONAL_RUNNER = r"""
import json as _wf_json
import signal as _wf_signal
import sys as _wf_sys

_wf_payload = _wf_json.loads(_wf_sys.stdin.read())

def _wf_report(result):
    print(__MARKER__ + _wf_json.dumps(result))
    _wf_sys.exit(0 if result.get("status") in ("passed", "probed") else 1)

def _wf_alarm(signum, frame):
    raise TimeoutError()

_wf_signal.signal(_wf_signal.SIGALRM, _wf_alarm)
_wf_ns = {"__name__": "__candidate__"}
try:
    exec(compile(_wf_payload["code"], "solution.py", "exec"), _wf_ns)
    _wf_method = getattr(_wf_ns["Solution"](), _wf_payload["func_name"])
except BaseException as _wf_exc:
    _wf_report({"status": "error", "index": -1, "error": type(_wf_exc).__name__})

_wf_probes = []
for _wf_index, _wf_test in enumerate(_wf_payload["tests"]):
    _wf_args = [_wf_json.loads(line) for line in _wf_test["input"].split("\n")]
    _wf_signal.setitimer(_wf_signal.ITIMER_REAL, _wf_payload["timeout"])
    try:
        _wf_value = _wf_method(*_wf_args)
    except TimeoutError:
        _wf_report({"status": "timeout", "index": _wf_index})
    except BaseException as _wf_exc:
        if _wf_payload["grade"]:
            _wf_report({"status": "error", "index": _wf_index, "error": type(_wf_exc).__name__})
        _wf_probes.append("exc:" + type(_wf_exc).__name__)
        continue
    finally:
        _wf_signal.setitimer(_wf_signal.ITIMER_REAL, 0)
    if isinstance(_wf_value, tuple):
        _wf_value = list(_wf_value)
    if _wf_payload["grade"]:
        if _wf_value != _wf_json.loads(_wf_test["output"]):
            _wf_report({"status": "failed", "index": _wf_index})
    else:
        _wf_probes.append("ok:" + repr(_wf_value)[:300])
if _wf_payload["grade"]:
    _wf_report({"status": "passed", "index": len(_wf_payload["tests"])})
_wf_report({"status": "probed", "outputs": _wf_probes})
"""


class _StringsOnly(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def decode_tests(raw: str) -> list[dict[str, str]]:
    """Public tests are JSON; hidden tests are base64(zlib(pickle(JSON string)))."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    decoded = _StringsOnly(io.BytesIO(zlib.decompress(base64.b64decode(raw.encode())))).load()
    return json.loads(decoded)


def _stripped_lines(text: str) -> list[str]:
    return [line.strip() for line in text.strip().split("\n")]


def _decimals(line: str) -> list[Decimal] | None:
    try:
        return [Decimal(token) for token in line.split()]
    except InvalidOperation:
        return None


def outputs_match(actual: str, expected: str) -> bool:
    """LiveCodeBench's standard-output check: same number of stripped lines, each equal as
    text or as a list of decimals."""
    got, want = _stripped_lines(actual), _stripped_lines(expected)
    if len(got) != len(want):
        return False
    for line, gold in zip(got, want, strict=True):
        if line == gold:
            continue
        left, right = _decimals(line), _decimals(gold)
        if left is None or right is None or left != right:
            return False
    return True


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


@register_task("livecodebench")
class LiveCodeBenchTask(Task):
    name = "livecodebench"

    def __init__(self, **options: Any):
        super().__init__(**options)
        self.timeout_s = float(options.get("timeout_s", 6.0))
        self.sandbox = Sandbox(max_workers=options.get("sandbox_workers"))
        self._tests: dict[str, list[dict[str, str]]] = {}
        self._score_cache: dict[tuple[str, str], Score] = {}
        self._key_cache: dict[tuple[str, str], str] = {}

    def load_items(self, n: int | None, seed: int) -> list[Item]:
        difficulties = set(self.options.get("difficulties", ["medium", "hard"]))
        platforms = set(self.options.get("platforms", ["atcoder", "leetcode"]))
        items = []
        with fetch(DATA_FILE).open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["difficulty"] not in difficulties or row["platform"] not in platforms:
                    continue
                metadata = json.loads(row["metadata"] or "{}")
                items.append(
                    Item(
                        benchmark=self.name,
                        item_id=f"{row['platform']}-{row['question_id']}",
                        data={
                            "title": row["question_title"],
                            "content": row["question_content"],
                            "platform": row["platform"],
                            "difficulty": row["difficulty"],
                            "contest_date": row["contest_date"],
                            "starter_code": row["starter_code"],
                            "func_name": metadata.get("func_name"),
                            "public_tests": json.loads(row["public_test_cases"]),
                            # Kept in the upstream encoding: decoding multiplies the size.
                            "hidden_tests": row["private_test_cases"],
                        },
                    )
                )
        return sample_items(items, n, seed)

    def _functional(self, item: Item) -> bool:
        return bool(item.data["func_name"])

    def problem_prompt(self, item: Item) -> str:
        data = item.data
        limits = (
            "on hidden tests with a time limit, so it must be fast enough for the largest "
            "inputs the constraints allow."
        )
        if self._functional(item):
            return (
                "Solve the following programming problem in Python 3 by completing the starter "
                f"code. Your method of class Solution is called {limits}\n\n"
                f"### Problem: {data['title']}\n\n{data['content']}\n\n"
                f"### Starter code\n```python\n{data['starter_code']}\n```"
            )
        return (
            "Solve the following competitive programming problem in Python 3. Write a complete "
            "program that reads the input from standard input and prints the answer to "
            f"standard output. It is run {limits}\n\n"
            f"### Problem: {data['title']}\n\n{data['content']}"
        )

    def answer_format(self) -> str:
        return (
            "After your reasoning, give your complete solution (with any imports it needs) as "
            "one Python code block:\n<answer>\n```python\n...\n```\n</answer>"
        )

    def parse_answer(self, answer_text: str | None, full_text: str, item: Item) -> str | None:
        code = extract_code(answer_text, full_text, item.data["func_name"])
        return code if code and code.strip() else None

    def display_answer(self, answer: Any | None) -> str:
        return "" if answer is None else f"<program {_digest(str(answer))}>"

    def tests(self, item: Item) -> list[dict[str, str]]:
        if item.item_id not in self._tests:
            hidden = decode_tests(item.data["hidden_tests"])
            self._tests[item.item_id] = [*item.data["public_tests"], *hidden]
        return self._tests[item.item_id]

    async def score(
        self, item: Item, answer: Any | None, evidence: Sequence[int] | None = None
    ) -> Score:
        if answer is None:
            return Score(correct=False, metrics={"pass": 0.0}, detail={"status": "no_answer"})
        cache_key = (item.item_id, _digest(str(answer)))
        if cache_key not in self._score_cache:
            tests = self.tests(item)
            if self._functional(item):
                result = await self._run_functional(item, str(answer), tests, grade=True)
            else:
                result = await self._grade_stdin(str(answer), tests)
            passed = result["status"] == "passed"
            self._score_cache[cache_key] = Score(
                correct=passed,
                metrics={"pass": float(passed)},
                detail={**result, "tests": len(tests)},
            )
        return self._score_cache[cache_key]

    async def _grade_stdin(self, code: str, tests: Sequence[dict[str, str]]) -> dict[str, Any]:
        program = PREAMBLE + code
        for index, test in enumerate(tests):
            result = await self.sandbox.run(
                program, self.timeout_s, stdin=test["input"], script=True
            )
            if result.status == "timeout":
                return {"status": "timeout", "index": index}
            if result.returncode != 0:
                return {"status": "error", "index": index, "stderr_tail": result.stderr[-300:]}
            if not outputs_match(result.stdout, test["output"]):
                return {"status": "failed", "index": index}
        return {"status": "passed", "index": len(tests)}

    async def _run_functional(
        self, item: Item, code: str, tests: Sequence[dict[str, str]], grade: bool
    ) -> dict[str, Any]:
        payload = {
            "code": PREAMBLE + code,
            "func_name": item.data["func_name"],
            "tests": list(tests),
            "timeout": self.timeout_s,
            "grade": grade,
        }
        result = await self.sandbox.run(
            _FUNCTIONAL_RUNNER.replace("__MARKER__", repr(MARKER)),
            self.timeout_s * len(tests) + 10,
            stdin=json.dumps(payload),
        )
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(MARKER):
                return json.loads(line[len(MARKER) :])
        return {"status": result.status if result.status == "timeout" else "error", "index": -1}

    async def equivalence_keys(
        self, item: Item, answers: Sequence[Any | None]
    ) -> list[Hashable | None]:
        return [None if a is None else await self._behaviour_key(item, str(a)) for a in answers]

    async def _behaviour_key(self, item: Item, code: str) -> str:
        cache_key = (item.item_id, _digest(code))
        if cache_key not in self._key_cache:
            public = item.data["public_tests"]
            if self._functional(item):
                result = await self._run_functional(item, code, public, grade=False)
                signature = json.dumps(result.get("outputs", result.get("status")))
            else:
                outputs = []
                for test in public:
                    run = await self.sandbox.run(
                        PREAMBLE + code, self.timeout_s, stdin=test["input"], script=True
                    )
                    outputs.append(
                        _stripped_lines(run.stdout) if run.status == "passed" else run.status
                    )
                signature = json.dumps(outputs)
            self._key_cache[cache_key] = "probe:" + _digest(signature)
        return self._key_cache[cache_key]

    async def aclose(self) -> None:
        self._tests.clear()
