"""HumanEval: 164 Python functions scored by executing the official hidden tests (pass@1).

Voting needs to know when two programs are "the same answer". Programs are clustered by
behaviour on the example calls that appear in the prompt's docstring (public information,
never the hidden tests), in the spirit of CodeT/AlphaCode clustering. When a prompt has no
usable example calls, programs are compared by their normalized syntax tree.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import re
from collections.abc import Hashable, Sequence
from typing import Any

from workforce.tasks.base import Item, Score, Task, sample_items
from workforce.tasks.download import RemoteFile, fetch
from workforce.tasks.registry import register_task
from workforce.tasks.sandbox import Sandbox

DATA_FILE = RemoteFile(
    url=(
        "https://github.com/openai/human-eval/raw/"
        "6d43fb980f9fee3c892a914eda09951f772ad10d/data/HumanEval.jsonl.gz"
    ),
    sha256="b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
    filename="HumanEval.jsonl.gz",
)

PROBE_MARKER = "@@WORKFORCE_PROBES@@"

_PROBE_RUNNER = r"""

import json as _wf_json
import signal as _wf_signal

def _wf_canon(value):
    if isinstance(value, float):
        return repr(round(value, 6))
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_wf_canon(v) for v in value)
        return ("[%s]" if isinstance(value, list) else "(%s)") % inner
    if isinstance(value, dict):
        items = sorted((_wf_canon(k), _wf_canon(v)) for k, v in value.items())
        return "{%s}" % ", ".join("%s: %s" % kv for kv in items)
    if isinstance(value, (set, frozenset)):
        return "{%s}" % ", ".join(sorted(_wf_canon(v) for v in value))
    return repr(value)[:300]

def _wf_alarm(signum, frame):
    raise TimeoutError()

_wf_signal.signal(_wf_signal.SIGALRM, _wf_alarm)
_wf_results = []
for _wf_expr in __PROBES__:
    try:
        _wf_signal.alarm(2)
        _wf_results.append("ok:" + _wf_canon(eval(_wf_expr)))
    except BaseException as _wf_exc:
        _wf_results.append("exc:" + type(_wf_exc).__name__)
    finally:
        _wf_signal.alarm(0)
print(__MARKER__ + _wf_json.dumps(_wf_results))
"""


def extract_code(
    answer_text: str | None, full_text: str, entry_point: str | None = None
) -> str | None:
    """Return the program inside the answer (fenced or raw). Without an answer tag, fall back
    to the last fenced block that defines ``entry_point``, else the last fenced block."""
    fence = r"```(?:python|py|Python)?[ \t]*\n(.*?)```"
    if answer_text is not None:
        blocks = re.findall(fence, answer_text, flags=re.DOTALL)
        if blocks:
            return blocks[-1]
        unterminated = re.search(r"```(?:python|py|Python)?[ \t]*\n(.*)\Z", answer_text, re.DOTALL)
        if unterminated:
            return unterminated.group(1)
        if answer_text.strip():
            return answer_text
    blocks = re.findall(fence, full_text, flags=re.DOTALL)
    if entry_point is not None:
        defining = [
            b for b in blocks if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", b, re.M)
        ]
        blocks = defining or blocks
    return blocks[-1] if blocks else None


def extract_probes(prompt: str, entry_point: str, limit: int = 8) -> list[str]:
    """Calls of ``entry_point`` that appear in the docstring examples, as Python expressions."""
    probes: list[str] = []
    for match in re.finditer(rf"(?<![\w.]){re.escape(entry_point)}\s*\(", prompt):
        if prompt[max(0, match.start() - 4) : match.start()] == "def ":
            continue
        args = _balanced(prompt, match.end() - 1)
        if args is None:
            continue
        call = entry_point + " ".join(args.split())
        try:
            ast.parse(call, mode="eval")
        except SyntaxError:
            continue
        if call not in probes:
            probes.append(call)
    return probes[:limit]


def _balanced(text: str, open_index: int) -> str | None:
    depth, quote, i = 0, None, open_index
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return text[open_index : i + 1]
        elif ch == "\n" and text[i + 1 : i + 2] == "\n":
            return None  # a blank line ends an example
        i += 1
    return None


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


@register_task("humaneval")
class HumanEvalTask(Task):
    name = "humaneval"

    def __init__(self, **options: Any):
        super().__init__(**options)
        self.timeout_s = float(options.get("timeout_s", 10.0))
        self.sandbox = Sandbox(max_workers=options.get("sandbox_workers"))
        self._score_cache: dict[tuple[str, str], Score] = {}
        self._key_cache: dict[tuple[str, str], str] = {}

    def load_items(self, n: int | None, seed: int) -> list[Item]:
        with gzip.open(fetch(DATA_FILE), "rt") as handle:
            rows = [json.loads(line) for line in handle]
        items = [
            Item(
                benchmark=self.name,
                item_id=row["task_id"],
                data={
                    "prompt": row["prompt"],
                    "entry_point": row["entry_point"],
                    "test": row["test"],
                },
            )
            for row in rows
        ]
        return sample_items(items, n, seed)

    def problem_prompt(self, item: Item) -> str:
        return (
            "Implement the following Python function so that it behaves exactly as its "
            "docstring describes, including edge cases.\n\n"
            f"```python\n{item.data['prompt']}```"
        )

    def answer_format(self) -> str:
        return (
            "After your reasoning, give the complete implementation (signature included, plus "
            "any imports or helper functions it needs) as one Python code block:\n"
            "<answer>\n```python\n...\n```\n</answer>"
        )

    def parse_answer(self, answer_text: str | None, full_text: str, item: Item) -> str | None:
        code = extract_code(answer_text, full_text, item.data["entry_point"])
        return code if code and code.strip() else None

    def display_answer(self, answer: Any | None) -> str:
        return "" if answer is None else f"<program {_digest(str(answer))}>"

    def program(self, item: Item, code: str) -> str:
        entry = re.escape(item.data["entry_point"])
        if re.search(rf"^\s*def\s+{entry}\s*\(", code, flags=re.MULTILINE):
            return f"{item.data['prompt']}\n\n{code}"
        return item.data["prompt"] + code  # a body-only completion

    async def score(
        self, item: Item, answer: Any | None, evidence: Sequence[int] | None = None
    ) -> Score:
        if answer is None:
            return Score(correct=False, metrics={"pass": 0.0}, detail={"status": "no_answer"})
        cache_key = (item.item_id, _digest(str(answer)))
        if cache_key not in self._score_cache:
            program = (
                f"{self.program(item, str(answer))}\n\n{item.data['test']}\n\n"
                f"check({item.data['entry_point']})\n"
            )
            result = await self.sandbox.run(program, self.timeout_s)
            self._score_cache[cache_key] = Score(
                correct=result.passed,
                metrics={"pass": float(result.passed)},
                detail={"status": result.status, "stderr_tail": result.stderr[-300:]},
            )
        return self._score_cache[cache_key]

    async def equivalence_keys(
        self, item: Item, answers: Sequence[Any | None]
    ) -> list[Hashable | None]:
        return [None if a is None else await self._behaviour_key(item, str(a)) for a in answers]

    async def _behaviour_key(self, item: Item, code: str) -> str:
        cache_key = (item.item_id, _digest(code))
        if cache_key in self._key_cache:
            return self._key_cache[cache_key]
        probes = extract_probes(item.data["prompt"], item.data["entry_point"])
        program = self.program(item, code)
        try:
            ast.parse(program)
        except SyntaxError:
            key = "syntax_error"
        else:
            if probes:
                key = await self._probe_signature(program, probes)
            else:
                key = "ast:" + _digest(ast.dump(ast.parse(code)))
        self._key_cache[cache_key] = key
        return key

    async def _probe_signature(self, program: str, probes: list[str]) -> str:
        runner = _PROBE_RUNNER.replace("__MARKER__", repr(PROBE_MARKER)).replace(
            "__PROBES__", repr(probes)
        )
        result = await self.sandbox.run(program + runner, self.timeout_s)
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(PROBE_MARKER):
                return "probe:" + _digest(line[len(PROBE_MARKER) :])
        return f"probe:{result.status}"
