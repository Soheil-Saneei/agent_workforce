"""Agent output format: free-form reasoning followed by tagged fields.

<critique>...</critique>        (discussion rounds only)
<evidence>3, 7</evidence>        (tasks that ask for cited evidence)
<answer>...</answer>
<confidence>0-100</confidence>  (elicited before discussion; also after, for analysis)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def _last_tag(text: str, tag: str) -> str | None:
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    # Tolerate an unterminated final tag (for example a truncated response).
    opened = re.findall(rf"<{tag}>(.*)\Z", text, flags=re.DOTALL | re.IGNORECASE)
    return opened[-1].strip() if opened else None


@dataclass(frozen=True)
class ParsedOutput:
    answer_text: str | None
    confidence: float | None
    evidence: tuple[int, ...] | None
    critique: str | None


def parse_confidence(raw: str | None) -> float | None:
    if raw is None:
        return None
    match = re.search(r"\d+(?:\.\d+)?", raw)
    if not match:
        return None
    value = float(match.group())
    if "%" not in raw and value <= 1.0 and "." in match.group():
        value *= 100  # a probability such as 0.8
    return max(0.0, min(100.0, value))


def parse_evidence(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    return tuple(sorted({int(n) for n in re.findall(r"\d+", raw)}))


def parse_output(text: str) -> ParsedOutput:
    return ParsedOutput(
        answer_text=_last_tag(text, "answer"),
        confidence=parse_confidence(_last_tag(text, "confidence")),
        evidence=parse_evidence(_last_tag(text, "evidence")),
        critique=_last_tag(text, "critique"),
    )


def strip_confidence(text: str) -> str:
    """Remove confidence statements before showing a solution to teammates.

    Confidence is elicited only for the tie-break; showing it would let confident agents
    dominate the discussion, which the anonymous protocol is designed to prevent.
    """
    text = re.sub(r"<confidence>.*?(</confidence>|\Z)", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()
