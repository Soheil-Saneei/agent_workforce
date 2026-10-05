"""Anonymous majority vote with a confidence-weighted tie-break.

Ballots are (answer key, pre-discussion confidence) pairs with the voter's identity dropped.
The most common key wins. Among tied keys, the one whose voters' summed pre-discussion
confidence is highest wins. A remaining tie is broken by a seeded random draw so that replays
reproduce the same outcome.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

# Weight for a ballot whose voter did not state a confidence (midpoint of the 0-100 scale).
DEFAULT_CONFIDENCE = 50.0


@dataclass(frozen=True)
class VoteOutcome:
    winner: str | None
    counts: dict[str, int]
    tie_break: str  # "none" | "confidence" | "random" | "no_votes"
    weights: dict[str, float]


def anonymous_vote(
    keys: Sequence[str | None], confidences: Sequence[float | None], rng: random.Random
) -> VoteOutcome:
    if len(keys) != len(confidences):
        raise ValueError("one confidence per ballot")
    ballots = [
        (key, DEFAULT_CONFIDENCE if conf is None else conf)
        for key, conf in zip(keys, confidences, strict=True)
        if key is not None
    ]
    if not ballots:
        return VoteOutcome(None, {}, "no_votes", {})
    counts = Counter(key for key, _ in ballots)
    top = max(counts.values())
    tied = sorted(key for key, count in counts.items() if count == top)
    if len(tied) == 1:
        return VoteOutcome(tied[0], dict(counts), "none", {})
    weights = {key: sum(conf for k, conf in ballots if k == key) for key in tied}
    best = max(weights.values())
    still_tied = [key for key in tied if weights[key] == best]
    if len(still_tied) == 1:
        return VoteOutcome(still_tied[0], dict(counts), "confidence", weights)
    return VoteOutcome(rng.choice(still_tied), dict(counts), "random", weights)
