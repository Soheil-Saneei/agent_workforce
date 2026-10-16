"""Solo-scaling metrics: accuracy, abstention and tokens used at each token checkpoint.

Pure functions of trial records that carry a ``solo`` section (see ``protocol/solo.py``).
Accuracy at a checkpoint counts a trajectory that has no answer yet as wrong, as a budget of
that size would; ``reached`` reports how many had one. CIs are percentile bootstraps over items
(replications of an item are averaged first), and gains are paired over items.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from workforce.analysis.metrics import (
    Trial,
    group_trials,
    item_estimate,
    paired_estimate,
    per_item,
)
from workforce.analysis.stats import mean

# A checkpoint is on the plateau when its accuracy is within this of the best checkpoint's.
PLATEAU_TOLERANCE = 0.02


def solo_trials(trials: Sequence[Trial]) -> list[Trial]:
    return [t for t in trials if "solo" in t]


def _at(trial: Trial, index: int) -> dict[str, Any]:
    return trial["solo"]["checkpoints"][index]


def _initial_correct(trial: Trial) -> float:
    return float(bool(trial["initial_correct"][0]))


def checkpoint_rows(trials: Sequence[Trial]) -> list[dict[str, Any]]:
    """One row per (benchmark, condition, checkpoint)."""
    rows = []
    for (benchmark, condition), group in group_trials(solo_trials(trials)).items():
        previous: dict[str, float] | None = None
        for index, checkpoint in enumerate(group[0]["solo"]["checkpoints"]):
            points = [_at(t, index) for t in group]
            reached = [t for t in group if _at(t, index)["steps"] > 0]
            correct = per_item(group, lambda t, i=index: float(_at(t, i)["correct"]))
            names = sorted({k for t in group for k in _at(t, index)["metrics"]})
            changed = [
                float(_at(t, index)["key"] != _at(t, index - 1)["key"])
                for t in group
                if index > 0 and _at(t, index - 1)["steps"] > 0
            ]
            rows.append(
                {
                    "benchmark": benchmark,
                    "condition": condition,
                    "tokens": checkpoint["tokens"],
                    "trials": len(group),
                    "reached": len(reached) / len(group),
                    "accuracy": item_estimate(correct),
                    "abstention": mean([float(_at(t, index)["abstained"]) for t in reached]),
                    "spent": mean([p["spent"] for p in points]),
                    "output_tokens": mean([p["output_tokens"] for p in points]),
                    "steps": mean([p["steps"] for p in points]),
                    "changed": mean(changed),
                    "metrics": {
                        n: mean([_at(t, index)["metrics"].get(n, 0.0) for t in group])
                        for n in names
                    },
                    "gain_vs_previous": (
                        paired_estimate(correct, previous) if previous is not None else None
                    ),
                    "gain_vs_initial": item_estimate(
                        per_item(
                            reached,
                            lambda t, i=index: float(_at(t, i)["correct"]) - _initial_correct(t),
                        )
                    ),
                }
            )
            previous = correct
    return rows


def _transitions(trial: Trial) -> tuple[int, int]:
    """(wrong -> right, right -> wrong) changes of the held answer across steps within the
    horizon. A step without an answer keeps the previous one."""
    within = trial["solo"]["steps_within_horizon"]
    held: bool | None = None
    up = down = 0
    for flag in trial["solo"]["step_correct"][:within]:
        if flag is None:
            continue
        if held is not None and flag != held:
            up += flag
            down += not flag
        held = flag
    return up, down


def curve_rows(trials: Sequence[Trial]) -> list[dict[str, Any]]:
    """One row per (benchmark, condition): the initial answer against the end of the
    trajectory, where the curve peaks and flattens, and how revisions moved answers."""
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in checkpoint_rows(trials):
        by_group.setdefault((row["benchmark"], row["condition"]), []).append(row)
    rows = []
    for (benchmark, condition), group in group_trials(solo_trials(trials)).items():
        points = by_group[(benchmark, condition)]
        accuracies = [p["accuracy"]["mean"] or 0.0 for p in points]
        best = max(accuracies)
        initial = per_item(group, _initial_correct)
        final = per_item(group, lambda t: float(t["team"]["correct"]))
        moves = [_transitions(t) for t in group]
        last = len(points) - 1
        rows.append(
            {
                "benchmark": benchmark,
                "condition": condition,
                "trials": len(group),
                "items": len({t["item_id"] for t in group}),
                "initial_accuracy": item_estimate(initial),
                "final_accuracy": item_estimate(final),
                "final_minus_initial": paired_estimate(final, initial),
                "best_tokens": points[accuracies.index(best)]["tokens"],
                "best_accuracy": best,
                "plateau_tokens": next(
                    p["tokens"]
                    for p, a in zip(points, accuracies, strict=True)
                    if a >= best - PLATEAU_TOLERANCE
                ),
                "initial_abstention": mean(
                    [float(t["solo"]["step_abstained"][0]) for t in group if t["solo"]["steps"]]
                ),
                "final_abstention": points[last]["abstention"],
                "steps": mean([t["solo"]["steps_within_horizon"] for t in group]),
                "tokens_used": mean([_at(t, last)["spent"] for t in group]),
                "overshoot": mean([t["solo"]["overshoot"] for t in group]),
                "wrong_to_right": sum(up for up, _ in moves),
                "right_to_wrong": sum(down for _, down in moves),
            }
        )
    return rows
