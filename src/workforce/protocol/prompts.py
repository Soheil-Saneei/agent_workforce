"""Message construction for every protocol phase.

Agents never learn teammates' procedures, models, identities or confidences: peers are shown
as "Teammate solution k" in an order shuffled independently for each viewer.
"""

from __future__ import annotations

from collections.abc import Sequence

from workforce.models.base import Message
from workforce.protocol.procedures import Procedure
from workforce.tasks.base import Item, Task

SYSTEM_TEMPLATE = (
    "You are a careful, rigorous problem solver.\n\n"
    "Follow this problem-solving procedure:\n{instructions}"
)
CONFIDENCE_INSTRUCTION = (
    "Finally, state how confident you are that your answer is correct, as an integer from 0 "
    "to 100:\n<confidence>...</confidence>"
)
PEER_HEADING = "=== Teammate solution {label} ==="
REVISION_HEADING = "Revision round {round} of {rounds}."


def allowance_line(tokens: int) -> str:
    rounded = max(50, tokens // 50 * 50)
    words = max(30, int(rounded * 0.7) // 10 * 10)
    return (
        f"Length limit: keep your entire response under about {rounded} tokens (roughly {words} "
        "words). Text beyond the limit is cut off, so make sure your response ends with the "
        "required tags."
    )


def system_message(procedure: Procedure) -> Message:
    return Message("system", SYSTEM_TEMPLATE.format(instructions=procedure.instructions))


def initial_message(task: Task, item: Item, allowance: int | None) -> Message:
    parts = [
        task.problem_prompt(item),
        "Work through the problem following your procedure.",
        task.answer_format(),
        CONFIDENCE_INSTRUCTION,
    ]
    if allowance is not None:
        parts.append(allowance_line(allowance))
    return Message("user", "\n\n".join(parts))


def discussion_message(
    task: Task,
    procedure: Procedure,
    round_index: int,
    rounds: int,
    peer_texts: Sequence[str],
    allowance: int | None,
) -> Message:
    if peer_texts:
        parts = [
            f"Discussion round {round_index} of {rounds}. All team members worked at the same "
            f"time. Here are your {len(peer_texts)} teammates' latest solutions, anonymized and "
            "in random order:",
            *(f"{PEER_HEADING.format(label=i + 1)}\n{text}" for i, text in enumerate(peer_texts)),
            "Critically review your own solution and your teammates' solutions, following your "
            "procedure: look for errors, unsupported steps and better-supported alternatives.",
        ]
    else:
        parts = [
            REVISION_HEADING.format(round=round_index, rounds=rounds),
            "Critically review your solution above, following your procedure: look for errors, "
            "unsupported steps and better-supported alternatives.",
        ]
    if procedure.discussion_focus:
        parts.append(procedure.discussion_focus)
    parts += [
        "Write your critique inside <critique>...</critique>. Then give your final answer, which "
        "may keep or change your previous answer, in exactly the same format as before:",
        task.answer_format(),
        CONFIDENCE_INSTRUCTION,
    ]
    if allowance is not None:
        parts.append(allowance_line(allowance))
    return Message("user", "\n\n".join(parts))


# Solo iteration ----------------------------------------------------------------------------
# A solo step's prompt depends only on the step index, never on the token horizon, so a
# trajectory's requests are the same however long it is allowed to run (see protocol/solo.py).
SOLO_COMMIT = (
    "The problem has a definite correct answer and a response without one is scored as wrong, "
    "so always commit to your single best answer, even if some doubt remains."
)
SOLO_CHECK_HEADING = "Self-check {step}."
SOLO_CHECK_METHODS = (
    "Solve the problem again from the beginning without relying on your earlier work, by a "
    "different route where possible (a different method, order of steps or reading of the key "
    "facts). Then compare the result with your latest answer and, if they differ, determine "
    "which one is correct.",
    "Audit your latest solution step by step: check every fact you used against the problem "
    "statement, redo every calculation, and confirm that each step follows from the previous "
    "ones. For a program, trace it by hand on the examples and on edge cases.",
    "Identify the most plausible alternative answer, or the most likely way your latest answer "
    "could be wrong, and test it seriously against the problem. Keep whichever answer holds up.",
)


def solo_initial_message(task: Task, item: Item) -> Message:
    parts = [
        task.problem_prompt(item),
        "Work through the problem following your procedure, writing out your reasoning in full "
        f"before you answer. {SOLO_COMMIT}",
        task.answer_format(),
        CONFIDENCE_INSTRUCTION,
    ]
    return Message("user", "\n\n".join(parts))


def solo_check_message(task: Task, step: int) -> Message:
    """The self-check after ``step`` - 1 earlier steps; verification methods rotate by step."""
    method = SOLO_CHECK_METHODS[(step - 1) % len(SOLO_CHECK_METHODS)]
    parts = [
        f"{SOLO_CHECK_HEADING.format(step=step)} Check your latest solution above with this "
        f"method:\n{method}",
        "Change your answer only if the check reveals a concrete mistake; if it confirms your "
        f"answer, keep it. {SOLO_COMMIT}",
        "Write your check inside <critique>...</critique>, showing your work. Then give your "
        "answer, which may keep or change your previous answer, in exactly the same format as "
        "before:",
        task.answer_format(),
        CONFIDENCE_INSTRUCTION,
    ]
    return Message("user", "\n\n".join(parts))
