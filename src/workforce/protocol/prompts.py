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
