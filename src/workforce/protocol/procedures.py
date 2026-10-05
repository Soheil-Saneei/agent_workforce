"""Problem-solving procedures (the diversity manipulation) and the default diversity levels.

Procedures change only *how* an agent works, never its persona, information, model or
authority. Experiment configs can override any procedure text or define new ones.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Procedure:
    name: str
    instructions: str
    discussion_focus: str = ""


DEFAULT_PROCEDURES: dict[str, Procedure] = {
    p.name: p
    for p in [
        # Identical condition: one generic procedure for every agent.
        Procedure(
            name="general",
            instructions=(
                "Read the problem carefully, reason through it step by step, check your "
                "reasoning, and then commit to the best answer."
            ),
        ),
        # Moderate diversity: distinct, broadly applicable procedures that can each solve
        # the whole task on their own.
        Procedure(
            name="decomposition",
            instructions=(
                "Break the problem into smaller subproblems. Solve the subproblems one at a "
                "time, stating each intermediate result explicitly, and then compose the "
                "intermediate results into the final answer."
            ),
        ),
        Procedure(
            name="evidence_first",
            instructions=(
                "Before reasoning toward any answer, extract and list every fact, requirement "
                "or example in the problem that could be relevant, quoting or closely "
                "paraphrasing the source. Only after the list is complete, reason from the "
                "listed facts to a conclusion, and do not rely on anything you did not list."
            ),
        ),
        Procedure(
            name="falsification",
            instructions=(
                "Generate two or three plausible candidate answers. For each candidate, "
                "actively search for counterexamples, contradictions or overlooked cases that "
                "would rule it out. Eliminate the candidates that fail and commit to the one "
                "that best survives the attempt to refute it."
            ),
        ),
        # Strong diversity: narrower occupational roles. Each still submits an answer,
        # because the horizontal protocol needs every agent's solution and vote.
        Procedure(
            name="solver",
            instructions=(
                "Your role is Solver. Produce a direct, complete solution to the problem as "
                "efficiently as possible."
            ),
            discussion_focus="As the Solver, make sure the team's answer is complete and direct.",
        ),
        Procedure(
            name="critic",
            instructions=(
                "Your role is Critic. Identify the most likely mistakes, traps and weak "
                "reasoning steps in solving this problem, and make sure your own answer avoids "
                "them."
            ),
            discussion_focus="As the Critic, concentrate on finding errors in every solution.",
        ),
        Procedure(
            name="alternative_hypothesis",
            instructions=(
                "Your role is Alternative-hypothesis generator. Deliberately consider answers "
                "other than the most obvious one, assess each against the problem, and accept "
                "the obvious answer only if every alternative fails."
            ),
            discussion_focus=(
                "As the Alternative-hypothesis generator, argue for the strongest alternative "
                "to the current majority answer and check whether it holds up."
            ),
        ),
        Procedure(
            name="fact_checker",
            instructions=(
                "Your role is Fact checker. Verify every factual claim and every reasoning step "
                "against the problem statement, accepting nothing that is not directly "
                "supported."
            ),
            discussion_focus=(
                "As the Fact checker, verify each claim in the solutions against the problem."
            ),
        ),
        Procedure(
            name="integrator",
            instructions=(
                "Your role is Integrator. Consider the different ways the problem could be "
                "approached and combine them into the single best-supported answer."
            ),
            discussion_focus=(
                "As the Integrator, combine the strongest parts of all solutions into one answer."
            ),
        ),
    ]
}

# Diversity level -> procedures, assigned to agents in order and cycled when N exceeds the list.
DEFAULT_DIVERSITY_LEVELS: dict[str, tuple[str, ...]] = {
    "identical": ("general",),
    "moderate": ("decomposition", "evidence_first", "falsification"),
    "strong": ("solver", "critic", "alternative_hypothesis", "fact_checker", "integrator"),
}
