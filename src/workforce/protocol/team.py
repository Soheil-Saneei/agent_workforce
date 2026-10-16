"""One trial: a team of N agents solving one item under one condition.

Horizontal protocol (no leader, no final-answer authority):

1. Every agent independently produces an initial solution and states its confidence.
2. All initial solutions are revealed simultaneously: no agent sees anything from a phase
   until every agent has finished that phase, so nobody anchors on a first speaker.
3. Each agent submits one critique-and-revision per discussion round (one round in the core
   protocol, two in the first experiment's variant). With N = 1 the round is a self-revision.
4. The team answer is an anonymous majority vote over the agents' final answers.
5. Ties go to the answer whose voters had the highest summed pre-discussion confidence.

With ``communication = independent`` the protocol stops after step 1 and votes, which is the
independent-ensemble control. Every trial also records the vote over its own initial answers
(a paired independent-ensemble control) and the best-individual oracle.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from workforce.design import Condition, ExperimentConfig
from workforce.gateway import CallContext, Gateway
from workforce.models.base import ChatRequest, Message
from workforce.protocol.budget import BudgetLedger, CallSlot
from workforce.protocol.parsing import parse_output, strip_confidence
from workforce.protocol.prompts import discussion_message, initial_message, system_message
from workforce.protocol.voting import VoteOutcome, anonymous_vote
from workforce.tasks.base import Item, Task

# Allowance used only while estimating input size; wider than any real allowance.
_PLACEHOLDER_ALLOWANCE = 99_999


def derive_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


@dataclass(frozen=True)
class TrialSpec:
    benchmark: str
    condition: Condition
    item: Item
    item_ordinal: int
    replication: int

    @property
    def key(self) -> str:
        return f"{self.benchmark}|{self.condition.id}|{self.item.item_id}|r{self.replication}"


@dataclass
class Turn:
    agent: int
    phase: int
    skipped: bool = False
    skip_reason: str | None = None
    text: str | None = None
    answer: Any = None
    answer_display: str = ""
    key: str | None = None
    confidence: float | None = None
    evidence: list[int] | None = None
    critique: str | None = None
    answer_tagged: bool = False  # the answer came from an <answer> tag, not a fallback
    allowance: int = 0
    max_output_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: str | None = None
    latency_s: float = 0.0
    call_key: str | None = None
    cached: bool = False
    cost_usd: float | None = None
    correct: bool | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def has_answer(self) -> bool:
        return not self.skipped and self.answer is not None


def canonical_key(key: Any) -> str | None:
    if key is None:
        return None
    if isinstance(key, str):
        return key
    if isinstance(key, int):
        return str(key)
    return "h:" + hashlib.sha1(repr(key).encode()).hexdigest()[:16]


async def score_turns(task: Task, item: Item, turns: Sequence[Turn]) -> None:
    """Set each answered turn's equivalence key, correctness and benchmark metrics."""
    answered = [t for t in turns if t.has_answer]
    keys = await task.equivalence_keys(item, [t.answer for t in answered])
    for turn, key in zip(answered, keys, strict=True):
        turn.key = canonical_key(key)
        score = await task.score(item, turn.answer, turn.evidence)
        turn.correct = score.correct
        turn.metrics = score.metrics


def vote_summary(
    chosen: Sequence[Turn | None], confidences: Sequence[float | None], seed: int
) -> dict[str, Any]:
    """Anonymous vote over ``chosen`` (one ballot per agent; None abstains), as recorded."""
    outcome: VoteOutcome = anonymous_vote(
        [t.key if t is not None else None for t in chosen],
        confidences,
        random.Random(derive_seed(seed, "vote")),
    )
    representative = None
    if outcome.winner is not None:
        members = [t for t in chosen if t is not None and t.key == outcome.winner]
        representative = min(members, key=lambda t: (-(confidences[t.agent] or 0.0), t.agent))
    return {
        "key": outcome.winner,
        "answer_display": representative.answer_display if representative else "",
        "correct": bool(representative and representative.correct),
        "metrics": representative.metrics if representative else {},
        "representative_agent": representative.agent if representative else None,
        "tie_break": outcome.tie_break,
        "counts": outcome.counts,
        "weights": outcome.weights,
    }


class TeamProtocol:
    def __init__(self, config: ExperimentConfig, task: Task, gateway: Gateway):
        self.config = config
        self.task = task
        self.gateway = gateway

    async def run(self, spec: TrialSpec) -> dict[str, Any]:
        started = time.perf_counter()
        cfg, cond, item = self.config, spec.condition, spec.item
        n, rounds = cond.team_size, cond.communication.rounds
        phases = 1 + rounds
        assignments = cfg.assign_agents(cond, spec.item_ordinal, spec.replication)
        procedures = [cfg.procedures[a.procedure] for a in assignments]
        model_names = [a.model for a in assignments]
        ceiling = min(
            [cfg.generation.max_output_tokens_per_call]
            + [cfg.models[m].max_output_tokens or 10**9 for m in model_names]
        )
        ledger = BudgetLedger(
            total=cfg.benchmarks[spec.benchmark].budget_tokens,
            team_size=n,
            phases=phases,
            count=cfg.budget.count,
            ceiling=ceiling,
            min_output=cfg.generation.min_output_tokens,
            input_margin=cfg.budget.input_margin,
        )
        trial_seed = derive_seed(cfg.seed, "trial", spec.key)
        histories: list[list[Message]] = [[system_message(p)] for p in procedures]
        active = [True] * n
        turns: list[list[Turn]] = []
        overhead = self._reveal_overhead(procedures, model_names, rounds, n)

        for phase in range(phases):
            peer_texts = self._reveal(turns, n, phase, trial_seed) if phase else [[]] * n

            def user_message(
                i: int,
                allowance: int | None,
                phase: int = phase,
                peer_texts: list[list[str]] = peer_texts,
            ) -> Message:
                if phase == 0:
                    return initial_message(self.task, item, allowance)
                return discussion_message(
                    self.task, procedures[i], phase, rounds, peer_texts[i], allowance
                )

            estimates = {
                i: ledger.estimate_input(
                    self.gateway.count_tokens(
                        model_names[i],
                        [*histories[i], user_message(i, _PLACEHOLDER_ALLOWANCE)],
                    )
                )
                for i in range(n)
                if active[i]
            }
            plan = ledger.plan_phase(phase, max(estimates.values(), default=0), overhead)
            disclosed = plan.allowance if cfg.budget.disclose_allowance else None

            launches: list[tuple[int, CallSlot, Message]] = []
            phase_turns: list[Turn] = []
            for i in range(n):
                if not active[i]:
                    ledger.skip(i, phase, "inactive")
                    phase_turns.append(Turn(i, phase, skipped=True, skip_reason="inactive"))
                    continue
                message = user_message(i, disclosed)
                estimate = ledger.estimate_input(
                    self.gateway.count_tokens(model_names[i], [*histories[i], message])
                )
                slot = ledger.reserve(i, phase, plan.allowance, estimate)
                if slot.skipped:
                    active[i] = False
                    phase_turns.append(Turn(i, phase, skipped=True, skip_reason=slot.skip_reason))
                    continue
                phase_turns.append(Turn(i, phase))
                launches.append((i, slot, message))

            results = await asyncio.gather(
                *(
                    self.gateway.complete(
                        model_names[i],
                        ChatRequest(
                            messages=(*histories[i], message),
                            max_output_tokens=slot.max_output_tokens,
                            temperature=cfg.generation.temperature,
                            seed=derive_seed(
                                cfg.seed,
                                "call",
                                spec.benchmark,
                                item.item_id,
                                spec.replication,
                                i,
                                phase,
                            ),
                        ),
                        CallContext(
                            trial_key=spec.key,
                            condition=cond.id,
                            benchmark=spec.benchmark,
                            item_id=item.item_id,
                            replication=spec.replication,
                            agent=i,
                            phase=phase,
                        ),
                    )
                    for i, slot, message in launches
                ),
                return_exceptions=True,
            )
            failure = next((r for r in results if isinstance(r, BaseException)), None)
            if failure is not None:
                raise failure

            for (i, slot, message), outcome in zip(launches, results, strict=True):
                response = outcome.response
                ledger.settle(
                    slot,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    response.finish_reason,
                )
                histories[i] += [message, Message("assistant", response.text)]
                parsed = parse_output(response.text)
                answer = self.task.parse_answer(parsed.answer_text, response.text, item)
                turn = phase_turns[i]
                turn.text = response.text
                turn.answer = answer
                turn.answer_display = self.task.display_answer(answer)
                turn.confidence = parsed.confidence
                turn.evidence = list(parsed.evidence) if parsed.evidence is not None else None
                turn.critique = parsed.critique
                turn.answer_tagged = parsed.answer_text is not None
                turn.allowance = slot.allowance
                turn.max_output_tokens = slot.max_output_tokens
                turn.input_tokens = response.usage.input_tokens
                turn.output_tokens = response.usage.output_tokens
                turn.cached_input_tokens = response.usage.cached_input_tokens
                turn.reasoning_tokens = response.usage.reasoning_tokens
                turn.finish_reason = response.finish_reason
                turn.latency_s = response.latency_s
                turn.call_key = outcome.key
                turn.cached = outcome.cached
                turn.cost_usd = outcome.cost_usd
            turns.append(phase_turns)

        await self._score(item, turns)
        return self._record(spec, assignments, turns, ledger, trial_seed, started)

    # ------------------------------------------------------------------------------------
    def _reveal_overhead(
        self, procedures: Sequence[Any], model_names: Sequence[str], rounds: int, n: int
    ) -> int:
        """Tokens a discussion message adds beyond the peer solutions it carries."""
        if rounds == 0:
            return 0
        return max(
            self.gateway.count_tokens(
                model,
                [
                    discussion_message(
                        self.task, proc, 1, rounds, [""] * (n - 1), _PLACEHOLDER_ALLOWANCE
                    )
                ],
            )
            for proc, model in zip(procedures, model_names, strict=True)
        )

    @staticmethod
    def _reveal(turns: list[list[Turn]], n: int, phase: int, trial_seed: int) -> list[list[str]]:
        """Each viewer's peers' latest solutions, confidence removed, in a viewer-specific
        random order. Built only after the previous phase has fully completed."""
        latest: list[str | None] = [None] * n
        for phase_turns in turns:
            for turn in phase_turns:
                if not turn.skipped and turn.text is not None:
                    latest[turn.agent] = strip_confidence(turn.text)
        views = []
        for viewer in range(n):
            peers = [latest[j] for j in range(n) if j != viewer and latest[j] is not None]
            random.Random(derive_seed(trial_seed, "reveal", phase, viewer)).shuffle(peers)
            views.append(peers)
        return views

    async def _score(self, item: Item, turns: list[list[Turn]]) -> None:
        await score_turns(self.task, item, [t for phase in turns for t in phase])

    def _record(
        self,
        spec: TrialSpec,
        assignments: Sequence[Any],
        turns: list[list[Turn]],
        ledger: BudgetLedger,
        trial_seed: int,
        started: float,
    ) -> dict[str, Any]:
        n = spec.condition.team_size
        initial = [t if t.has_answer else None for t in turns[0]]
        final: list[Turn | None] = []
        for agent in range(n):
            answered = [ph[agent] for ph in turns if ph[agent].has_answer]
            final.append(answered[-1] if answered else None)
        pre_confidence = [t.confidence if not t.skipped else None for t in turns[0]]
        all_turns = [t for phase in turns for t in phase]
        calls = [t for t in all_turns if not t.skipped]
        return {
            "trial_key": spec.key,
            "benchmark": spec.benchmark,
            "item_id": spec.item.item_id,
            "item_ordinal": spec.item_ordinal,
            "replication": spec.replication,
            "condition": spec.condition.to_dict(),
            "seed": trial_seed,
            "agents": [
                {
                    "index": a.index,
                    "procedure": a.procedure,
                    "model_name": a.model,
                    "model": self.config.models[a.model].model,
                }
                for a in assignments
            ],
            "turns": [[asdict(t) for t in phase] for phase in turns],
            "team": vote_summary(final, pre_confidence, trial_seed),
            "initial_vote": vote_summary(initial, pre_confidence, trial_seed),
            "initial_correct": [t.correct if t else None for t in initial],
            "final_correct": [t.correct if t else None for t in final],
            "oracle_initial": any(t is not None and t.correct for t in initial),
            "oracle_any": any(t.correct for t in all_turns if t.has_answer),
            "tokens": {
                "input": sum(t.input_tokens for t in calls),
                "output": sum(t.output_tokens for t in calls),
                "cached_input": sum(t.cached_input_tokens for t in calls),
                "reasoning": sum(t.reasoning_tokens for t in calls),
                "total": sum(t.input_tokens + t.output_tokens for t in calls),
            },
            "cost_usd": sum(t.cost_usd or 0.0 for t in calls),
            "calls": len(calls),
            "parse_failures": sum(1 for t in calls if t.answer is None),
            "untagged_answers": sum(1 for t in calls if not t.answer_tagged),
            "latency": {
                "critical_path_s": sum(
                    max((t.latency_s for t in phase if not t.skipped), default=0.0)
                    for phase in turns
                ),
                "wall_clock_s": time.perf_counter() - started,
            },
            "budget": ledger.to_dict(),
        }
