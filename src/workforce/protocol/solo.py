"""Solo iteration: one agent answers, then repeatedly checks and revises its own work.

This measures how a single agent's accuracy changes with the tokens it may spend alone, and it
is the solo phase that a collaboration-onset study branches from (team discussion after T solo
tokens).

Trajectory
----------
1. Step 0 is an initial solution: the problem, a request to write out the reasoning in full,
   the answer format and a confidence.
2. Every later step appends a self-check message to the agent's own conversation. The check
   method rotates (re-solve independently and compare, audit step by step, test the strongest
   alternative), and the agent keeps or revises its answer, always committing to one.
3. Each step is charged as the budget ledger charges a call (input + output, or output only
   with ``budget.count = "output"``), from provider-reported usage. A step starts only if its
   input estimate plus ``min_output_tokens`` still fits within the horizon (the benchmark's
   ``budget_tokens``). The last step can finish past the horizon; it is logged and paid for, but
   it counts toward no checkpoint.
4. The answer held at checkpoint ``c`` is the latest parsable answer of a step that finished
   with cumulative spend <= ``c``. A step without a parsable answer leaves the previous answer
   in place, as in the team protocol.

Prefix invariance
-----------------
A step's request (messages, seed, output ceiling) depends only on the trajectory so far and the
step index, never on the horizon or the checkpoints. So the state held at ``c`` is exactly what
a trajectory stopped at ``c`` would hold, one trajectory yields the whole accuracy-vs-tokens
curve, and a protocol that continues from ``SoloState.at(c)`` re-issues byte-identical requests
for the shared prefix. Calls are keyed by the trajectory (benchmark, item, replication, agent),
not by the experiment condition, so the gateway serves that prefix from its log.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from workforce.design import ExperimentConfig
from workforce.gateway import CallContext, Gateway
from workforce.models.base import ChatRequest, Message
from workforce.protocol.budget import charged_tokens, padded_input_estimate
from workforce.protocol.parsing import parse_output
from workforce.protocol.procedures import Procedure
from workforce.protocol.prompts import solo_check_message, solo_initial_message, system_message
from workforce.protocol.team import TrialSpec, Turn, derive_seed, score_turns, vote_summary
from workforce.tasks.base import Item, Task


@dataclass(frozen=True)
class SoloIdentity:
    """Which trajectory this is. Call contexts and seeds derive from it alone."""

    benchmark: str
    item_id: str
    replication: int
    agent: int = 0

    @property
    def key(self) -> str:
        return f"solo|{self.benchmark}|{self.item_id}|r{self.replication}|a{self.agent}"


@dataclass
class SoloState:
    """Everything needed to continue a trajectory: the agent's conversation and its steps.

    ``messages`` is ``[system, user_0, assistant_0, user_1, assistant_1, ...]`` and ``spent[k]``
    is the cumulative charged tokens after step ``k``.
    """

    messages: list[Message]
    turns: list[Turn] = field(default_factory=list)
    spent: list[int] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return self.spent[-1] if self.spent else 0

    def at(self, tokens: int) -> SoloState:
        """The state a trajectory stopped at ``tokens`` would hold (a prefix of this one)."""
        steps = sum(1 for spent in self.spent if spent <= tokens)
        return SoloState(
            list(self.messages[: 1 + 2 * steps]), list(self.turns[:steps]), self.spent[:steps]
        )

    def held(self) -> Turn | None:
        """The agent's current answer: its latest step with a parsable answer."""
        return next((t for t in reversed(self.turns) if t.answer is not None), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "turns": [asdict(t) for t in self.turns],
            "spent": list(self.spent),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SoloState:
        return cls(
            [Message(m["role"], m["content"]) for m in d["messages"]],
            [Turn(**t) for t in d["turns"]],
            list(d["spent"]),
        )


class SoloAgent:
    """Advances one agent's solo trajectory through the gateway, one step at a time."""

    def __init__(
        self,
        config: ExperimentConfig,
        task: Task,
        gateway: Gateway,
        item: Item,
        identity: SoloIdentity,
        procedure: Procedure,
        model_name: str,
        state: SoloState | None = None,
    ):
        self.config = config
        self.task = task
        self.gateway = gateway
        self.item = item
        self.identity = identity
        self.model_name = model_name
        self.state = state or SoloState([system_message(procedure)])
        # Fixed per-step output ceiling, never shrunk to fit a budget (prefix invariance).
        self.ceiling = min(
            config.generation.max_output_tokens_per_call,
            config.models[model_name].max_output_tokens or 10**9,
        )

    @property
    def count(self) -> str:
        return self.config.budget.count

    def next_message(self) -> Message:
        step = len(self.state.turns)
        if step == 0:
            return solo_initial_message(self.task, self.item)
        return solo_check_message(self.task, step)

    def fits(self, horizon: int) -> bool:
        """Can the next step still finish within ``horizon`` with ``min_output_tokens``?"""
        raw = self.gateway.count_tokens(
            self.model_name, [*self.state.messages, self.next_message()]
        )
        estimate = padded_input_estimate(raw, self.config.budget.input_margin)
        needed = charged_tokens(self.count, estimate, self.config.generation.min_output_tokens)
        return self.state.tokens + needed <= horizon

    async def step(self) -> Turn:
        cfg, ident = self.config, self.identity
        step = len(self.state.turns)
        message = self.next_message()
        outcome = await self.gateway.complete(
            self.model_name,
            ChatRequest(
                messages=(*self.state.messages, message),
                max_output_tokens=self.ceiling,
                temperature=cfg.generation.temperature,
                seed=derive_seed(
                    cfg.seed,
                    "solo",
                    ident.benchmark,
                    ident.item_id,
                    ident.replication,
                    ident.agent,
                    step,
                ),
            ),
            CallContext(
                trial_key=ident.key,
                condition="solo",
                benchmark=ident.benchmark,
                item_id=ident.item_id,
                replication=ident.replication,
                agent=ident.agent,
                phase=step,
            ),
        )
        response = outcome.response
        parsed = parse_output(response.text)
        answer = self.task.parse_answer(parsed.answer_text, response.text, self.item)
        turn = Turn(
            agent=ident.agent,
            phase=step,
            text=response.text,
            answer=answer,
            answer_display=self.task.display_answer(answer),
            confidence=parsed.confidence,
            evidence=list(parsed.evidence) if parsed.evidence is not None else None,
            critique=parsed.critique,
            answer_tagged=parsed.answer_text is not None,
            allowance=self.ceiling,
            max_output_tokens=self.ceiling,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cached_input_tokens=response.usage.cached_input_tokens,
            reasoning_tokens=response.usage.reasoning_tokens,
            finish_reason=response.finish_reason,
            latency_s=response.latency_s,
            call_key=outcome.key,
            cached=outcome.cached,
            cost_usd=outcome.cost_usd,
        )
        charge = charged_tokens(
            self.count, response.usage.input_tokens, response.usage.output_tokens
        )
        self.state.messages += [message, Message("assistant", response.text)]
        self.state.turns.append(turn)
        self.state.spent.append(self.state.tokens + charge)
        return turn

    async def run(self, horizon: int, max_steps: int) -> str:
        """Step until the next step cannot fit within ``horizon``. Returns the stop reason."""
        while len(self.state.turns) < max_steps:
            if not self.fits(horizon):
                return "horizon"
            await self.step()
        return "max_steps"


class SoloProtocol:
    """One trial: a single agent's solo trajectory, scored at every token checkpoint.

    The record keeps the team protocol's schema (``turns`` has one phase per step, ``team`` is
    the answer held at the horizon, ``initial_vote`` is step 0), so the standard tables apply,
    and adds a ``solo`` section with the checkpoints.
    """

    def __init__(self, config: ExperimentConfig, task: Task, gateway: Gateway):
        self.config = config
        self.task = task
        self.gateway = gateway

    async def run(self, spec: TrialSpec) -> dict[str, Any]:
        started = time.perf_counter()
        cfg = self.config
        assignment = cfg.assign_agents(spec.condition, spec.item_ordinal, spec.replication)[0]
        horizon = cfg.benchmarks[spec.benchmark].budget_tokens
        agent = SoloAgent(
            cfg,
            self.task,
            self.gateway,
            spec.item,
            SoloIdentity(spec.benchmark, spec.item.item_id, spec.replication, assignment.index),
            cfg.procedures[assignment.procedure],
            assignment.model,
        )
        stop_reason = await agent.run(horizon, cfg.solo.max_steps)
        await score_turns(self.task, spec.item, agent.state.turns)
        return self._record(spec, assignment, agent.state, horizon, stop_reason, started)

    def checkpoint(self, state: SoloState, tokens: int) -> dict[str, Any]:
        reached = state.at(tokens)
        held = reached.held()
        return {
            "tokens": tokens,
            "steps": len(reached.turns),
            "spent": reached.tokens,
            "output_tokens": sum(t.output_tokens for t in reached.turns),
            "answer_step": held.phase if held else None,
            "key": held.key if held else None,
            "answer_display": held.answer_display if held else "",
            "confidence": held.confidence if held else None,
            "correct": bool(held and held.correct),
            "abstained": bool(reached.turns)
            and self.task.is_abstention(held.answer if held else None),
            "metrics": held.metrics if held else {},
        }

    def _record(
        self,
        spec: TrialSpec,
        assignment: Any,
        state: SoloState,
        horizon: int,
        stop_reason: str,
        started: float,
    ) -> dict[str, Any]:
        cfg = self.config
        trial_seed = derive_seed(cfg.seed, "trial", spec.key)
        within = state.at(horizon)
        final = within.held()
        first = within.turns[0] if within.turns else None
        initial = first if first is not None and first.has_answer else None
        turns = state.turns
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
                    "index": assignment.index,
                    "procedure": assignment.procedure,
                    "model_name": assignment.model,
                    "model": cfg.models[assignment.model].model,
                }
            ],
            "turns": [[asdict(t)] for t in turns],
            "team": vote_summary([final], [final.confidence if final else None], trial_seed),
            "initial_vote": vote_summary(
                [initial], [initial.confidence if initial else None], trial_seed
            ),
            "initial_correct": [initial.correct if initial else None],
            "final_correct": [final.correct if final else None],
            "oracle_initial": bool(initial and initial.correct),
            "oracle_any": any(t.correct for t in within.turns if t.has_answer),
            "tokens": {
                "input": sum(t.input_tokens for t in turns),
                "output": sum(t.output_tokens for t in turns),
                "cached_input": sum(t.cached_input_tokens for t in turns),
                "reasoning": sum(t.reasoning_tokens for t in turns),
                "total": sum(t.input_tokens + t.output_tokens for t in turns),
            },
            "cost_usd": sum(t.cost_usd or 0.0 for t in turns),
            "calls": len(turns),
            "parse_failures": sum(1 for t in turns if t.answer is None),
            "untagged_answers": sum(1 for t in turns if not t.answer_tagged),
            "latency": {
                "critical_path_s": sum(t.latency_s for t in turns),
                "wall_clock_s": time.perf_counter() - started,
            },
            "budget": {
                "total": horizon,
                "count": cfg.budget.count,
                "used": state.tokens,
                "overrun": max(0, state.tokens - horizon),
                "plans": [],
                "calls": [
                    {
                        "agent": t.agent,
                        "phase": t.phase,
                        "allowance": t.allowance,
                        "max_output_tokens": t.max_output_tokens,
                        "input_tokens": t.input_tokens,
                        "output_tokens": t.output_tokens,
                        "finish_reason": t.finish_reason,
                        "hit_limit": t.finish_reason == "length",
                    }
                    for t in turns
                ],
                "any_agent_hit_limit": any(t.finish_reason == "length" for t in turns),
                "skipped_calls": 0,
            },
            "solo": {
                "horizon": horizon,
                "count": cfg.budget.count,
                "stop_reason": stop_reason,
                "steps": len(turns),
                "steps_within_horizon": len(within.turns),
                "spent": list(state.spent),
                "overshoot": max(0, state.tokens - horizon),
                "step_correct": [t.correct for t in turns],
                "step_abstained": [self.task.is_abstention(t.answer) for t in turns],
                "checkpoints": [self.checkpoint(state, c) for c in cfg.solo.checkpoints],
            },
        }
