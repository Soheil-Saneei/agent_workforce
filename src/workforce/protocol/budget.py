"""Fixed per-problem token budget, shared by the whole team.

Allocation policy ("planned uniform allowance, re-planned every phase")
-----------------------------------------------------------------------
At the start of phase ``p`` with ``m`` phases left (this one included), team size ``N``,
remaining budget ``R``, current per-agent input ``J`` (the largest exact input among this
phase's calls) and per-round reveal overhead ``h``, every agent gets the same output allowance
``a``: the largest value for which the remaining calls still fit in ``R`` even if every output
uses its full allowance. Later inputs grow by each round's outputs (own output plus the
N-1 revealed peer outputs plus ``h``), so the worst case is

    N*m*(J + a) + N*(N*a + h)*m*(m-1)/2 <= R
    =>  a = (R - N*m*J - N*h*m*(m-1)/2) / (N*m*(1 + N*(m-1)/2))

With ``count = "output"`` inputs are free and ``a = R / (N*m)``. Unused allowance rolls over to
later phases because the plan is recomputed from the actual remaining budget each phase. When
even ``min_output`` per call is unaffordable for all ``m`` phases, the plan drops phases from the
end (fewer discussion rounds) instead of starving the current phase.

Before each call the ledger reserves ``input + max_output`` (or just ``max_output``), so a
team cannot exceed the budget even with concurrent calls; a call whose room falls below
``min_output`` is skipped and the agent keeps its previous answer. The only way to exceed the
budget is an input-token underestimate, which is recorded as an overrun.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def padded_input_estimate(raw_count: int, margin: float) -> int:
    """Pre-call input estimate used for budget decisions: the tokenizer count plus a margin."""
    return math.ceil(raw_count * (1 + margin)) + 16


def charged_tokens(count: str, input_tokens: int, output_tokens: int) -> int:
    """Tokens a call costs against a budget: input + output (``count = "total"``) or output."""
    return output_tokens + (input_tokens if count == "total" else 0)


@dataclass
class PhasePlan:
    phase: int
    remaining: int
    phases_left: int
    planned_phases: int  # phases the allowance was planned for (< phases_left if unaffordable)
    base_input: int
    overhead: int
    allowance: int  # per-call output allowance after the ceiling (0 if the phase is skipped)
    raw_allowance: int  # before the ceiling


@dataclass
class CallSlot:
    agent: int
    phase: int
    allowance: int
    input_estimate: int
    max_output_tokens: int
    skipped: bool = False
    skip_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None

    @property
    def hit_limit(self) -> bool:
        """The agent ran into the budget: skipped, or cut off at its allowance."""
        return self.skipped or self.finish_reason == "length"


@dataclass
class BudgetLedger:
    total: int
    team_size: int
    phases: int
    count: str = "total"
    ceiling: int = 8192
    min_output: int = 200
    input_margin: float = 0.02
    used: int = 0
    reserved: int = 0
    plans: list[PhasePlan] = field(default_factory=list)
    slots: list[CallSlot] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return self.total - self.used

    @property
    def overrun(self) -> int:
        return max(0, self.used - self.total)

    def estimate_input(self, raw_count: int) -> int:
        return padded_input_estimate(raw_count, self.input_margin)

    def plan_phase(self, phase: int, base_input: int, overhead: int) -> PhasePlan:
        rem = self.remaining - self.reserved
        phases_left = self.phases - phase
        # If the budget cannot carry every remaining phase, drop phases from the end of the
        # plan: a team that cannot afford discussion still answers independently.
        for planned in range(phases_left, 0, -1):
            raw = self._raw_allowance(rem, planned, base_input, overhead)
            if raw >= self.min_output:
                break
        allowance = min(raw, self.ceiling) if raw >= self.min_output else 0
        plan = PhasePlan(phase, rem, phases_left, planned, base_input, overhead, allowance, raw)
        self.plans.append(plan)
        return plan

    def _raw_allowance(self, rem: int, m: int, base_input: int, overhead: int) -> int:
        n = self.team_size
        if self.count == "output":
            return rem // (n * m)
        numerator = rem - n * m * base_input - n * overhead * m * (m - 1) // 2
        return math.floor(numerator / (n * m * (1 + n * (m - 1) / 2)))

    def reserve(self, agent: int, phase: int, allowance: int, input_estimate: int) -> CallSlot:
        room = self.total - self.used - self.reserved
        if self.count == "total":
            room -= input_estimate
        max_output = min(allowance, room)
        slot = CallSlot(agent, phase, allowance, input_estimate, max(0, max_output))
        if allowance <= 0 or max_output < self.min_output:
            slot.skipped = True
            slot.skip_reason = "budget_exhausted"
            slot.max_output_tokens = 0
        else:
            self.reserved += self._charge(input_estimate, max_output)
        self.slots.append(slot)
        return slot

    def skip(self, agent: int, phase: int, reason: str) -> CallSlot:
        slot = CallSlot(agent, phase, 0, 0, 0, skipped=True, skip_reason=reason)
        self.slots.append(slot)
        return slot

    def settle(
        self, slot: CallSlot, input_tokens: int, output_tokens: int, finish_reason: str
    ) -> None:
        self.reserved -= self._charge(slot.input_estimate, slot.max_output_tokens)
        self.used += self._charge(input_tokens, output_tokens)
        slot.input_tokens, slot.output_tokens, slot.finish_reason = (
            input_tokens,
            output_tokens,
            finish_reason,
        )

    def release(self, slot: CallSlot) -> None:
        """Undo a reservation whose call failed."""
        self.reserved -= self._charge(slot.input_estimate, slot.max_output_tokens)

    def _charge(self, input_tokens: int, output_tokens: int) -> int:
        return charged_tokens(self.count, input_tokens, output_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "count": self.count,
            "used": self.used,
            "overrun": self.overrun,
            "plans": [asdict(p) for p in self.plans],
            "calls": [asdict(s) | {"hit_limit": s.hit_limit} for s in self.slots],
            "any_agent_hit_limit": any(s.hit_limit for s in self.slots),
            "skipped_calls": sum(s.skipped for s in self.slots),
        }
