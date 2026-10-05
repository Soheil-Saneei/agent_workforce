"""The single path from the protocol to a model: cache lookup, spend cap, concurrency, logging."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from workforce.models.base import (
    ChatRequest,
    ChatResponse,
    Message,
    ModelSpec,
    Provider,
    Usage,
)
from workforce.store import RunStore, now_iso


class SpendCapReached(Exception):
    """Starting the next call could push cumulative spend past the run's cap."""


class CacheMiss(Exception):
    """Offline replay needed a call that is not in the log."""


@dataclass(frozen=True)
class CallContext:
    trial_key: str
    condition: str
    benchmark: str
    item_id: str
    replication: int
    agent: int
    phase: int


@dataclass(frozen=True)
class CallOutcome:
    response: ChatResponse
    key: str
    cached: bool
    cost_usd: float | None


class SpendGuard:
    """Reserves the worst-case cost of a call before it starts, so concurrent calls can never
    jointly overshoot the cap; the reservation is replaced by the actual cost afterwards."""

    def __init__(self, cap_usd: float | None, already_spent: float = 0.0):
        self.cap = cap_usd
        self.spent = already_spent
        self.reserved = 0.0

    def reserve(self, worst_case: float) -> None:
        if self.cap is not None and self.spent + self.reserved + worst_case > self.cap:
            raise SpendCapReached(
                f"spend cap ${self.cap:.2f} reached (spent ${self.spent:.4f}, "
                f"in flight ${self.reserved:.4f}, next call up to ${worst_case:.4f})"
            )
        self.reserved += worst_case

    def settle(self, worst_case: float, actual: float) -> None:
        self.reserved -= worst_case
        self.spent += actual

    def release(self, worst_case: float) -> None:
        self.reserved -= worst_case


def call_key(spec: ModelSpec, request: ChatRequest, context: CallContext) -> str:
    material = {
        "model": spec.identity(),
        "request": request.to_dict(),
        "context": asdict(context),
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


class Gateway:
    def __init__(
        self,
        store: RunStore,
        specs: Mapping[str, ModelSpec],
        providers: Mapping[str, Provider],
        spend: SpendGuard,
        *,
        offline: bool = False,
    ):
        self.store = store
        self.specs = specs
        self.providers = providers
        self.spend = spend
        self.offline = offline
        self.stats = {"cached": 0, "paid": 0}
        self._semaphores: dict[tuple[int, str], asyncio.Semaphore] = {}

    def count_tokens(self, model_name: str, messages: Sequence[Message]) -> int:
        return self.providers[model_name].count_tokens(messages)

    def _semaphore(self, model_name: str) -> asyncio.Semaphore:
        slot = (id(asyncio.get_running_loop()), model_name)
        if slot not in self._semaphores:
            self._semaphores[slot] = asyncio.Semaphore(self.specs[model_name].concurrency)
        return self._semaphores[slot]

    async def complete(
        self, model_name: str, request: ChatRequest, context: CallContext
    ) -> CallOutcome:
        spec = self.specs[model_name]
        key = call_key(spec, request, context)
        record = self.store.cached_call(key)
        if record is not None:
            self.stats["cached"] += 1
            return CallOutcome(_response_from_record(record), key, True, record.get("cost_usd"))
        if self.offline:
            raise CacheMiss(f"call {key[:12]} ({context}) is not in the log")
        provider = self.providers[model_name]
        worst_case = 0.0
        if spec.price is not None:
            # 25% headroom on the input estimate; output can never exceed max_output_tokens.
            estimate = provider.count_tokens(request.messages)
            worst_case = spec.price.cost(int(estimate * 1.25), request.max_output_tokens)
        async with self._semaphore(model_name):
            self.spend.reserve(worst_case)
            started = now_iso()
            try:
                response = await provider.complete(request)
            except BaseException as exc:
                self.spend.release(worst_case)
                if isinstance(exc, Exception):
                    self.store.append_error(
                        {
                            "key": key,
                            "context": asdict(context),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                raise
            cost = (
                spec.price.cost(
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    response.usage.cached_input_tokens,
                )
                if spec.price is not None
                else None
            )
            self.spend.settle(worst_case, cost or 0.0)
        self.stats["paid"] += 1
        self.store.append_call(_record(key, spec, request, context, response, cost, started))
        return CallOutcome(response, key, False, cost)


def _record(
    key: str,
    spec: ModelSpec,
    request: ChatRequest,
    context: CallContext,
    response: ChatResponse,
    cost: float | None,
    started: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "started_at": started,
        "finished_at": now_iso(),
        **asdict(context),
        "model_name": spec.name,
        "provider": spec.provider,
        "model": spec.model,
        "params": spec.params,
        "seed": request.seed,
        "temperature": request.temperature,
        "max_output_tokens": request.max_output_tokens,
        "messages": [m.to_dict() for m in request.messages],
        "text": response.text,
        "finish_reason": response.finish_reason,
        "response_model": response.model,
        "response_id": response.response_id,
        "system_fingerprint": response.system_fingerprint,
        "usage": response.usage.to_dict(),
        "latency_s": response.latency_s,
        "attempts": response.attempts,
        "cost_usd": cost,
    }


def _response_from_record(record: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        text=record["text"],
        usage=Usage.from_dict(record["usage"]),
        finish_reason=record["finish_reason"],
        model=record.get("response_model") or record["model"],
        latency_s=record["latency_s"],
        attempts=record.get("attempts", 1),
        response_id=record.get("response_id"),
        system_fingerprint=record.get("system_fingerprint"),
    )
