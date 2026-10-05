"""Provider-agnostic chat-completion interface.

Every adapter turns a :class:`ChatRequest` into a :class:`ChatResponse` that carries the
text plus the provider-reported token usage. Retries, timeouts and rate-limit backoff live
here, once, so adapters only implement a single attempt (:meth:`Provider._complete_once`).
"""

from __future__ import annotations

import asyncio
import math
import random
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class Price:
    """USD per one million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None

    def cost(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
        cached = min(cached_input_tokens, input_tokens)
        cached_rate = (
            self.cached_input_per_mtok
            if self.cached_input_per_mtok is not None
            else self.input_per_mtok
        )
        return (
            (input_tokens - cached) * self.input_per_mtok
            + cached * cached_rate
            + output_tokens * self.output_per_mtok
        ) / 1_000_000

    def to_dict(self) -> dict[str, float | None]:
        return {
            "input_per_mtok": self.input_per_mtok,
            "output_per_mtok": self.output_per_mtok,
            "cached_input_per_mtok": self.cached_input_per_mtok,
        }


@dataclass(frozen=True)
class ModelSpec:
    """One model as an experiment sees it.

    ``name`` is the logical name used in configs and logs (for example ``base``); ``model`` is
    the provider-side identifier. ``params`` are extra provider-specific request fields (for
    example ``reasoning_effort``) and are part of the call cache key.
    """

    name: str
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: tuple[str, ...] = ()
    supports_temperature: bool = True
    supports_seed: bool = True
    max_output_tokens: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    price: Price | None = None
    timeout_s: float = 180.0
    max_retries: int = 8
    concurrency: int = 8

    def identity(self) -> dict[str, Any]:
        """Fields that change what the model returns (used for the call cache key)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "params": self.params,
        }


@dataclass(frozen=True)
class ChatRequest:
    messages: tuple[Message, ...]
    max_output_tokens: int
    temperature: float | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    source: str = "provider"  # "provider" when reported by the API, "estimated" otherwise

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Usage:
        return cls(
            input_tokens=int(d["input_tokens"]),
            output_tokens=int(d["output_tokens"]),
            cached_input_tokens=int(d.get("cached_input_tokens", 0)),
            reasoning_tokens=int(d.get("reasoning_tokens", 0)),
            source=d.get("source", "provider"),
        )


@dataclass(frozen=True)
class ChatResponse:
    text: str
    usage: Usage
    finish_reason: str  # "stop", "length", "content_filter", ...
    model: str  # model id reported by the provider
    latency_s: float
    attempts: int = 1
    response_id: str | None = None
    system_fingerprint: str | None = None


class ProviderError(Exception):
    """Base class for provider failures."""


class RetryableError(ProviderError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class FatalProviderError(ProviderError):
    """Not worth retrying (bad request, auth failure, unknown model)."""


class QuotaExhaustedError(FatalProviderError):
    """The account cannot pay for more calls. Stops the whole run."""


SleepFn = Callable[[float], Awaitable[None]]


class Provider(ABC):
    """A chat-completion backend. Subclasses implement one attempt; this class retries."""

    base_delay_s: float = 1.0
    max_delay_s: float = 60.0

    def __init__(self, spec: ModelSpec, *, sleep: SleepFn | None = None):
        self.spec = spec
        self._sleep: SleepFn = sleep or asyncio.sleep
        self._jitter = random.Random()

    @abstractmethod
    async def _complete_once(self, request: ChatRequest) -> ChatResponse:
        """Make exactly one attempt. Raise RetryableError or FatalProviderError on failure."""

    def check_credentials(self) -> None:  # noqa: B027 - optional hook
        """Raise FatalProviderError if the provider cannot authenticate (checked before a run)."""

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Estimate the input tokens of ``messages`` before sending them.

        Used only to plan the budget; the logged usage always comes from the provider.
        The default is a deliberately conservative character heuristic.
        """
        return sum(math.ceil(len(m.content) / 3) + 4 for m in messages) + 3

    async def complete(self, request: ChatRequest) -> ChatResponse:
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._complete_once(request)
            except RetryableError as exc:
                if attempt > self.spec.max_retries:
                    raise ProviderError(
                        f"{self.spec.name}: gave up after {attempt} attempts: {exc}"
                    ) from exc
                await self._sleep(self._backoff(attempt, exc.retry_after))
                continue
            return _with_attempts(response, attempt)

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        ceiling = min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1))
        delay = self._jitter.uniform(ceiling / 2, ceiling)
        if retry_after is not None:
            delay = max(delay, retry_after)
        return delay

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""


def _with_attempts(response: ChatResponse, attempts: int) -> ChatResponse:
    if response.attempts == attempts:
        return response
    return ChatResponse(
        text=response.text,
        usage=response.usage,
        finish_reason=response.finish_reason,
        model=response.model,
        latency_s=response.latency_s,
        attempts=attempts,
        response_id=response.response_id,
        system_fingerprint=response.system_fingerprint,
    )
