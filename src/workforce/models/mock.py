"""Deterministic offline provider used by the test suite and by ``--mock`` dry runs.

Tokens are whitespace-separated words, so budget arithmetic in tests is exact. The default
responder (:mod:`workforce.protocol.mock_agent`) speaks the protocol's output format and is a
pure function of (model, seed, messages): the same request always produces the same response.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Sequence

from workforce.models.base import (
    ChatRequest,
    ChatResponse,
    Message,
    ModelSpec,
    Provider,
    SleepFn,
    Usage,
)
from workforce.models.registry import register_provider

Responder = Callable[[ChatRequest, ModelSpec], str]


def mock_token_count(messages: Sequence[Message]) -> int:
    return sum(len(m.content.split()) + 4 for m in messages) + 3


def truncate_words(text: str, limit: int) -> tuple[str, int, bool]:
    """Keep the first ``limit`` whitespace-separated words, preserving the original layout."""
    words = list(re.finditer(r"\S+", text))
    if len(words) <= limit:
        return text, len(words), False
    if limit <= 0:
        return "", 0, True
    return text[: words[limit - 1].end()], limit, True


@register_provider("mock")
class MockProvider(Provider):
    def __init__(
        self, spec: ModelSpec, *, responder: Responder | None = None, sleep: SleepFn | None = None
    ):
        super().__init__(spec, sleep=sleep)
        if responder is None:
            from workforce.protocol.mock_agent import default_responder

            responder = default_responder
        self.responder = responder
        self.requests: list[ChatRequest] = []
        self.latency_s = float(spec.params.get("latency_s", 0.0))

    def count_tokens(self, messages: Sequence[Message]) -> int:
        return mock_token_count(messages)

    async def _complete_once(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        text = self.responder(request, self.spec)
        text, output_tokens, truncated = truncate_words(text, request.max_output_tokens)
        return ChatResponse(
            text=text,
            usage=Usage(
                input_tokens=mock_token_count(request.messages), output_tokens=output_tokens
            ),
            finish_reason="length" if truncated else "stop",
            model=self.spec.model,
            latency_s=self.latency_s,
        )
