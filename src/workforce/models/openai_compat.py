"""Adapter for any OpenAI-compatible ``/chat/completions`` endpoint (vLLM, Ollama, LM Studio,
OpenRouter, Together, ...). The first-party OpenAI adapter subclasses this one."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any

import httpx

from workforce.env import resolve_secret
from workforce.models.base import (
    ChatRequest,
    ChatResponse,
    FatalProviderError,
    Message,
    ModelSpec,
    Provider,
    QuotaExhaustedError,
    RetryableError,
    SleepFn,
    Usage,
)
from workforce.models.registry import register_provider

_QUOTA_CODES = {"insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached"}
# OpenAI's prompt-safety screen occasionally rejects benign prompts with HTTP 400
# "invalid_prompt"; in the team study 4 of ~4,000 HumanEval calls were flagged and all passed on a
# second attempt, so the flag is retried within the normal retry limit.
_FLAGGED_CODES = {"invalid_prompt"}
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")


def parse_reset_duration(value: str | None) -> float | None:
    """Parse ``Retry-After`` seconds or OpenAI reset durations such as ``6m0s`` or ``20ms``."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    total = 0.0
    matched = False
    for amount, unit in _DURATION.findall(value):
        matched = True
        total += float(amount) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total if matched else None


@register_provider("openai_compatible")
class OpenAICompatibleProvider(Provider):
    default_base_url: str | None = None
    default_key_env: tuple[str, ...] = ()
    requires_api_key = False
    max_tokens_field = "max_tokens"

    def __init__(
        self,
        spec: ModelSpec,
        *,
        client: httpx.AsyncClient | None = None,
        sleep: SleepFn | None = None,
    ):
        super().__init__(spec, sleep=sleep)
        base_url = spec.base_url or self.default_base_url
        if not base_url:
            raise ValueError(f"model {spec.name!r}: provider {spec.provider!r} needs base_url")
        self._key_names = spec.api_key_env or self.default_key_env
        api_key = resolve_secret(self._key_names) if self._key_names else None
        self._has_key = bool(api_key)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(spec.timeout_s, connect=15.0),
            limits=httpx.Limits(max_connections=max(4, spec.concurrency * 2)),
        )

    def check_credentials(self) -> None:
        if self.requires_api_key and not self._has_key:
            raise FatalProviderError(
                f"model {self.spec.name!r}: no API key found in any of "
                f"{', '.join(self._key_names)} (set it in the environment or pass --env-file)"
            )

    def build_body(self, request: ChatRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.spec.model,
            "messages": [m.to_dict() for m in request.messages],
            self.max_tokens_field: request.max_output_tokens,
        }
        if self.spec.supports_temperature and request.temperature is not None:
            body["temperature"] = request.temperature
        if self.spec.supports_seed and request.seed is not None:
            body["seed"] = request.seed
        body.update(self.spec.params)
        return body

    async def _complete_once(self, request: ChatRequest) -> ChatResponse:
        self.check_credentials()
        body = self.build_body(request)
        start = time.perf_counter()
        try:
            response = await self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise RetryableError(f"timeout: {type(exc).__name__}") from exc
        except httpx.TransportError as exc:
            raise RetryableError(f"transport error: {type(exc).__name__}: {exc}") from exc
        latency = time.perf_counter() - start
        self._raise_for_status(response)
        try:
            payload = response.json()
            choice = payload["choices"][0]
        except (ValueError, KeyError, IndexError) as exc:
            raise RetryableError(f"malformed response body: {response.text[:200]!r}") from exc
        text = (choice.get("message") or {}).get("content") or ""
        return ChatResponse(
            text=text,
            usage=self._usage(payload.get("usage"), request.messages, text),
            finish_reason=choice.get("finish_reason") or "unknown",
            model=payload.get("model", self.spec.model),
            latency_s=latency,
            response_id=payload.get("id"),
            system_fingerprint=payload.get("system_fingerprint"),
        )

    def _usage(self, usage: dict[str, Any] | None, messages: Sequence[Message], text: str) -> Usage:
        if not usage:
            return Usage(
                input_tokens=self.count_tokens(messages),
                output_tokens=self.count_tokens([Message("assistant", text)]),
                source="estimated",
            )
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        return Usage(
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            cached_input_tokens=int(prompt_details.get("cached_tokens") or 0),
            reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        try:
            error = response.json().get("error") or {}
        except ValueError:
            error = {}
        if not isinstance(error, dict):
            error = {"message": str(error)}
        message = f"HTTP {status}: {error.get('message') or response.text[:200]}"
        code = error.get("code") or error.get("type")
        if status == 429 and code in _QUOTA_CODES:
            raise QuotaExhaustedError(message)
        flagged = code in _FLAGGED_CODES or str(error.get("message", "")).startswith(
            "Invalid prompt:"
        )
        if status == 400 and flagged:
            raise RetryableError(message)
        if status in (408, 409, 429) or status >= 500:
            retry_after = parse_reset_duration(response.headers.get("retry-after"))
            if retry_after is None:
                retry_ms = response.headers.get("retry-after-ms")
                retry_after = float(retry_ms) / 1000 if retry_ms else None
            if retry_after is None and status == 429:
                retry_after = parse_reset_duration(
                    response.headers.get("x-ratelimit-reset-tokens")
                    or response.headers.get("x-ratelimit-reset-requests")
                )
            raise RetryableError(message, retry_after=retry_after)
        raise FatalProviderError(message)

    async def aclose(self) -> None:
        await self._client.aclose()
