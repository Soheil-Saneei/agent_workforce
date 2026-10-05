"""Provider adapters against a fake HTTP transport (no network)."""

import asyncio
import json

import httpx
import pytest

from workforce.env import load_env_file, parse_env_file
from workforce.models import create_provider
from workforce.models.base import (
    ChatRequest,
    FatalProviderError,
    Message,
    ModelSpec,
    ProviderError,
    QuotaExhaustedError,
)
from workforce.models.openai_compat import parse_reset_duration
from workforce.models.pricing import lookup_price

REQUEST = ChatRequest((Message("system", "s"), Message("user", "u")), 64, 0.7, 42)


def ok_body(text="hi", **usage):
    return {
        "id": "chatcmpl-1",
        "model": "gpt-6-luna-2026",
        "system_fingerprint": "fp",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": usage.get("cached", 0)},
            "completion_tokens_details": {"reasoning_tokens": usage.get("reasoning", 0)},
        },
    }


def provider(handler, monkeypatch, **spec_fields):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fields = {"name": "base", "provider": "openai", "model": "gpt-6-luna", **spec_fields}
    spec = ModelSpec(**fields)
    client = httpx.AsyncClient(
        base_url="https://api.test/v1",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer sk-test"},
    )
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    return create_provider(spec, client=client, sleep=fake_sleep), sleeps


def run(coro):
    return asyncio.run(coro)


def test_openai_request_body_and_usage(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json=ok_body(cached=8, reasoning=0))

    p, _ = provider(handler, monkeypatch, params={"reasoning_effort": "none"})
    response = run(p.complete(REQUEST))
    assert seen["max_completion_tokens"] == 64 and "max_tokens" not in seen
    assert seen["seed"] == 42 and seen["temperature"] == 0.7
    assert seen["reasoning_effort"] == "none"
    assert seen["messages"][1] == {"role": "user", "content": "u"}
    assert seen["auth"] == "Bearer sk-test"
    assert response.text == "hi" and response.model == "gpt-6-luna-2026"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (12, 3)
    assert response.usage.cached_input_tokens == 8
    assert response.system_fingerprint == "fp"


def test_unsupported_sampling_parameters_are_omitted(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=ok_body())

    p, _ = provider(handler, monkeypatch, supports_temperature=False, supports_seed=False)
    run(p.complete(REQUEST))
    assert "temperature" not in seen and "seed" not in seen


def test_rate_limits_and_server_errors_are_retried_with_backoff(monkeypatch):
    responses = iter(
        [
            httpx.Response(
                429,
                json={"error": {"code": "rate_limit_exceeded", "message": "slow"}},
                headers={"retry-after": "7"},
            ),
            httpx.Response(503, json={"error": {"message": "overloaded"}}),
            httpx.Response(200, json=ok_body()),
        ]
    )
    p, sleeps = provider(lambda r: next(responses), monkeypatch)
    response = run(p.complete(REQUEST))
    assert response.attempts == 3
    assert sleeps[0] >= 7  # Retry-After is honoured
    assert len(sleeps) == 2


def test_spurious_prompt_flags_are_retried(monkeypatch):
    flagged = {"error": {"code": "invalid_prompt", "message": "Invalid prompt: flagged"}}
    by_message = {"error": {"code": None, "message": "Invalid prompt: your prompt was flagged"}}
    responses = iter(
        [
            httpx.Response(400, json=flagged),
            httpx.Response(400, json=by_message),
            httpx.Response(200, json=ok_body()),
        ]
    )
    p, sleeps = provider(lambda r: next(responses), monkeypatch)
    assert run(p.complete(REQUEST)).attempts == 3
    assert len(sleeps) == 2


def test_timeouts_are_retried_then_give_up(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    p, sleeps = provider(handler, monkeypatch, max_retries=2)
    with pytest.raises(ProviderError, match="gave up after 3 attempts"):
        run(p.complete(REQUEST))
    assert len(sleeps) == 2


def test_quota_exhaustion_and_bad_requests_are_fatal(monkeypatch):
    quota = {"error": {"code": "insufficient_quota", "message": "no credits"}}
    p, sleeps = provider(lambda r: httpx.Response(429, json=quota), monkeypatch)
    with pytest.raises(QuotaExhaustedError):
        run(p.complete(REQUEST))
    p, sleeps = provider(
        lambda r: httpx.Response(400, json={"error": {"message": "bad"}}), monkeypatch
    )
    with pytest.raises(FatalProviderError, match="HTTP 400"):
        run(p.complete(REQUEST))
    assert sleeps == []


def test_openai_compatible_adapter_uses_max_tokens_and_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        seen["auth"] = request.headers.get("authorization")
        body = ok_body()
        del body["usage"]  # some local servers omit usage
        return httpx.Response(200, json=body)

    spec = ModelSpec(
        "local", "openai_compatible", "llama-3.1-8b", base_url="http://localhost:8000/v1"
    )
    p = create_provider(
        spec,
        client=httpx.AsyncClient(
            base_url="http://localhost:8000/v1", transport=httpx.MockTransport(handler)
        ),
    )
    p.check_credentials()
    response = run(p.complete(REQUEST))
    assert seen["max_tokens"] == 64 and seen["auth"] is None
    assert response.usage.source == "estimated" and response.usage.input_tokens > 0


def test_missing_key_is_reported_before_a_run(monkeypatch):
    for name in ("OPENAI_API_KEY", "OpenAI_API_Key"):
        monkeypatch.delenv(name, raising=False)
    p = create_provider(ModelSpec("base", "openai", "gpt-6-luna"))
    with pytest.raises(FatalProviderError, match="OPENAI_API_KEY, OpenAI_API_Key"):
        p.check_credentials()


def test_env_file_accepts_both_key_spellings(tmp_path, monkeypatch):
    for name in ("OPENAI_API_KEY", "OpenAI_API_Key", "WORKFORCE_ENV_FILE"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / "keys.env"
    env.write_text('# comment\nexport OpenAI_API_Key="sk-file"\nOTHER=1 # trailing\n')
    assert parse_env_file(env) == {"OpenAI_API_Key": "sk-file", "OTHER": "1"}
    monkeypatch.setenv("WORKFORCE_ENV_FILE", str(env))
    assert load_env_file() == env
    p = create_provider(ModelSpec("base", "openai", "gpt-6-luna"))
    p.check_credentials()  # found under the non-standard spelling


def test_reset_durations_and_prices():
    assert parse_reset_duration("6m0s") == 360
    assert parse_reset_duration("20ms") == pytest.approx(0.02)
    assert parse_reset_duration("1.5") == 1.5
    assert parse_reset_duration(None) is None
    assert lookup_price("openai", "gpt-4.1-mini-2025-04-14") == lookup_price(
        "openai", "gpt-4.1-mini"
    )
    assert lookup_price("openai_compatible", "llama") is None
    luna = lookup_price("openai", "gpt-6-luna")
    assert luna.cost(1_000_000, 1_000_000, 500_000) == pytest.approx(0.05 + 0.005 + 0.5)
