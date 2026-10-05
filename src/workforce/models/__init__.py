"""Pluggable model layer.

To add a provider, write one module with a :class:`Provider` subclass decorated with
``@register_provider("name")`` and import it below (or reference it from a config as
``provider = "package.module:ClassName"`` without touching this package).
"""

from workforce.models import mock, openai, openai_compat  # noqa: F401 - register adapters
from workforce.models.base import (
    ChatRequest,
    ChatResponse,
    FatalProviderError,
    Message,
    ModelSpec,
    Price,
    Provider,
    ProviderError,
    QuotaExhaustedError,
    RetryableError,
    Usage,
)
from workforce.models.registry import create_provider, register_provider, registered_providers

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "FatalProviderError",
    "Message",
    "ModelSpec",
    "Price",
    "Provider",
    "ProviderError",
    "QuotaExhaustedError",
    "RetryableError",
    "Usage",
    "create_provider",
    "register_provider",
    "registered_providers",
]
