"""First-party OpenAI adapter (Chat Completions API)."""

from __future__ import annotations

from collections.abc import Sequence

from workforce.models.base import Message
from workforce.models.openai_compat import OpenAICompatibleProvider
from workforce.models.registry import register_provider
from workforce.models.tokens import count_chat_tokens, encoding_for_model


@register_provider("openai")
class OpenAIProvider(OpenAICompatibleProvider):
    default_base_url = "https://api.openai.com/v1"
    # The project .env uses the second spelling; both are accepted.
    default_key_env = ("OPENAI_API_KEY", "OpenAI_API_Key")
    requires_api_key = True
    # Current OpenAI models reject the legacy ``max_tokens`` field.
    max_tokens_field = "max_completion_tokens"

    def count_tokens(self, messages: Sequence[Message]) -> int:
        return count_chat_tokens(messages, encoding_for_model(self.spec.model))
