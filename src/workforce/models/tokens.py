"""Pre-call token counting for budget planning."""

from __future__ import annotations

import functools
from collections.abc import Sequence

from workforce.models.base import Message

# Chat formatting overhead per message and for priming the reply (OpenAI chat format).
TOKENS_PER_MESSAGE = 4
REPLY_PRIMING_TOKENS = 3


@functools.cache
def _encoding(name: str):
    import tiktoken

    return tiktoken.get_encoding(name)


def encoding_for_model(model: str) -> str:
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model).name
    except KeyError:
        # Every current OpenAI chat model uses o200k_base.
        return "o200k_base"


def count_chat_tokens(messages: Sequence[Message], encoding: str = "o200k_base") -> int:
    enc = _encoding(encoding)
    total = REPLY_PRIMING_TOKENS
    for message in messages:
        total += TOKENS_PER_MESSAGE + len(enc.encode(message.content, disallowed_special=()))
    return total


def count_text_tokens(text: str, encoding: str = "o200k_base") -> int:
    return len(_encoding(encoding).encode(text, disallowed_special=()))
