"""Secrets loading. Values are never logged or printed; only variable names are."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path

ENV_FILE_VAR = "WORKFORCE_ENV_FILE"

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        match = _LINE.match(raw)
        if not match:
            continue
        key, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        values[key] = value
    return values


def load_env_file(path: str | Path | None = None) -> Path | None:
    """Load KEY=VALUE pairs into ``os.environ`` without overriding variables already set.

    Resolution order: explicit ``path``, then ``$WORKFORCE_ENV_FILE``, then ``./.env`` if present.
    Returns the file that was loaded, or None.
    """
    candidate = path or os.environ.get(ENV_FILE_VAR)
    if candidate:
        env_path = Path(candidate).expanduser()
        if not env_path.is_file():
            raise FileNotFoundError(f"env file not found: {env_path}")
    else:
        env_path = Path(".env")
        if not env_path.is_file():
            return None
    for key, value in parse_env_file(env_path).items():
        os.environ.setdefault(key, value)
    return env_path


def resolve_secret(names: Sequence[str]) -> str | None:
    """Return the first non-empty environment variable among ``names``."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None
