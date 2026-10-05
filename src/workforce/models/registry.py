"""Provider registry. Adapters register themselves by name with :func:`register_provider`.

A model spec may also name an out-of-tree adapter as ``"package.module:ClassName"``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from workforce.models.base import ModelSpec, Provider

_PROVIDERS: dict[str, type[Provider]] = {}


def register_provider(name: str) -> Callable[[type[Provider]], type[Provider]]:
    def decorator(cls: type[Provider]) -> type[Provider]:
        _PROVIDERS[name] = cls
        return cls

    return decorator


def provider_class(name: str) -> type[Provider]:
    if name in _PROVIDERS:
        return _PROVIDERS[name]
    if ":" in name:
        module_name, _, attr = name.partition(":")
        return getattr(importlib.import_module(module_name), attr)
    known = ", ".join(sorted(_PROVIDERS))
    raise KeyError(f"unknown provider {name!r} (known: {known}, or use 'module:Class')")


def create_provider(spec: ModelSpec, **kwargs: Any) -> Provider:
    return provider_class(spec.provider)(spec, **kwargs)


def registered_providers() -> list[str]:
    return sorted(_PROVIDERS)
