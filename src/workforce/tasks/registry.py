"""Benchmark registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from workforce.tasks.base import Task

_TASKS: dict[str, type[Task]] = {}


def register_task(name: str) -> Callable[[type[Task]], type[Task]]:
    def decorator(cls: type[Task]) -> type[Task]:
        _TASKS[name] = cls
        return cls

    return decorator


def create_task(name: str, **options: Any) -> Task:
    if name not in _TASKS:
        raise KeyError(f"unknown benchmark {name!r} (known: {', '.join(sorted(_TASKS))})")
    return _TASKS[name](**options)


def registered_tasks() -> list[str]:
    return sorted(_TASKS)
