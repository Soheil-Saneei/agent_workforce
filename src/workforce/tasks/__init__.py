"""Benchmarks behind the common :class:`~workforce.tasks.base.Task` interface."""

from workforce.tasks import (  # noqa: F401 - register tasks
    aime,
    humaneval,
    livecodebench,
    musique,
    musr,
    toy,
)
from workforce.tasks.base import Item, Score, Task
from workforce.tasks.registry import create_task, register_task, registered_tasks

__all__ = ["Item", "Score", "Task", "create_task", "register_task", "registered_tasks"]
