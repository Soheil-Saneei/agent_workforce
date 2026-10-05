"""Run single trials against scripted mock agents."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from workforce.design import ExperimentConfig
from workforce.gateway import Gateway, SpendGuard
from workforce.models.base import ChatRequest, ModelSpec
from workforce.models.mock import MockProvider
from workforce.protocol.team import TeamProtocol, TrialSpec, derive_seed
from workforce.store import RunStore
from workforce.tasks import create_task

Responder = Callable[[ChatRequest, ModelSpec], str]


def agent_seed_map(config: ExperimentConfig, spec: TrialSpec, phases: int) -> dict[int, tuple]:
    """seed -> (agent, phase) for every call of a trial (seeds are unique per agent/phase)."""
    return {
        derive_seed(
            config.seed, "call", spec.benchmark, spec.item.item_id, spec.replication, agent, phase
        ): (agent, phase)
        for agent in range(spec.condition.team_size)
        for phase in range(phases)
    }


def run_trial(
    config: ExperimentConfig,
    tmp_path: Path,
    responders: Mapping[str, Responder] | None = None,
    condition_id: str | None = None,
    item_ordinal: int = 0,
    provider_cls: type[MockProvider] = MockProvider,
) -> tuple[dict[str, Any], dict[str, MockProvider], TrialSpec]:
    store = RunStore.create(tmp_path / f"trial-{len(list(tmp_path.iterdir()))}", config)
    providers = {
        name: provider_cls(spec, responder=(responders or {}).get(name))
        for name, spec in config.models.items()
    }
    gateway = Gateway(store, config.models, providers, SpendGuard(None))
    benchmark = next(iter(config.benchmarks))
    task = create_task(benchmark)
    items = task.load_items(config.benchmarks[benchmark].items, config.seed)
    conditions = {c.id: c for c in config.conditions()}
    condition = conditions[condition_id] if condition_id else next(iter(conditions.values()))
    spec = TrialSpec(benchmark, condition, items[item_ordinal], item_ordinal, 0)
    record = asyncio.run(TeamProtocol(config, task, gateway).run(spec))
    return record, providers, spec
