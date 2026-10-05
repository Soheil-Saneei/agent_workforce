from __future__ import annotations

import asyncio
import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from workforce.design import ExperimentConfig, config_from_dict
from workforce.models.base import Provider
from workforce.runner import RunReport, execute
from workforce.store import RunStore

BASE_CONFIG: dict[str, Any] = {
    "experiment": {"name": "test", "seed": 11, "replications": 1},
    "benchmarks": {"toy_arithmetic": {"items": 4, "budget_tokens": 20000}},
    "factors": {
        "team_size": [1, 3],
        "diversity": ["identical", "moderate"],
        "communication": ["independent", "horizontal-2r"],
    },
    "generation": {"temperature": 0.7, "min_output_tokens": 20},
    "models": {"base": {"provider": "mock", "model": "mock-base"}},
}


def make_config(**sections: Any) -> ExperimentConfig:
    raw = copy.deepcopy(BASE_CONFIG)
    for key, value in sections.items():
        raw[key] = value
    return config_from_dict(raw)


def run_experiment(
    config: ExperimentConfig,
    run_dir: Path,
    providers: Mapping[str, Provider] | None = None,
    offline: bool = False,
) -> tuple[RunStore, RunReport]:
    store = (
        RunStore.open(run_dir)
        if (run_dir / "manifest.json").exists()
        else RunStore.create(run_dir, config)
    )
    report = asyncio.run(
        execute(store, config, providers=providers, offline=offline, log=lambda _: None)
    )
    return RunStore.open(run_dir), report


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "run"
