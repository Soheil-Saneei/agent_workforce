"""Declarative experiment configs (TOML) and their expansion into the factorial grid."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from workforce.models.base import ModelSpec, Price
from workforce.models.pricing import lookup_price
from workforce.protocol.procedures import (
    DEFAULT_DIVERSITY_LEVELS,
    DEFAULT_PROCEDURES,
    Procedure,
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Communication:
    mode: str  # "independent" | "horizontal"
    rounds: int

    @property
    def id(self) -> str:
        return "independent" if self.mode == "independent" else f"horizontal-{self.rounds}r"

    @classmethod
    def parse(cls, value: str) -> Communication:
        if value in ("independent", "none"):
            return cls("independent", 0)
        match = re.fullmatch(r"horizontal(?:-(\d+)r)?", value)
        if not match:
            raise ConfigError(
                f"communication {value!r}: use 'independent', 'horizontal' or 'horizontal-<k>r'"
            )
        rounds = int(match.group(1) or 1)
        if rounds < 1:
            raise ConfigError("horizontal communication needs at least one round")
        return cls("horizontal", rounds)


@dataclass(frozen=True)
class Condition:
    team_size: int
    diversity: str
    communication: Communication
    model_assignment: str

    @property
    def id(self) -> str:
        return f"n{self.team_size}-{self.diversity}-{self.communication.id}-{self.model_assignment}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "team_size": self.team_size,
            "diversity": self.diversity,
            "communication": self.communication.id,
            "rounds": self.communication.rounds,
            "model_assignment": self.model_assignment,
        }


@dataclass(frozen=True)
class DiversityLevel:
    name: str
    procedures: tuple[str, ...]
    by_team_size: dict[int, tuple[str, ...]] = field(default_factory=dict)
    counterbalance: bool = True

    def for_size(self, team_size: int) -> tuple[str, ...]:
        return self.by_team_size.get(team_size, self.procedures)


@dataclass(frozen=True)
class ModelAssignment:
    name: str
    models: tuple[str, ...]
    counterbalance: bool = True


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    items: int | None
    budget_tokens: int
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float | None = 0.7
    max_output_tokens_per_call: int = 8192
    min_output_tokens: int = 200


@dataclass(frozen=True)
class BudgetConfig:
    count: str = "total"  # "total" (input + output) or "output"
    disclose_allowance: bool = True
    input_margin: float = 0.02


@dataclass(frozen=True)
class RunConfig:
    trial_concurrency: int = 8
    max_spend_usd: float | None = None
    include_degenerate: bool = False


@dataclass(frozen=True)
class AgentAssignment:
    index: int
    procedure: str
    model: str


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    replications: int
    benchmarks: dict[str, BenchmarkConfig]
    team_sizes: tuple[int, ...]
    diversity_factor: tuple[str, ...]
    communication_factor: tuple[Communication, ...]
    model_assignment_factor: tuple[str, ...]
    generation: GenerationConfig
    budget: BudgetConfig
    run: RunConfig
    models: dict[str, ModelSpec]
    model_assignments: dict[str, ModelAssignment]
    diversity_levels: dict[str, DiversityLevel]
    procedures: dict[str, Procedure]
    raw: dict[str, Any]
    description: str = ""

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:16]

    def is_degenerate(self, condition: Condition) -> bool:
        """A single agent cannot be procedurally diverse or mixed-model."""
        if condition.team_size > 1:
            return False
        procedures = set(self.diversity_levels[condition.diversity].for_size(1))
        models = set(self.model_assignments[condition.model_assignment].models)
        return len(procedures) > 1 or len(models) > 1

    def conditions(self) -> list[Condition]:
        grid = [
            Condition(n, d, c, m)
            for n, d, c, m in itertools.product(
                self.team_sizes,
                self.diversity_factor,
                self.communication_factor,
                self.model_assignment_factor,
            )
        ]
        if self.run.include_degenerate:
            return grid
        return [c for c in grid if not self.is_degenerate(c)]

    def skipped_conditions(self) -> list[Condition]:
        if self.run.include_degenerate:
            return []
        return [
            Condition(n, d, c, m)
            for n, d, c, m in itertools.product(
                self.team_sizes,
                self.diversity_factor,
                self.communication_factor,
                self.model_assignment_factor,
            )
            if self.is_degenerate(Condition(n, d, c, m))
        ]

    def assign_agents(
        self, condition: Condition, item_ordinal: int, replication: int
    ) -> list[AgentAssignment]:
        """Procedure and model for each agent.

        With counterbalancing, which procedures are duplicated (N > number of procedures) and
        which procedure is paired with which model both rotate across items, so that over
        consecutive items every pairing occurs equally often and model effects cannot be
        confounded with procedure effects.
        """
        level = self.diversity_levels[condition.diversity]
        procedures = level.for_size(condition.team_size)
        models = self.model_assignments[condition.model_assignment]
        t = item_ordinal + replication
        p_offset = t % len(procedures) if level.counterbalance else 0
        m_offset = (t // len(procedures)) % len(models.models) if models.counterbalance else 0
        return [
            AgentAssignment(
                index=i,
                procedure=procedures[(i + p_offset) % len(procedures)],
                model=models.models[(i + m_offset) % len(models.models)],
            )
            for i in range(condition.team_size)
        ]


_TOP_LEVEL = {
    "experiment",
    "benchmarks",
    "factors",
    "generation",
    "budget",
    "run",
    "models",
    "model_assignments",
    "diversity",
    "procedures",
}


def _check_keys(table: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(table) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")


def _parse_price(raw: dict[str, Any] | None, provider: str, model: str) -> Price | None:
    if raw is None:
        return lookup_price(provider, model)
    _check_keys(raw, {"input", "output", "cached_input"}, f"price of {model}")
    return Price(float(raw["input"]), float(raw["output"]), raw.get("cached_input"))


def _parse_model(name: str, raw: dict[str, Any]) -> ModelSpec:
    allowed = {
        "provider",
        "model",
        "base_url",
        "api_key_env",
        "supports_temperature",
        "supports_seed",
        "max_output_tokens",
        "params",
        "price",
        "timeout_s",
        "max_retries",
        "concurrency",
    }
    _check_keys(raw, allowed, f"models.{name}")
    for required in ("provider", "model"):
        if required not in raw:
            raise ConfigError(f"models.{name}: missing {required!r}")
    key_env = raw.get("api_key_env", ())
    return ModelSpec(
        name=name,
        provider=raw["provider"],
        model=raw["model"],
        base_url=raw.get("base_url"),
        api_key_env=(key_env,) if isinstance(key_env, str) else tuple(key_env),
        supports_temperature=bool(raw.get("supports_temperature", True)),
        supports_seed=bool(raw.get("supports_seed", True)),
        max_output_tokens=raw.get("max_output_tokens"),
        params=dict(raw.get("params", {})),
        price=_parse_price(raw.get("price"), raw["provider"], raw["model"]),
        timeout_s=float(raw.get("timeout_s", 180.0)),
        max_retries=int(raw.get("max_retries", 8)),
        concurrency=int(raw.get("concurrency", 8)),
    )


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return config_from_dict(apply_overrides(raw, overrides or {}))


def apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """CLI overrides: ``mock`` (swap every model for the offline mock), ``items``, ``max_spend``."""
    raw = copy.deepcopy(raw)
    if overrides.get("mock"):
        for spec in raw.get("models", {}).values():
            spec["model"] = f"mock-{spec['model']}"
            spec["provider"] = "mock"
            spec.pop("base_url", None)
            spec.pop("price", None)
    if overrides.get("items") is not None:
        for bench in raw.get("benchmarks", {}).values():
            bench["items"] = int(overrides["items"])
    if overrides.get("max_spend") is not None:
        raw.setdefault("run", {})["max_spend_usd"] = float(overrides["max_spend"])
    return raw


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    _check_keys(raw, _TOP_LEVEL, "config")
    experiment = raw.get("experiment", {})
    _check_keys(experiment, {"name", "seed", "replications", "description"}, "experiment")

    procedures = dict(DEFAULT_PROCEDURES)
    for name, table in raw.get("procedures", {}).items():
        _check_keys(table, {"instructions", "discussion_focus"}, f"procedures.{name}")
        procedures[name] = Procedure(name, table["instructions"], table.get("discussion_focus", ""))

    diversity_levels = {
        name: DiversityLevel(name, procs) for name, procs in DEFAULT_DIVERSITY_LEVELS.items()
    }
    for name, table in raw.get("diversity", {}).items():
        _check_keys(table, {"procedures", "by_team_size", "counterbalance"}, f"diversity.{name}")
        base = diversity_levels.get(name)
        procs = tuple(table.get("procedures", base.procedures if base else ()))
        if not procs:
            raise ConfigError(f"diversity.{name}: needs a non-empty 'procedures' list")
        diversity_levels[name] = DiversityLevel(
            name,
            procs,
            {int(k): tuple(v) for k, v in table.get("by_team_size", {}).items()},
            bool(table.get("counterbalance", True)),
        )

    models = {name: _parse_model(name, table) for name, table in raw.get("models", {}).items()}
    if not models:
        raise ConfigError("config: define at least one model under [models.<name>]")
    assignments: dict[str, ModelAssignment] = {}
    for name, table in raw.get("model_assignments", {}).items():
        _check_keys(table, {"models", "counterbalance"}, f"model_assignments.{name}")
        assignments[name] = ModelAssignment(
            name, tuple(table["models"]), bool(table.get("counterbalance", True))
        )
    if not assignments:
        assignments["homogeneous"] = ModelAssignment("homogeneous", (next(iter(models)),))

    generation_raw = raw.get("generation", {})
    _check_keys(
        generation_raw,
        {"temperature", "max_output_tokens_per_call", "min_output_tokens"},
        "generation",
    )
    generation = GenerationConfig(**generation_raw)
    budget_raw = raw.get("budget", {})
    _check_keys(
        budget_raw, {"count", "disclose_allowance", "input_margin", "tokens_per_problem"}, "budget"
    )
    default_budget = budget_raw.get("tokens_per_problem")
    budget = BudgetConfig(**{k: v for k, v in budget_raw.items() if k != "tokens_per_problem"})
    if budget.count not in ("total", "output"):
        raise ConfigError("budget.count must be 'total' or 'output'")
    run_raw = raw.get("run", {})
    _check_keys(run_raw, {"trial_concurrency", "max_spend_usd", "include_degenerate"}, "run")
    run = RunConfig(**run_raw)

    benchmarks: dict[str, BenchmarkConfig] = {}
    for name, table in raw.get("benchmarks", {}).items():
        _check_keys(table, {"items", "budget_tokens", "options"}, f"benchmarks.{name}")
        tokens = table.get("budget_tokens", default_budget)
        if not tokens or tokens <= 0:
            raise ConfigError(f"benchmarks.{name}: needs a positive budget_tokens")
        benchmarks[name] = BenchmarkConfig(
            name, table.get("items"), int(tokens), dict(table.get("options", {}))
        )
    if not benchmarks:
        raise ConfigError("config: define at least one benchmark under [benchmarks.<name>]")

    factors = raw.get("factors", {})
    _check_keys(factors, {"team_size", "diversity", "communication", "model_assignment"}, "factors")
    config = ExperimentConfig(
        name=experiment.get("name", "experiment"),
        seed=int(experiment.get("seed", 0)),
        replications=int(experiment.get("replications", 1)),
        description=experiment.get("description", ""),
        benchmarks=benchmarks,
        team_sizes=tuple(int(n) for n in factors.get("team_size", [1, 3, 5])),
        diversity_factor=tuple(factors.get("diversity", ["identical", "moderate"])),
        communication_factor=tuple(
            Communication.parse(c) for c in factors.get("communication", ["independent"])
        ),
        model_assignment_factor=tuple(factors.get("model_assignment", list(assignments)[:1])),
        generation=generation,
        budget=budget,
        run=run,
        models=models,
        model_assignments=assignments,
        diversity_levels=diversity_levels,
        procedures=procedures,
        raw=raw,
    )
    _validate(config)
    return config


def _validate(config: ExperimentConfig) -> None:
    from workforce.tasks import registered_tasks

    for name in config.benchmarks:
        if name not in registered_tasks():
            raise ConfigError(f"unknown benchmark {name!r} (known: {registered_tasks()})")
    if any(n < 1 for n in config.team_sizes):
        raise ConfigError("team sizes must be >= 1")
    if config.replications < 1:
        raise ConfigError("replications must be >= 1")
    for level in config.diversity_factor:
        if level not in config.diversity_levels:
            raise ConfigError(f"unknown diversity level {level!r}")
        spec = config.diversity_levels[level]
        for procs in (spec.procedures, *spec.by_team_size.values()):
            for proc in procs:
                if proc not in config.procedures:
                    raise ConfigError(f"diversity.{level}: unknown procedure {proc!r}")
    for name in config.model_assignment_factor:
        if name not in config.model_assignments:
            raise ConfigError(f"unknown model assignment {name!r}")
        for model in config.model_assignments[name].models:
            if model not in config.models:
                raise ConfigError(f"model_assignments.{name}: unknown model {model!r}")
    if config.run.max_spend_usd is not None:
        missing = [
            spec.name
            for spec in config.models.values()
            if spec.price is None and _model_used(config, spec.name)
        ]
        if missing:
            raise ConfigError(
                f"run.max_spend_usd is set but no price is known for model(s) {missing}; "
                "add a price table to each"
            )


def _model_used(config: ExperimentConfig, model: str) -> bool:
    return any(model in config.model_assignments[a].models for a in config.model_assignment_factor)
