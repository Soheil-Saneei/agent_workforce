"""Runs every trial of an experiment, resumably, under a spend cap."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from workforce.design import ExperimentConfig
from workforce.gateway import CacheMiss, Gateway, SpendCapReached, SpendGuard
from workforce.models import create_provider
from workforce.models.base import Provider, QuotaExhaustedError
from workforce.protocol.team import TeamProtocol, TrialSpec
from workforce.store import RunStore, now_iso
from workforce.tasks import Item, Task, create_task

MAX_FAILURES = 25


@dataclass
class RunReport:
    run_dir: Path
    total_trials: int
    already_done: int
    completed: int = 0
    failed: int = 0
    stop_reason: str = "finished"
    spent_usd: float = 0.0
    calls_paid: int = 0
    calls_cached: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return self.total_trials - self.already_done - self.completed


def build_trials(config: ExperimentConfig, items: Mapping[str, list[Item]]) -> list[TrialSpec]:
    """All trials, ordered so that an interrupted run stays balanced across conditions:
    every condition of an item finishes before later items start."""
    conditions = config.conditions()
    longest = max(len(v) for v in items.values())
    trials = []
    for replication in range(config.replications):
        for ordinal in range(longest):
            for benchmark in config.benchmarks:
                if ordinal >= len(items[benchmark]):
                    continue
                for condition in conditions:
                    trials.append(
                        TrialSpec(
                            benchmark, condition, items[benchmark][ordinal], ordinal, replication
                        )
                    )
    return trials


def load_items(store: RunStore, config: ExperimentConfig, tasks: Mapping[str, Task]) -> dict:
    items = {}
    for name, bench in config.benchmarks.items():
        stored = store.items(name)
        if stored is None:
            stored = tasks[name].load_items(bench.items, config.seed)
            store.save_items(name, stored)
        items[name] = stored
    return items


def used_models(config: ExperimentConfig) -> list[str]:
    return sorted(
        {m for a in config.model_assignment_factor for m in config.model_assignments[a].models}
    )


async def execute(
    store: RunStore,
    config: ExperimentConfig,
    *,
    providers: Mapping[str, Provider] | None = None,
    offline: bool = False,
    log: Callable[[str], None] | None = None,
) -> RunReport:
    log = log or (lambda message: print(message, file=sys.stderr, flush=True))
    tasks = {name: create_task(name, **b.options) for name, b in config.benchmarks.items()}
    items = load_items(store, config, tasks)
    model_names = used_models(config)
    specs = {m: config.models[m] for m in model_names}
    owned = providers is None
    providers = providers or {m: create_provider(specs[m]) for m in model_names}
    if not offline:
        for provider in providers.values():
            provider.check_credentials()
    gateway = Gateway(
        store,
        specs,
        providers,
        SpendGuard(config.run.max_spend_usd, store.spent_usd),
        offline=offline,
    )
    protocols = {name: TeamProtocol(config, task, gateway) for name, task in tasks.items()}
    trials = build_trials(config, items)
    pending = [t for t in trials if not store.has_trial(t.key)]
    report = RunReport(store.dir, len(trials), len(trials) - len(pending))
    store.append_event(
        "run_started",
        pending=len(pending),
        done=report.already_done,
        spent_usd=store.spent_usd,
        offline=offline,
    )
    log(
        f"{store.dir}: {len(trials)} trials ({report.already_done} already done), "
        f"spent so far ${store.spent_usd:.4f}"
        + (f", cap ${config.run.max_spend_usd:.2f}" if config.run.max_spend_usd else "")
    )
    stop = asyncio.Event()
    gate = asyncio.Semaphore(config.run.trial_concurrency)

    def halt(reason: str, detail: str) -> None:
        if not stop.is_set():
            report.stop_reason = reason
            stop.set()
            log(f"stopping: {detail}")

    async def run_one(trial: TrialSpec) -> None:
        async with gate:
            if stop.is_set():
                return
            try:
                record = await protocols[trial.benchmark].run(trial)
            except SpendCapReached as exc:
                halt("spend_cap", str(exc))
                return
            except QuotaExhaustedError as exc:
                halt("quota_exhausted", str(exc))
                return
            except (CacheMiss, Exception) as exc:  # one trial failing must not kill the run
                report.failed += 1
                report.failures.append(f"{trial.key}: {type(exc).__name__}: {exc}")
                store.append_error(
                    {"trial_key": trial.key, "error": f"{type(exc).__name__}: {exc}"}
                )
                if report.failed >= MAX_FAILURES and not offline:
                    halt("too_many_failures", f"{report.failed} failed trials; last: {exc}")
                return
            store.append_trial(record)
            report.completed += 1
            if report.completed % 10 == 0 or report.completed == len(pending):
                log(
                    f"  {report.completed}/{len(pending)} trials, "
                    f"spent ${gateway.spend.spent:.4f}, calls paid {gateway.stats['paid']} "
                    f"cached {gateway.stats['cached']}"
                )

    try:
        await asyncio.gather(*(run_one(t) for t in pending))
    finally:
        if owned:
            for provider in providers.values():
                await provider.aclose()
        for task in tasks.values():
            await task.aclose()
    report.spent_usd = gateway.spend.spent
    report.calls_paid = gateway.stats["paid"]
    report.calls_cached = gateway.stats["cached"]
    if report.stop_reason == "finished" and report.failed:
        report.stop_reason = "finished_with_failures"
    store.append_event(
        "run_finished",
        reason=report.stop_reason,
        completed=report.completed,
        failed=report.failed,
        spent_usd=report.spent_usd,
        calls_paid=report.calls_paid,
        calls_cached=report.calls_cached,
    )
    return report


def rebuild_trials(store: RunStore, log: Callable[[str], None] | None = None) -> RunReport:
    """Regenerate trials.jsonl from calls.jsonl alone (no network), for example after a scorer
    or analysis fix. The previous trials file is kept as a timestamped backup."""
    if store.trials_path.exists():
        backup = store.trials_path.with_name(f"trials.{now_iso().replace(':', '')}.bak.jsonl")
        store.trials_path.replace(backup)
    fresh = RunStore.open(store.dir)
    return asyncio.run(execute(fresh, fresh.config(), offline=True, log=log))
