"""Cost estimates.

* ``budget_bound``: a hard upper bound straight from a config. The harness never lets a trial
  use more than its token budget, so items x replications x conditions x budget, priced as if
  every token were an output token, cannot be exceeded.
* ``project``: an expected cost from a finished (measured) run. Mean measured tokens per trial
  for each (benchmark, condition) are scaled to the target config's item counts and
  replications; the range is a bootstrap over the measured run's items.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from workforce.design import ExperimentConfig
from workforce.models.base import Price
from workforce.runner import used_models
from workforce.tasks.registry import create_task

# Number of items when a target config asks for the whole benchmark (items unset).
FULL_SIZES = {"musique": 2417, "musr": 756, "humaneval": 164, "aime": 60}


def target_items(config: ExperimentConfig, benchmark: str) -> int:
    items = config.benchmarks[benchmark].items
    if items is not None:
        return items
    if benchmark in FULL_SIZES:
        return FULL_SIZES[benchmark]
    return len(create_task(benchmark).load_items(None, config.seed))


def _price(config: ExperimentConfig, override: Price | None) -> Price:
    if override is not None:
        return override
    prices = [config.models[m].price for m in used_models(config)]
    if any(p is None for p in prices):
        raise ValueError("a model in the target config has no price; pass a price override")
    # Mixed-model teams: bound with the most expensive model.
    return max(prices, key=lambda p: (p.output_per_mtok, p.input_per_mtok))  # type: ignore[arg-type]


def budget_bound(config: ExperimentConfig, price: Price | None = None) -> dict[str, Any]:
    price = _price(config, price)
    conditions = config.conditions()
    # A solo trajectory's last step may finish past the budget by up to one output ceiling.
    overshoot = config.generation.max_output_tokens_per_call
    solo = sum(1 for c in conditions if c.communication.mode == "solo")
    rows = []
    for name, bench in config.benchmarks.items():
        per_condition = target_items(config, name) * config.replications
        trials = per_condition * len(conditions)
        tokens = trials * bench.budget_tokens + per_condition * solo * overshoot
        rows.append(
            {
                "benchmark": name,
                "trials": trials,
                "budget_tokens": bench.budget_tokens,
                "max_tokens": tokens,
                "max_usd": price.cost(0, tokens),
            }
        )
    return {"rows": rows, "total_usd": sum(r["max_usd"] for r in rows), "price": price}


@dataclass
class Projection:
    rows: list[dict[str, Any]]
    total_usd: float
    total_usd_no_cache: float
    ci_usd: tuple[float, float] | None
    missing: list[str]
    price: Price
    total_tokens: float


def project(
    trials: Sequence[dict[str, Any]],
    target: ExperimentConfig,
    price: Price | None = None,
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> Projection:
    price = _price(target, price)
    measured: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        measured[(trial["benchmark"], trial["condition"]["id"])].append(trial)

    def trial_cost(trial: dict[str, Any], cache: bool = True) -> float:
        tokens = trial["tokens"]
        return price.cost(tokens["input"], tokens["output"], tokens["cached_input"] if cache else 0)

    rows, missing = [], []
    per_item_cost: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    scale: dict[str, float] = {}
    for bench in target.benchmarks:
        n_target = target_items(target, bench) * target.replications
        for cond in target.conditions():
            group = measured.get((bench, cond.id))
            if not group:
                missing.append(f"{bench}/{cond.id}")
                continue
            mean_cost = sum(trial_cost(t) for t in group) / len(group)
            rows.append(
                {
                    "benchmark": bench,
                    "condition": cond.id,
                    "measured_trials": len(group),
                    "input_tokens": sum(t["tokens"]["input"] for t in group) / len(group),
                    "cached_input_tokens": sum(t["tokens"]["cached_input"] for t in group)
                    / len(group),
                    "output_tokens": sum(t["tokens"]["output"] for t in group) / len(group),
                    "usd_per_trial": mean_cost,
                    "usd_per_trial_no_cache": sum(trial_cost(t, False) for t in group) / len(group),
                    "target_trials": n_target,
                    "projected_usd": mean_cost * n_target,
                    "projected_tokens": n_target
                    * sum(t["tokens"]["total"] for t in group)
                    / len(group),
                }
            )
            # Per-item cost of one pass over all conditions, for the bootstrap range.
            reps: dict[str, list[float]] = defaultdict(list)
            for t in group:
                reps[t["item_id"]].append(trial_cost(t))
            for item, costs in reps.items():
                per_item_cost[bench][item] += sum(costs) / len(costs)
        scale[bench] = n_target

    rng = random.Random(seed)
    ci = None
    if all(len(items) >= 2 for items in per_item_cost.values()) and per_item_cost:
        totals = []
        for _ in range(resamples):
            total = 0.0
            for bench, items in per_item_cost.items():
                values = list(items.values())
                draw = [values[rng.randrange(len(values))] for _ in values]
                total += sum(draw) / len(draw) * scale[bench]
            totals.append(total)
        totals.sort()
        ci = (totals[int(0.025 * resamples)], totals[int(0.975 * resamples) - 1])
    return Projection(
        rows=rows,
        total_usd=sum(r["projected_usd"] for r in rows),
        total_usd_no_cache=sum(r["usd_per_trial_no_cache"] * r["target_trials"] for r in rows),
        ci_usd=ci,
        missing=missing,
        price=price,
        total_tokens=sum(r["projected_tokens"] for r in rows),
    )


def projection_markdown(projection: Projection, target: ExperimentConfig) -> str:
    p = projection.price
    lines = [
        f"Target: `{target.name}` ({len(target.conditions())} conditions x "
        f"{target.replications} replications). Price: ${p.input_per_mtok}/M input, "
        f"${p.cached_input_per_mtok}/M cached input, ${p.output_per_mtok}/M output.",
        "",
        "| Benchmark | Condition | Measured trials | Mean tokens in / cached / out | USD / trial "
        "| Target trials | Projected USD |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in projection.rows:
        lines.append(
            f"| {r['benchmark']} | {r['condition']} | {r['measured_trials']} | "
            f"{r['input_tokens']:,.0f} / {r['cached_input_tokens']:,.0f} / "
            f"{r['output_tokens']:,.0f} | {r['usd_per_trial']:.5f} | {r['target_trials']:,} | "
            f"{r['projected_usd']:.2f} |"
        )
    ci = projection.ci_usd
    lines += [
        "",
        f"**Projected total: ${projection.total_usd:,.2f}**"
        + (
            f" (95% bootstrap range over measured items ${ci[0]:,.2f} - ${ci[1]:,.2f})"
            if ci
            else ""
        )
        + f"; ${projection.total_usd_no_cache:,.2f} with no prompt-cache discount; "
        f"{projection.total_tokens / 1e6:,.1f}M tokens.",
    ]
    if projection.missing:
        lines.append(f"Not measured in the source run (excluded): {', '.join(projection.missing)}")
    return "\n".join(lines)
