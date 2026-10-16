"""Command line: run, resume, summarize, estimate-cost, grid, fetch-data, plot-solo."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from workforce.analysis.report import write_reports
from workforce.cost import budget_bound, project, projection_markdown
from workforce.design import ExperimentConfig, load_config
from workforce.env import load_env_file
from workforce.models.pricing import lookup_price
from workforce.runner import RunReport, execute, rebuild_trials
from workforce.store import RunStore
from workforce.tasks import create_task

EXIT_STOPPED = 2


def _load_env(path: str | None) -> None:
    loaded = load_env_file(path)
    if loaded is not None:
        print(f"loaded environment variables from {loaded}", file=sys.stderr)


def _finish(store: RunStore, report: RunReport) -> int:
    write_reports(store)
    print(
        f"\n{report.stop_reason}: {report.completed} trials completed now, "
        f"{report.already_done} earlier, {report.failed} failed, {report.remaining} remaining; "
        f"calls paid {report.calls_paid}, cached {report.calls_cached}; "
        f"run spend ${store.spent_usd:.4f}\nsummary: {store.dir / 'summary.md'}"
    )
    for failure in report.failures[:10]:
        print(f"  failed: {failure}", file=sys.stderr)
    if report.stop_reason in ("spend_cap", "quota_exhausted", "too_many_failures"):
        print(f"resume with: workforce resume {store.dir}", file=sys.stderr)
        return EXIT_STOPPED
    return 1 if report.failed else 0


def cmd_run(args: argparse.Namespace) -> int:
    _load_env(args.env_file)
    overrides = {"mock": args.mock, "items": args.items, "max_spend": args.max_spend}
    config = load_config(args.config, overrides)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else Path(args.runs_root) / (f"{config.name}{'-mock' if args.mock else ''}-{stamp}")
    )
    store = RunStore.create(run_dir, config, Path(args.config).read_text())
    return _finish(store, asyncio.run(execute(store, config)))


def cmd_resume(args: argparse.Namespace) -> int:
    _load_env(args.env_file)
    store = RunStore.open(Path(args.run_dir))
    config = store.config()
    if args.max_spend is not None:
        # The cap is a safety limit, not part of the experiment design, so it may change.
        from dataclasses import replace

        config = replace(config, run=replace(config.run, max_spend_usd=args.max_spend))
    return _finish(store, asyncio.run(execute(store, config)))


def cmd_summarize(args: argparse.Namespace) -> int:
    store = RunStore.open(Path(args.run_dir))
    if args.rebuild:
        report = rebuild_trials(store)
        print(f"rebuilt {report.completed} trials offline ({report.failed} could not be replayed)")
        store = RunStore.open(Path(args.run_dir))
    print(write_reports(store))
    return 0


def _price_override(model: str | None):
    if model is None:
        return None
    price = lookup_price("openai", model)
    if price is None:
        raise SystemExit(f"no built-in price for {model!r}")
    return price


def cmd_estimate_cost(args: argparse.Namespace) -> int:
    target = load_config(args.config)
    price = _price_override(args.price_model)
    bound = budget_bound(target, price)
    print(f"# Cost estimate for `{target.name}`\n")
    if args.from_run:
        trials = RunStore.open(Path(args.from_run)).trials()
        print(f"## Projection from measured tokens in `{args.from_run}`\n")
        print(projection_markdown(project(trials, target, price), target))
        print()
    print("## Hard upper bound from the token budgets\n")
    for row in bound["rows"]:
        overshoot = row["max_tokens"] - row["trials"] * row["budget_tokens"]
        print(
            f"- {row['benchmark']}: {row['trials']:,} trials x {row['budget_tokens']:,} tokens"
            + (f" + {overshoot:,} solo overshoot" if overshoot else "")
            + f" <= ${row['max_usd']:,.2f}"
        )
    print(f"- total <= ${bound['total_usd']:,.2f} (every token priced as output)")
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    config: ExperimentConfig = load_config(args.config)
    conditions = config.conditions()
    print(f"{config.name}: {len(conditions)} conditions x {config.replications} replications")
    for condition in conditions:
        print(f"  {condition.id}")
    for skipped in config.skipped_conditions():
        print(f"  (skipped, degenerate for one agent) {skipped.id}")
    for name, bench in config.benchmarks.items():
        print(f"  benchmark {name}: items={bench.items or 'all'}, budget={bench.budget_tokens:,}")
    return 0


def cmd_plot_solo(args: argparse.Namespace) -> int:
    from workforce.analysis.plot import plot_solo_curves

    # A later run replaces an earlier one's trials of the same benchmark and condition, so an
    # extended run (for example to a longer horizon) can be plotted beside the original.
    groups: dict[tuple[str, str], list[dict]] = {}
    for run in args.run_dirs:
        fresh: dict[tuple[str, str], list[dict]] = {}
        for trial in RunStore.open(Path(run)).trials():
            fresh.setdefault((trial["benchmark"], trial["condition"]["id"]), []).append(trial)
        groups.update(fresh)
    trials = [t for group in groups.values() for t in group]
    try:
        print(plot_solo_curves(trials, Path(args.out)))
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        raise SystemExit("plot-solo needs matplotlib: uv sync --extra plot") from None
    return 0


def cmd_fetch_data(args: argparse.Namespace) -> int:
    for name in args.benchmarks:
        items = create_task(name).load_items(None, 0)
        print(f"{name}: {len(items)} items available")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="workforce", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an experiment config")
    run.add_argument("config")
    run.add_argument("--run-dir", help="explicit run directory (default: runs/<name>-<time>)")
    run.add_argument("--runs-root", default="runs")
    run.add_argument("--env-file", help="file with API keys (default: $WORKFORCE_ENV_FILE or .env)")
    run.add_argument("--mock", action="store_true", help="replace every model with the mock")
    run.add_argument("--items", type=int, help="override items per benchmark")
    run.add_argument("--max-spend", type=float, help="override the USD spend cap")
    run.set_defaults(func=cmd_run)

    resume = sub.add_parser("resume", help="continue an interrupted run without re-paying")
    resume.add_argument("run_dir")
    resume.add_argument("--env-file")
    resume.add_argument("--max-spend", type=float, help="change the USD spend cap")
    resume.set_defaults(func=cmd_resume)

    summ = sub.add_parser("summarize", help="regenerate results tables from a run's logs")
    summ.add_argument("run_dir")
    summ.add_argument(
        "--rebuild", action="store_true", help="first re-derive trials from calls.jsonl offline"
    )
    summ.set_defaults(func=cmd_summarize)

    cost = sub.add_parser("estimate-cost", help="budget upper bound and pilot-based projection")
    cost.add_argument("--config", required=True, help="target experiment config")
    cost.add_argument("--from-run", help="run whose measured tokens to scale up")
    cost.add_argument("--price-model", help="price tokens as this OpenAI model instead")
    cost.set_defaults(func=cmd_estimate_cost)

    grid = sub.add_parser("grid", help="print the expanded factorial grid")
    grid.add_argument("config")
    grid.set_defaults(func=cmd_grid)

    plot = sub.add_parser("plot-solo", help="plot solo accuracy vs tokens (PNG)")
    plot.add_argument(
        "run_dirs",
        nargs="+",
        help="runs to plot together; a later run replaces an earlier one's benchmark/condition",
    )
    plot.add_argument("--out", required=True, help="output .png path")
    plot.set_defaults(func=cmd_plot_solo)

    fetch = sub.add_parser("fetch-data", help="download and verify benchmark files")
    fetch.add_argument("benchmarks", nargs="+")
    fetch.set_defaults(func=cmd_fetch_data)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
