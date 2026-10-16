"""Accuracy-vs-tokens figures as PNGs (need the ``plot`` extra): solo-iteration curves, and the
accuracy/cost trade-off between voting and discussing teams."""

from __future__ import annotations

import itertools
import math
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from workforce.analysis.metrics import condition_rows
from workforce.analysis.solo import checkpoint_rows

# Categorical slots in fixed order (validated default palette), then the chart chrome.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DISPLAY_NAMES = {
    "musique": "MuSiQue",
    "musr": "MuSR",
    "humaneval": "HumanEval",
    "aime": "AIME 2025-2026",
    "livecodebench": "LiveCodeBench v6",
    "toy_arithmetic": "Toy arithmetic",
}


def _label(tokens: int) -> str:
    return f"{tokens // 1000}k" if tokens >= 1000 and tokens % 1000 == 0 else f"{tokens:,}"


def _benchmark_order(benchmarks: set[str]) -> list[str]:
    order = list(DISPLAY_NAMES)
    return sorted(benchmarks, key=lambda b: (order.index(b) if b in order else len(order), b))


def _style(ax: Any, benchmark: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(DISPLAY_NAMES.get(benchmark, benchmark), color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def _save(fig: Any, out: Path) -> Path:
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    return out


def plot_solo_curves(trials: Sequence[dict[str, Any]], out: Path) -> Path:
    """One panel per benchmark: accuracy (95% bootstrap CI band) at each token checkpoint, one
    line per condition. A checkpoint is hollow when some trajectory had not finished a step."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = checkpoint_rows(trials)
    benchmarks = _benchmark_order({r["benchmark"] for r in rows})
    conditions = sorted({r["condition"] for r in rows})
    if not benchmarks:
        raise ValueError("no solo trials to plot")
    columns = min(3, len(benchmarks))
    lines = math.ceil(len(benchmarks) / columns)
    fig, axes = plt.subplots(
        lines, columns, figsize=(4.2 * columns, 3.3 * lines), squeeze=False, sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat[len(benchmarks) :]:
        ax.set_visible(False)
    for ax, benchmark in zip(axes.flat, benchmarks, strict=False):
        _style(ax, benchmark)
        for slot, condition in enumerate(conditions):
            points = [
                r for r in rows if r["benchmark"] == benchmark and r["condition"] == condition
            ]
            if not points:
                continue
            color = SERIES[slot % len(SERIES)]
            x = [p["tokens"] for p in points]
            y = [p["accuracy"]["mean"] for p in points]
            low = [(p["accuracy"]["ci"] or (m, m))[0] for p, m in zip(points, y, strict=True)]
            high = [(p["accuracy"]["ci"] or (m, m))[1] for p, m in zip(points, y, strict=True)]
            ax.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
            ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round", label=condition)
            for p, value in zip(points, y, strict=True):
                full = p["reached"] >= 1.0
                ax.plot(
                    p["tokens"],
                    value,
                    marker="o",
                    markersize=6,
                    markeredgewidth=2,
                    markeredgecolor=color,
                    markerfacecolor=color if full else SURFACE,
                    zorder=3,
                )
            ax.annotate(
                f"{y[-1]:.2f}",
                (x[-1], y[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=INK,
            )
        ticks = sorted({r["tokens"] for r in rows if r["benchmark"] == benchmark})
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [_label(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(ticks[0] / 1.3, ticks[-1] * 1.45)
        ax.set_ylim(0, 1)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy", color=MUTED, fontsize=9)
    total = next(t for t in trials if "solo" in t)["solo"]["count"] == "total"
    for column in range(columns):
        visible = [ax for ax in axes[:, column] if ax.get_visible()]
        visible[-1].set_xlabel(
            "tokens spent (input + output)" if total else "output tokens spent",
            color=MUTED,
            fontsize=9,
        )
    if len(conditions) > 1:
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower right", frameon=False)
    caption = textwrap.fill(
        "Band: 95% bootstrap CI over items. Hollow: some trajectories had not finished a step "
        "yet (no answer counts as wrong).",
        width=int(11 * 4.2 * columns),
    )
    fig.suptitle(
        f"Single-agent accuracy vs tokens spent alone\n{caption}",
        color=INK,
        fontsize=10,
        x=0.01,
        ha="left",
        linespacing=1.6,
    )
    _save(fig, out)
    plt.close(fig)
    return out


MARKERS = {1: "o", 3: "s", 5: "^"}


def plot_team_tradeoff(trials: Sequence[dict[str, Any]], out: Path) -> Path:
    """One panel per benchmark: team accuracy against mean tokens per trial. Each voting
    condition is joined to the discussing condition with the same team size and procedures, so
    each line shows what discussion adds in tokens and in accuracy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator

    rows = condition_rows([t for t in trials if "solo" not in t])
    if not rows:
        raise ValueError("no team trials to plot")
    benchmarks = _benchmark_order({r["benchmark"] for r in rows})
    fig, axes = plt.subplots(
        1, len(benchmarks), figsize=(4.2 * len(benchmarks), 3.6), squeeze=False, sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    vote, talk = SERIES[0], SERIES[1]
    for ax, benchmark in zip(axes.flat, benchmarks, strict=True):
        _style(ax, benchmark)
        group = [r for r in rows if r["benchmark"] == benchmark]
        for r in group:
            if r["communication"] != "independent":
                continue
            pair = [r] + [
                o
                for o in group
                if o["communication"] != "independent"
                and (o["team_size"], o["diversity"], o["model_assignment"])
                == (r["team_size"], r["diversity"], r["model_assignment"])
            ]
            for a, b in itertools.pairwise(pair):
                ax.plot(
                    [a["total_tokens"], b["total_tokens"]],
                    [a["accuracy"]["mean"], b["accuracy"]["mean"]],
                    color=GRID,
                    linewidth=2,
                    zorder=1,
                )
        for r in group:
            color = vote if r["communication"] == "independent" else talk
            ax.plot(
                r["total_tokens"],
                r["accuracy"]["mean"],
                marker=MARKERS.get(r["team_size"], "D"),
                markersize=8,
                markeredgewidth=2,
                markeredgecolor=color,
                markerfacecolor=color if r["diversity"] == "identical" else SURFACE,
                linestyle="none",
                zorder=3,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _label(round(v))))
        ax.minorticks_off()
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("mean tokens per problem (input + output)", color=MUTED, fontsize=9)
    axes[0, 0].set_ylabel("team accuracy", color=MUTED, fontsize=9)

    def key(marker: str, face: str, edge: str, label: str) -> Line2D:
        return Line2D(
            [],
            [],
            marker=marker,
            markersize=8,
            markeredgewidth=2,
            linestyle="none",
            markerfacecolor=face,
            markeredgecolor=edge,
            label=label,
        )

    handles = [
        key("o", vote, vote, "independent vote"),
        key("o", talk, talk, "two-round discussion"),
        *(key(m, MUTED, MUTED, f"{n} agent{'s' if n > 1 else ''}") for n, m in MARKERS.items()),
        key("o", MUTED, MUTED, "identical procedures"),
        key("o", SURFACE, MUTED, "diverse procedures"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    items = min(len({t["item_id"] for t in trials if t["benchmark"] == b}) for b in benchmarks)
    fig.suptitle(
        "Team accuracy vs tokens: grey lines join each voting team to the same team discussing\n"
        f"{items} problems per benchmark, so each accuracy has a 95% CI of roughly "
        f"+-{1.96 * (0.25 / items) ** 0.5:.2f}.",
        color=INK,
        fontsize=10,
        x=0.01,
        ha="left",
        linespacing=1.6,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out
