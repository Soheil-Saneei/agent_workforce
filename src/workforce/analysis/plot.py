"""Accuracy-vs-tokens curves of solo-iteration runs, as a PNG (needs the ``plot`` extra)."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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


def plot_solo_curves(trials: Sequence[dict[str, Any]], out: Path) -> Path:
    """One panel per benchmark: accuracy (95% bootstrap CI band) at each token checkpoint,
    one line per condition. Checkpoints that not every trajectory reached are hollow."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = checkpoint_rows(trials)
    benchmarks = sorted({r["benchmark"] for r in rows})
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
        ax.set_facecolor(SURFACE)
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
        ax.set_title(DISPLAY_NAMES.get(benchmark, benchmark), color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy", color=MUTED, fontsize=9)
    total = rows and next(t for t in trials if "solo" in t)["solo"]["count"] == "total"
    for ax in axes[-1, :]:
        ax.set_xlabel(
            "tokens spent (input + output)" if total else "output tokens spent",
            color=MUTED,
            fontsize=9,
        )
    if len(conditions) > 1:
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower right", frameon=False)
    caption = textwrap.fill(
        "Band: 95% bootstrap CI over items. Hollow marker: some trajectories had no answer "
        "yet at that checkpoint (counted as wrong).",
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
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out
