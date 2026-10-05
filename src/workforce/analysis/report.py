"""Markdown and CSV results tables, regenerated from a run's logs."""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from typing import Any

from workforce.analysis.metrics import condition_rows, gain_rows, procedure_rows
from workforce.store import RunStore


def _f(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _signed(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:+.{digits}f}"


def _estimate(est: dict[str, Any] | None, signed: bool = False) -> str:
    if not est or est.get("mean") is None:
        return "-"
    fmt = _signed if signed else _f
    text = fmt(est["mean"])
    if est.get("ci"):
        lo, hi = est["ci"]
        text += f" [{fmt(lo, 2)}, {fmt(hi, 2)}]"
    return text


def _tokens(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def outcome_table(rows: Sequence[dict[str, Any]]) -> str:
    return _table(
        [
            "Benchmark",
            "Condition",
            "Trials",
            "Team acc [95% CI]",
            "Mean indiv.",
            "Oracle",
            "Indep. vote (paired)",
            "Tokens in / out",
            "USD / trial",
            "Acc / 10k tok",
            "Latency s",
            "Budget hit",
            "Parse fail",
        ],
        [
            [
                r["benchmark"],
                r["condition"],
                str(r["trials"]),
                _estimate(r["accuracy"]),
                _f(r["individual_accuracy"]),
                _f(r["oracle_accuracy"]),
                _f(r["initial_vote_accuracy"]),
                f"{_tokens(r['input_tokens'])} / {_tokens(r['output_tokens'])}",
                _f(r["cost_usd"], 4),
                _f(r["accuracy_per_10k_tokens"]),
                _f(r["latency_s"], 1),
                _f(r["budget_hit_rate"], 2),
                _f(r["parse_failure_rate"], 3),
            ]
            for r in rows
        ],
    )


def gain_table(rows: Sequence[dict[str, Any]]) -> str:
    return _table(
        [
            "Benchmark",
            "Condition",
            "Generation gain",
            "Selection gain",
            "Coordination gain",
            "Coordination gain (paired)",
        ],
        [
            [
                r["benchmark"],
                r["condition"],
                _estimate(r.get("generation_gain"), signed=True),
                _estimate(r.get("selection_gain"), signed=True),
                _estimate(r.get("coordination_gain"), signed=True),
                _estimate(r.get("coordination_gain_paired"), signed=True),
            ]
            for r in rows
        ],
    )


def process_table(rows: Sequence[dict[str, Any]]) -> str:
    team_rows = [r for r in rows if r["team_size"] > 1]
    return _table(
        [
            "Benchmark",
            "Condition",
            "Disagree pre",
            "Disagree post",
            "Convergence",
            "Switch",
            "Error phi",
            "Same wrong ans.",
            "Lexical sim.",
            "Evidence Jaccard",
            "Contrib. Gini",
            "Critique redund.",
            "Correct minority",
            "Minority overturned",
            "Majority overturned",
        ],
        [
            [
                r["benchmark"],
                r["condition"],
                _f(r["disagreement_pre"]),
                _f(r["disagreement_post"]),
                _f(r["convergence"]),
                _f(r["switch_rate"]),
                _f(r["error_phi"]),
                _f(r["same_wrong_answer_rate"]),
                _f(r["similarity_initial"]),
                _f(r["evidence_jaccard"]),
                _f(r["contribution_gini"]),
                _f(r["critique_redundancy"]),
                _f(r["correct_minority"]),
                _f(r["minority_overturned"]),
                _f(r["majority_overturned"]),
            ]
            for r in team_rows
        ],
    )


def procedure_table(rows: Sequence[dict[str, Any]]) -> str:
    return _table(
        [
            "Benchmark",
            "Condition",
            "Procedure",
            "Answers",
            "Indiv. acc",
            "Unique correct",
            "Unique rate",
        ],
        [
            [
                r["benchmark"],
                r["condition"],
                r["procedure"],
                str(r["agent_answers"]),
                _f(r["individual_accuracy"]),
                str(r["unique_correct"]),
                _f(r["unique_correct_rate"]),
            ]
            for r in rows
        ],
    )


def summarize(store: RunStore) -> str:
    trials = store.trials()
    if not trials:
        return f"# {store.dir.name}\n\nNo completed trials yet.\n"
    rows = condition_rows(trials)
    calls = store.calls()
    tokens_in = sum(c["usage"]["input_tokens"] for c in calls)
    tokens_out = sum(c["usage"]["output_tokens"] for c in calls)
    cached = sum(c["usage"]["cached_input_tokens"] for c in calls)
    models = sorted({c["response_model"] or c["model"] for c in calls})
    parts = [
        f"# Results: {store.dir.name}",
        f"Config hash `{store.manifest['config_hash']}`, {len(trials)} trials, "
        f"{len(calls)} logged calls, models: {', '.join(models) or '-'}. "
        f"Tokens: {tokens_in:,} input ({cached:,} cached) + {tokens_out:,} output. "
        f"Spend: ${store.spent_usd:.4f}.",
        "Accuracy is the share of trials whose anonymous team vote is correct (MuSiQue: answer "
        "EM; MuSR: choice accuracy; HumanEval: hidden tests pass). CIs are 95% percentile "
        "bootstraps over items. Latency is the critical path (sum over phases of the slowest "
        "call). 'Budget hit' is the share of trials where some agent was cut off at its "
        "allowance or skipped for lack of budget.",
        "## Primary outcomes",
        outcome_table(rows),
        "## Gains",
        "Generation = oracle - single agent (N=1 independent identical); selection = team - "
        "mean individual; coordination = discussion - matched independent vote; paired = "
        "discussion - vote over the same trial's initial answers.",
        gain_table(gain_rows(trials)),
        "## Team process and diversity validation",
        process_table(rows),
        "## Procedures (initial answers)",
        procedure_table(procedure_rows(trials)),
    ]
    return "\n\n".join(parts) + "\n"


def conditions_csv(store: RunStore) -> str:
    rows = condition_rows(store.trials())
    flat = []
    for row in rows:
        out = {k: v for k, v in row.items() if k != "accuracy"}
        out["accuracy"] = row["accuracy"]["mean"]
        ci = row["accuracy"]["ci"] or (None, None)
        out["accuracy_ci_low"], out["accuracy_ci_high"] = ci
        flat.append(out)
    buffer = io.StringIO()
    if flat:
        writer = csv.DictWriter(buffer, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    return buffer.getvalue()


def write_reports(store: RunStore) -> str:
    text = summarize(store)
    (store.dir / "summary.md").write_text(text)
    (store.dir / "conditions.csv").write_text(conditions_csv(store))
    return text
