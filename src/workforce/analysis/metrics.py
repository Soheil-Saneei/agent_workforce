"""Outcome, gain, team-process and diversity-validation metrics computed from trial records.

Everything here is a pure function of ``trials.jsonl``, so tables can always be regenerated.

Gains (as defined in the design document), all with paired item-level bootstrap CIs:

* generation gain   = oracle team score - single-agent score (N=1, independent, identical,
                      same model assignment)
* selection gain    = team score - average individual (initial) score, same trials
* coordination gain = discussion score - independent-vote score of the matched condition
                      (same N, diversity, model assignment); compute-matched, separate samples
* paired coordination gain = discussion score - vote over the same trial's initial answers;
                      same samples, removes sampling noise but not compute-matched
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from itertools import combinations
from typing import Any

from workforce.analysis.similarity import TfidfModel
from workforce.analysis.stats import bootstrap_ci, gini, jaccard, mean, phi

Trial = dict[str, Any]


def condition_sort_key(condition: dict[str, Any]) -> tuple:
    return (
        condition["team_size"],
        condition["diversity"],
        condition["rounds"],
        condition["model_assignment"],
    )


def group_trials(trials: Iterable[Trial]) -> dict[tuple[str, str], list[Trial]]:
    groups: dict[tuple[str, str], list[Trial]] = defaultdict(list)
    for trial in trials:
        groups[(trial["benchmark"], trial["condition"]["id"])].append(trial)
    return dict(
        sorted(
            groups.items(),
            key=lambda kv: (kv[0][0], condition_sort_key(kv[1][0]["condition"])),
        )
    )


def per_item(trials: Iterable[Trial], value: Callable[[Trial], float | None]) -> dict[str, float]:
    """Mean of ``value`` per item (replications averaged), skipping undefined values."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for trial in trials:
        v = value(trial)
        if v is not None:
            buckets[trial["item_id"]].append(float(v))
    return {item: sum(vs) / len(vs) for item, vs in buckets.items()}


def item_estimate(item_values: dict[str, float]) -> dict[str, Any]:
    # Items in a fixed order, so the bootstrap does not depend on the order trials were logged.
    values = [item_values[item] for item in sorted(item_values)]
    return {"mean": mean(values), "ci": bootstrap_ci(values), "n_items": len(values)}


def paired_estimate(a: dict[str, float], b: dict[str, float]) -> dict[str, Any]:
    shared = sorted(set(a) & set(b))
    return item_estimate({i: a[i] - b[i] for i in shared})


# Per-trial values ---------------------------------------------------------------------------
def team_correct(trial: Trial) -> float:
    return float(trial["team"]["correct"])


def individual_mean(trial: Trial) -> float:
    flags = trial["initial_correct"]
    return sum(1.0 for c in flags if c) / len(flags)


def initial_vote_correct(trial: Trial) -> float:
    return float(trial["initial_vote"]["correct"])


def _final_keys(trial: Trial) -> list[str | None]:
    keys = []
    for agent in range(trial["condition"]["team_size"]):
        answered = [ph[agent] for ph in trial["turns"] if not ph[agent]["skipped"]]
        answered = [t for t in answered if t["answer"] is not None]
        keys.append(answered[-1]["key"] if answered else None)
    return keys


def _initial_keys(trial: Trial) -> list[str | None]:
    return [None if t["skipped"] or t["answer"] is None else t["key"] for t in trial["turns"][0]]


def pairwise_disagreement(keys: Sequence[str | None]) -> float | None:
    """Share of pairs of answering agents whose answers differ (abstentions excluded)."""
    pairs = list(combinations([k for k in keys if k is not None], 2))
    if not pairs:
        return None
    return sum(1 for a, b in pairs if a != b) / len(pairs)


def process_metrics(trial: Trial, tfidf: TfidfModel | None = None) -> dict[str, float | None]:
    n = trial["condition"]["team_size"]
    rounds = trial["condition"]["rounds"]
    out: dict[str, float | None] = {}
    if n < 2:
        return out
    initial_keys, final_keys = _initial_keys(trial), _final_keys(trial)
    out["disagreement_pre"] = pairwise_disagreement(initial_keys)
    initial = trial["turns"][0]
    if tfidf is not None:
        out["similarity_initial"] = tfidf.mean_pairwise([t["text"] for t in initial if t["text"]])
    evidence = [t["evidence"] for t in initial if t.get("evidence") is not None]
    if len(evidence) > 1:
        out["evidence_jaccard"] = mean([jaccard(a, b) for a, b in combinations(evidence, 2)])
    if rounds:
        post = pairwise_disagreement(final_keys)
        out["disagreement_post"] = post
        if out["disagreement_pre"] is not None and post is not None:
            out["convergence"] = out["disagreement_pre"] - post
        changed = [a != b for a, b in zip(initial_keys, final_keys, strict=True) if a and b]
        out["switch_rate"] = mean([float(c) for c in changed])
        discussion_tokens = [
            sum(trial["turns"][p][i]["output_tokens"] for p in range(1, rounds + 1))
            for i in range(n)
        ]
        out["contribution_gini"] = gini(discussion_tokens)
        if tfidf is not None:
            redundancy = [
                tfidf.mean_pairwise([t["critique"] for t in phase if t.get("critique")])
                for phase in trial["turns"][1:]
            ]
            out["critique_redundancy"] = mean([r for r in redundancy if r is not None])
    return out


def minority_outcomes(trial: Trial) -> dict[str, float | None]:
    """Did a correct initial minority win, and was a correct initial majority overturned?"""
    n = trial["condition"]["team_size"]
    k = sum(1 for c in trial["initial_correct"] if c)
    team = trial["team"]["correct"]
    return {
        "correct_minority": float(0 < k < n / 2),
        "minority_overturned": float(not team) if 0 < k < n / 2 else None,
        "majority_overturned": float(not team) if k > n / 2 else None,
    }


# Tables -----------------------------------------------------------------------------------
def fit_tfidf(trials: Sequence[Trial]) -> dict[str, TfidfModel]:
    corpora: dict[str, list[str]] = defaultdict(list)
    for trial in trials:
        for phase in trial["turns"]:
            corpora[trial["benchmark"]].extend(t["text"] for t in phase if t["text"])
    return {bench: TfidfModel(texts) for bench, texts in corpora.items()}


def condition_rows(trials: Sequence[Trial]) -> list[dict[str, Any]]:
    tfidf = fit_tfidf(trials)
    rows = []
    for (benchmark, _), group in group_trials(trials).items():
        cond = group[0]["condition"]
        total_tokens = sum(t["tokens"]["total"] for t in group)
        correct = sum(t["team"]["correct"] for t in group)
        process = [process_metrics(t, tfidf.get(benchmark)) for t in group]
        minority = [minority_outcomes(t) for t in group]
        budget_hits = [t["budget"]["any_agent_hit_limit"] for t in group]
        row: dict[str, Any] = {
            "benchmark": benchmark,
            "condition": cond["id"],
            "team_size": cond["team_size"],
            "diversity": cond["diversity"],
            "communication": cond["communication"],
            "model_assignment": cond["model_assignment"],
            "trials": len(group),
            "accuracy": item_estimate(per_item(group, team_correct)),
            "individual_accuracy": mean([individual_mean(t) for t in group]),
            "oracle_accuracy": mean([float(t["oracle_initial"]) for t in group]),
            "oracle_any_accuracy": mean([float(t["oracle_any"]) for t in group]),
            "initial_vote_accuracy": mean([initial_vote_correct(t) for t in group]),
            "input_tokens": mean([t["tokens"]["input"] for t in group]),
            "output_tokens": mean([t["tokens"]["output"] for t in group]),
            "total_tokens": mean([t["tokens"]["total"] for t in group]),
            "cached_input_tokens": mean([t["tokens"]["cached_input"] for t in group]),
            "cost_usd": mean([t["cost_usd"] for t in group]),
            "accuracy_per_10k_tokens": (correct / total_tokens * 10_000) if total_tokens else None,
            "latency_s": mean([t["latency"]["critical_path_s"] for t in group]),
            "budget_hit_rate": mean([float(h) for h in budget_hits]),
            "skipped_calls": mean([t["budget"]["skipped_calls"] for t in group]),
            "overruns": sum(1 for t in group if t["budget"]["overrun"] > 0),
            "untagged_answer_rate": sum(t.get("untagged_answers", 0) for t in group)
            / max(1, sum(t["calls"] for t in group)),
            "parse_failure_rate": sum(t["parse_failures"] for t in group)
            / max(1, sum(t["calls"] for t in group)),
            "tie_break_rate": mean(
                [float(t["team"]["tie_break"] in ("confidence", "random")) for t in group]
            ),
        }
        for name in (
            "disagreement_pre",
            "disagreement_post",
            "convergence",
            "switch_rate",
            "similarity_initial",
            "evidence_jaccard",
            "contribution_gini",
            "critique_redundancy",
        ):
            row[name] = mean([p[name] for p in process if p.get(name) is not None])
        for name in ("correct_minority", "minority_overturned", "majority_overturned"):
            row[name] = mean([m[name] for m in minority if m[name] is not None])
        row.update(error_correlation(group))
        rows.append(row)
    return rows


def error_correlation(group: Sequence[Trial]) -> dict[str, float | None]:
    """Pooled pairwise phi of initial-answer errors across items, and how often two agents
    who are both wrong give the same wrong answer."""
    pairs: list[tuple[bool, bool]] = []
    both_wrong = same_wrong = 0
    for trial in group:
        keys = _initial_keys(trial)
        flags = trial["initial_correct"]
        for i, j in combinations(range(len(flags)), 2):
            ei, ej = not flags[i], not flags[j]
            pairs += [(ei, ej), (ej, ei)]
            if ei and ej and keys[i] is not None and keys[j] is not None:
                both_wrong += 1
                same_wrong += keys[i] == keys[j]
    return {
        "error_phi": phi(pairs) if pairs else None,
        "same_wrong_answer_rate": same_wrong / both_wrong if both_wrong else None,
    }


def _baseline(conditions: dict[str, dict[str, Any]], cond: dict[str, Any]) -> str | None:
    for cid, other in conditions.items():
        if (
            other["team_size"] == 1
            and other["communication"] == "independent"
            and other["diversity"] == "identical"
            and other["model_assignment"] == cond["model_assignment"]
        ):
            return cid
    return None


def _matched_independent(conditions: dict[str, dict[str, Any]], cond: dict[str, Any]) -> str | None:
    for cid, other in conditions.items():
        if (
            other["communication"] == "independent"
            and other["team_size"] == cond["team_size"]
            and other["diversity"] == cond["diversity"]
            and other["model_assignment"] == cond["model_assignment"]
        ):
            return cid
    return None


def gain_rows(trials: Sequence[Trial]) -> list[dict[str, Any]]:
    by_bench: dict[str, dict[str, list[Trial]]] = defaultdict(dict)
    for (bench, cid), group in group_trials(trials).items():
        by_bench[bench][cid] = group
    rows = []
    for bench, groups in by_bench.items():
        conditions = {cid: g[0]["condition"] for cid, g in groups.items()}
        for cid, group in groups.items():
            cond = conditions[cid]
            team = per_item(group, team_correct)
            row: dict[str, Any] = {"benchmark": bench, "condition": cid, **_labels(cond)}
            baseline = _baseline(conditions, cond)
            row["generation_gain"] = (
                paired_estimate(
                    per_item(group, lambda t: float(t["oracle_initial"])),
                    per_item(groups[baseline], team_correct),
                )
                if baseline
                else None
            )
            row["selection_gain"] = paired_estimate(team, per_item(group, individual_mean))
            if cond["communication"] != "independent":
                matched = _matched_independent(conditions, cond)
                row["coordination_gain"] = (
                    paired_estimate(team, per_item(groups[matched], team_correct))
                    if matched
                    else None
                )
                row["coordination_gain_paired"] = paired_estimate(
                    team, per_item(group, initial_vote_correct)
                )
            rows.append(row)
    return rows


def benchmark_metric_rows(trials: Sequence[Trial]) -> list[dict[str, Any]]:
    """Mean of every benchmark-specific metric of the team answer (for example MuSiQue answer
    F1 and supporting-paragraph F1 alongside EM)."""
    rows = []
    for (benchmark, cid), group in group_trials(trials).items():
        names = sorted({k for t in group for k in t["team"]["metrics"]})
        rows.append(
            {
                "benchmark": benchmark,
                "condition": cid,
                **{n: mean([t["team"]["metrics"].get(n, 0.0) for t in group]) for n in names},
            }
        )
    return rows


def procedure_rows(trials: Sequence[Trial]) -> list[dict[str, Any]]:
    """Individual accuracy and unique correct contributions per procedure (initial answers)."""
    stats: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {"agents": 0, "correct": 0, "unique": 0}
    )
    for trial in trials:
        flags = trial["initial_correct"]
        if len(flags) < 2:
            continue
        for agent, flag in zip(trial["agents"], flags, strict=True):
            bucket = stats[(trial["benchmark"], trial["condition"]["id"], agent["procedure"])]
            bucket["agents"] += 1
            bucket["correct"] += bool(flag)
            others = [f for k, f in enumerate(flags) if k != agent["index"]]
            bucket["unique"] += bool(flag) and not any(others)
    return [
        {
            "benchmark": bench,
            "condition": cid,
            "procedure": proc,
            "agent_answers": int(s["agents"]),
            "individual_accuracy": s["correct"] / s["agents"],
            "unique_correct": int(s["unique"]),
            "unique_correct_rate": s["unique"] / s["agents"],
        }
        for (bench, cid, proc), s in sorted(stats.items())
    ]


def _labels(cond: dict[str, Any]) -> dict[str, Any]:
    return {
        "team_size": cond["team_size"],
        "diversity": cond["diversity"],
        "communication": cond["communication"],
        "model_assignment": cond["model_assignment"],
    }
