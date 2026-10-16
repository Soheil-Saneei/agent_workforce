"""Gain formulas and process metrics on hand-built trial records."""

import pytest

from workforce.analysis.metrics import (
    benchmark_metric_rows,
    condition_rows,
    error_correlation,
    gain_rows,
    minority_outcomes,
    pairwise_disagreement,
    procedure_rows,
)
from workforce.analysis.stats import gini, jaccard, phi


def trial(
    item,
    n,
    communication,
    initial,
    final,
    team,
    *,
    diversity="identical",
    initial_vote=None,
    procedures=None,
):
    """initial/final: answer keys per agent; correct answer is always 'ok'."""
    rounds = 0 if communication == "independent" else 2
    cid = f"n{n}-{diversity}-{communication}-homogeneous"

    def turn(agent, key, phase):
        return {
            "agent": agent,
            "phase": phase,
            "skipped": False,
            "answer": key,
            "key": key,
            "text": f"text {key} {agent}",
            "evidence": None,
            "critique": "c" if phase else None,
            "output_tokens": 10 * (agent + 1),
            "correct": key == "ok",
        }

    turns = [[turn(i, k, 0) for i, k in enumerate(initial)]]
    for phase in range(1, rounds + 1):
        turns.append([turn(i, k, phase) for i, k in enumerate(final)])
    return {
        "trial_key": f"b|{cid}|{item}|r0",
        "benchmark": "b",
        "item_id": item,
        "condition": {
            "id": cid,
            "team_size": n,
            "diversity": diversity,
            "communication": communication,
            "rounds": rounds,
            "model_assignment": "homogeneous",
        },
        "agents": [{"index": i, "procedure": (procedures or ["general"] * n)[i]} for i in range(n)],
        "turns": turns,
        "team": {"correct": team == "ok", "tie_break": "none"},
        "initial_vote": {"correct": (initial_vote or team) == "ok"},
        "initial_correct": [k == "ok" for k in initial],
        "oracle_initial": "ok" in initial,
        "oracle_any": "ok" in initial or "ok" in final,
        "tokens": {"input": 900, "output": 100, "cached_input": 0, "total": 1000},
        "cost_usd": 0.001,
        "calls": n * (rounds + 1),
        "parse_failures": 0,
        "latency": {"critical_path_s": 1.0},
        "budget": {"any_agent_hit_limit": False, "skipped_calls": 0, "overrun": 0},
    }


@pytest.fixture
def trials():
    out = []
    # Two items. Single agent: right on item 1, wrong on item 2.
    out += [
        trial("1", 1, "independent", ["ok"], ["ok"], "ok"),
        trial("2", 1, "independent", ["x"], ["x"], "x"),
    ]
    # N=3 independent: item 1 majority right; item 2 one right agent is outvoted.
    out += [
        trial("1", 3, "independent", ["ok", "ok", "x"], ["ok", "ok", "x"], "ok"),
        trial("2", 3, "independent", ["ok", "x", "x"], ["ok", "x", "x"], "x"),
    ]
    # N=3 discussion: item 1 right; item 2 the minority persuades everyone.
    out += [
        trial(
            "1", 3, "horizontal-2r", ["ok", "x", "x"], ["ok", "ok", "ok"], "ok", initial_vote="x"
        ),
        trial("2", 3, "horizontal-2r", ["ok", "x", "x"], ["ok", "ok", "x"], "ok", initial_vote="x"),
    ]
    return out


def by_condition(rows):
    return {r["condition"]: r for r in rows}


def test_gain_formulas(trials):
    gains = by_condition(gain_rows(trials))
    single = 0.5  # N=1 accuracy over the two items
    indep = gains["n3-identical-independent-homogeneous"]
    # generation = oracle (1.0 on both items) - single agent (0.5)
    assert indep["generation_gain"]["mean"] == pytest.approx(1.0 - single)
    # selection = team (0.5) - mean individual ((2/3 + 1/3) / 2 = 0.5)
    assert indep["selection_gain"]["mean"] == pytest.approx(0.0)
    disc = gains["n3-identical-horizontal-2r-homogeneous"]
    # coordination = discussion (1.0) - matched independent vote (0.5)
    assert disc["coordination_gain"]["mean"] == pytest.approx(0.5)
    # paired coordination = discussion (1.0) - vote over its own initial answers (0.0)
    assert disc["coordination_gain_paired"]["mean"] == pytest.approx(1.0)
    assert disc["selection_gain"]["mean"] == pytest.approx(1.0 - 1 / 3)
    assert "coordination_gain" not in indep


def test_condition_rows_and_process_metrics(trials):
    rows = by_condition(condition_rows(trials))
    disc = rows["n3-identical-horizontal-2r-homogeneous"]
    assert disc["accuracy"]["mean"] == 1.0
    assert disc["oracle_accuracy"] == 1.0
    assert disc["initial_vote_accuracy"] == 0.0
    assert disc["accuracy_per_10k_tokens"] == pytest.approx(2 / 2000 * 10_000)
    assert disc["disagreement_pre"] == pytest.approx(2 / 3)
    assert disc["disagreement_post"] == pytest.approx((0 + 2 / 3) / 2)
    assert disc["correct_minority"] == 1.0
    assert disc["minority_overturned"] == 0.0
    indep = rows["n3-identical-independent-homogeneous"]
    assert indep["minority_overturned"] == 1.0  # item 2: the correct minority lost the vote
    assert indep["majority_overturned"] == 0.0
    assert indep["disagreement_post"] is None
    assert disc["contribution_gini"] == pytest.approx(gini([40, 80, 120]))


def test_pairwise_disagreement_ignores_abstentions():
    assert pairwise_disagreement(["a", "a", "b"]) == pytest.approx(2 / 3)
    assert pairwise_disagreement(["a", None, "a"]) == 0.0
    assert pairwise_disagreement([None, "a"]) is None


def test_minority_outcomes():
    t = trial("1", 5, "horizontal-2r", ["ok", "x", "x", "x", "y"], ["x"] * 5, "x")
    assert minority_outcomes(t) == {
        "correct_minority": 1.0,
        "minority_overturned": 1.0,
        "majority_overturned": None,
    }


def test_error_correlation_and_unique_contributions():
    group = [
        trial(
            "1",
            2,
            "independent",
            ["ok", "ok"],
            ["ok", "ok"],
            "ok",
            procedures=["decomposition", "falsification"],
        ),
        trial(
            "2",
            2,
            "independent",
            ["x", "x"],
            ["x", "x"],
            "x",
            procedures=["decomposition", "falsification"],
        ),
        trial(
            "3",
            2,
            "independent",
            ["ok", "y"],
            ["ok", "y"],
            "ok",
            procedures=["decomposition", "falsification"],
        ),
    ]
    result = error_correlation(group)
    assert result["same_wrong_answer_rate"] == 1.0  # both wrong only on item 2, same answer
    assert result["error_phi"] == pytest.approx(
        phi(
            [(False, False), (True, True), (False, True)] * 1
            + [(False, False), (True, True), (True, False)]
        )
    )
    rows = {r["procedure"]: r for r in procedure_rows(group)}
    assert rows["decomposition"]["unique_correct"] == 1
    assert rows["falsification"]["unique_correct"] == 0


def test_stats_helpers():
    assert gini([1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 3]) == pytest.approx(2 / 3)
    assert phi([(True, True), (False, False)]) == pytest.approx(1.0)
    assert phi([(True, False), (False, True)]) == pytest.approx(-1.0)
    assert phi([(True, True)]) is None
    assert jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)


def test_benchmark_metric_means():
    a = trial("1", 1, "independent", ["ok"], ["ok"], "ok")
    b = trial("2", 1, "independent", ["x"], ["x"], "x")
    a["team"]["metrics"], b["team"]["metrics"] = {"em": 1.0, "f1": 1.0}, {"em": 0.0, "f1": 0.5}
    (row,) = benchmark_metric_rows([a, b])
    assert row["em"] == 0.5 and row["f1"] == 0.75


def test_item_estimates_do_not_depend_on_trial_order():
    from workforce.analysis.metrics import item_estimate

    values = {f"item{i}": float(i % 3 == 0) for i in range(30)}
    shuffled = dict(reversed(list(values.items())))
    assert item_estimate(values) == item_estimate(shuffled)
