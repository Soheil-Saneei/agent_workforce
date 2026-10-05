import pytest

from workforce.protocol.budget import BudgetLedger

from conftest import make_config
from helpers import run_trial


def test_allowance_follows_the_closed_form():
    ledger = BudgetLedger(total=10_000, team_size=3, phases=3, ceiling=10**6, min_output=1)
    plan = ledger.plan_phase(0, base_input=500, overhead=100)
    # a = (R - N*m*J - N*h*m(m-1)/2) / (N*m*(1 + N(m-1)/2)) = (10000 - 4500 - 900) / 36
    assert plan.allowance == 4600 // 36
    assert plan.planned_phases == 3


def test_output_mode_ignores_inputs():
    ledger = BudgetLedger(total=9_000, team_size=3, phases=3, count="output", min_output=1)
    assert ledger.plan_phase(0, base_input=50_000, overhead=999).allowance == 1_000


def test_reservation_caps_concurrent_calls_and_skips_when_room_is_too_small():
    ledger = BudgetLedger(total=1_000, team_size=2, phases=1, min_output=100)
    first = ledger.reserve(0, 0, allowance=600, input_estimate=200)
    assert first.max_output_tokens == 600 and not first.skipped
    second = ledger.reserve(1, 0, allowance=600, input_estimate=150)
    assert second.skipped and second.skip_reason == "budget_exhausted"  # only 50 tokens left


def test_input_underestimate_is_recorded_as_overrun():
    ledger = BudgetLedger(total=1_000, team_size=1, phases=1, min_output=10)
    slot = ledger.reserve(0, 0, allowance=500, input_estimate=100)
    ledger.settle(slot, input_tokens=900, output_tokens=500, finish_reason="length")
    assert ledger.overrun == 400
    assert ledger.to_dict()["any_agent_hit_limit"]


def verbose(request, spec):
    """An agent that always writes far past its allowance."""
    return "filler " * 50_000 + "<answer>1</answer><confidence>50</confidence>"


@pytest.mark.parametrize("team_size", [1, 3, 5])
@pytest.mark.parametrize("communication", ["independent", "horizontal-2r"])
def test_team_never_exceeds_its_budget(tmp_path, team_size, communication):
    config = make_config(
        benchmarks={"toy_arithmetic": {"items": 1, "budget_tokens": 4_000}},
        factors={
            "team_size": [team_size],
            "diversity": ["identical"],
            "communication": [communication],
        },
    )
    record, providers, _ = run_trial(config, tmp_path, {"base": verbose})
    assert record["tokens"]["total"] <= 4_000
    assert record["budget"]["used"] == record["tokens"]["total"]
    assert record["budget"]["overrun"] == 0
    assert record["budget"]["any_agent_hit_limit"]
    # every request's max_output_tokens was planned before it was sent
    for request in providers["base"].requests:
        assert request.max_output_tokens >= config.generation.min_output_tokens


def test_larger_teams_get_smaller_per_call_allowances(tmp_path):
    allowances = {}
    for n in (1, 3, 5):
        config = make_config(
            benchmarks={"toy_arithmetic": {"items": 1, "budget_tokens": 30_000}},
            factors={
                "team_size": [n],
                "diversity": ["identical"],
                "communication": ["horizontal-2r"],
            },
            generation={"min_output_tokens": 20, "max_output_tokens_per_call": 10**6},
        )
        record, _, _ = run_trial(config, tmp_path)
        allowances[n] = record["budget"]["plans"][0]["allowance"]
    assert allowances[1] > allowances[3] > allowances[5] > 0


def test_unaffordable_rounds_are_dropped_from_the_end(tmp_path):
    # Enough for initial answers from five agents, not for two discussion rounds.
    config = make_config(
        benchmarks={"toy_arithmetic": {"items": 1, "budget_tokens": 2_200}},
        factors={"team_size": [5], "diversity": ["identical"], "communication": ["horizontal-2r"]},
    )
    record, _, _ = run_trial(config, tmp_path)
    initial, *rounds = record["turns"]
    assert all(not t["skipped"] for t in initial)
    assert record["budget"]["plans"][0]["planned_phases"] < 3
    assert any(t["skipped"] for phase in rounds for t in phase)
    assert record["budget"]["skipped_calls"] > 0
    assert record["team"]["key"] is not None  # the team still votes on its standing answers
    assert record["tokens"]["total"] <= 2_200
