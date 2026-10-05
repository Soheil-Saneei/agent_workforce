import asyncio
import re
import time
from typing import ClassVar

from workforce.models.mock import MockProvider
from workforce.protocol.prompts import PEER_HEADING, REVISION_HEADING
from workforce.protocol.team import TrialSpec
from workforce.tasks import create_task

from conftest import make_config
from helpers import agent_seed_map, run_trial

MODERATE = {
    "team_size": [3],
    "diversity": ["moderate"],
    "communication": ["horizontal-2r"],
}


def uid_responder(request, spec):
    """Every response carries a unique id derived from the call's seed."""
    return (
        f"working uid{request.seed} done\n<answer>{request.seed % 3}</answer>\n"
        f"<confidence>{request.seed % 100}</confidence>"
    )


class TimedMock(MockProvider):
    """Records when each call starts and ends; calls take a little while."""

    log: ClassVar[list[tuple[int, float, float]]] = []

    async def _complete_once(self, request):
        start = time.perf_counter()
        await asyncio.sleep(0.01 + (request.seed % 5) / 500)
        response = await super()._complete_once(request)
        TimedMock.log.append((request.seed, start, time.perf_counter()))
        return response


def test_reveal_is_simultaneous_and_contains_only_completed_phases(tmp_path):
    config = make_config(factors={**MODERATE, "team_size": [5]})
    TimedMock.log = []
    _, providers, spec = run_trial(
        config, tmp_path, {"base": uid_responder}, provider_cls=TimedMock
    )
    seeds = agent_seed_map(config, spec, phases=3)
    # 1. No call of phase p+1 starts before every call of phase p has finished.
    by_phase: dict[int, list[tuple[float, float]]] = {}
    for seed, start, end in TimedMock.log:
        by_phase.setdefault(seeds[seed][1], []).append((start, end))
    for phase in (0, 1):
        assert max(e for _, e in by_phase[phase]) <= min(s for s, _ in by_phase[phase + 1])
    # 2. A phase-p request only contains outputs from phases < p.
    for request in providers["base"].requests:
        _, phase = seeds[request.seed]
        seen = {int(u) for m in request.messages for u in re.findall(r"uid(\d+)", m.content)}
        assert all(seeds[s][1] < phase for s in seen)
        if phase > 0:  # and every teammate's previous-phase output is there
            assert len([s for s in seen if seeds[s][1] == phase - 1]) == 5


def test_reveal_is_anonymous_and_hides_confidence(tmp_path):
    config = make_config(factors={**MODERATE, "team_size": [5]})
    _, providers, _ = run_trial(config, tmp_path, {"base": uid_responder})
    procedures = {
        p: config.procedures[p].instructions
        for p in ("decomposition", "evidence_first", "falsification")
    }
    orders = set()
    for request in providers["base"].requests:
        system = request.messages[0].content
        reveal = request.messages[-1].content
        if PEER_HEADING.format(label=1) not in reveal:
            continue
        for name, text in procedures.items():
            assert text not in reveal and name not in reveal
            if text not in system:  # teammates' procedures never leak into the history
                assert all(text not in m.content for m in request.messages)
        assert "mock-base" not in reveal
        assert not re.search(r"<confidence>\s*\d", reveal)  # peers' stated confidence is hidden
        orders.add(tuple(re.findall(r"uid(\d+)", reveal)))
    assert len(orders) > 1  # each viewer sees its own random order


def test_tie_break_uses_confidence_stated_before_discussion(tmp_path):
    config = make_config(
        factors={
            **MODERATE,
            "team_size": [3],
            "diversity": ["identical"],
            "communication": ["horizontal"],
        }
    )
    plan = {
        # agent: (answer, confidence before, confidence after)
        0: ("101", 90, 5),
        1: ("202", 40, 99),
        2: ("303", 60, 50),
    }
    seeds: dict[int, tuple[int, int]] = {}

    def responder(request, spec):
        agent, phase = seeds[request.seed]
        answer, before, after = plan[agent]
        return (
            f"<answer>{answer}</answer><confidence>{before if phase == 0 else after}</confidence>"
        )

    item = create_task("toy_arithmetic").load_items(4, config.seed)[0]
    spec = TrialSpec("toy_arithmetic", config.conditions()[0], item, 0, 0)
    seeds.update(agent_seed_map(config, spec, phases=2))
    record, _, _ = run_trial(config, tmp_path, {"base": responder})
    assert record["team"]["tie_break"] == "confidence"
    assert record["team"]["answer_display"] == "101"  # highest *pre*-discussion confidence
    assert record["team"]["weights"] == {"101": 90.0, "202": 40.0, "303": 60.0}


def test_majority_beats_confidence(tmp_path):
    config = make_config(
        factors={"team_size": [3], "diversity": ["identical"], "communication": ["independent"]}
    )
    answers = iter(["7", "7", "9"])
    confidences = iter([10, 10, 100])

    def responder(request, spec):
        return f"<answer>{next(answers)}</answer><confidence>{next(confidences)}</confidence>"

    record, _, _ = run_trial(config, tmp_path, {"base": responder})
    assert record["team"]["answer_display"] == "7"
    assert record["team"]["tie_break"] == "none"
    assert record["team"]["counts"] == {"7": 2, "9": 1}


def test_independent_condition_is_one_simultaneous_phase(tmp_path):
    config = make_config(
        factors={"team_size": [5], "diversity": ["moderate"], "communication": ["independent"]}
    )
    record, providers, _ = run_trial(config, tmp_path)
    assert len(record["turns"]) == 1
    assert len(providers["base"].requests) == 5
    assert record["team"] == record["initial_vote"]


def test_single_agent_discussion_is_self_revision(tmp_path):
    config = make_config(
        factors={"team_size": [1], "diversity": ["identical"], "communication": ["horizontal-2r"]}
    )
    _, providers, _ = run_trial(config, tmp_path)
    requests = providers["base"].requests
    assert len(requests) == 3
    for round_index, request in enumerate(requests[1:], start=1):
        assert request.messages[-1].content.startswith(
            REVISION_HEADING.format(round=round_index, rounds=2)
        )
        assert PEER_HEADING.format(label=1) not in request.messages[-1].content


def test_mixed_model_team_routes_each_agent_to_its_own_model(tmp_path):
    config = make_config(
        factors={
            "team_size": [3],
            "diversity": ["moderate"],
            "communication": ["horizontal"],
            "model_assignment": ["mixed"],
        },
        models={
            "strong": {"provider": "mock", "model": "mock-strong", "params": {"accuracy": 1.0}},
            "weak": {"provider": "mock", "model": "mock-weak", "params": {"accuracy": 0.0}},
        },
        model_assignments={"mixed": {"models": ["strong", "weak"]}},
    )
    record, providers, _ = run_trial(config, tmp_path)
    models = [a["model_name"] for a in record["agents"]]
    assert sorted(models) == ["strong", "strong", "weak"]
    assert len(providers["strong"].requests) == 2 * 2  # two agents x two phases
    assert len(providers["weak"].requests) == 1 * 2
    for agent, correct in zip(record["agents"], record["initial_correct"], strict=True):
        assert correct is (agent["model_name"] == "strong")
    assert record["oracle_initial"] is True
