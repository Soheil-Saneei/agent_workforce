"""Solo iteration: the step loop, checkpoint recording, token accounting and resumability."""

import asyncio
import json
import math

import pytest

from workforce.analysis.report import write_reports
from workforce.analysis.solo import checkpoint_rows, curve_rows
from workforce.design import Communication, ConfigError
from workforce.gateway import Gateway, SpendGuard
from workforce.models.mock import MockProvider, mock_token_count
from workforce.protocol.prompts import SOLO_CHECK_HEADING, SOLO_CHECK_METHODS, SOLO_COMMIT
from workforce.protocol.solo import SoloAgent, SoloIdentity, SoloProtocol, SoloState
from workforce.protocol.team import TrialSpec
from workforce.runner import rebuild_trials
from workforce.store import RunStore
from workforce.tasks import create_task

from conftest import make_config, run_experiment

SOLO = {"team_size": [1], "diversity": ["identical"], "communication": ["solo"]}


def solo_config(budget=1200, checkpoints=(100, 200, 400, 800, 1200), count="total", **extra):
    return make_config(
        factors=SOLO,
        benchmarks={"toy_arithmetic": {"items": 3, "budget_tokens": budget}},
        solo={"checkpoints": list(checkpoints)},
        budget={"count": count},
        **extra,
    )


def step_of(request):
    return sum(1 for m in request.messages if m.role == "assistant")


def numbered(request, spec):
    """Answers with its step number; 20 words of output per step."""
    words = " ".join(["word"] * 18)
    return f"{words} <answer>{step_of(request)}</answer> <confidence>50</confidence>"


class Harness:
    def __init__(self, config, tmp_path, responder=numbered, name="store"):
        self.config = config
        self.store = RunStore.create(tmp_path / name, config)
        self.provider = MockProvider(config.models["base"], responder=responder)
        self.gateway = Gateway(self.store, config.models, {"base": self.provider}, SpendGuard(None))
        self.task = create_task("toy_arithmetic")
        self.items = self.task.load_items(3, config.seed)

    def trial(self, item_ordinal=0):
        condition = self.config.conditions()[0]
        spec = TrialSpec("toy_arithmetic", condition, self.items[item_ordinal], item_ordinal, 0)
        protocol = SoloProtocol(self.config, self.task, self.gateway)
        return asyncio.run(protocol.run(spec))

    def agent(self, state=None):
        return SoloAgent(
            self.config,
            self.task,
            self.gateway,
            self.items[0],
            SoloIdentity("toy_arithmetic", self.items[0].item_id, 0),
            self.config.procedures["general"],
            "base",
            state,
        )


def test_checkpoints_hold_the_answer_of_the_last_step_finished_within_them(tmp_path):
    config = solo_config()
    harness = Harness(config, tmp_path)
    record = harness.trial()
    requests = harness.provider.requests
    # Spend is charged per call exactly as the budget ledger charges it: input + output.
    charges = [mock_token_count(r.messages) + 20 for r in requests]
    cumulative = [sum(charges[: k + 1]) for k in range(len(charges))]
    solo = record["solo"]
    assert solo["spent"] == cumulative
    assert record["tokens"]["total"] == cumulative[-1]
    for point in solo["checkpoints"]:
        finished = [k for k, spent in enumerate(cumulative) if spent <= point["tokens"]]
        assert point["steps"] == len(finished)
        assert point["spent"] == (cumulative[finished[-1]] if finished else 0)
        assert point["output_tokens"] == 20 * len(finished)
        expected = str(finished[-1]) if finished else ""
        assert point["answer_display"] == expected
        assert point["answer_step"] == (finished[-1] if finished else None)
    assert solo["checkpoints"][0]["steps"] == 0  # 100 tokens cannot hold the first step
    # The trial's team answer is the answer held at the horizon (the budget).
    within = [k for k, spent in enumerate(cumulative) if spent <= 1200]
    assert record["team"]["answer_display"] == str(within[-1])
    assert solo["steps_within_horizon"] == len(within)
    assert solo["overshoot"] == max(0, cumulative[-1] - 1200)


def test_steps_start_only_while_they_can_finish_within_the_horizon(tmp_path):
    config = solo_config(budget=1000, checkpoints=(500, 1000))
    harness = Harness(config, tmp_path)
    record = harness.trial()
    requests = harness.provider.requests
    spent_before = [0, *record["solo"]["spent"]]
    min_output = config.generation.min_output_tokens

    def needed(messages):
        return math.ceil(mock_token_count(messages) * 1.02) + 16 + min_output

    for k, request in enumerate(requests):
        assert spent_before[k] + needed(request.messages) <= 1000
    # The step after the last one would not have fit.
    agent = harness.agent()
    asyncio.run(agent.run(1000, 64))
    assert record["solo"]["stop_reason"] == "horizon"
    assert not agent.fits(1000)
    assert len(agent.state.turns) == len(requests)


def test_output_count_charges_only_output_tokens(tmp_path):
    config = solo_config(budget=200, checkpoints=(50, 100, 200), count="output")
    record = Harness(config, tmp_path).trial()
    spent = record["solo"]["spent"]
    assert spent == [20 * (k + 1) for k in range(len(spent))]
    # Steps continue while another min_output (20) still fits: 200 / 20 = 10 steps.
    assert len(spent) == 10
    assert [p["steps"] for p in record["solo"]["checkpoints"]] == [2, 5, 10]


def test_max_steps_bounds_a_trajectory(tmp_path):
    config = make_config(
        factors=SOLO,
        benchmarks={"toy_arithmetic": {"items": 3, "budget_tokens": 200}},
        solo={"checkpoints": [200], "max_steps": 3},
        budget={"count": "output"},
    )
    record = Harness(config, tmp_path).trial()
    assert record["solo"]["steps"] == 3 and record["solo"]["stop_reason"] == "max_steps"


def test_requests_do_not_depend_on_the_horizon(tmp_path):
    short = Harness(solo_config(budget=600, checkpoints=(200, 600)), tmp_path, name="short")
    long = Harness(solo_config(budget=1200, checkpoints=(200, 600, 1200)), tmp_path, name="long")
    short_record, long_record = short.trial(), long.trial()
    a, b = short.provider.requests, long.provider.requests
    assert 0 < len(a) < len(b)
    assert [r.to_dict() for r in a] == [r.to_dict() for r in b[: len(a)]]
    # So the state at a shared checkpoint is the same, whatever the horizon.
    for index in (0, 1):
        assert (
            short_record["solo"]["checkpoints"][index] == long_record["solo"]["checkpoints"][index]
        )


def test_a_trajectory_continues_from_a_checkpoint_state_from_the_call_log(tmp_path):
    config = solo_config()
    harness = Harness(config, tmp_path)
    agent = harness.agent()
    asyncio.run(agent.run(1200, 64))
    full = agent.state
    paid = len(harness.provider.requests)

    # Serialize the state held at a checkpoint, restore it and continue to the horizon.
    snapshot = json.loads(json.dumps(full.at(400).to_dict()))
    restored = SoloState.from_dict(snapshot)
    assert len(restored.turns) == sum(1 for s in full.spent if s <= 400) > 0
    assert len(restored.messages) == 1 + 2 * len(restored.turns)
    resumed = harness.agent(restored)
    asyncio.run(resumed.run(1200, 64))
    assert len(harness.provider.requests) == paid  # every call served from the log
    assert resumed.state.spent == full.spent
    assert [t.text for t in resumed.state.turns] == [t.text for t in full.turns]
    assert resumed.state.messages == full.messages


def test_self_checks_rotate_methods_and_carry_the_whole_trajectory(tmp_path):
    config = solo_config(budget=3000, checkpoints=(3000,))
    harness = Harness(config, tmp_path)
    harness.trial()
    requests = harness.provider.requests
    assert len(requests) >= 4
    first = requests[0].messages[-1].content
    assert "Compute: " in first and "writing out your reasoning in full" in first
    assert SOLO_COMMIT in first
    for k, request in enumerate(requests[1:], start=1):
        last = request.messages[-1].content
        assert last.startswith(SOLO_CHECK_HEADING.format(step=k))
        assert SOLO_CHECK_METHODS[(k - 1) % len(SOLO_CHECK_METHODS)] in last
        assert SOLO_COMMIT in last
        assert len(request.messages) == 2 + 2 * k  # system, then every earlier exchange
        assert request.max_output_tokens == config.generation.max_output_tokens_per_call
    assert not any("Length limit" in m.content for r in requests for m in r.messages)
    assert len({r.seed for r in requests}) == len(requests)


def test_unparsable_steps_keep_the_previous_answer_and_abstentions_are_recorded(tmp_path):
    def flaky(request, spec):
        step = step_of(request)
        if step == 1:
            return "I lost track. " + " ".join(["word"] * 30)
        return f"{' '.join(['word'] * 17)} <answer>{step}</answer> <confidence>50</confidence>"

    config = solo_config(budget=2000, checkpoints=(2000,))
    record = Harness(config, tmp_path, responder=flaky).trial()
    solo = record["solo"]
    assert solo["step_abstained"][:3] == [False, True, False]
    assert record["parse_failures"] == 1

    def mute(request, spec):
        return "no answer at all " + " ".join(["word"] * 20)

    record = Harness(config, tmp_path, responder=mute, name="mute").trial()
    point = record["solo"]["checkpoints"][0]
    assert point["steps"] > 0 and point["abstained"] and not point["correct"]
    assert record["team"]["tie_break"] == "no_votes"


def test_held_answer_survives_an_unparsable_step(tmp_path):
    def flaky(request, spec):
        if step_of(request) == 1:
            return " ".join(["word"] * 30)
        return f"{' '.join(['word'] * 17)} <answer>{step_of(request)}</answer>"

    config = solo_config(budget=2000, checkpoints=(2000,))
    harness = Harness(config, tmp_path, responder=flaky)
    agent = harness.agent()
    asyncio.run(agent.step())
    asyncio.run(agent.step())
    assert agent.state.held().answer == 0  # step 1 had no answer; step 0's is still held
    point = SoloProtocol(config, harness.task, harness.gateway).checkpoint(agent.state, 2000)
    assert point["answer_step"] == 0 and point["steps"] == 2 and not point["abstained"]


def test_solo_runs_through_the_runner_and_rebuilds_offline(tmp_path):
    config = make_config(
        factors={"team_size": [1, 3], "diversity": ["identical"], "communication": ["solo"]},
        benchmarks={"toy_arithmetic": {"items": 4, "budget_tokens": 1500}},
        solo={"checkpoints": [500, 1000, 1500]},
        run={"trial_concurrency": 2},
    )
    # A solo cell with three agents is degenerate and skipped.
    assert [c.id for c in config.conditions()] == ["n1-identical-solo-homogeneous"]
    assert [c.id for c in config.skipped_conditions()] == ["n3-identical-solo-homogeneous"]
    store, report = run_experiment(config, tmp_path / "run")
    assert report.completed == 4 and report.failed == 0
    text = write_reports(store)
    assert "## Solo scaling" in text and "| 1,500 |" in text
    assert (store.dir / "solo_checkpoints.csv").read_text().startswith("benchmark,condition")
    calls = store.calls()
    assert all(c["trial_key"].startswith("solo|toy_arithmetic|") for c in calls)

    def outcomes(s):
        return {t["trial_key"]: t["solo"]["checkpoints"] for t in s.trials()}

    before = outcomes(store)
    rebuilt = rebuild_trials(store, log=lambda _: None)
    assert rebuilt.failed == 0 and rebuilt.calls_paid == 0
    assert outcomes(RunStore.open(store.dir)) == before


def test_solo_config_validation():
    assert Communication.parse("solo") == Communication("solo", 0)
    assert solo_config().solo.checkpoints == (100, 200, 400, 800, 1200)
    with pytest.raises(ConfigError, match="must cover the largest checkpoint"):
        solo_config(budget=1000, checkpoints=(500, 2000))
    with pytest.raises(ConfigError, match="strictly increasing"):
        solo_config(checkpoints=(400, 200))
    with pytest.raises(ConfigError, match="unknown key"):
        make_config(factors=SOLO, solo={"checkpoint": [1]})
    # Non-solo experiments ignore the (default) checkpoints entirely.
    assert make_config().solo.checkpoints[-1] == 32000


def _trial(item, steps_correct, checkpoints, initial=None):
    """A minimal solo trial record for the analysis functions."""
    return {
        "benchmark": "b",
        "item_id": item,
        "condition": {
            "id": "solo",
            "team_size": 1,
            "diversity": "identical",
            "rounds": 0,
            "model_assignment": "m",
        },
        "initial_correct": [steps_correct[0] if initial is None else initial],
        "team": {"correct": checkpoints[-1][1]},
        "solo": {
            "steps": len(steps_correct),
            "steps_within_horizon": len(steps_correct),
            "step_correct": steps_correct,
            "step_abstained": [False] * len(steps_correct),
            "overshoot": 0,
            "checkpoints": [
                {
                    "tokens": tokens,
                    "steps": steps,
                    "spent": tokens // 2 * bool(steps),
                    "output_tokens": 10 * steps,
                    "key": key,
                    "correct": correct,
                    "abstained": False,
                    "metrics": {"f1": 0.5 + 0.5 * correct} if steps else {},
                }
                for tokens, correct, steps, key in checkpoints
            ],
        },
    }


def test_checkpoint_and_curve_rows():
    trials = [
        # tokens, correct, steps, key
        _trial(
            "i1", [False, True], [(100, False, 0, None), (200, False, 1, "a"), (400, True, 2, "b")]
        ),
        _trial(
            "i2", [True, True], [(100, False, 0, None), (200, True, 1, "c"), (400, True, 2, "c")]
        ),
        _trial(
            "i3",
            [True, None, False],
            [(100, False, 0, None), (200, True, 1, "d"), (400, False, 3, "e")],
        ),
    ]
    rows = checkpoint_rows(trials)
    assert [r["tokens"] for r in rows] == [100, 200, 400]
    assert [r["reached"] for r in rows] == [0.0, 1.0, 1.0]
    assert [r["accuracy"]["mean"] for r in rows] == pytest.approx([0.0, 2 / 3, 2 / 3])
    assert rows[1]["gain_vs_previous"]["mean"] == pytest.approx(2 / 3)
    assert rows[2]["changed"] == pytest.approx(2 / 3)
    assert rows[2]["gain_vs_initial"]["mean"] == pytest.approx(0.0)  # +1, 0, -1
    assert rows[0]["metrics"] == {} and rows[2]["metrics"]["f1"] == pytest.approx(2.5 / 3)
    (curve,) = curve_rows(trials)
    assert curve["wrong_to_right"] == 1 and curve["right_to_wrong"] == 1
    assert curve["best_tokens"] == 200 and curve["plateau_tokens"] == 200
    assert curve["final_minus_initial"]["mean"] == pytest.approx(0.0)


def test_plot_solo_curves_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")
    from workforce.analysis.plot import plot_solo_curves

    config = solo_config(budget=1200, checkpoints=(400, 800, 1200))
    store, _ = run_experiment(config, tmp_path / "run")
    out = plot_solo_curves(store.trials(), tmp_path / "plots" / "curves.png")
    assert out.read_bytes().startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="no solo trials"):
        plot_solo_curves([], tmp_path / "empty.png")


def test_cost_bound_includes_the_last_step_overshooting_the_horizon():
    from workforce.cost import budget_bound

    config = solo_config(budget=1200, checkpoints=(1200,))
    (row,) = budget_bound(config)["rows"]
    ceiling = config.generation.max_output_tokens_per_call
    assert row["max_tokens"] == row["trials"] * (1200 + ceiling)
