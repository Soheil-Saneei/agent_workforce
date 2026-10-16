import asyncio

import pytest

from workforce.cost import budget_bound, project
from workforce.models.base import ChatRequest, Message, ModelSpec, Price
from workforce.models.mock import MockProvider, truncate_words
from workforce.protocol.parsing import parse_output, strip_confidence

from conftest import make_config, run_experiment


def test_parse_output_reads_the_last_tags():
    text = (
        "draft <answer>wrong</answer>\n<critique>P2 is misread</critique>\n"
        "<evidence>P3, 12, 3</evidence>\n<answer> Paris </answer>\n<confidence>85%</confidence>"
    )
    parsed = parse_output(text)
    assert parsed.answer_text == "Paris"
    assert parsed.confidence == 85
    assert parsed.evidence == (3, 12)
    assert parsed.critique == "P2 is misread"
    assert parse_output("<confidence>0.7</confidence>").confidence == pytest.approx(70)
    assert parse_output("reasoning <answer>Lyon").answer_text == "Lyon"  # truncated tag
    assert parse_output("no tags").answer_text is None


def test_strip_confidence():
    assert strip_confidence("x <answer>1</answer>\n<confidence>99</confidence>") == (
        "x <answer>1</answer>"
    )


def test_mock_is_deterministic_and_truncates_exactly():
    spec = ModelSpec("base", "mock", "mock-base")
    request = ChatRequest((Message("user", "Compute: (2 + 3) * 4"),), 1000, 0.7, 5)
    a = asyncio.run(MockProvider(spec).complete(request))
    b = asyncio.run(MockProvider(spec).complete(request))
    assert a.text == b.text and a.usage == b.usage
    assert truncate_words("a  b\nc d", 3) == ("a  b\nc", 3, True)
    short = asyncio.run(MockProvider(spec).complete(ChatRequest(request.messages, 5, 0.7, 5)))
    assert short.finish_reason == "length" and short.usage.output_tokens == 5


def test_cost_projection_scales_measured_tokens(tmp_path):
    priced = {
        "provider": "mock",
        "model": "mock-base",
        "price": {"input": 1.0, "output": 2.0, "cached_input": 0.5},
    }
    measured = make_config(models={"base": priced})
    store, _ = run_experiment(
        measured, tmp_path / "measured", {"base": MockProvider(measured.models["base"])}
    )
    target = make_config(
        models={"base": priced},
        experiment={"name": "full", "seed": 11, "replications": 3},
        benchmarks={"toy_arithmetic": {"items": 40, "budget_tokens": 20000}},
    )
    projection = project(store.trials(), target)
    assert projection.missing == []
    measured_cost_per_pass = sum(t["cost_usd"] for t in store.trials()) / 4  # 4 measured items
    assert projection.total_usd == pytest.approx(measured_cost_per_pass * 40 * 3)
    assert projection.ci_usd[0] <= projection.total_usd <= projection.ci_usd[1]
    bound = budget_bound(target)
    assert bound["total_usd"] == pytest.approx(
        len(target.conditions()) * 40 * 3 * 20000 * 2.0 / 1e6
    )
    assert projection.total_usd < bound["total_usd"]
    assert budget_bound(target, Price(0, 1))["total_usd"] < bound["total_usd"]
