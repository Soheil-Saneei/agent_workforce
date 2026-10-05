from dataclasses import replace

from workforce.models.mock import MockProvider

from conftest import make_config, run_experiment


def test_spend_cap_stops_before_it_is_exceeded_and_resume_continues(tmp_path):
    priced = {"provider": "mock", "model": "mock-priced", "price": {"input": 1.0, "output": 4.0}}
    config = make_config(
        models={"base": priced},
        generation={"min_output_tokens": 20, "max_output_tokens_per_call": 500},
        run={"max_spend_usd": 0.02},
    )
    provider = MockProvider(config.models["base"])
    store, report = run_experiment(config, tmp_path / "run", {"base": provider})
    assert report.stop_reason == "spend_cap"
    assert 0 < store.spent_usd <= 0.02
    assert report.remaining > 0
    assert all(c["cost_usd"] > 0 for c in store.calls())

    raised = replace(config, run=replace(config.run, max_spend_usd=100.0))
    store, report = run_experiment(raised, tmp_path / "run", {"base": provider})
    assert report.stop_reason == "finished" and report.remaining == 0
