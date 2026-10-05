"""Interrupted runs resume without paying again for completed calls."""

from workforce.models.base import FatalProviderError
from workforce.models.mock import MockProvider
from workforce.runner import rebuild_trials
from workforce.store import RunStore

from conftest import make_config, run_experiment


class CrashingMock(MockProvider):
    """Fails every call after the first ``budget`` calls (simulates a crash or outage)."""

    def __init__(self, spec, budget, **kwargs):
        super().__init__(spec, **kwargs)
        self.budget = budget

    async def _complete_once(self, request):
        if len(self.requests) >= self.budget:
            raise FatalProviderError("simulated outage")
        return await super()._complete_once(request)


def outcomes(store):
    return {
        t["trial_key"]: (t["team"]["key"], t["team"]["correct"], t["tokens"]["total"])
        for t in store.trials()
    }


def test_resume_replays_logged_calls_and_matches_an_uninterrupted_run(tmp_path):
    config = make_config()
    spec = config.models["base"]

    clean = MockProvider(spec)
    clean_store, clean_report = run_experiment(config, tmp_path / "clean", {"base": clean})
    total_calls = len(clean.requests)
    assert clean_report.completed == len(clean_store.trials()) > 0

    crashing = CrashingMock(spec, budget=total_calls // 2)
    first_store, first = run_experiment(config, tmp_path / "resumed", {"base": crashing})
    paid_before_crash = len(first_store.calls())
    assert first.failed > 0 and paid_before_crash == total_calls // 2

    fresh = MockProvider(spec)
    store, second = run_experiment(config, tmp_path / "resumed", {"base": fresh})
    assert second.failed == 0 and second.remaining == 0
    # Only the calls missing from the log were made again.
    assert len(fresh.requests) == total_calls - paid_before_crash
    assert second.calls_cached >= paid_before_crash - sum(t["calls"] for t in first_store.trials())
    assert len(store.calls()) == total_calls
    assert outcomes(store) == outcomes(clean_store)

    # A completed run resumes as a no-op.
    idle = MockProvider(spec)
    _, third = run_experiment(config, tmp_path / "resumed", {"base": idle})
    assert third.completed == 0 and idle.requests == []


def test_torn_final_log_line_is_ignored(tmp_path):
    config = make_config(
        factors={"team_size": [3], "diversity": ["identical"], "communication": ["independent"]}
    )
    store, _ = run_experiment(
        config, tmp_path / "run", {"base": MockProvider(config.models["base"])}
    )
    with store.calls_path.open("a") as handle:
        handle.write('{"key": "half-writ')
    reopened = RunStore.open(store.dir)
    assert len(reopened.calls()) == len(store.calls())


def test_rebuild_regenerates_trials_offline_from_the_call_log(tmp_path):
    config = make_config()
    store, _ = run_experiment(
        config, tmp_path / "run", {"base": MockProvider(config.models["base"])}
    )
    before = outcomes(store)
    report = rebuild_trials(store, log=lambda _: None)
    assert report.failed == 0 and report.calls_paid == 0
    assert outcomes(RunStore.open(store.dir)) == before
