from collections import Counter
from pathlib import Path

import pytest

from workforce.design import Communication, ConfigError, apply_overrides, load_config

from conftest import make_config


def test_factorial_expansion_crosses_all_factors_and_skips_degenerate_cells():
    config = make_config(
        factors={
            "team_size": [1, 3, 5],
            "diversity": ["identical", "moderate", "strong"],
            "communication": ["independent", "horizontal"],
        }
    )
    ids = [c.id for c in config.conditions()]
    # 3 x 3 x 2 = 18 cells; a single agent cannot be moderately or strongly diverse.
    assert len(ids) == 18 - 4
    assert len(set(ids)) == len(ids)
    assert "n1-identical-independent-homogeneous" in ids
    assert "n5-strong-horizontal-1r-homogeneous" in ids
    assert {c.id for c in config.skipped_conditions()} == {
        "n1-moderate-independent-homogeneous",
        "n1-moderate-horizontal-1r-homogeneous",
        "n1-strong-independent-homogeneous",
        "n1-strong-horizontal-1r-homogeneous",
    }


def test_include_degenerate_keeps_the_full_grid():
    config = make_config(
        factors={
            "team_size": [1, 3],
            "diversity": ["identical", "moderate"],
            "communication": ["independent"],
        },
        run={"include_degenerate": True},
    )
    assert len(config.conditions()) == 4
    assert config.skipped_conditions() == []


def test_first_experiment_grid_has_ten_conditions():
    config = load_config(Path(__file__).parent.parent / "configs" / "team-study.toml")
    assert len(config.conditions()) == 10
    assert {c.communication.rounds for c in config.conditions()} == {0, 2}


def test_communication_levels_parse():
    assert Communication.parse("independent") == Communication("independent", 0)
    assert Communication.parse("none").id == "independent"
    assert Communication.parse("horizontal") == Communication("horizontal", 1)
    assert Communication.parse("horizontal-2r").rounds == 2
    with pytest.raises(ConfigError):
        Communication.parse("hierarchical")


def test_unknown_keys_are_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        make_config(generation={"temprature": 0.5})
    with pytest.raises(ConfigError, match="unknown diversity"):
        make_config(factors={"diversity": ["extreme"]})


def test_spend_cap_requires_known_prices():
    with pytest.raises(ConfigError, match="no price"):
        make_config(
            models={
                "local": {
                    "provider": "openai_compatible",
                    "model": "llama",
                    "base_url": "http://localhost:8000/v1",
                }
            },
            run={"max_spend_usd": 1.0},
        )


def test_moderate_procedures_rotate_so_no_procedure_is_always_doubled():
    config = make_config(
        factors={"team_size": [5], "diversity": ["moderate"], "communication": ["independent"]}
    )
    condition = config.conditions()[0]
    doubled = Counter()
    for ordinal in range(3):
        procs = Counter(a.procedure for a in config.assign_agents(condition, ordinal, 0))
        assert sorted(procs.values()) == [1, 2, 2]
        doubled.update(p for p, count in procs.items() if count == 2)
    assert set(doubled.values()) == {2}  # each procedure doubled in two of three items


def test_procedure_model_pairing_is_counterbalanced_in_mixed_teams():
    config = make_config(
        factors={
            "team_size": [3],
            "diversity": ["moderate"],
            "communication": ["independent"],
            "model_assignment": ["mixed"],
        },
        models={
            "a": {"provider": "mock", "model": "mock-a"},
            "b": {"provider": "mock", "model": "mock-b"},
            "c": {"provider": "mock", "model": "mock-c"},
        },
        model_assignments={"mixed": {"models": ["a", "b", "c"]}},
    )
    condition = config.conditions()[0]
    pairings = Counter()
    for ordinal in range(9):
        agents = config.assign_agents(condition, ordinal, 0)
        assert sorted(a.model for a in agents) == ["a", "b", "c"]
        pairings.update((a.procedure, a.model) for a in agents)
    # Over 9 items every (procedure, model) pairing occurs equally often.
    assert len(pairings) == 9 and set(pairings.values()) == {3}


def test_mock_override_replaces_every_model():
    raw = {
        "models": {
            "base": {
                "provider": "openai",
                "model": "gpt-6-luna",
                "price": {"input": 1, "output": 2},
            }
        }
    }
    out = apply_overrides(raw, {"mock": True, "items": 3})
    assert out["models"]["base"] == {"provider": "mock", "model": "mock-gpt-6-luna"}
    assert raw["models"]["base"]["provider"] == "openai"  # input untouched
