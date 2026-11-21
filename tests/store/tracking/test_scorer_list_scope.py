from dataclasses import FrozenInstanceError

import pytest

from mlflow.store.tracking.scorer_list_scope import (
    ScorerListScope,
    parse_scorer_row_key,
    scorer_row_key,
)


def test_scope_membership_and_immutability():
    pairs = [("1", "a"), ("1", "a")]
    scope = ScorerListScope(
        experiment_ids=["2"],
        scorer_pairs=pairs,
        excluded_experiment_ids={"2"},
        excluded_scorer_pairs={("1", "b")},
    )
    pairs.append(("3", "a"))
    assert scope.permits("1", "a")
    assert not any(scope.permits(e, n) for e, n in [("2", "a"), ("1", "b"), ("3", "a")])
    assert not ScorerListScope().permits("1", "a")
    assert ScorerListScope(experiment_ids=None).permits("1", "a")
    with pytest.raises(FrozenInstanceError, match="cannot assign"):
        scope.experiment_ids = None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"experiment_ids": "1"},
        {"scorer_pairs": [["1", "a"]]},
        {"excluded_experiment_ids": None},
        {"experiment_ids": ["01"]},
        {"scorer_pairs": [("1", "")]},
        {"scorer_pairs": [("1", 3)]},
    ],
)
def test_scope_rejects_bad_shapes(kwargs):
    with pytest.raises(ValueError, match="Invalid"):
        ScorerListScope(**kwargs)


@pytest.mark.parametrize("name", ["a/b", "%", "a%2Fb", "é", "*", " a ", "a\tb"])
@pytest.mark.parametrize("experiment", ["1", "opaque-id"])
def test_row_key_round_trip(experiment, name):
    assert parse_scorer_row_key(scorer_row_key(experiment, name)) == (experiment, name)


@pytest.mark.parametrize(
    "pattern",
    [
        "1/a/b",
        "1_0/a",
        "1٢/a",
        "1/a%2fb",
        "1/%61",
        "01/a",
        "1/",
        "/a",
        "1.0/a",
        "1/%FF",
        "1/ ",
        "9" * 10000 + "/a",
    ],
)
def test_dead_patterns(pattern):
    assert parse_scorer_row_key(pattern) is None
