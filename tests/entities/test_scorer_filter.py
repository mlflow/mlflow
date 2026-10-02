from dataclasses import FrozenInstanceError

import pytest

from mlflow.entities.scorer_filter import ScorerFilter
from mlflow.exceptions import MlflowException


def test_scorer_filter_normalizes_literal_names():
    selection = ScorerFilter(
        experiment_ids=["1", "1"],
        scorers=[("1", "redundant"), ("2", "*/%2F/'毒性"), ("2", "*/%2F/'毒性")],
    )
    assert selection.experiment_ids == frozenset({"1"})
    assert selection.scorers == frozenset({("2", "*/%2F/'毒性")})
    assert selection.candidate_experiment_ids == {"1", "2"}
    assert ScorerFilter().candidate_experiment_ids == set()
    with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
        selection.experiment_ids = frozenset()


@pytest.mark.parametrize("experiment_id", [None, 1, "", "invalid id"])
@pytest.mark.parametrize("whole_experiment", [False, True])
def test_scorer_filter_rejects_invalid_experiment_ids(experiment_id, whole_experiment):
    kwargs = (
        {"experiment_ids": [experiment_id]}
        if whole_experiment
        else {"scorers": [(experiment_id, "name")]}
    )
    with pytest.raises(MlflowException, match="nonempty|Invalid experiment ID"):
        ScorerFilter(**kwargs)


@pytest.mark.parametrize("name", [None, 1, "", "  "])
def test_scorer_filter_rejects_invalid_names(name):
    with pytest.raises(MlflowException, match="nonempty"):
        ScorerFilter(scorers=[("1", name)])


@pytest.mark.parametrize("name", ["*", "a/b", "%2F", "毒性", " name "])
def test_scorer_filter_preserves_exact_names(name):
    selection = ScorerFilter(scorers=[("1", name), ("2", "other")])
    assert selection.scorers == {("1", name), ("2", "other")}
    assert selection.experiment_ids == set()
