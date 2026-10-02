import pytest

from mlflow.entities.scorer_filter import ScorerFilter
from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import ScorerFilter as ProtoScorerFilter


def test_scorer_filter_intersection_matches_set_intersection():
    universe = {(eid, name) for eid in ["1", "2", "3"] for name in ["a", "b"]}
    selections = [
        ScorerFilter(),
        ScorerFilter(experiment_ids={"1", "2"}),
        ScorerFilter(scorers={("1", "a"), ("2", "b")}),
        ScorerFilter(experiment_ids={"2", "3"}, scorers={("1", "b")}),
    ]

    def selected(selection):
        return {p for p in universe if p[0] in selection.experiment_ids or p in selection.scorers}

    for left in selections:
        for right in selections:
            assert selected(left.intersect(right)) == selected(left) & selected(right)
            assert left.intersect(right) == right.intersect(left)


def test_scorer_filter_normalizes_and_round_trips_literal_names():
    selection = ScorerFilter(
        experiment_ids=["1", "1"],
        scorers=[("1", "redundant"), ("2", "*/%2F/'毒性"), ("2", "*/%2F/'毒性")],
    )
    assert selection.scorers == {("2", "*/%2F/'毒性")}
    assert selection.candidate_experiment_ids == {"1", "2"}
    assert ScorerFilter.from_proto(selection.to_proto()) == selection
    assert ScorerFilter.from_proto(ProtoScorerFilter()) == ScorerFilter()


@pytest.mark.parametrize(
    "selector",
    [
        ProtoScorerFilter.Scorer(scorer_name="a"),
        ProtoScorerFilter.Scorer(experiment_id="1"),
        ProtoScorerFilter.Scorer(experiment_id="invalid id", scorer_name="a"),
        ProtoScorerFilter.Scorer(experiment_id="1", scorer_name="  "),
    ],
)
def test_scorer_filter_rejects_invalid_pairs(selector):
    with pytest.raises(MlflowException, match="nonempty|Invalid experiment ID"):
        ScorerFilter.from_proto(ProtoScorerFilter(scorers=[selector]))
