from __future__ import annotations

from unittest import mock

import pytest

from mlflow.exceptions import MlflowException


def test_check_evidently_installed_raises_when_missing():
    with mock.patch.dict("sys.modules", {"evidently": None}):
        from mlflow.genai.scorers.evidently.utils import check_evidently_installed

        with pytest.raises(MlflowException, match="evidently"):
            check_evidently_installed()


def test_check_evidently_installed_succeeds():
    pytest.importorskip("evidently")
    from mlflow.genai.scorers.evidently.utils import check_evidently_installed

    # Should not raise when evidently is installed
    check_evidently_installed()


def test_get_metric_class_valid():
    pytest.importorskip("evidently")
    from mlflow.genai.scorers.evidently.registry import get_metric_class

    metric_class = get_metric_class("MissingValueCount")
    assert metric_class is not None


def test_get_metric_class_invalid():
    pytest.importorskip("evidently")
    from mlflow.genai.scorers.evidently.registry import get_metric_class

    with pytest.raises(MlflowException, match="Unknown Evidently metric"):
        get_metric_class("NonExistentMetric")


def test_map_scorer_inputs_to_dataframe_with_dict_outputs():
    from mlflow.genai.scorers.evidently.utils import map_scorer_inputs_to_dataframe

    current_df, reference_df = map_scorer_inputs_to_dataframe(
        outputs={"feature_1": 0.5, "feature_2": 1.0},
    )
    assert list(current_df.columns) == ["feature_1", "feature_2"]
    assert current_df.iloc[0]["feature_1"] == 0.5
    assert reference_df is None


def test_map_scorer_inputs_to_dataframe_with_reference():
    from mlflow.genai.scorers.evidently.utils import map_scorer_inputs_to_dataframe

    current_df, reference_df = map_scorer_inputs_to_dataframe(
        outputs={"feature_1": 0.5},
        expectations={"reference_data": [{"feature_1": 0.1}, {"feature_1": 0.2}]},
    )
    assert len(current_df) == 1
    assert reference_df is not None
    assert len(reference_df) == 2


def test_map_scorer_inputs_to_dataframe_raises_without_data():
    from mlflow.genai.scorers.evidently.utils import map_scorer_inputs_to_dataframe

    with pytest.raises(MlflowException, match="require either"):
        map_scorer_inputs_to_dataframe()


def test_evidently_scorer_returns_feedback():
    pytest.importorskip("evidently")
    from mlflow.genai.scorers.evidently import EvidentlyScorer

    scorer = EvidentlyScorer(metric_name="MissingValueCount", column="feature_1")
    assert scorer.name == "MissingValueCount"

    feedback = scorer(outputs={"feature_1": None})
    assert feedback.name == "MissingValueCount"
    assert feedback.source.source_id == "evidently/MissingValueCount"


def test_evidently_scorer_error_returns_feedback_with_error():
    pytest.importorskip("evidently")
    from mlflow.genai.scorers.evidently import EvidentlyScorer

    scorer = EvidentlyScorer(metric_name="MissingValueCount", column="feature_1")

    # Pass invalid data that will cause Evidently to fail
    with mock.patch(
        "mlflow.genai.scorers.evidently.map_scorer_inputs_to_dataframe",
        side_effect=ValueError("test error"),
    ):
        feedback = scorer(outputs={"feature_1": 0.5})

    assert feedback.error is not None


def test_get_scorer_factory():
    pytest.importorskip("evidently")
    from mlflow.genai.scorers.evidently import get_scorer

    scorer = get_scorer("MissingValueCount", column="feature_1")
    assert scorer.name == "MissingValueCount"


def test_concrete_scorer_classes():
    pytest.importorskip("evidently")
    from mlflow.genai.scorers.evidently import MissingValues, UniqueValues

    scorer = MissingValues(column="col1")
    assert scorer.name == "MissingValueCount"

    scorer = UniqueValues(column="col1")
    assert scorer.name == "UniqueValueCount"


@pytest.mark.parametrize(
    ("scorer_class_name", "expected_metric"),
    [
        ("ValueDrift", "ValueDrift"),
        ("MissingValues", "MissingValueCount"),
        ("UniqueValues", "UniqueValueCount"),
    ],
)
def test_concrete_scorer_metric_names(scorer_class_name: str, expected_metric: str):
    pytest.importorskip("evidently")
    import mlflow.genai.scorers.evidently as evidently_module

    scorer_class = getattr(evidently_module, scorer_class_name)
    assert scorer_class.metric_name == expected_metric
