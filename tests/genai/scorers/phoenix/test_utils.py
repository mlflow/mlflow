import sys
from unittest.mock import Mock, patch

import pytest

from mlflow.exceptions import MlflowException


def test_check_phoenix_installed_raises_without_phoenix():
    with patch.dict("sys.modules", {"phoenix": None, "phoenix.evals": None}):
        for mod in list(sys.modules.keys()):
            if "mlflow.genai.scorers.phoenix" in mod:
                del sys.modules[mod]

        from mlflow.genai.scorers.phoenix.utils import check_phoenix_installed

        with pytest.raises(MlflowException, match="arize-phoenix-evals"):
            check_phoenix_installed()


def test_map_scorer_inputs_basic():
    from mlflow.genai.scorers.phoenix.utils import map_scorer_inputs_to_phoenix_record

    record = map_scorer_inputs_to_phoenix_record(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
    )

    assert record["input"] == "What is MLflow?"
    assert record["output"] == "MLflow is a platform"
    assert "reference" not in record


def test_map_scorer_inputs_with_expected_response():
    from mlflow.genai.scorers.phoenix.utils import map_scorer_inputs_to_phoenix_record

    record = map_scorer_inputs_to_phoenix_record(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
        expectations={"expected_response": "MLflow is an ML platform."},
    )

    assert record["input"] == "What is MLflow?"
    assert record["output"] == "MLflow is a platform"
    assert record["reference"] == "MLflow is an ML platform."


def test_map_scorer_inputs_with_context():
    from mlflow.genai.scorers.phoenix.utils import map_scorer_inputs_to_phoenix_record

    record = map_scorer_inputs_to_phoenix_record(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
        expectations={"context": "MLflow context here."},
    )

    assert record["reference"] == "MLflow context here."


def test_map_scorer_inputs_expected_response_priority():
    from mlflow.genai.scorers.phoenix.utils import map_scorer_inputs_to_phoenix_record

    record = map_scorer_inputs_to_phoenix_record(
        inputs="test",
        outputs="test output",
        expectations={
            "expected_response": "priority value",
            "context": "should be ignored",
            "reference": "also ignored",
        },
    )

    assert record["reference"] == "priority value"


def test_map_scorer_inputs_with_trace():
    from mlflow.genai.scorers.phoenix.utils import map_scorer_inputs_to_phoenix_record

    mock_trace = Mock()

    with (
        patch(
            "mlflow.genai.scorers.phoenix.utils.resolve_inputs_from_trace",
            return_value="resolved input",
        ),
        patch(
            "mlflow.genai.scorers.phoenix.utils.resolve_outputs_from_trace",
            return_value="resolved output",
        ),
        patch(
            "mlflow.genai.scorers.phoenix.utils.resolve_expectations_from_trace",
            return_value={"expected_response": "resolved reference"},
        ),
    ):
        record = map_scorer_inputs_to_phoenix_record(trace=mock_trace)

        assert record["input"] == "resolved input"
        assert record["output"] == "resolved output"
        assert record["reference"] == "resolved reference"


def test_map_scorer_inputs_trace_fallback_context():
    from mlflow.genai.scorers.phoenix.utils import map_scorer_inputs_to_phoenix_record

    mock_trace = Mock()

    with (
        patch(
            "mlflow.genai.scorers.phoenix.utils.resolve_inputs_from_trace",
            return_value="input",
        ),
        patch(
            "mlflow.genai.scorers.phoenix.utils.resolve_outputs_from_trace",
            return_value="output",
        ),
        patch(
            "mlflow.genai.scorers.phoenix.utils.resolve_expectations_from_trace",
            return_value=None,
        ),
        patch(
            "mlflow.genai.scorers.phoenix.utils.extract_retrieval_context_from_trace",
            return_value={"span1": [{"content": "context from trace"}]},
        ),
    ):
        record = map_scorer_inputs_to_phoenix_record(trace=mock_trace)

        assert record["reference"] == "context from trace"


def test_map_scorer_inputs_none_values():
    from mlflow.genai.scorers.phoenix.utils import map_scorer_inputs_to_phoenix_record

    record = map_scorer_inputs_to_phoenix_record()

    assert "input" not in record
    assert "output" not in record
    assert "reference" not in record
