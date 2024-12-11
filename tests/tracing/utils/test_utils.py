import pytest

from mlflow.entities import LiveSpan
from mlflow.exceptions import MlflowTracingException
from mlflow.models.dependencies_schemas import DEPENDENCIES_SCHEMA_KEY, DependenciesSchemasType
from mlflow.models.model import ModelInfo
from mlflow.tracing.utils import (
    deduplicate_span_names_in_place,
    encode_span_id,
    maybe_get_dependencies_schemas,
    maybe_get_request_id,
)

from tests.tracing.helper import create_mock_otel_span


def test_deduplicate_span_names():
    span_names = ["red", "red", "blue", "red", "green", "blue"]

    spans = [
        LiveSpan(create_mock_otel_span("trace_id", span_id=i, name=span_name), request_id="tr-123")
        for i, span_name in enumerate(span_names)
    ]
    deduplicate_span_names_in_place(spans)

    assert [span.name for span in spans] == [
        "red_1",
        "red_2",
        "blue_1",
        "red_3",
        "green",
        "blue_2",
    ]
    # Check if the span order is preserved
    assert [span.span_id for span in spans] == [encode_span_id(i) for i in [0, 1, 2, 3, 4, 5]]


def test_maybe_get_request_id():
    assert maybe_get_request_id(is_evaluate=True) is None

    try:
        from mlflow.pyfunc.context import Context, set_prediction_context
    except ImportError:
        pytest.skip("Skipping the rest of tests as mlflow.pyfunc module is not available.")

    with set_prediction_context(Context(request_id="eval", is_evaluate=True)):
        assert maybe_get_request_id(is_evaluate=True) == "eval"

    with set_prediction_context(Context(request_id="non_eval", is_evaluate=False)):
        assert maybe_get_request_id(is_evaluate=True) is None

    with pytest.raises(MlflowTracingException, match="Missing request_id for context"):
        with set_prediction_context(Context(request_id=None, is_evaluate=True)):
            maybe_get_request_id(is_evaluate=True)


def test_maybe_get_dependencies_schemas():
    try:
        from mlflow.pyfunc.context import Context, set_prediction_context
    except ImportError:
        pytest.skip("Skipping the rest of tests as mlflow.pyfunc module is not available.")

    model_info = ModelInfo(
        artifact_path="model",
        flavors={
            "python_function": {"loader_module": "mlflow.pyfunc", "pickled_model": "model.pkl"},
            "lang": {"loader_module": "mlflow.lang", "pickled_model": "model.pkl"},
        },
        model_uri="models:/model",
        model_uuid="model_uuid",
        run_id="run_id",
        saved_input_example_info=None,
        signature=None,
        utc_time_created="2021-01-01",
        mlflow_version="1.0.0",
        metadata={
            DEPENDENCIES_SCHEMA_KEY: {
                DependenciesSchemasType.RETRIEVERS.value: [
                    {
                        "name": "retriever",
                        "primary_key": "primary-key",
                        "text_column": "text-column",
                        "doc_uri": "doc-uri",
                        "other_columns": ["column1", "column2"],
                    }
                ]
            }
        },
    )

    assert maybe_get_dependencies_schemas() is None

    with set_prediction_context(Context(model_info=model_info)):
        assert maybe_get_dependencies_schemas() == {
            DependenciesSchemasType.RETRIEVERS.value: [
                {
                    "name": "retriever",
                    "primary_key": "primary-key",
                    "text_column": "text-column",
                    "doc_uri": "doc-uri",
                    "other_columns": ["column1", "column2"],
                }
            ]
        }

    assert maybe_get_dependencies_schemas() is None
