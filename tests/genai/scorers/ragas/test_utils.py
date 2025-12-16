import pytest
from langchain_core.documents import Document

import mlflow
from mlflow.entities.span import SpanType
from mlflow.genai.scorers.ragas.utils import (
    create_mlflow_error_message_from_ragas_param,
    map_scorer_inputs_to_ragas_sample,
)


def test_map_scorer_inputs_to_ragas_sample_basic():
    sample = map_scorer_inputs_to_ragas_sample(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
    )

    assert sample.user_input == "What is MLflow?"
    assert sample.response == "MLflow is a platform"
    assert sample.reference is None
    assert sample.retrieved_contexts is None


def test_map_scorer_inputs_to_ragas_sample_with_expectations():
    expectations = {
        "expected_output": "MLflow is an open source platform",
    }

    sample = map_scorer_inputs_to_ragas_sample(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
        expectations=expectations,
    )

    assert sample.reference == "MLflow is an open source platform"


def test_map_scorer_inputs_to_ragas_sample_with_trace():
    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def retrieve_docs():
        return [
            Document(page_content="Document 1", metadata={}),
            Document(page_content="Document 2", metadata={}),
        ]

    retrieve_docs()
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    sample = map_scorer_inputs_to_ragas_sample(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
        trace=trace,
    )

    assert sample.retrieved_contexts is not None
    assert len(sample.retrieved_contexts) == 2
    assert "Document 1" in str(sample.retrieved_contexts)
    assert "Document 2" in str(sample.retrieved_contexts)


@pytest.mark.parametrize(
    ("ragas_param", "expected_mlflow_param", "expected_guidance"),
    [
        ("user_input", "inputs", "judge(inputs='What is MLflow?'"),
        (
            "response",
            "outputs",
            "judge(inputs='...', outputs='MLflow is a platform'",
        ),
        (
            "reference",
            "expectations['expected_output']",
            "expectations={'expected_output':",
        ),
        (
            "retrieved_contexts",
            "trace with retrieval spans",
            "retrieval spans",
        ),
        (
            "reference_contexts",
            "trace with retrieval spans",
            "retrieval spans",
        ),
    ],
)
def test_create_mlflow_error_message_from_ragas_param(
    ragas_param, expected_mlflow_param, expected_guidance
):
    metric_name = "TestMetric"
    error_message = create_mlflow_error_message_from_ragas_param(ragas_param, metric_name)

    assert metric_name in error_message
    assert expected_mlflow_param in error_message
    assert expected_guidance in error_message
