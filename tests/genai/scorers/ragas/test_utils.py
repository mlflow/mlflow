import pytest
from langchain_core.documents import Document
from ragas.dataset_schema import MultiTurnSample, SingleTurnSample

import mlflow
from mlflow.entities.span import SpanType
from mlflow.genai.scorers.ragas.utils import (
    create_mlflow_error_message_from_ragas_param,
    map_scorer_inputs_to_ragas_sample,
)


@pytest.fixture
def sample_trace():
    with mlflow.start_span(name="root", span_type=SpanType.CHAIN) as root:
        root.set_inputs({"messages": [{"role": "user", "content": "Hello"}]})
        with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as r:
            r.set_outputs(
                [
                    Document(page_content="Document 1"),
                    Document(page_content="Document 2"),
                ]
            )
        with mlflow.start_span(name="tool", span_type=SpanType.TOOL) as t:
            t.set_inputs({"x": 1})
            t.set_outputs({"y": 2})
        root.set_outputs("Response")
    return mlflow.get_trace(root.trace_id)


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


def test_map_scorer_inputs_to_ragas_sample_with_trace(sample_trace):
    sample = map_scorer_inputs_to_ragas_sample(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
        trace=sample_trace,
    )

    assert sample.retrieved_contexts is not None
    assert len(sample.retrieved_contexts) == 2
    assert "Document 1" in str(sample.retrieved_contexts)
    assert "Document 2" in str(sample.retrieved_contexts)


def test_map_scorer_inputs_with_rubrics():
    rubrics_dict = {
        "0": "Poor response",
        "1": "Good response",
    }
    expectations = {
        "rubrics": rubrics_dict,
        "expected_output": "MLflow is a platform",
    }

    sample = map_scorer_inputs_to_ragas_sample(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
        expectations=expectations,
    )

    assert sample.rubrics == rubrics_dict
    assert sample.reference == "MLflow is a platform"
    assert sample.user_input == "What is MLflow?"
    assert sample.response == "MLflow is a platform"


def test_map_scorer_inputs_with_only_rubrics():
    rubrics_dict = {
        "0": "Incorrect answer",
        "1": "Correct answer",
    }
    expectations = {"rubrics": rubrics_dict}

    sample = map_scorer_inputs_to_ragas_sample(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
        expectations=expectations,
    )

    assert sample.rubrics == rubrics_dict
    assert sample.reference is None


@pytest.mark.parametrize(
    ("ragas_param", "expected_mlflow_param", "expected_guidance"),
    [
        ("user_input", "inputs", "judge(inputs='What is MLflow?'"),
        ("response", "outputs", "judge(inputs='...', outputs='MLflow is a platform'"),
        (
            "reference",
            "expectations['expected_output']",
            "expectations={'expected_output':",
        ),
        ("retrieved_contexts", "trace with retrieval spans", "retrieval spans"),
        ("reference_contexts", "trace with retrieval spans", "retrieval spans"),
        ("rubrics", "expectations['rubrics']", "expectations={'rubrics':"),
        (
            "reference_tool_calls",
            "expectations['expected_tool_calls']",
            "expected_tool_calls",
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


@pytest.mark.parametrize(
    ("is_agentic", "expected_type"),
    [
        (True, MultiTurnSample),
        (False, SingleTurnSample),
    ],
)
def test_map_scorer_inputs_sample_type_based_on_is_agentic(is_agentic, expected_type):
    sample = map_scorer_inputs_to_ragas_sample(
        inputs="What is the weather?",
        outputs="It's sunny today.",
        is_agentic=is_agentic,
    )
    assert isinstance(sample, expected_type)


@pytest.mark.parametrize(
    ("expectations", "assertion_fn"),
    [
        (
            {
                "expected_tool_calls": [
                    {"name": "weather_check", "arguments": {"location": "Paris"}},
                ]
            },
            lambda s: (
                len(s.reference_tool_calls) == 1
                and s.reference_tool_calls[0].name == "weather_check"
                and s.reference_tool_calls[0].args == {"location": "Paris"}
            ),
        ),
        (
            {"expected_output": "Table booked at a Chinese restaurant for 8pm"},
            lambda s: s.reference == "Table booked at a Chinese restaurant for 8pm",
        ),
        (
            {"reference_topics": ["machine learning", "data science", "MLflow"]},
            lambda s: s.reference_topics == ["machine learning", "data science", "MLflow"],
        ),
    ],
)
def test_map_scorer_inputs_agentic_with_expectations(expectations, assertion_fn):
    sample = map_scorer_inputs_to_ragas_sample(
        is_agentic=True,
        expectations=expectations,
    )

    assert isinstance(sample, MultiTurnSample)
    assert assertion_fn(sample)
