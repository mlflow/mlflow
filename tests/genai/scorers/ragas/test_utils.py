import pytest
from langchain_core.documents import Document
from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.messages import AIMessage, HumanMessage, ToolCall

import mlflow
from mlflow.entities.span import SpanType
from mlflow.genai.scorers.ragas.utils import (
    create_mlflow_error_message_from_ragas_param,
    extract_reference_tool_calls_from_expectations,
    map_scorer_inputs_to_ragas_sample,
    map_session_to_ragas_messages,
    map_trace_to_ragas_messages,
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


@pytest.mark.parametrize(
    ("expectations", "expected_result"),
    [
        (None, []),
        ({}, []),
        ({"expected_output": "some output"}, []),
        ({"expected_tool_calls": []}, []),
        (
            {"expected_tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]},
            [ToolCall(name="get_weather", args={"city": "Paris"})],
        ),
        (
            {
                "expected_tool_calls": [
                    {"name": "search", "arguments": {"query": "MLflow"}},
                    {"name": "fetch", "arguments": {"url": "https://mlflow.org"}},
                ]
            },
            [
                ToolCall(name="search", args={"query": "MLflow"}),
                ToolCall(name="fetch", args={"url": "https://mlflow.org"}),
            ],
        ),
        ({"expected_tool_calls": [{"arguments": {"x": 1}}]}, []),
        ({"expected_tool_calls": [{"name": "tool1"}]}, []),
    ],
)
def test_extract_reference_tool_calls_from_expectations(expectations, expected_result):
    result = extract_reference_tool_calls_from_expectations(expectations)

    assert len(result) == len(expected_result)
    for actual, expected in zip(result, expected_result):
        assert actual.name == expected.name
        assert actual.args == expected.args


def test_map_trace_to_ragas_messages_with_tool_call(sample_trace):
    messages = map_trace_to_ragas_messages(sample_trace)

    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert len(messages[1].tool_calls) == 1
    assert messages[1].tool_calls[0].name == "tool"


def test_map_session_to_ragas_messages_multi_turn(sample_trace):
    messages = map_session_to_ragas_messages([sample_trace, sample_trace])

    assert len(messages) == 4  # 2 turns * 2 messages each
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], HumanMessage)
    assert isinstance(messages[3], AIMessage)
