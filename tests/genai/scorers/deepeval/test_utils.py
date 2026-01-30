from unittest.mock import Mock

import pytest

from mlflow.entities.span import Span, SpanAttributeKey, SpanType
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.deepeval.models import create_deepeval_model
from mlflow.genai.scorers.deepeval.utils import (
    _convert_to_deepeval_tool_calls,
    _extract_tool_calls_from_trace,
    map_scorer_inputs_to_deepeval_test_case,
)


def test_create_deepeval_model_databricks():
    model = create_deepeval_model("databricks")
    assert model.__class__.__name__ == "DatabricksDeepEvalLLM"
    assert model.get_model_name() == "databricks"


def test_create_deepeval_model_databricks_serving_endpoint():
    model = create_deepeval_model("databricks:/my-endpoint")
    assert model.__class__.__name__ == "LiteLLMModel"
    assert model.name == "databricks/my-endpoint"


def test_create_deepeval_model_openai():
    model = create_deepeval_model("openai:/gpt-4")
    assert model.__class__.__name__ == "LiteLLMModel"
    assert model.name == "openai/gpt-4"


def test_create_deepeval_model_with_provider_no_slash():
    model = create_deepeval_model("openai:gpt-4")
    assert model.__class__.__name__ == "LiteLLMModel"
    assert model.name == "openai/gpt-4"


def test_create_deepeval_model_rejects_model_name_only():
    with pytest.raises(MlflowException, match="Malformed model uri"):
        create_deepeval_model("gpt-4")


def test_convert_to_deepeval_tool_calls():
    tool_call_dicts = [
        {
            "name": "search",
            "description": "Search the web",
            "reasoning": "Need to find information",
            "output": "Search results",
            "input_parameters": {"query": "MLflow"},
        },
        {
            "name": "calculator",
            "output": "42",
            "input_parameters": {"expression": "6*7"},
        },
    ]

    tool_calls = _convert_to_deepeval_tool_calls(tool_call_dicts)

    assert len(tool_calls) == 2
    assert tool_calls[0].name == "search"
    assert tool_calls[0].description == "Search the web"
    assert tool_calls[0].output == "Search results"
    assert tool_calls[0].input_parameters == {"query": "MLflow"}
    assert tool_calls[1].name == "calculator"


def test_extract_tool_calls_from_trace():
    span1 = Mock(spec=Span)
    span1.name = "search_tool"
    span1.attributes = {
        SpanAttributeKey.INPUTS: {"query": "test"},
        SpanAttributeKey.OUTPUTS: {"results": ["result1", "result2"]},
    }

    trace = Mock()
    trace.search_spans.return_value = [span1]

    tool_calls = _extract_tool_calls_from_trace(trace)

    assert len(tool_calls) == 1
    assert tool_calls[0].name == "search_tool"
    assert tool_calls[0].input_parameters == {"query": "test"}
    assert tool_calls[0].output == {"results": ["result1", "result2"]}
    trace.search_spans.assert_called_once_with(span_type=SpanType.TOOL)


def test_extract_tool_calls_from_trace_returns_none_when_no_tools():
    trace = Mock()
    trace.search_spans.return_value = []
    assert _extract_tool_calls_from_trace(trace) is None


def test_map_mlflow_to_test_case_basic():
    test_case = map_scorer_inputs_to_deepeval_test_case(
        metric_name="AnswerRelevancy",
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
    )

    assert test_case.input == "What is MLflow?"
    assert test_case.actual_output == "MLflow is a platform"
    assert test_case.expected_output is None
    assert test_case.retrieval_context == []


def test_map_mlflow_to_test_case_with_expectations():
    expectations = {
        "expected_output": "MLflow is an open source platform",
        "other_key": "other_value",
    }

    test_case = map_scorer_inputs_to_deepeval_test_case(
        metric_name="AnswerRelevancy",
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
        expectations=expectations,
    )

    assert test_case.expected_output == "MLflow is an open source platform"
    assert "expected_output: MLflow is an open source platform" in test_case.context


def test_map_mlflow_to_test_case_with_expected_tool_calls():
    expectations = {
        "expected_tool_calls": [
            {"name": "search", "input_parameters": {"query": "test"}},
        ]
    }

    test_case = map_scorer_inputs_to_deepeval_test_case(
        metric_name="ToolCorrectness",
        inputs="Search for test",
        outputs="Found results",
        expectations=expectations,
    )

    assert test_case.expected_tools is not None
    assert len(test_case.expected_tools) == 1
    assert test_case.expected_tools[0].name == "search"
