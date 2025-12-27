import json
from unittest import mock

import litellm
import pytest
from litellm import RetryPolicy
from litellm.types.utils import ModelResponse
from pydantic import BaseModel, Field

from mlflow.entities.assessment import AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
    InvokeDatabricksModelOutput,
)
from mlflow.genai.judges.adapters.litellm_adapter import _MODEL_RESPONSE_FORMAT_CAPABILITIES
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.judges.utils.invocation_utils import (
    _invoke_databricks_structured_output,
    get_chat_completions_with_structured_output,
    invoke_judge_model,
)
from mlflow.tracing.constant import AssessmentMetadataKey
from mlflow.types.llm import ChatMessage


@pytest.fixture(autouse=True)
def clear_model_capabilities_cache():
    _MODEL_RESPONSE_FORMAT_CAPABILITIES.clear()


@pytest.fixture
def mock_response():
    content = json.dumps({"result": "yes", "rationale": "The response meets all criteria."})
    response = ModelResponse(choices=[{"message": {"content": content}}])
    response._hidden_params = {"response_cost": 0.123}
    return response


@pytest.fixture
def mock_tool_response():
    tool_calls = [{"id": "call_123", "function": {"name": "get_trace_info", "arguments": "{}"}}]
    response = ModelResponse(choices=[{"message": {"tool_calls": tool_calls, "content": None}}])
    response._hidden_params = {"response_cost": 0.05}
    return response


@pytest.fixture
def mock_trace():
    trace_info = TraceInfo(
        trace_id="test-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
    )
    return Trace(info=trace_info, data=None)


@pytest.mark.parametrize("num_retries", [None, 3])
def test_invoke_judge_model_successful_with_litellm(num_retries, mock_response):
    with mock.patch("litellm.completion", return_value=mock_response) as mock_litellm:
        kwargs = {
            "model_uri": "openai:/gpt-4",
            "prompt": "Evaluate this response",
            "assessment_name": "quality_check",
        }
        if num_retries is not None:
            kwargs["num_retries"] = num_retries

        feedback = invoke_judge_model(**kwargs)

    expected_retries = 10 if num_retries is None else num_retries
    expected_retry_policy = RetryPolicy(
        TimeoutErrorRetries=expected_retries,
        RateLimitErrorRetries=expected_retries,
        InternalServerErrorRetries=expected_retries,
        ContentPolicyViolationErrorRetries=expected_retries,
        BadRequestErrorRetries=0,
        AuthenticationErrorRetries=0,
    )

    # Check that the messages were converted to litellm.Message objects
    call_args = mock_litellm.call_args
    assert len(call_args.kwargs["messages"]) == 1
    msg = call_args.kwargs["messages"][0]
    assert isinstance(msg, litellm.Message)
    assert msg.role == "user"
    assert msg.content == "Evaluate this response"

    call_kwargs = mock_litellm.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-4"
    assert call_kwargs["tools"] is None
    assert call_kwargs["tool_choice"] is None
    assert call_kwargs["retry_policy"] == expected_retry_policy
    assert call_kwargs["retry_strategy"] == "exponential_backoff_retry"
    assert call_kwargs["max_retries"] == 0
    assert call_kwargs["drop_params"] is True

    response_format = call_kwargs["response_format"]
    assert issubclass(response_format, BaseModel)
    assert "result" in response_format.model_fields
    assert "rationale" in response_format.model_fields

    assert feedback.name == "quality_check"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The response meets all criteria."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4"
    assert feedback.trace_id is None
    assert feedback.metadata is not None
    assert feedback.metadata[AssessmentMetadataKey.JUDGE_COST] == pytest.approx(0.123)


def test_invoke_judge_model_with_chat_messages(mock_response):
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant"),
        ChatMessage(role="user", content="Evaluate this response"),
    ]

    with mock.patch("litellm.completion", return_value=mock_response) as mock_litellm:
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt=messages,
            assessment_name="quality_check",
        )

    mock_litellm.assert_called_once()
    call_args = mock_litellm.call_args
    messages_arg = call_args.kwargs["messages"]

    assert len(messages_arg) == 2
    assert isinstance(messages_arg[0], litellm.Message)
    assert messages_arg[0].role == "system"
    assert messages_arg[0].content == "You are a helpful assistant"
    assert isinstance(messages_arg[1], litellm.Message)
    assert messages_arg[1].role == "user"
    assert messages_arg[1].content == "Evaluate this response"

    assert feedback.name == "quality_check"
    assert feedback.value == CategoricalRating.YES
    assert feedback.trace_id is None


def test_invoke_judge_model_successful_with_native_provider():
    mock_response = json.dumps({"result": "yes", "rationale": "The response meets all criteria."})

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter._is_litellm_available", return_value=False
        ),
        mock.patch(
            "mlflow.metrics.genai.model_utils.score_model_on_payload", return_value=mock_response
        ) as mock_score_model_on_payload,
    ):
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate this response",
            assessment_name="quality_check",
        )

    mock_score_model_on_payload.assert_called_once_with(
        model_uri="openai:/gpt-4",
        payload="Evaluate this response",
        eval_parameters=None,
        endpoint_type="llm/v1/chat",
    )

    assert feedback.name == "quality_check"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The response meets all criteria."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4"
    assert feedback.trace_id is None
    assert feedback.metadata is None


def test_invoke_judge_model_with_unsupported_provider():
    with pytest.raises(MlflowException, match=r"No suitable adapter found"):
        with mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter._is_litellm_available", return_value=False
        ):
            invoke_judge_model(
                model_uri="unsupported:/model", prompt="Test prompt", assessment_name="test"
            )


def test_invoke_judge_model_with_trace_requires_litellm(mock_trace):
    with pytest.raises(MlflowException, match=r"LiteLLM is required for using traces with judges"):
        with mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter._is_litellm_available", return_value=False
        ):
            invoke_judge_model(
                model_uri="openai:/gpt-4",
                prompt="Test prompt",
                assessment_name="test",
                trace=mock_trace,
            )


def test_invoke_judge_model_invalid_json_response():
    mock_content = "This is not valid JSON"
    mock_response = ModelResponse(choices=[{"message": {"content": mock_content}}])

    with mock.patch("litellm.completion", return_value=mock_response):
        with pytest.raises(MlflowException, match=r"Failed to parse"):
            invoke_judge_model(
                model_uri="openai:/gpt-4", prompt="Test prompt", assessment_name="test"
            )


def test_invoke_judge_model_with_trace_passes_tools(mock_trace, mock_response):
    with (
        mock.patch("litellm.completion", return_value=mock_response) as mock_litellm,
        mock.patch("mlflow.genai.judges.tools.list_judge_tools") as mock_list_tools,
    ):
        # Mock some tools being available
        mock_tool1 = mock.Mock()
        mock_tool1.name = "get_trace_info"
        mock_tool1.get_definition.return_value.to_dict.return_value = {
            "name": "get_trace_info",
            "description": "Get trace info",
        }

        mock_tool2 = mock.Mock()
        mock_tool2.name = "list_spans"
        mock_tool2.get_definition.return_value.to_dict.return_value = {
            "name": "list_spans",
            "description": "List spans",
        }

        mock_list_tools.return_value = [mock_tool1, mock_tool2]

        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate this response",
            assessment_name="quality_check",
            trace=mock_trace,
        )

    # Verify tools were passed to litellm completion
    mock_litellm.assert_called_once()
    call_kwargs = mock_litellm.call_args.kwargs
    assert call_kwargs["tools"] == [
        {"name": "get_trace_info", "description": "Get trace info"},
        {"name": "list_spans", "description": "List spans"},
    ]
    assert call_kwargs["tool_choice"] == "auto"
    assert feedback.trace_id == "test-trace"


def test_invoke_judge_model_tool_calling_loop(mock_trace):
    # First call: model requests tool call
    mock_tool_call_response = ModelResponse(
        choices=[
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {"name": "get_trace_info", "arguments": "{}"},
                        }
                    ],
                    "content": None,
                }
            }
        ],
    )
    mock_tool_call_response._hidden_params = {"response_cost": 0.05}

    # Second call: model provides final answer
    mock_final_response = ModelResponse(
        choices=[
            {
                "message": {
                    "content": json.dumps({"result": "yes", "rationale": "The trace looks good."})
                }
            }
        ],
    )
    mock_final_response._hidden_params = {"response_cost": 0.15}

    with (
        mock.patch(
            "litellm.completion", side_effect=[mock_tool_call_response, mock_final_response]
        ) as mock_litellm,
        mock.patch("mlflow.genai.judges.tools.list_judge_tools") as mock_list_tools,
        mock.patch(
            "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
        ) as mock_invoke_tool,
    ):
        mock_tool = mock.Mock()
        mock_tool.name = "get_trace_info"
        mock_tool.get_definition.return_value.to_dict.return_value = {"name": "get_trace_info"}
        mock_list_tools.return_value = [mock_tool]

        mock_invoke_tool.return_value = {"trace_id": "test-trace", "state": "OK"}

        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate this response",
            assessment_name="quality_check",
            trace=mock_trace,
        )

    # Verify litellm.completion was called twice (tool call + final response)
    assert mock_litellm.call_count == 2

    # Verify tool was invoked
    mock_invoke_tool.assert_called_once()
    tool_call_arg = mock_invoke_tool.call_args.kwargs["tool_call"]
    from mlflow.types.llm import ToolCall

    assert isinstance(tool_call_arg, ToolCall)
    assert tool_call_arg.function.name == "get_trace_info"

    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The trace looks good."
    assert feedback.trace_id == "test-trace"
    assert feedback.metadata is not None
    assert feedback.metadata[AssessmentMetadataKey.JUDGE_COST] == pytest.approx(0.20)


@pytest.mark.parametrize("env_var_value", ["3", None])
def test_invoke_judge_model_completion_iteration_limit(mock_trace, monkeypatch, env_var_value):
    if env_var_value is not None:
        monkeypatch.setenv("MLFLOW_JUDGE_MAX_ITERATIONS", env_var_value)
        expected_limit = int(env_var_value)
    else:
        monkeypatch.delenv("MLFLOW_JUDGE_MAX_ITERATIONS", raising=False)
        expected_limit = 30
    mock_tool_call_response = ModelResponse(
        choices=[
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {"name": "get_trace_info", "arguments": "{}"},
                        }
                    ],
                    "content": None,
                }
            }
        ]
    )

    with (
        mock.patch("litellm.completion", return_value=mock_tool_call_response) as mock_litellm,
        mock.patch("mlflow.genai.judges.tools.list_judge_tools") as mock_list_tools,
        mock.patch(
            "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
        ) as mock_invoke_tool,
    ):
        mock_tool = mock.Mock()
        mock_tool.name = "get_trace_info"
        mock_tool.get_definition.return_value.to_dict.return_value = {"name": "get_trace_info"}
        mock_list_tools.return_value = [mock_tool]
        mock_invoke_tool.return_value = {"trace_id": "test-trace", "state": "OK"}

        with pytest.raises(
            MlflowException, match="Completion iteration limit.*exceeded"
        ) as exc_info:
            invoke_judge_model(
                model_uri="openai:/gpt-4",
                prompt="Evaluate this response",
                assessment_name="quality_check",
                trace=mock_trace,
            )

        error_msg = str(exc_info.value)
        assert f"Completion iteration limit of {expected_limit} exceeded" in error_msg
        assert "model is not powerful enough" in error_msg
        assert mock_litellm.call_count == expected_limit


def test_invoke_judge_model_with_custom_response_format():
    class ResponseFormat(BaseModel):
        result: int
        rationale: str

    # Mock litellm to return structured JSON
    mock_response = ModelResponse(
        choices=[
            {
                "message": {
                    "content": '{"result": 8, "rationale": "High quality"}',
                    "tool_calls": None,
                }
            }
        ]
    )

    with mock.patch("litellm.completion", return_value=mock_response) as mock_completion:
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt=[ChatMessage(role="user", content="Evaluate this")],
            assessment_name="test_judge",
            response_format=ResponseFormat,
        )

    # Verify the result was correctly parsed and converted to dict
    assert feedback.name == "test_judge"
    assert feedback.value == 8
    assert feedback.rationale == "High quality"

    # Verify response_format was passed to litellm.completion
    call_kwargs = mock_completion.call_args.kwargs
    assert "response_format" in call_kwargs
    assert call_kwargs["response_format"] == ResponseFormat


# Tests for Databricks adapter integration with invoke_judge_model
@pytest.mark.parametrize(
    ("model_uri", "expected_model_name"),
    [
        ("databricks:/test-model", "test-model"),
        ("endpoints:/databricks-gpt-oss-120b", "databricks-gpt-oss-120b"),
    ],
)
@pytest.mark.parametrize("with_trace", [False, True])
def test_invoke_judge_model_databricks_success_not_in_databricks(
    model_uri: str, expected_model_name: str, with_trace: bool, mock_trace
) -> None:
    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter._invoke_databricks_serving_endpoint",
            return_value=InvokeDatabricksModelOutput(
                response='{"result": "yes", "rationale": "Good response"}',
                request_id="req-123",
                num_prompt_tokens=10,
                num_completion_tokens=5,
            ),
        ) as mock_invoke_db,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter._record_judge_model_usage_success_databricks_telemetry"
        ) as mock_success_telemetry,
    ):
        kwargs = {
            "model_uri": model_uri,
            "prompt": "Test prompt",
            "assessment_name": "test_assessment",
        }
        if with_trace:
            kwargs["trace"] = mock_trace

        feedback = invoke_judge_model(**kwargs)

        mock_invoke_db.assert_called_once_with(
            model_name=expected_model_name,
            prompt="Test prompt",
            num_retries=10,
            response_format=None,
            inference_params=None,
        )
        mock_success_telemetry.assert_called_once()

    assert feedback.name == "test_assessment"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "Good response"
    assert feedback.trace_id == ("test-trace" if with_trace else None)
    assert feedback.source.source_id == f"databricks:/{expected_model_name}"
    assert feedback.metadata is None


@pytest.mark.parametrize(
    ("model_uri", "expected_model_name"),
    [
        ("databricks:/test-model", "test-model"),
        ("endpoints:/databricks-gpt-oss-120b", "databricks-gpt-oss-120b"),
    ],
)
def test_invoke_judge_model_databricks_success_in_databricks(
    model_uri: str, expected_model_name: str
) -> None:
    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter._invoke_databricks_serving_endpoint",
            return_value=InvokeDatabricksModelOutput(
                response='{"result": "no", "rationale": "Bad response"}',
                request_id="req-456",
                num_prompt_tokens=15,
                num_completion_tokens=8,
            ),
        ) as mock_invoke_db,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter._record_judge_model_usage_success_databricks_telemetry"
        ) as mock_success_telemetry,
    ):
        feedback = invoke_judge_model(
            model_uri=model_uri,
            prompt="Test prompt",
            assessment_name="test_assessment",
        )

        # Verify telemetry was called
        mock_success_telemetry.assert_called_once_with(
            request_id="req-456",
            model_provider="databricks",
            endpoint_name=expected_model_name,
            num_prompt_tokens=15,
            num_completion_tokens=8,
        )
        mock_invoke_db.assert_called_once_with(
            model_name=expected_model_name,
            prompt="Test prompt",
            num_retries=10,
            response_format=None,
            inference_params=None,
        )

    assert feedback.value == CategoricalRating.NO
    assert feedback.rationale == "Bad response"
    assert feedback.trace_id is None
    assert feedback.metadata is None


@pytest.mark.parametrize(
    "model_uri", ["databricks:/test-model", "endpoints:/databricks-gpt-oss-120b"]
)
def test_invoke_judge_model_databricks_source_id(model_uri: str) -> None:
    with mock.patch(
        "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter._invoke_databricks_serving_endpoint",
        return_value=InvokeDatabricksModelOutput(
            response='{"result": "yes", "rationale": "Great response"}',
            request_id="req-789",
            num_prompt_tokens=4,
            num_completion_tokens=2,
        ),
    ) as mock_invoke_db:
        feedback = invoke_judge_model(
            model_uri=model_uri,
            prompt="Test prompt",
            assessment_name="test_assessment",
        )

    expected_model_name = (
        "test-model" if model_uri.startswith("databricks") else "databricks-gpt-oss-120b"
    )
    mock_invoke_db.assert_called_once_with(
        model_name=expected_model_name,
        prompt="Test prompt",
        num_retries=10,
        response_format=None,
        inference_params=None,
    )
    assert feedback.source.source_id == f"databricks:/{expected_model_name}"


@pytest.mark.parametrize(
    ("model_uri", "expected_model_name"),
    [
        ("databricks:/test-model", "test-model"),
        ("endpoints:/databricks-gpt-oss-120b", "databricks-gpt-oss-120b"),
    ],
)
def test_invoke_judge_model_databricks_failure_in_databricks(
    model_uri: str, expected_model_name: str
) -> None:
    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter._invoke_databricks_serving_endpoint",
            side_effect=MlflowException("Model invocation failed"),
        ) as mock_invoke_db,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter._record_judge_model_usage_failure_databricks_telemetry"
        ) as mock_failure_telemetry,
    ):
        with pytest.raises(MlflowException, match="Model invocation failed"):
            invoke_judge_model(
                model_uri=model_uri,
                prompt="Test prompt",
                assessment_name="test_assessment",
            )

        # Verify failure telemetry was called
        mock_failure_telemetry.assert_called_once_with(
            model_provider="databricks",
            endpoint_name=expected_model_name,
            error_code="UNKNOWN",
            error_message=mock.ANY,
        )
        mock_invoke_db.assert_called_once_with(
            model_name=expected_model_name,
            prompt="Test prompt",
            num_retries=10,
            response_format=None,
            inference_params=None,
        )

        # Verify error message contains the traceback
        call_args = mock_failure_telemetry.call_args[1]
        assert "Model invocation failed" in call_args["error_message"]


@pytest.mark.parametrize(
    ("model_uri", "expected_model_name"),
    [
        ("databricks:/test-model", "test-model"),
        ("endpoints:/databricks-gpt-oss-120b", "databricks-gpt-oss-120b"),
    ],
)
def test_invoke_judge_model_databricks_telemetry_error_handling(
    model_uri: str, expected_model_name: str
) -> None:
    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter._invoke_databricks_serving_endpoint",
            return_value=InvokeDatabricksModelOutput(
                response='{"result": "yes", "rationale": "Good"}',
                request_id="req-789",
                num_prompt_tokens=5,
                num_completion_tokens=3,
            ),
        ) as mock_invoke_db,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter._record_judge_model_usage_success_databricks_telemetry",
            side_effect=Exception("Telemetry failed"),
        ) as mock_success_telemetry,
    ):
        # Should still return feedback despite telemetry failure
        feedback = invoke_judge_model(
            model_uri=model_uri,
            prompt="Test prompt",
            assessment_name="test_assessment",
        )

        mock_success_telemetry.assert_called_once_with(
            request_id="req-789",
            model_provider="databricks",
            endpoint_name=expected_model_name,
            num_prompt_tokens=5,
            num_completion_tokens=3,
        )
        mock_invoke_db.assert_called_once_with(
            model_name=expected_model_name,
            prompt="Test prompt",
            num_retries=10,
            response_format=None,
            inference_params=None,
        )

    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "Good"
    assert feedback.trace_id is None
    assert feedback.metadata is None


# Tests for LiteLLM adapter integration with invoke_judge_model
def test_litellm_nonfatal_error_messages_suppressed():
    suppression_state_during_call = {}

    def mock_completion(**kwargs):
        # Capture the state of litellm flags during the call
        suppression_state_during_call["set_verbose"] = litellm.set_verbose
        suppression_state_during_call["suppress_debug_info"] = litellm.suppress_debug_info

        return ModelResponse(
            choices=[{"message": {"content": '{"result": "pass", "rationale": "Test completed"}'}}]
        )

    with mock.patch("litellm.completion", side_effect=mock_completion):
        result = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Test prompt for suppression",
            assessment_name="suppression_test",
        )

        # Verify suppression was active during the litellm.completion call
        assert suppression_state_during_call["set_verbose"] is False
        assert suppression_state_during_call["suppress_debug_info"] is True

        # Verify the call succeeded
        assert result.value == "pass"


def test_unsupported_response_format_handling_supports_multiple_threads():
    model_key = "openai/gpt-4-race-bug"
    _MODEL_RESPONSE_FORMAT_CAPABILITIES.clear()

    bad_request_error = litellm.BadRequestError(
        message="response_format not supported", model=model_key, llm_provider="openai"
    )

    call_count = 0
    capabilities_cache_call_count = 0

    def mock_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        if "response_format" in kwargs:
            raise bad_request_error
        else:
            return ModelResponse(
                choices=[{"message": {"content": '{"result": "yes", "rationale": "Success"}'}}]
            )

    class MockCapabilitiesCache(dict):
        def get(self, key, default=None):
            nonlocal capabilities_cache_call_count
            capabilities_cache_call_count += 1

            if capabilities_cache_call_count == 1:
                return True
            elif capabilities_cache_call_count == 2:
                return False
            else:
                return False

    with (
        mock.patch("litellm.completion", side_effect=mock_completion),
        mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter._MODEL_RESPONSE_FORMAT_CAPABILITIES",
            MockCapabilitiesCache(),
        ),
    ):
        result = invoke_judge_model(
            model_uri=f"openai:/{model_key}",
            prompt="Test prompt",
            assessment_name="race_bug_test",
        )

        assert call_count == 2, "Should make 2 calls: initial (fails) + retry (succeeds)"
        assert capabilities_cache_call_count == 1
        assert result.value == "yes"


@pytest.mark.parametrize(
    ("error_type", "error_class"),
    [
        ("BadRequestError", litellm.BadRequestError),
        ("UnsupportedParamsError", litellm.UnsupportedParamsError),
    ],
)
def test_invoke_judge_model_retries_without_response_format_on_bad_request(error_type, error_class):
    mock_response = ModelResponse(
        choices=[{"message": {"content": '{"result": "yes", "rationale": "Test rationale"}'}}]
    )
    error = error_class(
        message="response_format not supported", model="openai/gpt-4", llm_provider="openai"
    )

    with mock.patch("litellm.completion", side_effect=[error, mock_response]) as mock_litellm:
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Test prompt",
            assessment_name="test",
        )

        # Should have been called twice - once with response_format, once without
        assert mock_litellm.call_count == 2

        # First call should include response_format
        first_call_kwargs = mock_litellm.call_args_list[0].kwargs
        assert "response_format" in first_call_kwargs

        # Second call should not include response_format
        second_call_kwargs = mock_litellm.call_args_list[1].kwargs
        assert "response_format" not in second_call_kwargs

        # Should still return valid feedback
        assert feedback.name == "test"
        assert feedback.value == "yes"
        assert feedback.trace_id is None


def test_invoke_judge_model_stops_trying_response_format_after_failure():
    bad_request_error = litellm.BadRequestError(
        message="response_format not supported", model="openai/gpt-4", llm_provider="openai"
    )

    # Mock responses for: initial fail, retry success, tool call 1, tool call 2
    tool_call_response = ModelResponse(
        choices=[
            {
                "message": {
                    "tool_calls": [
                        {"id": "call_123", "function": {"name": "test_tool", "arguments": "{}"}}
                    ],
                    "content": None,
                }
            }
        ]
    )

    success_response = ModelResponse(
        choices=[{"message": {"content": '{"result": "yes", "rationale": "Test rationale"}'}}]
    )

    with (
        mock.patch(
            "litellm.completion",
            side_effect=[
                bad_request_error,
                tool_call_response,
                success_response,
            ],
        ) as mock_litellm,
        mock.patch("mlflow.genai.judges.tools.list_judge_tools") as mock_list_tools,
        mock.patch("mlflow.genai.judges.tools.registry._judge_tool_registry.invoke") as mock_invoke,
    ):
        mock_tool = mock.Mock()
        mock_tool.get_definition.return_value.to_dict.return_value = {"name": "test_tool"}
        mock_list_tools.return_value = [mock_tool]
        mock_invoke.return_value = {"result": "tool executed"}

        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Test prompt",
            assessment_name="test",
            trace=mock.Mock(),
        )

        # Should have been called 3 times total
        assert mock_litellm.call_count == 3

        # First call should include response_format and fail
        first_call_kwargs = mock_litellm.call_args_list[0].kwargs
        assert "response_format" in first_call_kwargs

        # Second call should not include response_format and succeed with tool call
        second_call_kwargs = mock_litellm.call_args_list[1].kwargs
        assert "response_format" not in second_call_kwargs

        # Third call (after tool execution) should also not include response_format
        third_call_kwargs = mock_litellm.call_args_list[2].kwargs
        assert "response_format" not in third_call_kwargs

        assert feedback.name == "test"


def test_invoke_judge_model_caches_capabilities_globally():
    bad_request_error = litellm.BadRequestError(
        message="response_format not supported", model="openai/gpt-4", llm_provider="openai"
    )

    success_response = ModelResponse(
        choices=[{"message": {"content": '{"result": "yes", "rationale": "Test rationale"}'}}]
    )

    # First call - should try response_format and cache the failure
    with mock.patch(
        "litellm.completion", side_effect=[bad_request_error, success_response]
    ) as mock_litellm:
        feedback1 = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Test prompt 1",
            assessment_name="test1",
        )

        # Should have been called twice (initial fail + retry)
        assert mock_litellm.call_count == 2
        assert feedback1.name == "test1"
        assert feedback1.trace_id is None

        # Verify capability was cached
        assert _MODEL_RESPONSE_FORMAT_CAPABILITIES.get("openai/gpt-4") is False

    # Second call - should directly use cached capability (no response_format)
    with mock.patch("litellm.completion", return_value=success_response) as mock_litellm_2:
        feedback2 = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Test prompt 2",
            assessment_name="test2",
        )

        # Should only be called once (no retry needed)
        assert mock_litellm_2.call_count == 1

        # Should not include response_format
        call_kwargs = mock_litellm_2.call_args.kwargs
        assert "response_format" not in call_kwargs

        assert feedback2.name == "test2"
        assert feedback2.trace_id is None


def test_get_chat_completions_with_structured_output():
    class FieldExtraction(BaseModel):
        inputs: str = Field(description="The user's original request")
        outputs: str = Field(description="The system's final response")

    mock_response = ModelResponse(
        choices=[
            {
                "message": {
                    "content": '{"inputs": "What is MLflow?", "outputs": "MLflow is a platform"}',
                    "tool_calls": None,
                }
            }
        ]
    )
    mock_response._hidden_params = {"response_cost": 0.05}

    with mock.patch("litellm.completion", return_value=mock_response) as mock_completion:
        result = get_chat_completions_with_structured_output(
            model_uri="openai:/gpt-4",
            messages=[
                ChatMessage(role="system", content="Extract fields"),
                ChatMessage(role="user", content="Find inputs and outputs"),
            ],
            output_schema=FieldExtraction,
        )

    assert isinstance(result, FieldExtraction)
    assert result.inputs == "What is MLflow?"
    assert result.outputs == "MLflow is a platform"

    call_kwargs = mock_completion.call_args.kwargs
    assert "response_format" in call_kwargs
    assert call_kwargs["response_format"] == FieldExtraction


def test_get_chat_completions_with_structured_output_with_trace(mock_trace):
    class FieldExtraction(BaseModel):
        inputs: str = Field(description="The user's original request")
        outputs: str = Field(description="The system's final response")

    tool_call_response = ModelResponse(
        choices=[
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {
                                "name": "get_trace_info",
                                "arguments": "{}",
                            },
                        }
                    ],
                    "content": None,
                }
            }
        ]
    )
    tool_call_response._hidden_params = {"response_cost": 0.05}

    final_response = ModelResponse(
        choices=[
            {
                "message": {
                    "content": '{"inputs": "question from trace", "outputs": "answer from trace"}',
                    "tool_calls": None,
                }
            }
        ]
    )
    final_response._hidden_params = {"response_cost": 0.10}

    with (
        mock.patch(
            "litellm.completion", side_effect=[tool_call_response, final_response]
        ) as mock_completion,
        mock.patch("mlflow.genai.judges.tools.list_judge_tools") as mock_list_tools,
        mock.patch("mlflow.genai.judges.tools.registry._judge_tool_registry.invoke") as mock_invoke,
    ):
        mock_tool = mock.Mock()
        mock_tool.get_definition.return_value.to_dict.return_value = {"name": "get_trace_info"}
        mock_list_tools.return_value = [mock_tool]
        mock_invoke.return_value = {"trace_id": "test-trace", "state": "OK"}

        result = get_chat_completions_with_structured_output(
            model_uri="openai:/gpt-4",
            messages=[
                ChatMessage(role="system", content="Extract fields"),
                ChatMessage(role="user", content="Find inputs and outputs"),
            ],
            output_schema=FieldExtraction,
            trace=mock_trace,
        )

    assert isinstance(result, FieldExtraction)
    assert result.inputs == "question from trace"
    assert result.outputs == "answer from trace"

    assert mock_completion.call_count == 2
    mock_invoke.assert_called_once()


@pytest.mark.parametrize(
    "inference_params",
    [
        None,
        {"temperature": 0},
        {"temperature": 0.5, "max_tokens": 100},
        {"temperature": 0.5, "top_p": 0.9, "max_tokens": 500, "presence_penalty": 0.1},
    ],
)
def test_invoke_judge_model_with_inference_params(mock_response, inference_params):
    with mock.patch("litellm.completion", return_value=mock_response) as mock_litellm:
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate this",
            assessment_name="test",
            inference_params=inference_params,
        )

    assert feedback.name == "test"
    call_kwargs = mock_litellm.call_args.kwargs

    if inference_params:
        for key, value in inference_params.items():
            assert call_kwargs[key] == value
    else:
        assert "temperature" not in call_kwargs


def test_get_chat_completions_with_inference_params():
    class OutputSchema(BaseModel):
        result: str

    mock_response_obj = ModelResponse(choices=[{"message": {"content": '{"result": "pass"}'}}])

    inference_params = {"temperature": 0.1}

    with mock.patch("litellm.completion", return_value=mock_response_obj) as mock_litellm:
        result = get_chat_completions_with_structured_output(
            model_uri="openai:/gpt-4",
            messages=[ChatMessage(role="user", content="Test")],
            output_schema=OutputSchema,
            inference_params=inference_params,
        )

    assert result.result == "pass"
    call_kwargs = mock_litellm.call_args.kwargs
    assert call_kwargs["temperature"] == 0.1


def test_inference_params_in_tool_calling_loop(mock_trace):
    inference_params = {"temperature": 0.2}

    tool_call_response = ModelResponse(
        choices=[
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "get_trace_info", "arguments": "{}"},
                        }
                    ],
                    "content": None,
                }
            }
        ]
    )

    final_response = ModelResponse(
        choices=[{"message": {"content": '{"result": "yes", "rationale": "OK"}'}}]
    )

    with (
        mock.patch(
            "litellm.completion", side_effect=[tool_call_response, final_response]
        ) as mock_litellm,
        mock.patch("mlflow.genai.judges.tools.list_judge_tools") as mock_list_tools,
        mock.patch("mlflow.genai.judges.tools.registry._judge_tool_registry.invoke") as mock_invoke,
    ):
        mock_tool = mock.Mock()
        mock_tool.get_definition.return_value.to_dict.return_value = {"name": "get_trace_info"}
        mock_list_tools.return_value = [mock_tool]
        mock_invoke.return_value = {"result": "info"}

        invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate",
            assessment_name="test",
            trace=mock_trace,
            inference_params=inference_params,
        )

    # Both calls should have temperature set
    assert mock_litellm.call_count == 2
    for call in mock_litellm.call_args_list:
        assert call.kwargs["temperature"] == 0.2


# Tests for _invoke_databricks_structured_output


def test_structured_output_schema_injection_with_existing_system_message():
    class TestSchema(BaseModel):
        outputs: str = Field(description="The outputs")

    captured_messages = []

    def mock_loop(messages, trace, on_final_answer):
        captured_messages.extend(messages)
        return on_final_answer('{"outputs": "test result"}')

    with mock.patch(
        "mlflow.genai.judges.utils.invocation_utils._run_databricks_agentic_loop",
        side_effect=mock_loop,
    ):
        result = _invoke_databricks_structured_output(
            messages=[
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="Extract the outputs"),
            ],
            output_schema=TestSchema,
            trace=None,
        )

    # Verify schema was appended to existing system message
    assert len(captured_messages) == 2
    assert captured_messages[0].role == "system"
    assert "You are a helpful assistant." in captured_messages[0].content
    assert "You must return your response as JSON matching this schema:" in (
        captured_messages[0].content
    )
    assert '"outputs"' in captured_messages[0].content

    assert isinstance(result, TestSchema)
    assert result.outputs == "test result"


def test_structured_output_schema_injection_without_system_message():
    class TestSchema(BaseModel):
        inputs: str = Field(description="The inputs")
        outputs: str = Field(description="The outputs")

    captured_messages = []

    def mock_loop(messages, trace, on_final_answer):
        captured_messages.extend(messages)
        return on_final_answer('{"inputs": "hello", "outputs": "world"}')

    with mock.patch(
        "mlflow.genai.judges.utils.invocation_utils._run_databricks_agentic_loop",
        side_effect=mock_loop,
    ):
        result = _invoke_databricks_structured_output(
            messages=[
                ChatMessage(role="user", content="Extract fields from the trace"),
            ],
            output_schema=TestSchema,
            trace=None,
        )

    # Verify schema was inserted as new system message at the beginning
    assert len(captured_messages) == 2
    assert captured_messages[0].role == "system"
    assert "You must return your response as JSON matching this schema:" in (
        captured_messages[0].content
    )
    assert captured_messages[1].role == "user"
    assert captured_messages[1].content == "Extract fields from the trace"

    assert isinstance(result, TestSchema)
    assert result.inputs == "hello"
    assert result.outputs == "world"
