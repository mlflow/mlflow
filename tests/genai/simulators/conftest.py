from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_trace():
    return Mock()


@pytest.fixture
def simulation_mocks(mock_trace):
    """Fixture providing common mocks for conversation simulation tests."""
    with (
        patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke,
        patch("mlflow.trace") as mock_trace_decorator,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace") as mock_update_trace,
        patch(
            "mlflow.tracing.client.TracingClient",
            return_value=Mock(get_trace=lambda _: mock_trace),
        ),
    ):
        mock_get_trace_id.return_value = "trace_123"
        mock_trace_decorator.return_value = lambda fn: fn

        yield {
            "invoke": mock_invoke,
            "trace_decorator": mock_trace_decorator,
            "get_trace_id": mock_get_trace_id,
            "update_trace": mock_update_trace,
            "trace": mock_trace,
        }


@pytest.fixture
def mock_llm_response():
    return "This is a test response from the user agent."


@pytest.fixture
def mock_predict_fn():
    def predict_fn(input=None, messages=None, context=None, **kwargs):
        return {
            "output": [
                {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "This is a mock response from the agent.",
                        }
                    ],
                }
            ]
        }

    return predict_fn


@pytest.fixture
def mock_predict_fn_with_context():
    def predict_fn(input=None, messages=None, **kwargs):
        context_info = f" Context: {kwargs}" if kwargs else ""

        return {
            "output": [
                {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"Mock response.{context_info}",
                        }
                    ],
                }
            ]
        }

    return predict_fn


@pytest.fixture
def simple_test_case():
    return {
        "goal": "Learn about MLflow tracing",
    }


@pytest.fixture
def test_case_with_persona():
    return {
        "goal": "Understand model deployment",
        "persona": "You are an expert who asks direct questions.",
    }


@pytest.fixture
def test_case_with_context():
    return {
        "goal": "Debug an error",
        "context": {"user_id": "U001", "session_id": "S001"},
    }
