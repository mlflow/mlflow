from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_trace():
    trace = Mock()
    trace.info.trace_metadata = {}
    trace.info.tags = {}
    return trace


@pytest.fixture
def simulation_mocks(mock_trace):
    """Fixture providing common mocks for conversation simulation tests."""
    # Use a counter to return unique trace IDs for each call
    trace_id_counter = {"count": 0}

    def unique_trace_id(*args, **kwargs):
        trace_id_counter["count"] += 1
        return f"trace_{trace_id_counter['count']}"

    # Track metadata/tags passed to tracing.context and apply them to mock traces
    captured_context_calls = []

    @contextmanager
    def mock_context(metadata=None, tags=None, enabled=None, session_id=None, user=None):
        captured_context_calls.append({
            "metadata": metadata,
            "tags": tags,
            "session_id": session_id,
            "user": user,
        })
        # Apply metadata/tags to the mock trace so tests can assert on them
        if metadata:
            mock_trace.info.trace_metadata.update(metadata)
        if session_id is not None:
            mock_trace.info.trace_metadata["mlflow.trace.session"] = session_id
        if tags:
            mock_trace.info.tags.update(tags)
        yield

    with (
        patch("mlflow.genai.simulators.simulator.invoke_model_without_tracing") as mock_invoke,
        patch("mlflow.get_last_active_trace_id", side_effect=unique_trace_id) as mock_get_trace_id,
        patch("mlflow.tracing.context", side_effect=mock_context),
        patch(
            "mlflow.tracing.client.TracingClient",
            return_value=Mock(get_trace=lambda _: mock_trace),
        ),
    ):
        yield {
            "invoke": mock_invoke,
            "get_trace_id": mock_get_trace_id,
            "context_calls": captured_context_calls,
            "trace": mock_trace,
        }


@pytest.fixture
def mock_llm_response():
    return "This is a test response from the user agent."


@pytest.fixture
def mock_predict_fn():
    def predict_fn(input=None, **kwargs):
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
    def predict_fn(input=None, **kwargs):
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


@pytest.fixture
def test_case_with_simulation_guidelines():
    return {
        "goal": "Learn about ML pipelines",
        "simulation_guidelines": "Ask clarifying questions before proceeding",
    }
