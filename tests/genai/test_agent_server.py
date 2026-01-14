import contextvars
from typing import AsyncGenerator
from unittest.mock import Mock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from mlflow.genai.agent_server import (
    AgentServer,
    get_invoke_function,
    get_request_headers,
    get_stream_function,
    invoke,
    set_request_headers,
    stream,
)
from mlflow.genai.agent_server.validator import ResponsesAgentValidator
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test to ensure test isolation."""
    import mlflow.genai.agent_server.server

    mlflow.genai.agent_server.server._invoke_function = None
    mlflow.genai.agent_server.server._stream_function = None


async def responses_invoke(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    return ResponsesAgentResponse(
        output=[
            {
                "type": "message",
                "id": "msg-123",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello from ResponsesAgent!"}],
            }
        ]
    )


async def responses_stream(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    yield ResponsesAgentStreamEvent(
        type="response.output_item.done",
        item={
            "type": "message",
            "id": "msg-123",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello from ResponsesAgent stream!"}],
        },
    )


async def arbitrary_invoke(request: dict) -> dict:
    return {
        "response": "Hello from ArbitraryDictAgent!",
        "arbitrary_field": "custom_value",
        "nested": {"data": "some nested content"},
    }


async def arbitrary_stream(request: dict) -> AsyncGenerator[dict, None]:
    yield {"type": "custom_event", "data": "First chunk"}
    yield {"type": "custom_event", "data": "Second chunk", "final": True}


def test_invoke_decorator_single_registration():
    @invoke()
    def my_invoke_function(request):
        return {"result": "success"}

    registered_function = get_invoke_function()
    assert registered_function is not None
    result = registered_function({"test": "request"})
    assert result == {"result": "success"}


def test_stream_decorator_single_registration():
    @stream()
    async def my_stream_function(request):
        yield {"delta": {"content": "hello"}}

    registered_function = get_stream_function()
    assert registered_function is not None


def test_multiple_invoke_registrations_raises_error():
    @invoke()
    def first_function(request):
        return {"result": "first"}

    with pytest.raises(ValueError, match="invoke decorator can only be used once"):

        @invoke()
        def second_function(request):
            return {"result": "second"}


def test_multiple_stream_registrations_raises_error():
    @stream()
    def first_stream(request):
        yield {"delta": {"content": "first"}}

    with pytest.raises(ValueError, match="stream decorator can only be used once"):

        @stream()
        def second_stream(request):
            yield {"delta": {"content": "second"}}


def test_get_invoke_function_returns_registered():
    def my_function(request):
        return {"test": "data"}

    @invoke()
    def registered_function(request):
        return my_function(request)

    result = get_invoke_function()
    assert result is not None
    test_result = result({"input": "test"})
    assert test_result == {"test": "data"}


def test_decorator_preserves_function_metadata():
    @invoke()
    def function_with_metadata(request):
        """This is a test function with documentation."""
        return {"result": "success"}

    # Get the wrapper function
    wrapper = get_invoke_function()

    # Verify that functools.wraps preserved the metadata
    assert wrapper.__name__ == "function_with_metadata"
    assert wrapper.__doc__ == "This is a test function with documentation."

    @stream()
    async def stream_with_metadata(request):
        """This is a test stream function."""
        yield {"delta": {"content": "hello"}}

    stream_wrapper = get_stream_function()
    assert stream_wrapper.__name__ == "stream_with_metadata"
    assert stream_wrapper.__doc__ == "This is a test stream function."


def test_validator_request_dict_responses_agent():
    validator_responses = ResponsesAgentValidator()
    request_data = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ]
    }
    result = validator_responses.validate_and_convert_request(request_data)
    assert isinstance(result, ResponsesAgentRequest)


def test_validator_invalid_request_dict_raises_error():
    validator_responses = ResponsesAgentValidator()
    invalid_data = {"invalid": "structure"}

    with pytest.raises(ValueError, match="Invalid data for ResponsesAgentRequest"):
        validator_responses.validate_and_convert_request(invalid_data)


def test_validator_none_type_returns_data_unchanged():
    validator_responses = ResponsesAgentValidator()
    request_data = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ]
    }
    result = validator_responses.validate_and_convert_request(request_data)
    assert isinstance(result, ResponsesAgentRequest)


def test_validator_response_dict_format():
    validator_responses = ResponsesAgentValidator()
    response_dict = {
        "output": [
            {
                "type": "message",
                "id": "123",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello"}],
            }
        ]
    }

    result = validator_responses.validate_and_convert_result(response_dict)
    assert isinstance(result, dict)
    assert result == response_dict


def test_validator_response_pydantic_format():
    validator_responses = ResponsesAgentValidator()
    response_pydantic = ResponsesAgentResponse(
        output=[
            {
                "type": "message",
                "id": "123",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello"}],
            }
        ]
    )

    result = validator_responses.validate_and_convert_result(response_pydantic)
    assert isinstance(result, dict)
    assert "output" in result


def test_validator_response_dataclass_format():
    validator_responses = ResponsesAgentValidator()
    valid_response = ResponsesAgentResponse(
        output=[
            {
                "type": "message",
                "id": "123",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello"}],
            }
        ]
    )
    result = validator_responses.validate_and_convert_result(valid_response)
    assert isinstance(result, dict)
    assert "output" in result


def test_validator_stream_response_formats():
    validator_responses = ResponsesAgentValidator()
    # Test streaming response validation for different agent types
    stream_event = ResponsesAgentStreamEvent(
        type="response.output_item.done",
        item={
            "type": "message",
            "id": "123",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello"}],
        },
    )

    result = validator_responses.validate_and_convert_result(stream_event, stream=True)
    assert isinstance(result, dict)


def test_arbitrary_dict_agent_fails_responses_validation():
    validator_responses = ResponsesAgentValidator()
    arbitrary_response = {
        "response": "Hello from ArbitraryDictAgent!",
        "arbitrary_field": "custom_value",
        "nested": {"data": "some nested content"},
    }

    # This should fail validation because it doesn't match ResponsesAgentResponse schema
    with pytest.raises(ValueError, match="Invalid data for ResponsesAgentResponse"):
        validator_responses.validate_and_convert_result(arbitrary_response)


def test_responses_agent_passes_validation():
    validator_responses = ResponsesAgentValidator()
    valid_response = {
        "output": [
            {
                "type": "message",
                "id": "123",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello"}],
            }
        ]
    }

    # This should pass validation
    result = validator_responses.validate_and_convert_result(valid_response)
    assert isinstance(result, dict)
    assert "output" in result


def test_agent_server_initialization():
    server = AgentServer()
    assert server.agent_type is None
    assert server.validator is not None
    assert server.app.title == "Agent Server"


def test_agent_server_with_agent_type():
    server = AgentServer("ResponsesAgent")
    assert server.agent_type == "ResponsesAgent"


def test_agent_server_routes_registration():
    server = AgentServer()
    routes = [route.path for route in server.app.routes]
    assert "/invocations" in routes
    assert "/health" in routes


def test_invocations_endpoint_malformed_json():
    server = AgentServer()
    client = TestClient(server.app)

    response = client.post("/invocations", data="malformed json")
    assert response.status_code == 400
    response_json = response.json()
    assert "Invalid JSON in request body" in response_json["detail"]


def test_invocations_endpoint_missing_invoke_function():
    server = AgentServer()
    client = TestClient(server.app)

    response = client.post("/invocations", json={"test": "data"})
    assert response.status_code == 500
    response_json = response.json()
    assert "No invoke function registered" in response_json["detail"]


def test_invocations_endpoint_validation_error():
    server = AgentServer("ResponsesAgent")
    client = TestClient(server.app)

    # Send invalid request data for responses agent
    invalid_data = {"invalid": "structure"}
    response = client.post("/invocations", json=invalid_data)
    assert response.status_code == 400
    response_json = response.json()
    assert "Invalid parameters for ResponsesAgent" in response_json["detail"]


def test_invocations_endpoint_success_invoke():
    with patch("mlflow.start_span") as mock_span:
        # Mock the span context manager
        mock_span_instance = Mock()
        mock_span_instance.__enter__ = Mock(return_value=mock_span_instance)
        mock_span_instance.__exit__ = Mock(return_value=None)
        mock_span_instance.trace_id = "test-trace-id"
        mock_span.return_value = mock_span_instance

        @invoke()
        def test_invoke(request):
            return {
                "output": [
                    {
                        "type": "message",
                        "id": "123",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    }
                ]
            }

        server = AgentServer("ResponsesAgent")
        client = TestClient(server.app)

        request_data = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ]
        }

        response = client.post("/invocations", json=request_data)
        assert response.status_code == 200
        response_json = response.json()
        assert "output" in response_json


def test_invocations_endpoint_success_stream():
    with patch("mlflow.start_span") as mock_span:
        # Mock the span context manager
        mock_span_instance = Mock()
        mock_span_instance.__enter__ = Mock(return_value=mock_span_instance)
        mock_span_instance.__exit__ = Mock(return_value=None)
        mock_span_instance.trace_id = "test-trace-id"
        mock_span.return_value = mock_span_instance

        @stream()
        def test_stream(request):
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item={
                    "type": "message",
                    "id": "123",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}],
                },
            )

        server = AgentServer("ResponsesAgent")
        client = TestClient(server.app)

        request_data = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ],
            "stream": True,
        }

        response = client.post("/invocations", json=request_data)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


def test_health_endpoint_returns_status():
    server = AgentServer()
    client = TestClient(server.app)

    response = client.get("/health")
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "healthy"


def test_request_headers_isolation():
    # Test that headers are isolated between contexts
    set_request_headers({"test": "value1"})
    assert get_request_headers()["test"] == "value1"

    # In a different context, headers should be independent
    ctx = contextvars.copy_context()

    def test_different_context():
        set_request_headers({"test": "value2"})
        return get_request_headers()["test"]

    result = ctx.run(test_different_context)
    assert result == "value2"
    # Original context should be unchanged
    assert get_request_headers()["test"] == "value1"


def test_tracing_span_creation():
    with patch("mlflow.start_span") as mock_span:
        mock_span_instance = Mock()
        mock_span_instance.__enter__ = Mock(return_value=mock_span_instance)
        mock_span_instance.__exit__ = Mock(return_value=None)
        mock_span.return_value = mock_span_instance

        @invoke()
        def test_function(request):
            return {"result": "success"}

        server = AgentServer()
        client = TestClient(server.app)

        client.post("/invocations", json={"test": "data"})
        # Verify span was created with correct name
        mock_span.assert_called_once_with(name="test_function")


def test_tracing_attributes_setting():
    with patch("mlflow.start_span") as mock_span:
        mock_span_instance = Mock()
        mock_span_instance.__enter__ = Mock(return_value=mock_span_instance)
        mock_span_instance.__exit__ = Mock(return_value=None)
        mock_span.return_value = mock_span_instance

        @invoke()
        def test_function(request):
            return {
                "output": [
                    {
                        "type": "message",
                        "id": "123",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    }
                ]
            }

        server = AgentServer("ResponsesAgent")
        client = TestClient(server.app)

        request_data = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ]
        }

        client.post("/invocations", json=request_data)

        # Verify span was created (this is the main functionality we can reliably test)
        mock_span.assert_called_once_with(name="test_function")
        # Verify the span context manager was used
        mock_span_instance.__enter__.assert_called_once()
        mock_span_instance.__exit__.assert_called_once()


def test_chat_proxy_disabled_by_default():
    server = AgentServer()
    assert not hasattr(server, "proxy_client")


def test_chat_proxy_enabled():
    server = AgentServer(enable_chat_proxy=True)
    assert hasattr(server, "proxy_client")
    assert server.proxy_client is not None
    assert server.chat_proxy_timeout == 300.0


def test_chat_proxy_custom_timeout(monkeypatch):
    monkeypatch.setenv("CHAT_PROXY_TIMEOUT_SECONDS", "60.0")
    server = AgentServer(enable_chat_proxy=True)
    assert server.proxy_client is not None
    assert server.chat_proxy_timeout == 60.0


@pytest.mark.asyncio
async def test_chat_proxy_forwards_allowed_paths():
    @invoke()
    def test_invoke(request):
        return {
            "output": [
                {
                    "type": "message",
                    "id": "123",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}],
                }
            ]
        }

    server = AgentServer("ResponsesAgent", enable_chat_proxy=True)
    client = TestClient(server.app)

    mock_response = Mock()
    mock_response.content = b'{"chat": "response"}'
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}

    with patch.object(server.proxy_client, "request", return_value=mock_response) as mock_request:
        response = client.get("/assets/index.js")
        assert response.status_code == 200
        assert response.content == b'{"chat": "response"}'
        mock_request.assert_called_once()


@pytest.mark.asyncio
async def test_chat_proxy_does_not_forward_matched_routes():
    @invoke()
    def test_invoke(request):
        return {
            "output": [
                {
                    "type": "message",
                    "id": "123",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}],
                }
            ]
        }

    server = AgentServer("ResponsesAgent", enable_chat_proxy=True)
    client = TestClient(server.app)

    with patch.object(server.proxy_client, "request") as mock_request:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
        mock_request.assert_not_called()


@pytest.mark.asyncio
async def test_chat_proxy_handles_connect_error():
    server = AgentServer(enable_chat_proxy=True)
    client = TestClient(server.app)

    with patch.object(
        server.proxy_client, "request", side_effect=httpx.ConnectError("Connection failed")
    ):
        response = client.get("/")
        assert response.status_code == 503
        assert response.text == "Service unavailable"


@pytest.mark.asyncio
async def test_chat_proxy_handles_general_error():
    server = AgentServer(enable_chat_proxy=True)
    client = TestClient(server.app)

    with patch.object(server.proxy_client, "request", side_effect=Exception("Unexpected error")):
        response = client.get("/")
        assert response.status_code == 502
        assert "Proxy error: Unexpected error" in response.text


@pytest.mark.asyncio
async def test_chat_proxy_forwards_post_requests_with_body():
    server = AgentServer(enable_chat_proxy=True)
    client = TestClient(server.app)

    mock_response = Mock()
    mock_response.content = b'{"result": "success"}'
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}

    # POST to root path (allowed) to test body forwarding
    with patch.object(server.proxy_client, "request", return_value=mock_response) as mock_request:
        response = client.post("/", json={"message": "hello"})
        assert response.status_code == 200
        assert response.content == b'{"result": "success"}'

        call_args = mock_request.call_args
        assert call_args.kwargs["method"] == "POST"
        assert call_args.kwargs["content"] is not None


@pytest.mark.asyncio
async def test_chat_proxy_respects_chat_app_port_env_var(monkeypatch):
    monkeypatch.setenv("CHAT_APP_PORT", "8080")
    server = AgentServer(enable_chat_proxy=True)
    client = TestClient(server.app)

    mock_response = Mock()
    mock_response.content = b"test"
    mock_response.status_code = 200
    mock_response.headers = {}

    with patch.object(server.proxy_client, "request", return_value=mock_response) as mock_request:
        client.get("/assets/test.js")
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args.kwargs["url"] == "http://localhost:8080/assets/test.js"


def test_responses_create_endpoint_invoke():
    with patch("mlflow.start_span") as mock_span:
        mock_span_instance = Mock()
        mock_span_instance.__enter__ = Mock(return_value=mock_span_instance)
        mock_span_instance.__exit__ = Mock(return_value=None)
        mock_span.return_value = mock_span_instance

        @invoke()
        def test_invoke(request):
            return {
                "output": [
                    {
                        "type": "message",
                        "id": "123",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    }
                ]
            }

        server = AgentServer("ResponsesAgent")
        client = TestClient(server.app)

        request_data = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ]
        }

        response = client.post("/responses", json=request_data)
        assert response.status_code == 200
        assert "output" in response.json()


def test_responses_create_endpoint_stream():
    with patch("mlflow.start_span") as mock_span:
        mock_span_instance = Mock()
        mock_span_instance.__enter__ = Mock(return_value=mock_span_instance)
        mock_span_instance.__exit__ = Mock(return_value=None)
        mock_span.return_value = mock_span_instance

        @stream()
        def test_stream(request):
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item={
                    "type": "message",
                    "id": "123",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}],
                },
            )

        server = AgentServer("ResponsesAgent")
        client = TestClient(server.app)

        request_data = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ],
            "stream": True,
        }

        response = client.post("/responses", json=request_data)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


def test_responses_create_with_custom_inputs_and_context():
    with patch("mlflow.start_span") as mock_span:
        mock_span_instance = Mock()
        mock_span_instance.__enter__ = Mock(return_value=mock_span_instance)
        mock_span_instance.__exit__ = Mock(return_value=None)
        mock_span.return_value = mock_span_instance

        @invoke()
        def test_invoke(request):
            assert request.custom_inputs == {"key": "value"}
            assert request.context.user_id == "test-user"
            return {
                "output": [
                    {
                        "type": "message",
                        "id": "123",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    }
                ]
            }

        server = AgentServer("ResponsesAgent")
        client = TestClient(server.app)

        request_data = {
            "input": [{"role": "user", "content": "Hello"}],
            "custom_inputs": {"key": "value"},
            "context": {"user_id": "test-user", "conversation_id": "conv-123"},
        }

        response = client.post("/responses", json=request_data)
        assert response.status_code == 200


def test_responses_create_validation_error():
    server = AgentServer("ResponsesAgent")
    client = TestClient(server.app)

    invalid_data = {"invalid": "structure"}
    response = client.post("/responses", json=invalid_data)
    assert response.status_code == 400
    assert "Invalid parameters for ResponsesAgent" in response.json()["detail"]


def test_responses_create_malformed_json():
    server = AgentServer("ResponsesAgent")
    client = TestClient(server.app)

    response = client.post("/responses", data="malformed json")
    assert response.status_code == 400
    assert "Invalid JSON in request body" in response.json()["detail"]


def test_agent_info_endpoint_responses_agent():
    import mlflow

    server = AgentServer("ResponsesAgent")
    client = TestClient(server.app)

    response = client.get("/agent/info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "mlflow_agent_server"
    assert data["use_case"] == "agent"
    assert data["mlflow_version"] == mlflow.__version__
    assert data["agent_api"] == "responses"


def test_agent_info_endpoint_no_agent_type():
    import mlflow

    server = AgentServer()
    client = TestClient(server.app)

    response = client.get("/agent/info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "mlflow_agent_server"
    assert data["use_case"] == "agent"
    assert data["mlflow_version"] == mlflow.__version__
    assert "agent_api" not in data


def test_agent_info_endpoint_custom_app_name(monkeypatch):
    import mlflow

    monkeypatch.setenv("DATABRICKS_APP_NAME", "custom_agent")
    server = AgentServer("ResponsesAgent")
    client = TestClient(server.app)

    response = client.get("/agent/info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "custom_agent"
    assert data["use_case"] == "agent"
    assert data["mlflow_version"] == mlflow.__version__
    assert data["agent_api"] == "responses"


def test_agent_server_routes_registration_responses_agent():
    server = AgentServer("ResponsesAgent")
    routes = [route.path for route in server.app.routes]
    assert "/invocations" in routes
    assert "/responses" in routes
    assert "/agent/info" in routes
    assert "/health" in routes


def test_agent_server_routes_registration_no_responses_route():
    server = AgentServer()  # No agent_type
    routes = [route.path for route in server.app.routes]
    assert "/invocations" in routes
    assert "/responses" not in routes  # Should NOT be present
    assert "/agent/info" in routes
    assert "/health" in routes


def test_responses_not_available_for_non_responses_agent():
    server = AgentServer()  # No agent_type
    client = TestClient(server.app)

    request_data = {"input": [{"role": "user", "content": "Hello"}]}

    response = client.post("/responses", json=request_data)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_proxy_forwards_root_path():
    server = AgentServer(enable_chat_proxy=True)
    client = TestClient(server.app)

    mock_response = Mock()
    mock_response.content = b"<html>index</html>"
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html"}

    with patch.object(server.proxy_client, "request", return_value=mock_response) as mock_request:
        response = client.get("/")
        assert response.status_code == 200
        mock_request.assert_called_once()
        assert mock_request.call_args.kwargs["url"] == "http://localhost:3000/"


@pytest.mark.asyncio
async def test_chat_proxy_forwards_assets_path():
    server = AgentServer(enable_chat_proxy=True)
    client = TestClient(server.app)

    mock_response = Mock()
    mock_response.content = b"console.log('test');"
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/javascript"}

    with patch.object(server.proxy_client, "request", return_value=mock_response) as mock_request:
        response = client.get("/assets/index-abc123.js")
        assert response.status_code == 200
        mock_request.assert_called_once()
        assert (
            mock_request.call_args.kwargs["url"] == "http://localhost:3000/assets/index-abc123.js"
        )


@pytest.mark.asyncio
async def test_chat_proxy_forwards_favicon():
    server = AgentServer(enable_chat_proxy=True)
    client = TestClient(server.app)

    mock_response = Mock()
    mock_response.content = b"favicon data"
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "image/x-icon"}

    with patch.object(server.proxy_client, "request", return_value=mock_response) as mock_request:
        response = client.get("/favicon.ico")
        assert response.status_code == 200
        mock_request.assert_called_once()
        assert mock_request.call_args.kwargs["url"] == "http://localhost:3000/favicon.ico"


@pytest.mark.asyncio
async def test_chat_proxy_blocks_api_paths():
    server = AgentServer(enable_chat_proxy=True)
    client = TestClient(server.app)

    with patch.object(server.proxy_client, "request") as mock_request:
        response = client.get("/api/session")
        assert response.status_code == 404
        assert response.text == "Not found"
        mock_request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ["/api/chat", "/api/config", "/some/random/path", "/admin", "/.env"],
)
async def test_chat_proxy_blocks_arbitrary_paths(path):
    server = AgentServer(enable_chat_proxy=True)
    client = TestClient(server.app)

    with patch.object(server.proxy_client, "request") as mock_request:
        response = client.get(path)
        assert response.status_code == 404
        assert response.text == "Not found"
        mock_request.assert_not_called()
