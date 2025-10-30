import json
import pytest
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock, patch
from typing import AsyncGenerator

from fastapi.testclient import TestClient

from mlflow.pyfunc.agent_server import (
    AgentServer,
    AgentValidator,
    get_invoke_function,
    invoke,
    stream,
)
from mlflow.types.agent import ChatAgentChunk, ChatAgentRequest
from mlflow.types.llm import (
    ChatCompletionChunk,
    ChatCompletionRequest,
)
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)


class TestDecoratorRegistration:
    def setup_method(self):
        # Reset global state before each test
        import mlflow.pyfunc.agent_server

        mlflow.pyfunc.agent_server._invoke_function = None
        mlflow.pyfunc.agent_server._stream_function = None

    def test_invoke_decorator_single_registration(self):
        @invoke()
        def my_invoke_function(request):
            return {"result": "success"}

        assert get_invoke_function() == my_invoke_function

    def test_stream_decorator_single_registration(self):
        @stream()
        async def my_stream_function(request):
            yield {"delta": {"content": "hello"}}

        import mlflow.pyfunc.agent_server

        assert mlflow.pyfunc.agent_server._stream_function == my_stream_function

    def test_multiple_invoke_registrations_raises_error(self):
        @invoke()
        def first_function(request):
            return {"result": "first"}

        with pytest.raises(ValueError, match="invoke decorator can only be used once"):

            @invoke()
            def second_function(request):
                return {"result": "second"}

    def test_multiple_stream_registrations_raises_error(self):
        @stream()
        def first_stream(request):
            yield {"delta": {"content": "first"}}

        with pytest.raises(ValueError, match="stream decorator can only be used once"):

            @stream()
            def second_stream(request):
                yield {"delta": {"content": "second"}}

    def test_get_invoke_function_returns_registered(self):
        def my_function(request):
            return {"test": "data"}

        @invoke()
        def registered_function(request):
            return my_function(request)

        result = get_invoke_function()
        assert result == registered_function


class TestAgentValidator:
    def setup_method(self):
        self.validator_responses = AgentValidator("agent/v1/responses")
        self.validator_chat_v1 = AgentValidator("agent/v1/chat")
        self.validator_chat_v2 = AgentValidator("agent/v2/chat")
        self.validator_none = AgentValidator(None)

    def test_validator_request_dict_responses_agent(self):
        request_data = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ]
        }
        result = self.validator_responses.validate_and_convert_request(request_data)
        assert isinstance(result, ResponsesAgentRequest)

    def test_validator_request_dict_chat_v1(self):
        request_data = {"messages": [{"role": "user", "content": "Hello"}]}
        result = self.validator_chat_v1.validate_and_convert_request(request_data)
        assert isinstance(result, ChatCompletionRequest)

    def test_validator_request_dict_chat_v2(self):
        request_data = {"messages": [{"role": "user", "content": "Hello"}]}
        result = self.validator_chat_v2.validate_and_convert_request(request_data)
        assert isinstance(result, ChatAgentRequest)

    def test_validator_invalid_request_dict_raises_error(self):
        invalid_data = {"invalid": "structure"}

        with pytest.raises(ValueError, match="Invalid data for ResponsesAgentRequest"):
            self.validator_responses.validate_and_convert_request(invalid_data)

    def test_validator_none_type_returns_data_unchanged(self):
        request_data = {"any": "data"}
        result = self.validator_none.validate_and_convert_request(request_data)
        assert result == request_data

    def test_validator_response_dict_format(self):
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

        result = self.validator_responses.validate_and_convert_result(response_dict)
        assert isinstance(result, dict)
        assert result == response_dict

    def test_validator_response_pydantic_format(self):
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

        result = self.validator_responses.validate_and_convert_result(response_pydantic)
        assert isinstance(result, dict)
        assert "output" in result

    def test_validator_response_dataclass_format(self):
        @dataclass
        class TestDataclass:
            message: str
            status: str

        response_dataclass = TestDataclass(message="hello", status="completed")

        result = self.validator_none.validate_and_convert_result(response_dataclass)
        assert isinstance(result, dict)
        assert result == {"message": "hello", "status": "completed"}

    def test_validator_unsupported_output_type_raises_error(self):
        unsupported_output = ["not", "a", "dict", "or", "model"]

        with pytest.raises(
            ValueError, match="Result needs to be a pydantic model, dataclass, or dict"
        ):
            self.validator_none.validate_and_convert_result(unsupported_output)

    def test_validator_stream_response_formats(self):
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

        result = self.validator_responses.validate_and_convert_result(stream_event, stream=True)
        assert isinstance(result, dict)

    def test_validator_chat_v1_stream_response(self):
        chunk = ChatCompletionChunk(
            id="123",
            choices=[{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
            created=1234567890,
            model="test",
            object="chat.completion.chunk",
        )

        result = self.validator_chat_v1.validate_and_convert_result(chunk, stream=True)
        assert isinstance(result, dict)

    def test_validator_chat_v2_stream_response(self):
        chunk = ChatAgentChunk(delta={"content": "hello"})

        result = self.validator_chat_v2.validate_and_convert_result(chunk, stream=True)
        assert isinstance(result, dict)


class TestAgentServerInitialization:
    def test_agent_server_initialization(self):
        server = AgentServer()
        assert server.agent_type is None
        assert server.validator is not None
        assert server.app.title == "Agent Server"
        assert server.app.version == "0.0.1"

    def test_agent_server_with_agent_type(self):
        server = AgentServer(agent_type="agent/v1/responses")
        assert server.agent_type == "agent/v1/responses"
        assert server.validator.agent_type == "agent/v1/responses"

    def test_agent_server_cors_middleware_setup(self):
        server = AgentServer()
        # Check that CORS middleware is configured
        middlewares = [middleware.cls.__name__ for middleware in server.app.user_middleware]
        assert "CORSMiddleware" in middlewares

    @patch("mlflow.pyfunc.agent_server.Path")
    def test_agent_server_static_files_setup_exists(self, mock_path):
        mock_ui_path = Mock()
        mock_ui_path.exists.return_value = True
        mock_path.return_value.parent.parent.parent = mock_ui_path

        server = AgentServer()
        # Verify that static file routes are set up when UI exists
        routes = [route.path for route in server.app.routes]
        assert "/" in routes
        assert "/databricks.svg" in routes

    @patch("mlflow.pyfunc.agent_server.Path")
    def test_agent_server_static_files_setup_missing(self, mock_path):
        mock_ui_path = Mock()
        mock_ui_path.exists.return_value = False
        mock_path.return_value.parent.parent.parent = mock_ui_path

        with patch.object(AgentServer, "_setup_static_files") as mock_setup:
            mock_setup.return_value = None
            server = AgentServer()
            # Verify setup was called (implementation would log warning)
            assert server.app is not None

    def test_agent_server_routes_registration(self):
        server = AgentServer()
        routes = [route.path for route in server.app.routes]
        assert "/invocations" in routes
        assert "/health" in routes


class TestRequestHandling:
    def setup_method(self):
        # Reset global state before each test
        import mlflow.pyfunc.agent_server

        mlflow.pyfunc.agent_server._invoke_function = None
        mlflow.pyfunc.agent_server._stream_function = None

    def test_invocations_endpoint_malformed_json(self):
        server = AgentServer()
        client = TestClient(server.app)

        response = client.post("/invocations", data="malformed json")
        assert response.status_code == 400
        response_json = response.json()
        assert "Invalid JSON in request body" in response_json["detail"]

    def test_invocations_endpoint_missing_invoke_function(self):
        server = AgentServer()
        client = TestClient(server.app)

        response = client.post("/invocations", json={"test": "data"})
        assert response.status_code == 500
        response_json = response.json()
        assert "No invoke function registered" in response_json["detail"]

    def test_invocations_endpoint_validation_error(self):
        server = AgentServer(agent_type="agent/v1/responses")
        client = TestClient(server.app)

        # Send invalid request data for responses agent
        invalid_data = {"invalid": "structure"}
        response = client.post("/invocations", json=invalid_data)
        assert response.status_code == 400
        response_json = response.json()
        assert "Invalid parameters for agent/v1/responses" in response_json["detail"]

    @patch("mlflow.start_span")
    def test_invocations_endpoint_success_invoke(self, mock_span):
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

        server = AgentServer(agent_type="agent/v1/responses")
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

    @patch("mlflow.start_span")
    def test_invocations_endpoint_success_stream(self, mock_span):
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

        server = AgentServer(agent_type="agent/v1/responses")
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

    def test_health_endpoint_returns_status(self):
        server = AgentServer()
        client = TestClient(server.app)

        response = client.get("/health")
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["status"] == "healthy"
        assert response_json["version"] == "0.0.1"


class TestContextManagement:
    def test_request_headers_isolation(self):
        from mlflow.pyfunc.agent_server.utils import set_request_headers, get_request_headers

        # Test that headers are isolated between contexts
        set_request_headers({"test": "value1"})
        assert get_request_headers()["test"] == "value1"

        # In a different context, headers should be independent
        import contextvars

        ctx = contextvars.copy_context()

        def test_different_context():
            set_request_headers({"test": "value2"})
            return get_request_headers()["test"]

        result = ctx.run(test_different_context)
        assert result == "value2"
        # Original context should be unchanged
        assert get_request_headers()["test"] == "value1"

    def test_forwarded_access_token_extraction(self):
        from mlflow.pyfunc.agent_server.utils import (
            set_request_headers,
            get_forwarded_access_token,
        )

        test_token = "test-token-123"
        set_request_headers({"x-forwarded-access-token": test_token})

        assert get_forwarded_access_token() == test_token

    def test_forwarded_access_token_missing(self):
        from mlflow.pyfunc.agent_server.utils import (
            set_request_headers,
            get_forwarded_access_token,
        )

        set_request_headers({"other-header": "value"})
        assert get_forwarded_access_token() is None

    @patch("mlflow.pyfunc.agent_server.utils.WorkspaceClient")
    def test_obo_workspace_client(self, mock_workspace_client):
        from mlflow.pyfunc.agent_server.utils import (
            set_request_headers,
            get_obo_workspace_client,
        )

        test_token = "test-token-123"
        set_request_headers({"x-forwarded-access-token": test_token})

        get_obo_workspace_client()
        mock_workspace_client.assert_called_once_with(token=test_token, auth_type="pat")


class TestMLflowIntegration:
    @patch("mlflow.start_span")
    def test_tracing_span_creation(self, mock_span):
        mock_span_instance = Mock()
        mock_span_instance.__enter__ = Mock(return_value=mock_span_instance)
        mock_span_instance.__exit__ = Mock(return_value=None)
        mock_span.return_value = mock_span_instance

        @invoke()
        def test_function(request):
            return {"result": "success"}

        server = AgentServer()
        client = TestClient(server.app)

        response = client.post("/invocations", json={"test": "data"})
        # Verify span was created with correct name
        mock_span.assert_called_once_with(name="test_function_invoke")

    @patch("mlflow.start_span")
    def test_tracing_attributes_setting(self, mock_span):
        mock_span_instance = Mock()
        mock_span_instance.__enter__ = Mock(return_value=mock_span_instance)
        mock_span_instance.__exit__ = Mock(return_value=None)
        mock_span.return_value = mock_span_instance

        @invoke()
        def test_function(request):
            return {"result": "success"}

        server = AgentServer(agent_type="agent/v1/responses")
        client = TestClient(server.app)

        response = client.post("/invocations", json={"test": "data"})

        # Verify span attributes were set
        mock_span_instance.set_inputs.assert_called_once()
        mock_span_instance.set_outputs.assert_called_once()
        mock_span_instance.set_attribute.assert_called()

    @patch("mlflow.start_span")
    @patch("mlflow.pyfunc.agent_server.InMemoryTraceManager")
    def test_databricks_output_integration(self, mock_trace_manager, mock_span):
        mock_span_instance = Mock()
        mock_span_instance.__enter__ = Mock(return_value=mock_span_instance)
        mock_span_instance.__exit__ = Mock(return_value=None)
        mock_span_instance.trace_id = "test-trace-id"
        mock_span.return_value = mock_span_instance

        # Mock trace manager
        mock_trace = Mock()
        mock_trace.to_mlflow_trace.return_value.to_dict.return_value = {"trace": "data"}
        mock_trace_manager.get_instance.return_value.get_trace.return_value.__enter__ = Mock(
            return_value=mock_trace
        )
        mock_trace_manager.get_instance.return_value.get_trace.return_value.__exit__ = Mock(
            return_value=None
        )

        @invoke()
        def test_function(request):
            return {"result": "success"}

        server = AgentServer()
        client = TestClient(server.app)

        request_data = {"test": "data", "databricks_options": {"return_trace": True}}

        response = client.post("/invocations", json=request_data)
        response_json = response.json()

        # Verify databricks_output is included in response
        assert "databricks_output" in response_json
        assert "trace" in response_json["databricks_output"]

