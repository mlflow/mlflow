import importlib
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest

import mlflow.pydanticai._pydanticai_autolog as autolog_module
from mlflow.pydanticai._pydanticai_autolog import (
    _GLOBAL_SESSION_KEY,
    DEFAULT_SESSION_NAME,
    FLAVOUR_NAME,
    AutoLoggingConfig,
    LiveSpan,
    SpanEvent,
    SpanStatusCode,
    SpanType,
    TraceMetadataKey,
    _capture_inputs,
    _end_span_err,
    _end_span_ok,
    _get_or_create_session_span,
    _patched_end_run,
    _start_span,
    _with_span_async,
    _with_span_sync,
    autolog,
)


class MockAgent:
    async def run(self):
        return "async result"

    def run_sync(self):
        return "sync result"


class MockInstrumentedModel:
    provider_name = "mock_provider"
    wrapped = MagicMock(__class__=MagicMock(__name__="WrappedMockModel"))

    async def request(self):
        return "model request result"

    async def request_stream(self):
        yield "chunk1"
        yield "chunk2"

    model_name = "mock-model-name"


class MockTool:
    name = "mock_tool"

    async def run(self):
        return "tool result"


class MockMCPServer:
    async def call_tool(self, tool_name):
        return f"called {tool_name}"

    async def list_tools(self):
        return ["tool1", "tool2"]


class MockToolMessage:
    tool_call_id = "call_123"

    def args_as_json_str(self):
        return json.dumps({"arg1": "value1"})


@dataclass
class MockRunContextUsage:
    request_tokens: int = 10
    response_tokens: int = 20
    total_tokens: int = 30
    details: dict | None = None


@dataclass
class MockMessage:
    role: str = "user"
    content: str = "hello"


@dataclass
class MockRunContext:
    model: Any = field(default_factory=MockInstrumentedModel)
    prompt: Any = "Test Prompt"
    messages: list = field(default_factory=lambda: [MockMessage()])
    usage: MockRunContextUsage = field(default_factory=MockRunContextUsage)
    retry: int = 0
    run_step: int = 1


MOCK_RUN_CONTEXT = MockRunContext()

pydantic_ai_available = True
mock_pydantic_ai_module = MagicMock()
mock_pydantic_ai_module.Agent = MockAgent
mock_pydantic_ai_module.models = MagicMock()
mock_pydantic_ai_module.models.instrumented = MagicMock()
mock_pydantic_ai_module.models.instrumented.InstrumentedModel = MockInstrumentedModel
mock_pydantic_ai_module.tools = MagicMock()
mock_pydantic_ai_module.tools.Tool = MockTool
mock_pydantic_ai_module.mcp = MagicMock()
mock_pydantic_ai_module.mcp.MCPServer = MockMCPServer


@pytest.fixture(autouse=True)
def reset_global_state_and_config():
    autolog_module._session_spans.clear()
    autolog_module._session_name = DEFAULT_SESSION_NAME
    if hasattr(AutoLoggingConfig, "_config_cache"):
        AutoLoggingConfig._config_cache.clear()
    yield
    autolog_module._session_spans.clear()
    autolog_module._session_name = DEFAULT_SESSION_NAME
    if hasattr(AutoLoggingConfig, "_config_cache"):
        AutoLoggingConfig._config_cache.clear()


@pytest.fixture
def mock_mlflow_client():
    client = MagicMock(spec=autolog_module.mlflow.MlflowClient)
    client.end_span = MagicMock()
    return client


@pytest.fixture
def mock_live_span():
    span = MagicMock(spec=LiveSpan)
    span.request_id = "req-123"
    span.span_id = "span-abc"
    span.add_event = MagicMock()
    return span


@pytest.fixture
def mock_active_run():
    run = MagicMock()
    run.info.run_id = "run-xyz"
    return run


@pytest.fixture
def mock_autologging_config():
    with patch.object(autolog_module.AutoLoggingConfig, "init") as mock_init:
        config_instance = MagicMock(spec=AutoLoggingConfig)
        config_instance.log_traces = True
        mock_init.return_value = config_instance
        yield mock_init, config_instance


@pytest.fixture
def mock_dependencies(mock_mlflow_client, mock_live_span, mock_active_run, mock_autologging_config):
    with (
        patch.object(autolog_module, "_mlclient", return_value=mock_mlflow_client),
        patch.object(
            autolog_module, "start_client_span_or_trace", return_value=mock_live_span
        ) as mock_start_trace,
        patch.object(autolog_module, "end_client_span_or_trace") as mock_end_trace,
        patch.object(
            autolog_module, "set_span_in_context", return_value="mock_token"
        ) as mock_set_context,
        patch.object(autolog_module, "detach_span_from_context") as mock_detach_context,
        patch.object(
            autolog_module.mlflow, "active_run", return_value=None
        ) as mock_active_run_func,
        patch.object(
            autolog_module.mlflow.utils.autologging_utils, "safe_patch"
        ) as mock_safe_patch,
        patch.object(autolog_module.atexit, "register") as mock_atexit_register,
        patch.object(autolog_module.InMemoryTraceManager, "get_instance") as mock_get_trace_manager,
    ):
        mock_trace_manager_instance = MagicMock()
        mock_trace_manager_instance.set_request_metadata = MagicMock()
        mock_get_trace_manager.return_value = mock_trace_manager_instance
        yield {
            "client": mock_mlflow_client,
            "live_span": mock_live_span,
            "start_trace": mock_start_trace,
            "end_trace": mock_end_trace,
            "set_context": mock_set_context,
            "detach_context": mock_detach_context,
            "active_run_func": mock_active_run_func,
            "active_run_obj": mock_active_run,
            "safe_patch": mock_safe_patch,
            "atexit_register": mock_atexit_register,
            "autologging_config_init": mock_autologging_config[0],
            "autologging_config_instance": mock_autologging_config[1],
            "trace_manager": mock_trace_manager_instance,
        }


def test_start_span_no_run(mock_dependencies):
    inputs = {"key": "value"}
    span = _start_span(name="test_span", span_type=SpanType.AGENT, inputs=inputs)
    mock_dependencies["start_trace"].assert_called_once_with(
        mock_dependencies["client"], name="test_span", span_type=SpanType.AGENT, inputs=inputs
    )
    assert span == mock_dependencies["live_span"]
    mock_dependencies["trace_manager"].set_request_metadata.assert_not_called()


def test_start_span_with_run(mock_dependencies):
    inputs = {"key": "value"}
    run_id = mock_dependencies["active_run_obj"].info.run_id
    span = _start_span(name="test_span", span_type=SpanType.LLM, inputs=inputs, run_id=run_id)
    mock_dependencies["start_trace"].assert_called_once_with(
        mock_dependencies["client"], name="test_span", span_type=SpanType.LLM, inputs=inputs
    )
    assert span == mock_dependencies["live_span"]
    mock_dependencies["trace_manager"].set_request_metadata.assert_called_once_with(
        mock_dependencies["live_span"].request_id, TraceMetadataKey.SOURCE_RUN, run_id
    )


def test_end_span_ok(mock_dependencies):
    outputs = {"result": "success"}
    _end_span_ok(mock_dependencies["live_span"], outputs)
    mock_dependencies["end_trace"].assert_called_once_with(
        mock_dependencies["client"], mock_dependencies["live_span"], outputs=outputs
    )


def test_end_span_ok_handles_exception(mock_dependencies, caplog):
    mock_dependencies["end_trace"].side_effect = Exception("End span failed")
    outputs = {"result": "success"}
    logger_name = autolog_module.__name__
    with caplog.at_level(autolog_module.logging.DEBUG, logger=logger_name):
        _end_span_ok(mock_dependencies["live_span"], outputs)
    mock_dependencies["end_trace"].assert_called_once_with(
        mock_dependencies["client"], mock_dependencies["live_span"], outputs=outputs
    )


def test_end_span_err(mock_dependencies):
    exc = ValueError("Something went wrong")
    _end_span_err(mock_dependencies["live_span"], exc)
    mock_dependencies["live_span"].add_event.assert_called_once()
    event_arg = mock_dependencies["live_span"].add_event.call_args[0][0]
    assert isinstance(event_arg, SpanEvent)
    assert "ValueError: Something went wrong" in event_arg.attributes["exception.stacktrace"]
    mock_dependencies["client"].end_span.assert_called_once_with(
        mock_dependencies["live_span"].request_id,
        mock_dependencies["live_span"].span_id,
        status=SpanStatusCode.ERROR,
    )


@pytest.mark.asyncio
async def test_with_span_async_success(mock_dependencies):
    original_result = "async success"
    original_func = AsyncMock(return_value=original_result)
    mock_self = MagicMock()
    mock_self.__class__.__name__ = "MyAsyncClass"
    test_inputs = {"arg1": 1, "kwarg1": "test"}
    with patch.object(
        autolog_module, "construct_full_inputs", return_value=test_inputs
    ) as mock_construct:
        decorator_instance = _with_span_async(
            span_name="test_async_span",
            span_type=SpanType.CHAIN,
            capture_inputs=lambda args, kwargs: mock_construct(
                original_func, args[0], *args[1:], **kwargs
            ),
        )

        def decorated_func(*args, **kwargs):
            return decorator_instance(original_func, *args, **kwargs)

        result = await decorated_func(mock_self, 1, kwarg1="test")
    assert result == original_result
    original_func.assert_awaited_once_with(mock_self, 1, kwarg1="test")
    mock_construct.assert_called_once_with(original_func, mock_self, 1, kwarg1="test")
    mock_dependencies["autologging_config_init"].assert_called_once_with(flavor_name=FLAVOUR_NAME)
    assert autolog_module._session_spans
    calls = mock_dependencies["start_trace"].call_args_list
    assert len(calls) == 2
    _, sess_kwargs = calls[0]
    assert sess_kwargs["name"] == DEFAULT_SESSION_NAME
    assert sess_kwargs["span_type"] == SpanType.CHAIN
    assert sess_kwargs["inputs"] == {}
    _, span_kwargs = calls[1]
    assert span_kwargs["name"] == "test_async_span"
    assert span_kwargs["span_type"] == SpanType.CHAIN
    assert span_kwargs["inputs"] == test_inputs
    mock_dependencies["client"].end_span.assert_not_called()


@pytest.mark.asyncio
async def test_with_span_async_exception(mock_dependencies):
    test_exception = ValueError("Async error")
    original_func = AsyncMock(side_effect=test_exception)
    mock_self = MagicMock()
    mock_self.__class__.__name__ = "MyErrorClass"
    test_inputs = {"arg1": 1}
    with patch.object(
        autolog_module, "construct_full_inputs", return_value=test_inputs
    ) as mock_construct:
        decorator_instance = _with_span_async(
            span_name=lambda s: f"{s.__class__.__name__}.method",
            span_type=SpanType.CHAIN,
            capture_inputs=lambda args, kwargs: mock_construct(
                original_func, args[0], *args[1:], **kwargs
            ),
        )

        def decorated_func(*args, **kwargs):
            return decorator_instance(original_func, *args, **kwargs)

        with pytest.raises(ValueError, match="Async error"):
            await decorated_func(mock_self, 1)
    original_func.assert_awaited_once_with(mock_self, 1)
    mock_construct.assert_called_once_with(original_func, mock_self, 1)
    calls = mock_dependencies["start_trace"].call_args_list
    assert len(calls) == 2
    _, span_kwargs = calls[1]
    assert span_kwargs["name"] == "MyErrorClass.method"
    assert span_kwargs["span_type"] == SpanType.CHAIN
    assert span_kwargs["inputs"] == test_inputs
    calls = mock_dependencies["set_context"].call_args_list
    assert len(calls) == 2
    assert calls[1][0][0] == mock_dependencies["live_span"]
    mock_dependencies["live_span"].add_event.assert_called_once()
    mock_dependencies["client"].end_span.assert_called_once_with(
        mock_dependencies["live_span"].request_id,
        mock_dependencies["live_span"].span_id,
        status=SpanStatusCode.ERROR,
    )
    mock_dependencies["end_trace"].assert_not_called()
    mock_dependencies["detach_context"].assert_called_once_with("mock_token")


@pytest.mark.asyncio
async def test_with_span_async_log_traces_disabled(mock_dependencies):
    mock_dependencies["autologging_config_instance"].log_traces = False
    original_result = "no trace async"
    original_func = AsyncMock(return_value=original_result)
    mock_self = MagicMock()
    mock_self.__class__.__name__ = "MyAsyncClass"
    capture_inputs_mock = MagicMock()
    decorator_instance = _with_span_async(
        span_name="test_async_span",
        span_type=SpanType.CHAIN,
        capture_inputs=capture_inputs_mock,
    )

    def decorated_func(*args, **kwargs):
        return decorator_instance(original_func, *args, **kwargs)

    result = await decorated_func(mock_self, 1)
    assert result == original_result
    original_func.assert_awaited_once_with(mock_self, 1)
    mock_dependencies["autologging_config_init"].assert_called_once_with(flavor_name=FLAVOUR_NAME)
    mock_dependencies["start_trace"].assert_not_called()
    mock_dependencies["set_context"].assert_not_called()
    mock_dependencies["detach_context"].assert_not_called()
    mock_dependencies["end_trace"].assert_not_called()
    mock_dependencies["client"].end_span.assert_not_called()
    capture_inputs_mock.assert_not_called()
    assert not autolog_module._session_spans


def test_with_span_sync_success(mock_dependencies):
    original_result = "sync success"
    original_func = MagicMock(return_value=original_result)
    mock_self = MagicMock()
    mock_self.__class__.__name__ = "MySyncClass"
    test_inputs = {"arg1": 2, "kwarg1": "sync_test"}
    with patch.object(
        autolog_module, "construct_full_inputs", return_value=test_inputs
    ) as mock_construct:
        decorator_instance = _with_span_sync(
            span_name="test_sync_span",
            span_type=SpanType.TOOL,
            capture_inputs=lambda args, kwargs: mock_construct(
                original_func, args[0], *args[1:], **kwargs
            ),
        )

        def decorated_func(*args, **kwargs):
            return decorator_instance(original_func, *args, **kwargs)

        result = decorated_func(mock_self, 2, kwarg1="sync_test")
    assert result == original_result
    original_func.assert_called_once_with(mock_self, 2, kwarg1="sync_test")
    mock_construct.assert_called_once_with(original_func, mock_self, 2, kwarg1="sync_test")
    mock_dependencies["autologging_config_init"].assert_called_once_with(flavor_name=FLAVOUR_NAME)
    assert autolog_module._session_spans
    calls = mock_dependencies["start_trace"].call_args_list
    assert len(calls) == 2
    _, span_kwargs = calls[1]
    assert span_kwargs["name"] == "test_sync_span"
    assert span_kwargs["span_type"] == SpanType.TOOL
    assert span_kwargs["inputs"] == test_inputs
    assert len(mock_dependencies["set_context"].call_args_list) == 2
    mock_dependencies["detach_context"].assert_called_once_with("mock_token")
    mock_dependencies["end_trace"].assert_called_once_with(
        mock_dependencies["client"], mock_dependencies["live_span"], outputs=original_result
    )
    mock_dependencies["client"].end_span.assert_not_called()


def test_with_span_sync_exception(mock_dependencies):
    test_exception = TypeError("Sync error")
    original_func = MagicMock(side_effect=test_exception)
    mock_self = MagicMock()
    mock_self.__class__.__name__ = "MySyncErrorClass"
    test_inputs = {"arg1": 1}
    with patch.object(
        autolog_module, "construct_full_inputs", return_value=test_inputs
    ) as mock_construct:
        decorator_instance = _with_span_sync(
            span_name=lambda s: f"{s.__class__.__name__}.sync_method",
            span_type=SpanType.CHAIN,
            capture_inputs=lambda args, kwargs: mock_construct(
                original_func, args[0], *args[1:], **kwargs
            ),
        )

        def decorated_func(*args, **kwargs):
            return decorator_instance(original_func, *args, **kwargs)

        with pytest.raises(TypeError, match="Sync error"):
            decorated_func(mock_self, 1)
    original_func.assert_called_once_with(mock_self, 1)
    mock_construct.assert_called_once_with(original_func, mock_self, 1)
    calls = mock_dependencies["start_trace"].call_args_list
    assert len(calls) == 2
    _, span_kwargs = calls[1]
    assert span_kwargs["name"] == "MySyncErrorClass.sync_method"
    assert span_kwargs["span_type"] == SpanType.CHAIN
    assert span_kwargs["inputs"] == test_inputs
    assert len(mock_dependencies["set_context"].call_args_list) == 2
    mock_dependencies["live_span"].add_event.assert_called_once()
    mock_dependencies["client"].end_span.assert_called_once_with(
        mock_dependencies["live_span"].request_id,
        mock_dependencies["live_span"].span_id,
        status=SpanStatusCode.ERROR,
    )
    mock_dependencies["end_trace"].assert_not_called()
    mock_dependencies["detach_context"].assert_called_once_with("mock_token")


def test_with_span_sync_log_traces_disabled(mock_dependencies):
    mock_dependencies["autologging_config_instance"].log_traces = False
    original_result = "no trace sync"
    original_func = MagicMock(return_value=original_result)
    mock_self = MagicMock()
    capture_inputs_mock = MagicMock()
    decorator_instance = _with_span_sync(
        span_name="test_sync_span",
        span_type=SpanType.CHAIN,
        capture_inputs=capture_inputs_mock,
    )

    def decorated_func(*args, **kwargs):
        return decorator_instance(original_func, *args, **kwargs)

    result = decorated_func(mock_self, 1)
    assert result == original_result
    original_func.assert_called_once_with(mock_self, 1)
    mock_dependencies["autologging_config_init"].assert_called_once_with(flavor_name=FLAVOUR_NAME)
    mock_dependencies["start_trace"].assert_not_called()
    mock_dependencies["set_context"].assert_not_called()
    mock_dependencies["detach_context"].assert_not_called()
    mock_dependencies["end_trace"].assert_not_called()
    mock_dependencies["client"].end_span.assert_not_called()
    capture_inputs_mock.assert_not_called()
    assert not autolog_module._session_spans


def test_get_or_create_session_span_global(mock_dependencies):
    mock_dependencies["active_run_func"].return_value = None
    _get_or_create_session_span()
    mock_dependencies["start_trace"].assert_called_once_with(
        mock_dependencies["client"], name=DEFAULT_SESSION_NAME, span_type=SpanType.CHAIN, inputs={}
    )
    mock_dependencies["set_context"].assert_called_once_with(mock_dependencies["live_span"])
    assert _GLOBAL_SESSION_KEY in autolog_module._session_spans
    span_tuple = autolog_module._session_spans[_GLOBAL_SESSION_KEY]
    assert span_tuple == (mock_dependencies["live_span"], "mock_token")
    mock_dependencies["atexit_register"].assert_called_once()
    registered_func = mock_dependencies["atexit_register"].call_args[0][0]
    mock_dependencies["start_trace"].reset_mock()
    mock_dependencies["set_context"].reset_mock()
    mock_dependencies["atexit_register"].reset_mock()
    _get_or_create_session_span()
    mock_dependencies["start_trace"].assert_not_called()
    mock_dependencies["set_context"].assert_not_called()
    mock_dependencies["atexit_register"].assert_not_called()
    assert _GLOBAL_SESSION_KEY in autolog_module._session_spans
    mock_dependencies["detach_context"].reset_mock()
    mock_dependencies["end_trace"].reset_mock()
    registered_func()
    mock_dependencies["detach_context"].assert_called_once_with(span_tuple[1])
    mock_dependencies["end_trace"].assert_called_once_with(
        mock_dependencies["client"], span_tuple[0], outputs=None
    )
    assert _GLOBAL_SESSION_KEY not in autolog_module._session_spans


def test_get_or_create_session_span_with_run(mock_dependencies):
    mock_dependencies["active_run_func"].return_value = mock_dependencies["active_run_obj"]
    run_id = mock_dependencies["active_run_obj"].info.run_id
    _get_or_create_session_span()
    mock_dependencies["start_trace"].assert_called_once_with(
        mock_dependencies["client"], name=DEFAULT_SESSION_NAME, span_type=SpanType.CHAIN, inputs={}
    )
    mock_dependencies["set_context"].assert_called_once_with(mock_dependencies["live_span"])
    assert run_id in autolog_module._session_spans
    assert autolog_module._session_spans[run_id] == (
        mock_dependencies["live_span"],
        "mock_token",
    )
    mock_dependencies["trace_manager"].set_request_metadata.assert_called_once_with(
        mock_dependencies["live_span"].request_id, TraceMetadataKey.SOURCE_RUN, run_id
    )
    mock_dependencies["atexit_register"].assert_not_called()
    mock_dependencies["start_trace"].reset_mock()
    _get_or_create_session_span()
    mock_dependencies["start_trace"].assert_not_called()


def test_patched_end_run_closes_session_span(mock_dependencies):
    mock_dependencies["active_run_func"].return_value = mock_dependencies["active_run_obj"]
    run_id = mock_dependencies["active_run_obj"].info.run_id
    _get_or_create_session_span()
    assert run_id in autolog_module._session_spans
    span_tuple = autolog_module._session_spans[run_id]
    mock_original_end_run = MagicMock(return_value="original end run result")
    result = _patched_end_run(mock_original_end_run, status="FINISHED", arg1="test")
    mock_original_end_run.assert_called_once_with("FINISHED", arg1="test")
    assert result == "original end run result"
    mock_dependencies["detach_context"].assert_called_with(span_tuple[1])
    mock_dependencies["end_trace"].assert_called_with(
        mock_dependencies["client"], span_tuple[0], outputs=None
    )
    assert run_id not in autolog_module._session_spans


def test_patched_end_run_no_active_run(mock_dependencies):
    mock_dependencies["active_run_func"].return_value = None
    mock_original_end_run = MagicMock(return_value="original end run result")
    result = _patched_end_run(mock_original_end_run, status="FINISHED")
    mock_original_end_run.assert_called_once_with("FINISHED")
    assert result == "original end run result"
    mock_dependencies["detach_context"].assert_not_called()
    mock_dependencies["end_trace"].assert_not_called()


def test_patched_end_run_no_matching_span(mock_dependencies):
    mock_dependencies["active_run_func"].return_value = mock_dependencies["active_run_obj"]
    assert mock_dependencies["active_run_obj"].info.run_id not in autolog_module._session_spans
    mock_original_end_run = MagicMock(return_value="original end run result")
    result = _patched_end_run(mock_original_end_run, status="FINISHED")
    mock_original_end_run.assert_called_once_with("FINISHED")
    assert result == "original end run result"
    mock_dependencies["detach_context"].assert_not_called()
    mock_dependencies["end_trace"].assert_not_called()


def test_capture_inputs():
    def sample_func(self, pos1, pos2, kwarg1="a", kwarg2="b"):
        pass

    mock_self = MagicMock()
    args = (mock_self, 1, 2)
    kwargs = {"kwarg1": "x"}

    with patch.object(autolog_module, "construct_full_inputs") as mock_construct:
        mock_construct.return_value = {"mocked": "value"}
        actual_inputs = _capture_inputs(sample_func, *args, **kwargs)
        mock_construct.assert_called_once_with(sample_func, mock_self, 1, 2, kwarg1="x")
        assert actual_inputs == {"mocked": "value"}

    kwargs = {"kwarg1": "x", "kwarg2": "b"}
    actual_inputs_real = _capture_inputs(sample_func, *args, **kwargs)
    assert actual_inputs_real == {"pos1": 1, "pos2": 2, "kwarg1": "x", "kwarg2": "b"}


@patch.dict("sys.modules", {"pydantic_ai": mock_pydantic_ai_module})
def test_patch_agent_methods(mock_dependencies):
    importlib.reload(autolog_module)
    autolog_module._patch_agent_methods()
    expected_calls = [
        call(FLAVOUR_NAME, mock_pydantic_ai_module.Agent, "run_sync", ANY),
        call(FLAVOUR_NAME, mock_pydantic_ai_module.Agent, "run", ANY),
    ]
    mock_dependencies["safe_patch"].assert_has_calls(expected_calls, any_order=True)
    assert mock_dependencies["safe_patch"].call_count >= 2


@patch.dict(
    "sys.modules",
    {
        "pydantic_ai": mock_pydantic_ai_module,
        "pydantic_ai.models": mock_pydantic_ai_module.models,
        "pydantic_ai.models.instrumented": mock_pydantic_ai_module.models.instrumented,
    },
)
def test_patch_instrumented_model(mock_dependencies):
    importlib.reload(autolog_module)
    autolog_module._patch_instrumented_model()
    expected_calls = [
        call(
            FLAVOUR_NAME,
            mock_pydantic_ai_module.models.instrumented.InstrumentedModel,
            "request",
            ANY,
        ),
    ]
    if hasattr(mock_pydantic_ai_module.models.instrumented.InstrumentedModel, "request_stream"):
        expected_calls.append(
            call(
                FLAVOUR_NAME,
                mock_pydantic_ai_module.models.instrumented.InstrumentedModel,
                "request_stream",
                ANY,
            )
        )
    mock_dependencies["safe_patch"].assert_has_calls(expected_calls, any_order=True)


@patch.dict(
    "sys.modules",
    {"pydantic_ai": mock_pydantic_ai_module, "pydantic_ai.tools": mock_pydantic_ai_module.tools},
)
def test_patch_tool_run(mock_dependencies):
    importlib.reload(autolog_module)
    autolog_module._patch_tool_run()
    mock_dependencies["safe_patch"].assert_any_call(
        FLAVOUR_NAME, mock_pydantic_ai_module.tools.Tool, "run", ANY
    )


@pytest.mark.asyncio
@patch.dict(
    "sys.modules",
    {
        "pydantic_ai": mock_pydantic_ai_module,
        "pydantic_ai.tools": mock_pydantic_ai_module.tools,
    },
)
async def test_tool_run_wrapper_logic(mock_dependencies):
    importlib.reload(autolog_module)
    autolog_module._mlclient = lambda: mock_dependencies["client"]
    autolog_module.start_client_span_or_trace = mock_dependencies["start_trace"]
    autolog_module.end_client_span_or_trace = mock_dependencies["end_trace"]
    autolog_module.set_span_in_context = mock_dependencies["set_context"]
    autolog_module.detach_span_from_context = mock_dependencies["detach_context"]

    autolog_module.safe_patch = mock_dependencies["safe_patch"]
    autolog_module._patch_tool_run()

    wrapper = None
    for args, _ in mock_dependencies["safe_patch"].call_args_list:
        _, target_cls, method, patch_fn = args
        if target_cls is mock_pydantic_ai_module.tools.Tool and method == "run":
            wrapper = patch_fn
            break
    assert wrapper is not None

    original_tool_run = AsyncMock(return_value="original tool result")
    mock_tool_instance = mock_pydantic_ai_module.tools.Tool()
    mock_tool_instance.name = "mock_tool"
    mock_message = MockToolMessage()
    test_tracer = MagicMock()
    test_run_context = MOCK_RUN_CONTEXT

    result = await wrapper(
        original_tool_run,
        mock_tool_instance,
        mock_message,
        test_run_context,
        test_tracer,
    )
    assert result == "original tool result"
    original_tool_run.assert_awaited_once_with(
        mock_tool_instance, mock_message, test_run_context, test_tracer
    )

    calls = mock_dependencies["start_trace"].call_args_list
    assert len(calls) == 1
    _, span_kwargs = calls[0]
    assert span_kwargs["name"] == "mock_tool"
    assert span_kwargs["span_type"] == SpanType.TOOL
    assert span_kwargs["inputs"]["tool_name"] == "mock_tool"
    assert span_kwargs["inputs"]["tool_call_id"] == "call_123"
    assert span_kwargs["inputs"]["tool_arguments"] == {"arg1": "value1"}
    assert "run_context" in span_kwargs["inputs"]
    assert span_kwargs["inputs"]["run_context"]["model_class"] == "MockInstrumentedModel"

    mock_dependencies["end_trace"].assert_called_once_with(
        mock_dependencies["client"],
        mock_dependencies["live_span"],
        outputs="original tool result",
    )


@patch.dict(
    "sys.modules",
    {"pydantic_ai": mock_pydantic_ai_module, "pydantic_ai.mcp": mock_pydantic_ai_module.mcp},
)
def test_patch_mcp_server(mock_dependencies):
    importlib.reload(autolog_module)
    autolog_module._patch_mcp_server()
    expected_calls = [
        call(FLAVOUR_NAME, mock_pydantic_ai_module.mcp.MCPServer, "call_tool", ANY),
        call(FLAVOUR_NAME, mock_pydantic_ai_module.mcp.MCPServer, "list_tools", ANY),
    ]
    mock_dependencies["safe_patch"].assert_has_calls(expected_calls, any_order=True)


@pytest.mark.asyncio
@patch.dict(
    "sys.modules",
    {"pydantic_ai": mock_pydantic_ai_module, "pydantic_ai.mcp": mock_pydantic_ai_module.mcp},
)
async def test_mcp_wrapper_logic(mock_dependencies):
    importlib.reload(autolog_module)
    autolog_module._mlclient = lambda: mock_dependencies["client"]
    autolog_module.start_client_span_or_trace = mock_dependencies["start_trace"]
    autolog_module.end_client_span_or_trace = mock_dependencies["end_trace"]
    autolog_module.set_span_in_context = mock_dependencies["set_context"]
    autolog_module.detach_span_from_context = mock_dependencies["detach_context"]
    autolog_module.safe_patch = mock_dependencies["safe_patch"]
    autolog_module._patch_mcp_server()

    wrappers = {}
    for c in mock_dependencies["safe_patch"].call_args_list:
        target_class, method_name, wrapper = c[0][1], c[0][2], c[0][3]
        if target_class == mock_pydantic_ai_module.mcp.MCPServer:
            wrappers[method_name] = wrapper

    call_tool_wrapper = wrappers["call_tool"]
    list_tools_wrapper = wrappers["list_tools"]

    original_call_tool = AsyncMock(return_value="original call_tool result")
    mcp_instance = mock_pydantic_ai_module.mcp.MCPServer()
    mcp_instance.__class__.__name__ = "MyMCPServer"

    await call_tool_wrapper(original_call_tool, mcp_instance, "tool_name", {"arg": "val"})
    assert autolog_module._session_spans
    assert mock_dependencies["start_trace"].call_count == 2
    _, kwargs = mock_dependencies["start_trace"].call_args_list[1]
    assert kwargs["name"] == "MyMCPServer.call_tool"
    assert kwargs["span_type"] == SpanType.CHAIN
    assert kwargs["inputs"] == {"tool_name": "tool_name", "arguments": {"arg": "val"}}
    mock_dependencies["end_trace"].assert_called_once_with(
        ANY, mock_dependencies["live_span"], outputs="original call_tool result"
    )
    original_call_tool.assert_awaited_once_with(mcp_instance, "tool_name", {"arg": "val"})

    mock_dependencies["start_trace"].reset_mock()
    mock_dependencies["end_trace"].reset_mock()
    autolog_module._session_spans.clear()

    original_list_tools = AsyncMock(return_value=["t1", "t2"])
    await list_tools_wrapper(original_list_tools, mcp_instance)
    assert autolog_module._session_spans
    assert mock_dependencies["start_trace"].call_count == 2
    _, kwargs = mock_dependencies["start_trace"].call_args_list[1]
    assert kwargs["name"] == "MyMCPServer.list_tools"
    assert kwargs["span_type"] == SpanType.CHAIN
    assert kwargs["inputs"] == {}
    mock_dependencies["end_trace"].assert_called_once_with(
        ANY, mock_dependencies["live_span"], outputs=["t1", "t2"]
    )
    original_list_tools.assert_awaited_once_with(mcp_instance)


@patch.object(autolog_module, "_patch_agent_methods")
@patch.object(autolog_module, "_patch_instrumented_model")
@patch.object(autolog_module, "_patch_tool_run")
@patch.object(autolog_module, "_patch_mcp_server")
def test_autolog_enables_and_patches(
    mock_patch_mcp, mock_patch_tool, mock_patch_model, mock_patch_agent, mock_dependencies
):
    autolog_module.safe_patch = mock_dependencies["safe_patch"]
    autolog(session_name="TestSession", log_traces=True)
    mock_dependencies["autologging_config_init"].assert_called_once_with(flavor_name=FLAVOUR_NAME)
    assert mock_dependencies["autologging_config_instance"].log_traces
    assert autolog_module._session_name == "TestSession"
    calls = [args for args, _ in mock_dependencies["safe_patch"].call_args_list]
    assert any(
        flavor == FLAVOUR_NAME and target is autolog_module._fluent and method == "end_run"
        for flavor, target, method, _ in calls
    )
    assert any(
        flavor == FLAVOUR_NAME and target is autolog_module.mlflow and method == "end_run"
        for flavor, target, method, _ in calls
    )
    mock_patch_agent.assert_called_once()
    mock_patch_model.assert_called_once()
    mock_patch_tool.assert_called_once()
    mock_patch_mcp.assert_called_once()


@patch.object(autolog_module, "_patch_agent_methods")
@patch.object(autolog_module, "_patch_instrumented_model")
@patch.object(autolog_module, "_patch_tool_run")
@patch.object(autolog_module, "_patch_mcp_server")
def test_autolog_disable(
    mock_patch_mcp, mock_patch_tool, mock_patch_model, mock_patch_agent, mock_dependencies
):
    autolog(disable=True)
    mock_patch_agent.assert_not_called()
    mock_patch_model.assert_not_called()
    mock_patch_tool.assert_not_called()
    mock_patch_mcp.assert_not_called()
