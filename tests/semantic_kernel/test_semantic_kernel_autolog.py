import asyncio
import json
from unittest import mock

import openai
import pytest
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.text_content import TextContent
from semantic_kernel.exceptions import FunctionExecutionException, KernelInvokeException
from semantic_kernel.functions.function_result import FunctionResult
from semantic_kernel.functions.kernel_function_metadata import KernelFunctionMetadata
from semantic_kernel.utils.telemetry.agent_diagnostics import (
    gen_ai_attributes as agent_gen_ai_attributes,
)
from semantic_kernel.utils.telemetry.model_diagnostics import (
    gen_ai_attributes as model_gen_ai_attributes,
)

import mlflow.semantic_kernel
from mlflow.entities import SpanType
from mlflow.entities.span_status import SpanStatusCode
from mlflow.semantic_kernel.autolog import SemanticKernelSpanProcessor
from mlflow.semantic_kernel.tracing_utils import (
    _serialize_chat_output,
    _serialize_kernel_output,
    _serialize_text_output,
)
from mlflow.tracing.constant import (
    SpanAttributeKey,
    TokenUsageKey,
)

from tests.semantic_kernel.resources import (
    _create_and_invoke_chat_agent,
    _create_and_invoke_chat_completion_direct,
    _create_and_invoke_embeddings,
    _create_and_invoke_kernel_complex,
    _create_and_invoke_kernel_function,
    _create_and_invoke_kernel_function_object,
    _create_and_invoke_kernel_simple,
    _create_and_invoke_text_completion,
)
from tests.tracing.helper import get_traces

lock = asyncio.Lock()


@pytest.fixture(autouse=True)
async def lock_fixture():
    async with lock:
        yield


@pytest.mark.asyncio
async def test_sk_invoke_simple(mock_openai):
    mlflow.semantic_kernel.autolog()
    _ = await _create_and_invoke_kernel_simple(mock_openai)

    # Trace
    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.request_id
    assert trace.info.experiment_id == "0"
    assert trace.info.timestamp_ms > 0
    assert len(trace.data.spans) >= 2
    assert trace.data.response
    assert not trace.data.response.startswith("<coroutine")

    trace_inputs_str = trace.info.trace_metadata.get("mlflow.traceInputs", "")
    trace_outputs_str = trace.info.trace_metadata.get("mlflow.traceOutputs", "")

    trace_inputs = json.loads(trace_inputs_str)
    assert trace_inputs["messages"][0]["content"] == "Is sushi the best food ever?"

    trace_outputs = json.loads(trace_outputs_str)

    assert len(trace_outputs["messages"]) == 1
    assert trace_outputs["messages"][0]["role"] == "assistant"
    assert (
        trace_outputs["messages"][0]["content"]
        == '[{"role": "user", "content": "Is sushi the best food ever?"}]'
    )

    root_span = next((s for s in trace.data.spans if s.parent_id is None), None)
    child_span = next((s for s in trace.data.spans if s.parent_id == root_span.span_id), None)

    # Root span
    assert SpanAttributeKey.REQUEST_ID in root_span.attributes
    assert not str(root_span.get_attribute(SpanAttributeKey.OUTPUTS)).startswith("<coroutine")
    output = root_span.get_attribute(SpanAttributeKey.OUTPUTS)
    assert isinstance(output["messages"], list)
    assert len(output["messages"]) == 1
    assert output["messages"][0]["role"] == "assistant"
    assert "content" in output["messages"][0]

    # Child span
    assert child_span.name == "chat.completions gpt-4o-mini"
    assert "gen_ai.operation.name" in child_span.attributes

    inputs = child_span.get_attribute(SpanAttributeKey.INPUTS)
    assert inputs == {"messages": [{"role": "user", "content": "Is sushi the best food ever?"}]}

    outputs = child_span.get_attribute(SpanAttributeKey.OUTPUTS)

    assert outputs == {
        "messages": [
            {
                "role": "assistant",
                "content": '[{"role": "user", "content": "Is sushi the best food ever?"}]',
                "finish_reason": "stop",
            }
        ]
    }

    chat_usage = child_span.get_attribute(SpanAttributeKey.CHAT_USAGE)
    assert chat_usage[TokenUsageKey.INPUT_TOKENS] == 9
    assert chat_usage[TokenUsageKey.OUTPUT_TOKENS] == 12
    assert chat_usage[TokenUsageKey.TOTAL_TOKENS] == 21
    assert child_span.get_attribute(SpanAttributeKey.SPAN_TYPE) == SpanType.CHAT_MODEL


@pytest.mark.asyncio
async def test_sk_invoke_simple_with_sk_initialization_of_tracer(
    mock_openai, dummy_otel_span_processor
):
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace import get_tracer_provider, set_tracer_provider

    resource = Resource.create({ResourceAttributes.SERVICE_NAME: "telemetry-console-quickstart"})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(dummy_otel_span_processor)
    set_tracer_provider(tracer_provider)

    mlflow.semantic_kernel.autolog()
    _tracer_provider = get_tracer_provider()
    assert isinstance(_tracer_provider, TracerProvider)
    span_processors = _tracer_provider._active_span_processor._span_processors
    assert len(span_processors) == 2
    assert any(isinstance(p, SemanticKernelSpanProcessor) for p in span_processors)

    _ = await _create_and_invoke_kernel_simple(mock_openai)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.request_id
    assert len(trace.data.spans) == 2


@pytest.mark.asyncio
async def test_sk_invoke_complex(mock_openai):
    mlflow.semantic_kernel.autolog()
    _ = await _create_and_invoke_kernel_complex(mock_openai)

    # Trace
    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]

    assert trace.data.response
    assert not trace.data.response.startswith("<coroutine")
    assert trace.info.tags.get("mlflow.traceName")

    spans = trace.data.spans
    assert len(spans) == 2

    # Root span
    root_span = next((s for s in trace.data.spans if s.parent_id is None), None)

    assert root_span.name == "execute_tool ChatBot-Chat"
    assert root_span.get_attribute(SpanAttributeKey.REQUEST_ID) == trace.info.request_id
    assert root_span.get_attribute(SpanAttributeKey.SPAN_TYPE)

    # Child span
    child_span = next(s for s in spans if s.parent_id == root_span.span_id)
    assert child_span.name == "chat.completions gpt-4o-mini"
    assert child_span.parent_id == root_span.span_id

    assert child_span.get_attribute(SpanAttributeKey.SPAN_TYPE) == SpanType.CHAT_MODEL
    assert child_span.get_attribute(model_gen_ai_attributes.OPERATION) == "chat.completions"
    assert child_span.get_attribute(model_gen_ai_attributes.SYSTEM) == "openai"
    assert child_span.get_attribute(model_gen_ai_attributes.MODEL) == "gpt-4o-mini"
    assert child_span.get_attribute(model_gen_ai_attributes.RESPONSE_ID) == "chatcmpl-123"
    assert child_span.get_attribute(model_gen_ai_attributes.FINISH_REASON) == "FinishReason.STOP"
    assert child_span.get_attribute(model_gen_ai_attributes.INPUT_TOKENS) == 9
    assert child_span.get_attribute(model_gen_ai_attributes.OUTPUT_TOKENS) == 12

    inputs = child_span.get_attribute(SpanAttributeKey.INPUTS)
    assert any(
        "I want to find a hotel in Seattle with free wifi and a pool." in m.get("content", "")
        for m in inputs.get("messages", [])
    )

    outputs = child_span.get_attribute(SpanAttributeKey.OUTPUTS)
    assert isinstance(outputs["messages"], list)

    chat_usage = child_span.get_attribute(SpanAttributeKey.CHAT_USAGE)
    assert chat_usage[TokenUsageKey.INPUT_TOKENS] == 9
    assert chat_usage[TokenUsageKey.OUTPUT_TOKENS] == 12
    assert chat_usage[TokenUsageKey.TOTAL_TOKENS] == 21


@pytest.mark.asyncio
async def test_sk_invoke_agent(mock_openai):
    mlflow.semantic_kernel.autolog()
    _ = await _create_and_invoke_chat_agent(mock_openai)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    spans = trace.data.spans
    assert len(spans) == 3

    root_span = next(s for s in spans if s.parent_id is None)
    child_span = next(s for s in spans if s.parent_id == root_span.span_id)
    grandchild_span = next(s for s in spans if s.parent_id == child_span.span_id)

    assert root_span.name == "invoke_agent sushi_agent"
    assert root_span.span_type == SpanType.CHAT_MODEL
    assert root_span.get_attribute(model_gen_ai_attributes.OPERATION) == "invoke_agent"
    assert root_span.get_attribute(agent_gen_ai_attributes.AGENT_NAME) == "sushi_agent"

    assert child_span.name == "AutoFunctionInvocationLoop"
    assert child_span.span_type == SpanType.UNKNOWN
    assert "sk.available_functions" in child_span.attributes

    assert grandchild_span.name.startswith("chat.completions gpt-4o-mini")
    assert grandchild_span.span_type == SpanType.CHAT_MODEL
    assert grandchild_span.get_attribute(model_gen_ai_attributes.MODEL) == "gpt-4o-mini"
    inputs = grandchild_span.get_attribute(SpanAttributeKey.INPUTS)
    assert isinstance(inputs["messages"], list)
    outputs = grandchild_span.get_attribute(SpanAttributeKey.OUTPUTS)
    assert isinstance(outputs["messages"], list)
    assert (
        grandchild_span.get_attribute(model_gen_ai_attributes.FINISH_REASON) == "FinishReason.STOP"
    )


@pytest.mark.asyncio
async def test_sk_autolog_trace_on_exception(mock_openai):
    mlflow.semantic_kernel.autolog()
    openai_client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)

    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            service_id="chat-gpt",
            ai_model_id="gpt-4o-mini",
            async_client=openai_client,
        )
    )

    error_message = "thiswillfail"
    with mock.patch.object(
        openai_client.chat.completions, "create", side_effect=RuntimeError(error_message)
    ):
        with pytest.raises(
            KernelInvokeException, match="Error occurred while invoking function"
        ) as exc_info:
            await kernel.invoke_prompt("Hello?")

        assert isinstance(exc_info.value.__cause__, FunctionExecutionException)

    traces = get_traces()
    assert traces, "No traces recorded"
    assert len(traces) == 1
    trace = traces[0]
    assert len(trace.data.spans) == 2
    assert trace.info.status == "ERROR"

    parent_span = next(s for s in trace.data.spans if s.parent_id is None)
    child_span = next(s for s in trace.data.spans if s.parent_id == parent_span.span_id)
    assert child_span.status.status_code == SpanStatusCode.ERROR
    assert child_span.events[0].name == "exception"
    assert error_message in child_span.events[0].attributes["exception.message"]


@pytest.mark.asyncio
async def test_tracing_autolog_with_active_span(mock_openai):
    mlflow.semantic_kernel.autolog()

    with mlflow.start_span("parent"):
        response = await _create_and_invoke_kernel_simple(mock_openai)

    assert isinstance(response, FunctionResult)

    traces = get_traces()
    assert len(traces) == 1

    trace = traces[0]
    spans = trace.data.spans
    assert len(spans) == 3

    assert trace.info.request_id is not None
    assert trace.info.status == "OK"
    assert trace.info.tags["mlflow.traceName"] == "parent"

    parent = trace.data.spans[0]
    assert parent.name == "parent"
    assert parent.parent_id is None
    assert parent.get_attribute(SpanAttributeKey.SPAN_TYPE) == SpanType.UNKNOWN

    child = trace.data.spans[1]
    assert child.parent_id == parent.span_id
    assert child.get_attribute(SpanAttributeKey.SPAN_TYPE) == SpanType.CHAT_MODEL

    grandchild = trace.data.spans[2]
    assert grandchild.name == "chat.completions gpt-4o-mini"
    assert grandchild.parent_id == child.span_id
    assert grandchild.get_attribute(SpanAttributeKey.SPAN_TYPE) == SpanType.CHAT_MODEL
    assert grandchild.get_attribute(model_gen_ai_attributes.OPERATION) == "chat.completions"
    assert grandchild.get_attribute(model_gen_ai_attributes.SYSTEM) == "openai"
    assert grandchild.get_attribute(model_gen_ai_attributes.MODEL) == "gpt-4o-mini"
    assert (
        grandchild.get_attribute(SpanAttributeKey.INPUTS)["messages"][0]["content"]
        == "Is sushi the best food ever?"
    )


@pytest.mark.asyncio
async def test_tracing_attribution_with_threaded_calls(mock_openai):
    mlflow.semantic_kernel.autolog()

    n = 3
    openai_client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)

    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            service_id="chat-gpt",
            ai_model_id="gpt-4o-mini",
            async_client=openai_client,
        )
    )

    async def call(prompt: str):
        return await kernel.invoke_prompt(prompt)

    prompts = [f"What is this number: {i}" for i in range(n)]
    _ = await asyncio.gather(*(call(p) for p in prompts))

    traces = get_traces()
    assert len(traces) == n

    unique_messages = set()
    for trace in traces:
        spans = trace.data.spans
        assert len(spans) == 2

        parent_span = next((s for s in spans if s.parent_id is None), None)
        assert parent_span
        child_span = next((s for s in spans if s.parent_id is not None), None)
        assert child_span

        inputs = child_span.get_attribute(SpanAttributeKey.INPUTS)
        assert inputs
        message = inputs["messages"][0]["content"]
        assert message.startswith("What is this number: ")
        unique_messages.add(message)
        assert child_span.get_attribute(SpanAttributeKey.OUTPUTS)

    assert len(unique_messages) == n


@pytest.mark.asyncio
async def test_sk_output_serialization():
    chat_msg = ChatMessageContent(role="assistant", content="Hello", finish_reason="stop")
    result = _serialize_chat_output(chat_msg)
    assert result["messages"][0]["content"] == "Hello"
    assert result["messages"][0]["role"] == "assistant"

    msgs = [
        ChatMessageContent(role="assistant", content=f"Message {i}", finish_reason="stop")
        for i in range(2)
    ]
    result = _serialize_chat_output(msgs)
    assert len(result["messages"]) == 2
    assert result["messages"][0]["content"] == "Message 0"
    assert result["messages"][1]["content"] == "Message 1"

    assert _serialize_chat_output(None) is None
    assert _serialize_chat_output("not a chat") == "null"

    # Test text output serialization
    text_content = TextContent(text="Hello world")
    result = _serialize_text_output(text_content)
    parsed = json.loads(result)
    assert parsed["type"] == "text"
    assert parsed["text"] == "Hello world"

    assert json.loads(_serialize_text_output(None)) is None
    assert json.loads(_serialize_text_output("plain string")) == "plain string"

    func_metadata = KernelFunctionMetadata(name="test_function", is_prompt=False)
    kernel_result = FunctionResult(function=func_metadata, value="result value")
    result = _serialize_kernel_output(kernel_result)
    assert result == "result value"

    nested_result = FunctionResult(function=func_metadata, value={"key": "value"})
    result = _serialize_kernel_output(nested_result)
    assert result == {"key": "value"}

    assert json.loads(_serialize_kernel_output(None)) is None
    assert _serialize_kernel_output([1, 2, 3]) == [1, 2, 3]


@pytest.mark.parametrize(
    ("create_and_invoke_func", "span_name_pattern", "expected_span_input_keys"),
    [
        (
            _create_and_invoke_kernel_simple,
            "chat.completions",
            ["messages"],
        ),
        (
            _create_and_invoke_kernel_function,
            "execute_tool",
            ["messages"],
        ),
        (
            _create_and_invoke_text_completion,
            "text.completions",
            ["messages"],
        ),
        (
            _create_and_invoke_chat_completion_direct,
            "chat.completions",
            ["messages"],
        ),
    ],
)
@pytest.mark.asyncio
async def test_sk_input_parsing(
    mock_openai, create_and_invoke_func, span_name_pattern, expected_span_input_keys
):
    mlflow.semantic_kernel.autolog()

    _ = await create_and_invoke_func(mock_openai)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]

    target_span = None
    for span in trace.data.spans:
        if span_name_pattern in span.name:
            target_span = span
            break

    assert target_span is not None, f"No span found with pattern '{span_name_pattern}'"

    span_inputs = target_span.get_attribute("mlflow.spanInputs")
    if isinstance(span_inputs, str):
        span_inputs = json.loads(span_inputs)

    for key in expected_span_input_keys:
        assert key in span_inputs, (
            f"Expected '{key}' in span inputs for {target_span.name}, got: {span_inputs}"
        )


@pytest.mark.asyncio
async def test_sk_invoke_with_kernel_arguments(mock_openai):
    mlflow.semantic_kernel.autolog()
    _ = await _create_and_invoke_kernel_function_object(mock_openai)
    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]

    # Check that kernel arguments were passed through to the prompt
    child_span = next(s for s in trace.data.spans if "chat.completions" in s.name)
    inputs = child_span.get_attribute(SpanAttributeKey.INPUTS)
    assert inputs["messages"][0]["content"] == "Add 5 and 3"


@pytest.mark.asyncio
async def test_sk_embeddings(mock_openai):
    mlflow.semantic_kernel.autolog()

    result = await _create_and_invoke_embeddings(mock_openai)

    assert result is not None
    assert len(result) == 3

    # NOTE: Semantic Kernel currently does not instrument embeddings with OpenTelemetry
    # spans, so no traces are generated for embedding operations
    traces = get_traces()
    assert len(traces) == 0


@pytest.mark.asyncio
async def test_kernel_invoke_function_object(mock_openai):
    """Test that kernel.invoke with function object works correctly"""
    mlflow.semantic_kernel.autolog()

    _ = await _create_and_invoke_kernel_function_object(mock_openai)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]

    # Verify trace structure
    assert len(trace.data.spans) == 2

    # Root span should be execute_tool
    root_span = next(s for s in trace.data.spans if s.parent_id is None)
    assert "execute_tool" in root_span.name
    assert "MathPlugin-Add" in root_span.name

    # Child span should be chat completion
    child_span = next(s for s in trace.data.spans if s.parent_id == root_span.span_id)
    assert "chat.completions" in child_span.name

    # Verify inputs are captured (same as other kernel.invoke patterns)
    root_inputs = root_span.get_attribute(SpanAttributeKey.INPUTS)
    assert "messages" in root_inputs
    assert root_inputs["messages"][0]["content"] == "Add 5 and 3"
