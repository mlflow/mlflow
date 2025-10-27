import asyncio
from unittest import mock

import openai
import pytest
from semantic_kernel import Kernel
from semantic_kernel.agents import AgentResponseItem
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.exceptions import FunctionExecutionException, KernelInvokeException
from semantic_kernel.functions.function_result import FunctionResult
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
from mlflow.tracing.constant import (
    SpanAttributeKey,
    TokenUsageKey,
)

from tests.semantic_kernel.resources import (
    _create_and_invoke_chat_agent,
    _create_and_invoke_chat_completion_direct,
    _create_and_invoke_embeddings,
    _create_and_invoke_kernel_complex,
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


@pytest.fixture(params=[True, False])
def with_openai_autolog(request):
    # Test with OpenAI autologging enabled and disabled
    if request.param:
        mlflow.openai.autolog()
    else:
        mlflow.openai.autolog(disable=True)

    return request.param


@pytest.mark.asyncio
async def test_sk_invoke_simple(mock_openai, with_openai_autolog):
    mlflow.semantic_kernel.autolog()
    result = await _create_and_invoke_kernel_simple(mock_openai)

    # The mock OpenAI endpoint echos the user message back
    prompt = "Is sushi the best food ever?"
    expected_content = '[{"role": "user", "content": "Is sushi the best food ever?"}]'

    # Validate the result is not mutated by tracing logic
    assert isinstance(result, FunctionResult)
    assert isinstance(result.value[0], ChatMessageContent)
    assert result.value[0].items[0].text == expected_content

    # Trace
    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.request_id
    assert trace.info.experiment_id == "0"
    assert trace.info.timestamp_ms > 0
    assert trace.info.status == "OK"
    assert "Is sushi the best food ever?" in trace.info.request_preview
    assert "Is sushi the best food ever?" in trace.info.response_preview

    spans = trace.data.spans
    assert len(spans) == (5 if with_openai_autolog else 4)

    # Kernel.invoke_prompt
    assert spans[0].name == "Kernel.invoke_prompt"
    assert spans[0].span_type == SpanType.AGENT
    assert spans[0].inputs == {"prompt": prompt}
    assert spans[0].outputs == [{"role": "assistant", "content": expected_content}]

    # Kernel.invoke_prompt
    assert spans[1].name == "Kernel.invoke"
    assert spans[1].span_type == SpanType.AGENT
    assert spans[1].inputs["function"] is not None
    assert spans[1].outputs == [{"role": "assistant", "content": expected_content}]

    # Execute LLM as a tool
    assert spans[2].name.startswith("execute_tool")
    assert spans[2].span_type == SpanType.TOOL

    # Actual LLM call
    assert spans[3].name == "chat.completions gpt-4o-mini"
    assert "gen_ai.operation.name" in spans[3].attributes
    assert spans[3].inputs == {"messages": [{"role": "user", "content": prompt}]}
    assert spans[3].outputs == {"messages": [{"role": "assistant", "content": expected_content}]}

    chat_usage = spans[3].get_attribute(SpanAttributeKey.CHAT_USAGE)
    assert chat_usage[TokenUsageKey.INPUT_TOKENS] == 9
    assert chat_usage[TokenUsageKey.OUTPUT_TOKENS] == 12
    assert chat_usage[TokenUsageKey.TOTAL_TOKENS] == 21
    assert spans[3].get_attribute(SpanAttributeKey.SPAN_TYPE) == SpanType.CHAT_MODEL

    # OpenAI autologging
    if with_openai_autolog:
        assert spans[4].name == "AsyncCompletions"
        assert spans[4].span_type == SpanType.CHAT_MODEL
        assert spans[4].parent_id == spans[3].span_id
        assert spans[4].inputs == {
            "messages": [{"role": "user", "content": prompt}],
            "model": "gpt-4o-mini",
            "stream": False,
        }
        assert spans[4].get_attribute(SpanAttributeKey.CHAT_USAGE) == {
            "input_tokens": 9,
            "output_tokens": 12,
            "total_tokens": 21,
        }

    # Trace level token usage should not double-count
    assert trace.info.token_usage == {
        "input_tokens": 9,
        "output_tokens": 12,
        "total_tokens": 21,
    }


@pytest.mark.asyncio
async def test_sk_invoke_simple_with_sk_initialization_of_tracer(mock_openai):
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace import get_tracer_provider, set_tracer_provider

    resource = Resource.create({ResourceAttributes.SERVICE_NAME: "telemetry-console-quickstart"})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
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
    assert len(trace.data.spans) == 4


@pytest.mark.asyncio
async def test_sk_invoke_complex(mock_openai):
    mlflow.semantic_kernel.autolog()
    result = await _create_and_invoke_kernel_complex(mock_openai)

    # Validate the result is not mutated by tracing logic
    assert isinstance(result, FunctionResult)
    assert isinstance(result.value[0], ChatMessageContent)
    assert result.value[0].items[0].text.startswith('[{"role": "system",')

    # Trace
    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert len(spans) == 3  # Kernel.invoke, execute_tool, chat.completions

    kernel_span, tool_span, chat_span = spans
    assert kernel_span.name == "Kernel.invoke"
    assert kernel_span.span_type == SpanType.AGENT
    function_metadata = kernel_span.inputs["function"]["metadata"]
    assert function_metadata["name"] == "Chat"
    assert function_metadata["plugin_name"] == "ChatBot"
    prompt = kernel_span.inputs["function"]["prompt_template"]["prompt_template_config"]
    assert prompt["template"] == "{{$chat_history}}{{$user_input}}"
    arguments = kernel_span.inputs["arguments"]
    assert arguments["user_input"] == "I want to find a hotel in Seattle with free wifi and a pool."
    assert len(arguments["chat_history"]) == 2

    assert tool_span.name == "execute_tool ChatBot-Chat"
    assert tool_span.span_type == SpanType.TOOL
    assert tool_span.parent_id == kernel_span.span_id

    assert chat_span.name == "chat.completions gpt-4o-mini"
    assert chat_span.parent_id == tool_span.span_id
    assert chat_span.span_type == SpanType.CHAT_MODEL
    assert chat_span.get_attribute(model_gen_ai_attributes.OPERATION) == "chat.completions"
    assert chat_span.get_attribute(model_gen_ai_attributes.SYSTEM) == "openai"
    assert chat_span.get_attribute(model_gen_ai_attributes.MODEL) == "gpt-4o-mini"
    assert chat_span.get_attribute(model_gen_ai_attributes.RESPONSE_ID) == "chatcmpl-123"
    assert chat_span.get_attribute(model_gen_ai_attributes.FINISH_REASON) == "FinishReason.STOP"
    assert chat_span.get_attribute(model_gen_ai_attributes.INPUT_TOKENS) == 9
    assert chat_span.get_attribute(model_gen_ai_attributes.OUTPUT_TOKENS) == 12

    assert any(
        "I want to find a hotel in Seattle with free wifi and a pool." in m.get("content", "")
        for m in chat_span.inputs.get("messages", [])
    )
    assert isinstance(chat_span.outputs["messages"], list)

    chat_usage = chat_span.get_attribute(SpanAttributeKey.CHAT_USAGE)
    assert chat_usage[TokenUsageKey.INPUT_TOKENS] == 9
    assert chat_usage[TokenUsageKey.OUTPUT_TOKENS] == 12
    assert chat_usage[TokenUsageKey.TOTAL_TOKENS] == 21


@pytest.mark.asyncio
async def test_sk_invoke_agent(mock_openai):
    mlflow.semantic_kernel.autolog()
    result = await _create_and_invoke_chat_agent(mock_openai)
    assert isinstance(result, AgentResponseItem)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    spans = trace.data.spans
    assert len(spans) == 3

    root_span, child_span, grandchild_span = spans

    assert root_span.name == "invoke_agent sushi_agent"
    assert root_span.span_type == SpanType.AGENT
    assert root_span.get_attribute(model_gen_ai_attributes.OPERATION) == "invoke_agent"
    assert root_span.get_attribute(agent_gen_ai_attributes.AGENT_NAME) == "sushi_agent"

    assert child_span.name == "AutoFunctionInvocationLoop"
    assert child_span.span_type == SpanType.UNKNOWN
    assert "sk.available_functions" in child_span.attributes

    assert grandchild_span.name.startswith("chat.completions gpt-4o-mini")
    assert grandchild_span.span_type == SpanType.CHAT_MODEL
    assert grandchild_span.get_attribute(model_gen_ai_attributes.MODEL) == "gpt-4o-mini"
    assert isinstance(grandchild_span.inputs["messages"], list)
    assert isinstance(grandchild_span.outputs["messages"], list)
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
    assert len(trace.data.spans) == 4
    assert trace.info.status == "ERROR"

    _, _, _, llm_span = trace.data.spans
    assert llm_span.status.status_code == SpanStatusCode.ERROR
    assert llm_span.events[0].name == "exception"
    assert error_message in llm_span.events[0].attributes["exception.message"]


@pytest.mark.asyncio
async def test_tracing_autolog_with_active_span(mock_openai, with_openai_autolog):
    mlflow.semantic_kernel.autolog()

    with mlflow.start_span("parent"):
        response = await _create_and_invoke_kernel_simple(mock_openai)

    assert isinstance(response, FunctionResult)

    traces = get_traces()
    assert len(traces) == 1

    trace = traces[0]
    spans = trace.data.spans
    assert len(spans) == (6 if with_openai_autolog else 5)

    assert trace.info.request_id is not None
    assert trace.info.status == "OK"
    assert trace.info.tags["mlflow.traceName"] == "parent"

    parent = trace.data.spans[0]
    assert parent.name == "parent"
    assert parent.parent_id is None
    assert parent.span_type == SpanType.UNKNOWN

    assert spans[1].name == "Kernel.invoke_prompt"
    assert spans[1].parent_id == parent.span_id
    assert spans[2].name == "Kernel.invoke"
    assert spans[2].parent_id == spans[1].span_id
    assert spans[3].name.startswith("execute_tool")
    assert spans[3].parent_id == spans[2].span_id
    assert spans[4].name == "chat.completions gpt-4o-mini"
    assert spans[4].parent_id == spans[3].span_id

    if with_openai_autolog:
        assert spans[5].name == "AsyncCompletions"
        assert spans[5].parent_id == spans[4].span_id


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
        assert len(spans) == 4

        assert spans[0].span_type == SpanType.AGENT
        assert spans[1].span_type == SpanType.AGENT
        assert spans[2].span_type == SpanType.TOOL
        assert spans[3].span_type == SpanType.CHAT_MODEL

        message = spans[3].inputs["messages"][0]["content"]
        assert message.startswith("What is this number: ")
        unique_messages.add(message)
        assert spans[3].outputs["messages"][0]["content"]

    assert len(unique_messages) == n


@pytest.mark.parametrize(
    ("create_and_invoke_func", "span_name_pattern", "expected_span_input_keys"),
    [
        (
            _create_and_invoke_kernel_simple,
            "chat.completions",
            ["messages"],
        ),
        (
            _create_and_invoke_text_completion,
            "text.completions",
            # Text completion input should be stored as a raw string
            None,
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
    if expected_span_input_keys:
        for key in expected_span_input_keys:
            assert key in target_span.inputs, (
                f"Expected '{key}' in span inputs for {target_span.name}, got: {target_span.inputs}"
            )
    else:
        assert isinstance(target_span.inputs, str)


@pytest.mark.asyncio
async def test_sk_invoke_with_kernel_arguments(mock_openai):
    mlflow.semantic_kernel.autolog()
    _ = await _create_and_invoke_kernel_function_object(mock_openai)
    traces = get_traces()
    assert len(traces) == 1

    # Check that kernel arguments were passed through to the prompt
    child_span = next(s for s in traces[0].data.spans if "chat.completions" in s.name)
    assert child_span.inputs["messages"][0]["content"] == "Add 5 and 3"


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

    await _create_and_invoke_kernel_function_object(mock_openai)

    traces = get_traces()
    assert len(traces) == 1

    # Verify trace structure
    assert len(traces[0].data.spans) == 3

    # Root span should be execute_tool
    kernel_span, tool_span, chat_span = traces[0].data.spans

    assert kernel_span.name == "Kernel.invoke"
    assert kernel_span.span_type == SpanType.AGENT
    assert kernel_span.inputs["function"] is not None
    assert kernel_span.outputs is not None

    assert tool_span.name == "execute_tool MathPlugin-Add"
    assert tool_span.span_type == SpanType.TOOL

    # Child span should be chat completion
    assert chat_span.name == "chat.completions gpt-4o-mini"
    assert chat_span.span_type == SpanType.CHAT_MODEL
