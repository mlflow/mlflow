import asyncio
import inspect
import random
from dataclasses import asdict
from typing import List
from unittest.mock import ANY

import importlib_metadata
import llama_index.core
import openai
import pytest
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import Settings
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from packaging.version import Version

import mlflow
import mlflow.tracking._tracking_service
from mlflow.entities.span import SpanType
from mlflow.entities.span_status import SpanStatusCode
from mlflow.entities.trace import Trace
from mlflow.entities.trace_status import TraceStatus
from mlflow.llama_index.tracer import remove_llama_index_tracer, set_llama_index_tracer
from mlflow.tracking._tracking_service.utils import _use_tracking_uri
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID

llama_core_version = Version(importlib_metadata.version("llama-index-core"))
llama_oai_version = Version(importlib_metadata.version("llama-index-llms-openai"))


@pytest.fixture(autouse=True)
def set_handlers():
    set_llama_index_tracer()
    yield
    # Clean up
    remove_llama_index_tracer()


def _get_all_traces() -> List[Trace]:
    """Utility function to get all traces in the test experiment."""
    return mlflow.MlflowClient().search_traces(experiment_ids=[DEFAULT_EXPERIMENT_ID])


@pytest.mark.parametrize("is_async", [True, False])
def test_trace_llm_complete(is_async):
    # By default llama-index uses "gpt-3.5-turbo" model that only has chat interface,
    # and llama-index redirects completion call to chat endpoint. We use non-chat
    # model here to test trace for completion.
    model_name = "gpt-3.5-turbo-instruct"
    llm = OpenAI(model=model_name)

    response = asyncio.run(llm.acomplete("Hello")) if is_async else llm.complete("Hello")
    assert response.text == "Hello"

    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK

    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "OpenAI.{}complete".format("a" if is_async else "")
    assert spans[0].span_type == SpanType.LLM
    assert spans[0].inputs == {"args": ["Hello"]}
    assert spans[0].outputs["text"] == "Hello"

    attr = spans[0].attributes
    assert (
        attr["usage"].items()
        >= {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        }.items()
    )
    assert attr["prompt"] == "Hello"
    assert attr["invocation_params"]["model_name"] == model_name
    assert attr["model_dict"]["model"] == model_name


def test_trace_llm_complete_stream():
    model_name = "gpt-3.5-turbo"
    llm = OpenAI(model=model_name)

    response_gen = llm.stream_complete("Hello")
    # No trace should be created until the generator is consumed
    assert len(_get_all_traces()) == 0
    assert inspect.isgenerator(response_gen)

    response = [r.text for r in response_gen]
    assert response == ["Hello", "Hello world"]

    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK

    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "OpenAI.stream_complete"
    assert spans[0].span_type == SpanType.LLM
    assert spans[0].inputs == {"args": ["Hello"]}
    assert spans[0].outputs["text"] == "Hello world"

    attr = spans[0].attributes
    assert (
        attr["usage"].items()
        >= {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        }.items()
    )
    assert attr["prompt"] == "Hello"
    assert attr["invocation_params"]["model_name"] == model_name
    assert attr["model_dict"]["model"] == model_name


@pytest.mark.parametrize("is_async", [True, False])
def test_trace_llm_chat(is_async):
    llm = OpenAI()
    message = ChatMessage(role="system", content="Hello")

    response = asyncio.run(llm.achat([message])) if is_async else llm.chat([message])
    assert isinstance(response.message, ChatMessage)
    assert response.message.content == '[{"role": "system", "content": "Hello"}]'

    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK

    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "OpenAI.achat" if is_async else "OpenAI.chat"
    assert spans[0].span_type == SpanType.CHAT_MODEL
    assert spans[0].inputs == {
        "messages": [{"role": "system", "content": "Hello", "additional_kwargs": {}}]
    }
    # `addtional_kwargs` was broken until 0.1.30 release of llama-index-llms-openai
    expected_kwargs = (
        {"completion_tokens": 12, "prompt_tokens": 9, "total_tokens": 21}
        if llama_oai_version >= Version("0.1.30")
        else {}
    )
    assert spans[0].outputs == {
        "message": {
            "role": "assistant",
            "content": '[{"role": "system", "content": "Hello"}]',
            "additional_kwargs": {},
        },
        "raw": ANY,
        "delta": None,
        "logprobs": None,
        "additional_kwargs": expected_kwargs,
    }

    attr = spans[0].attributes
    assert (
        attr["usage"].items()
        >= {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        }.items()
    )
    assert attr["invocation_params"]["model_name"] == llm.metadata.model_name
    assert attr["model_dict"]["model"] == llm.metadata.model_name


def test_trace_llm_chat_stream():
    llm = OpenAI()
    message = ChatMessage(role="system", content="Hello")

    response_gen = llm.stream_chat([message])
    # No trace should be created until the generator is consumed
    assert len(_get_all_traces()) == 0
    assert inspect.isgenerator(response_gen)

    chunks = list(response_gen)
    assert len(chunks) == 2
    assert all(isinstance(c.message, ChatMessage) for c in chunks)
    assert [c.message.content for c in chunks] == ["Hello", "Hello world"]

    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK

    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "OpenAI.stream_chat"
    assert spans[0].span_type == SpanType.CHAT_MODEL
    assert spans[0].inputs == {
        "messages": [{"role": "system", "content": "Hello", "additional_kwargs": {}}]
    }
    # `addtional_kwargs` was broken until 0.1.30 release of llama-index-llms-openai
    expected_kwargs = (
        {"completion_tokens": 12, "prompt_tokens": 9, "total_tokens": 21}
        if llama_oai_version >= Version("0.1.30")
        else {}
    )
    assert spans[0].outputs == {
        "message": {
            "role": "assistant",
            "content": "Hello world",
            "additional_kwargs": {},
        },
        "raw": ANY,
        "delta": " world",
        "logprobs": None,
        "additional_kwargs": expected_kwargs,
    }

    attr = spans[0].attributes
    assert (
        attr["usage"].items()
        >= {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        }.items()
    )
    assert attr["invocation_params"]["model_name"] == llm.metadata.model_name
    assert attr["model_dict"]["model"] == llm.metadata.model_name


@pytest.mark.parametrize("is_stream", [True, False])
def test_trace_llm_error(monkeypatch, is_stream):
    # Disable callback as it doesn't handle error response well
    monkeypatch.setattr(Settings, "callback_manager", None)

    llm = OpenAI(
        api_base="http://localhost:1234/invalid/base",
        timeout=1,
        max_retries=0,
    )
    message = ChatMessage(role="system", content="Hello")

    with pytest.raises(openai.APIConnectionError, match="Connection error."):  # noqa PT012
        if is_stream:
            next(llm.stream_chat([message]))
        else:
            llm.chat([message])

    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.ERROR
    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "OpenAI.stream_chat" if is_stream else "OpenAI.chat"
    assert spans[0].span_type == SpanType.CHAT_MODEL
    assert spans[0].inputs == {"messages": [message.dict()]}
    assert spans[0].outputs is None
    events = traces[0].data.spans[0].events
    assert len(events) == 1
    assert events[0].attributes["exception.message"] == "Connection error."


@pytest.mark.parametrize("is_async", [True, False])
def test_trace_retriever(multi_index, is_async):
    retriever = VectorIndexRetriever(multi_index, similarity_top_k=3)

    if is_async:
        retrieved = asyncio.run(retriever.aretrieve("apple"))
    else:
        retrieved = retriever.retrieve("apple")
    assert len(retrieved) == 1

    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK

    spans = traces[0].data.spans
    assert len(spans) == 4
    for i in range(1, 4):
        assert spans[i].parent_id == spans[i - 1].span_id

    assert spans[0].name == "BaseRetriever.aretrieve" if is_async else "BaseRetriever.retrieve"
    assert spans[0].span_type == SpanType.RETRIEVER
    assert spans[0].inputs == {"str_or_query_bundle": "apple"}
    assert len(spans[0].outputs) == 1
    assert spans[0].outputs[0]["page_content"] == retrieved[0].text

    assert spans[1].name.startswith("VectorIndexRetriever")
    assert spans[1].span_type == SpanType.RETRIEVER
    assert spans[1].inputs["query_bundle"]["query_str"] == "apple"
    assert spans[1].outputs == spans[0].outputs

    assert spans[2].name.startswith("BaseEmbedding")
    assert spans[2].span_type == SpanType.EMBEDDING
    assert spans[2].inputs == {"query": "apple"}
    assert len(spans[2].outputs) == 1536  # embedding size
    assert spans[2].attributes["model_name"] == Settings.embed_model.model_name

    assert spans[3].name.startswith("OpenAIEmbedding")
    assert spans[3].span_type == SpanType.EMBEDDING
    assert spans[3].inputs == {"query": "apple"}
    assert len(spans[3].outputs) == 1536  # embedding size
    assert spans[3].attributes["model_name"] == Settings.embed_model.model_name


@pytest.mark.parametrize("is_stream", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
def test_trace_query_engine(multi_index, is_stream, is_async):
    if is_stream and is_async:
        pytest.skip("Async stream is not supported yet")

    engine = multi_index.as_query_engine(streaming=is_stream)

    if is_stream:
        response = engine.query("Hello")
        assert isinstance(response, StreamingResponse)
        assert len(_get_all_traces()) == 0
        response = "".join(response.response_gen)
        assert response == "Hello world"
    else:
        response = asyncio.run(engine.aquery("Hello")) if is_async else engine.query("Hello")
        assert response.response.startswith('[{"role": "system", "content": "You are an')
        response = asdict(response)
        if Version(llama_index.core.__version__) > Version("0.10.68"):
            response["source_nodes"] = [n.dict() for n in response["source_nodes"]]

    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK

    # Async methods have "a" prefix
    prefix = "a" if is_async else ""

    # Validate span attributes for some key spans
    spans = traces[0].data.spans
    assert spans[0].name == f"BaseQueryEngine.{prefix}query"
    assert spans[0].span_type == SpanType.CHAIN
    assert spans[0].inputs == {"str_or_query_bundle": "Hello"}
    assert spans[0].outputs == response


def test_trace_agent():
    # Mock LLM to return deterministic responses and let the agent use a tool
    class MockLLMForAgent(OpenAI, extra="allow"):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._mock_response = iter(
                [
                    ChatResponse(
                        message=ChatMessage(
                            role="assistant",
                            content=None,
                            additional_kwargs={
                                "tool_calls": [
                                    ChatCompletionMessageToolCall(
                                        id="test",
                                        function={
                                            "name": "add",
                                            "arguments": '{"a": 1, "b": 2}',
                                        },
                                        type="function",
                                    )
                                ]
                            },
                        )
                    ),
                    ChatResponse(
                        message=ChatMessage(
                            role="assistant",
                            content="The result is 3",
                        )
                    ),
                ]
            )

        def chat(self, *args, **kwargs):
            return next(self._mock_response)

    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer"""
        return a + b

    add_tool = FunctionTool.from_defaults(fn=add)

    llm = MockLLMForAgent()
    agent = OpenAIAgent.from_tools([add_tool], llm=llm)
    response = agent.chat("What is 1 + 2?").response

    assert response == "The result is 3"

    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK

    spans = traces[0].data.spans
    name_to_span = {span.name: span for span in spans}
    tool_span = name_to_span["FunctionTool.call"]
    assert tool_span.span_type == SpanType.TOOL
    assert tool_span.inputs == {"kwargs": {"a": 1, "b": 2}}
    assert tool_span.outputs["content"] == "3"
    assert tool_span.attributes["name"] == "add"
    assert tool_span.attributes["description"] is not None
    assert tool_span.attributes["parameters"] is not None


@pytest.mark.parametrize("is_stream", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
def test_trace_chat_engine(multi_index, is_stream, is_async):
    if is_stream and is_async:
        pytest.skip("Async stream is not supported yet")

    engine = multi_index.as_chat_engine()

    if is_stream:
        response_gen = engine.stream_chat("Hello").response_gen
        response = "".join(response_gen)
        assert response == "Hello world"
    else:
        response = asyncio.run(engine.achat("Hello")) if is_async else engine.chat("Hello")
        assert response.response == '[{"role": "user", "content": "Hello"}]'

    # Since chat engine is a complex agent-based system, it is challenging to strictly
    # validate the trace structure and attributes. The detailed validation is done in
    # other tests for individual components.
    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK
    root_span = traces[0].data.spans[0]
    assert root_span.inputs == {"message": "Hello"}


def test_tracer_handle_tracking_uri_update(tmp_path):
    OpenAI().complete("Hello")
    assert len(_get_all_traces()) == 1

    # Set different tracking URI and initialize the tracer
    with _use_tracking_uri(tmp_path / "dummy"):
        assert len(_get_all_traces()) == 0

        # The new trace will be logged to the updated tracking URI
        OpenAI().complete("Hello")
        assert len(_get_all_traces()) == 1


@pytest.mark.skipif(
    llama_core_version >= Version("0.11.10"),
    reason="Workflow tracing does not work correctly in >= 0.11.10 until "
    "https://github.com/run-llama/llama_index/issues/16283 is fixed",
)
@pytest.mark.skipif(
    llama_core_version < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
@pytest.mark.asyncio
async def test_tracer_simple_workflow():
    from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

    class MyWorkflow(Workflow):
        @step
        async def my_step(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="Hi, world!")

    w = MyWorkflow(timeout=10, verbose=False)
    await w.run()

    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK
    assert all(s.status.status_code == SpanStatusCode.OK for s in traces[0].data.spans)


@pytest.mark.skipif(
    llama_core_version >= Version("0.11.10"),
    reason="Workflow tracing does not work correctly in >= 0.11.10 until "
    "https://github.com/run-llama/llama_index/issues/16283 is fixed",
)
@pytest.mark.skipif(
    llama_core_version < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
@pytest.mark.asyncio
async def test_tracer_parallel_workflow():
    from llama_index.core.workflow import (
        Context,
        Event,
        StartEvent,
        StopEvent,
        Workflow,
        step,
    )

    class ProcessEvent(Event):
        data: str

    class ResultEvent(Event):
        result: str

    class ParallelWorkflow(Workflow):
        @step
        async def start(self, ctx: Context, ev: StartEvent) -> ProcessEvent:
            await ctx.set("num_to_collect", len(ev.inputs))
            for item in ev.inputs:
                ctx.send_event(ProcessEvent(data=item))
            return None

        @step(num_workers=3)
        async def process_data(self, ev: ProcessEvent) -> ResultEvent:
            # Simulate some time-consuming processing
            await asyncio.sleep(random.randint(1, 2))
            return ResultEvent(result=ev.data)

        @step
        async def combine_results(self, ctx: Context, ev: ResultEvent) -> StopEvent:
            num_to_collect = await ctx.get("num_to_collect")
            results = ctx.collect_events(ev, [ResultEvent] * num_to_collect)
            if results is None:
                return None

            combined_result = ", ".join(sorted([event.result for event in results]))
            return StopEvent(result=combined_result)

    w = ParallelWorkflow()
    result = await w.run(inputs=["apple", "grape", "orange", "banana"])
    assert result == "apple, banana, grape, orange"

    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK
    for s in traces[0].data.spans:
        assert s.status.status_code == SpanStatusCode.OK

    root_span = traces[0].data.spans[0]
    assert root_span.inputs == {"kwargs": {"inputs": ["apple", "grape", "orange", "banana"]}}
    assert root_span.outputs == "apple, banana, grape, orange"
