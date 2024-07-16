import asyncio
from dataclasses import asdict
from typing import List
from unittest.mock import ANY

import openai
import pytest
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import (
    Settings,
)
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall

import mlflow
from mlflow.entities.span import SpanType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_status import TraceStatus
from mlflow.llama_index.tracer import MlflowEventHandler, MlflowSpanHandler
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID


@pytest.fixture(autouse=True)
def set_handlers():
    span_handler = MlflowSpanHandler(mlflow.MlflowClient())
    event_handler = MlflowEventHandler(span_handler)

    dsp = get_dispatcher()
    dsp.add_span_handler(span_handler)
    dsp.add_event_handler(event_handler)

    yield

    # Clean up
    dsp.span_handlers = []
    dsp.event_handlers = []


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
    assert spans[0].name == "OpenAI.acomplete" if is_async else "OpenAI.complete"

    attr = spans[0].attributes
    assert attr[SpanAttributeKey.SPAN_TYPE] == SpanType.LLM
    assert attr[SpanAttributeKey.INPUTS] == {"args": ["Hello"]}
    assert attr[SpanAttributeKey.OUTPUTS]["text"] == "Hello"
    assert attr["usage"] == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
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

    attr = spans[0].attributes
    assert attr[SpanAttributeKey.SPAN_TYPE] == SpanType.CHAT_MODEL
    assert attr[SpanAttributeKey.INPUTS] == {
        "messages": [{"role": "system", "content": "Hello", "additional_kwargs": {}}]
    }
    assert attr[SpanAttributeKey.OUTPUTS] == {
        "message": {
            "role": "assistant",
            "content": '[{"role": "system", "content": "Hello"}]',
            "additional_kwargs": {},
        },
        "raw": ANY,
        "delta": None,
        "logprobs": None,
        "additional_kwargs": {},
    }
    assert attr["usage"] == {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
    assert attr["invocation_params"]["model_name"] == llm.metadata.model_name
    assert attr["model_dict"]["model"] == llm.metadata.model_name


def test_trace_llm_error(monkeypatch):
    # Disable callback as it doesn't handle error response well
    monkeypatch.setattr(Settings, "callback_manager", None)

    llm = OpenAI(
        api_base="http://localhost:1234/invalid/base",
        timeout=1,
        max_retries=0,
    )

    with pytest.raises(openai.APIConnectionError, match="Connection error."):
        llm.chat([ChatMessage(role="system", content="Hello")])

    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.ERROR
    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "OpenAI.chat"
    assert spans[0].attributes[SpanAttributeKey.SPAN_TYPE] == SpanType.CHAT_MODEL
    assert spans[0].attributes[SpanAttributeKey.INPUTS] == {
        "messages": [{"role": "system", "content": "Hello", "additional_kwargs": {}}]
    }
    assert SpanAttributeKey.OUTPUTS not in spans[0].attributes
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
    assert spans[0].attributes[SpanAttributeKey.SPAN_TYPE] == SpanType.RETRIEVER
    assert spans[0].attributes[SpanAttributeKey.INPUTS] == {"str_or_query_bundle": "apple"}
    output = spans[0].attributes[SpanAttributeKey.OUTPUTS]
    assert len(output) == 1
    assert output[0]["node"]["text"] == retrieved[0].text

    assert spans[1].name.startswith("VectorIndexRetriever")
    assert spans[1].attributes[SpanAttributeKey.SPAN_TYPE] == SpanType.RETRIEVER
    assert spans[1].attributes[SpanAttributeKey.INPUTS]["query_bundle"]["query_str"] == "apple"
    assert (
        spans[1].attributes[SpanAttributeKey.OUTPUTS]
        == spans[0].attributes[SpanAttributeKey.OUTPUTS]
    )

    assert spans[2].name.startswith("BaseEmbedding")
    assert spans[2].attributes[SpanAttributeKey.SPAN_TYPE] == SpanType.EMBEDDING
    assert spans[2].attributes[SpanAttributeKey.INPUTS] == {"query": "apple"}
    assert len(spans[2].attributes[SpanAttributeKey.OUTPUTS]) == 1536  # embedding size
    assert spans[2].attributes["model_name"] == Settings.embed_model.model_name

    assert spans[3].name.startswith("OpenAIEmbedding")
    assert spans[3].attributes[SpanAttributeKey.SPAN_TYPE] == SpanType.EMBEDDING
    assert spans[3].attributes[SpanAttributeKey.INPUTS] == {"query": "apple"}
    assert len(spans[3].attributes[SpanAttributeKey.OUTPUTS]) == 1536  # embedding size
    assert spans[3].attributes["model_name"] == Settings.embed_model.model_name


@pytest.mark.parametrize("is_async", [True, False])
def test_trace_query_engine(multi_index, is_async):
    engine = multi_index.as_query_engine()

    response = asyncio.run(engine.aquery("Hello")) if is_async else engine.query("Hello")
    assert response.response.startswith('[{"role": "system", "content": "You are an')
    response = asdict(response)

    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK

    spans = traces[0].data.spans
    assert len(spans) == 13 if is_async else 14

    # Validate the tree structure
    # 0 -- 1 -- 2 -- 3 -- 4 -- 5
    #   \- 6 -- 7 -- 8
    #             \- 9 -- 10
    #                  \- 11 -- 12 (-- 13)
    for i in range(1, 6):
        assert spans[i].parent_id == spans[i - 1].span_id
    assert spans[6].parent_id == spans[1].span_id
    assert spans[7].parent_id == spans[6].span_id
    assert spans[8].parent_id == spans[7].span_id
    assert spans[9].parent_id == spans[7].span_id
    assert spans[10].parent_id == spans[9].span_id
    assert spans[11].parent_id == spans[9].span_id
    assert spans[12].parent_id == spans[11].span_id
    if not is_async:
        assert spans[13].parent_id == spans[12].span_id

    # Async methods have "a" prefix
    prefix = "a" if is_async else ""

    # Validate span attributes for some key spans
    assert spans[0].name == f"BaseQueryEngine.{prefix}query"
    assert spans[0].attributes[SpanAttributeKey.SPAN_TYPE] == SpanType.CHAIN
    assert spans[0].attributes[SpanAttributeKey.INPUTS] == {"str_or_query_bundle": "Hello"}
    assert spans[0].attributes[SpanAttributeKey.OUTPUTS] == response

    assert spans[2].name == f"BaseRetriever.{prefix}retrieve"
    assert spans[2].attributes[SpanAttributeKey.SPAN_TYPE] == SpanType.RETRIEVER

    assert spans[6].name == f"BaseSynthesizer.{prefix}synthesize"
    assert spans[6].attributes[SpanAttributeKey.SPAN_TYPE] == SpanType.CHAIN
    assert spans[6].attributes[SpanAttributeKey.INPUTS] == {"query": ANY, "nodes": ANY}

    assert spans[9].name == f"Refine.{prefix}get_response"
    assert spans[9].attributes[SpanAttributeKey.SPAN_TYPE] == SpanType.CHAIN
    assert spans[9].attributes[SpanAttributeKey.INPUTS] == {
        "query_str": "Hello",
        "text_chunks": ANY,
        "prev_response": None,
    }

    llm_method_name = f"OpenAI.{prefix}chat"
    assert spans[-1].name == llm_method_name
    assert spans[-1].attributes[SpanAttributeKey.SPAN_TYPE] == SpanType.CHAT_MODEL
    assert spans[-1].attributes[SpanAttributeKey.INPUTS] == {
        "messages": [
            {"role": "system", "content": ANY, "additional_kwargs": {}},
            {"role": "user", "content": ANY, "additional_kwargs": {}},
        ]
    }


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
    assert tool_span.attributes[SpanAttributeKey.SPAN_TYPE] == SpanType.TOOL
    assert tool_span.attributes[SpanAttributeKey.INPUTS] == {"kwargs": {"a": 1, "b": 2}}
    assert tool_span.attributes[SpanAttributeKey.OUTPUTS]["content"] == "3"
    assert tool_span.attributes["name"] == "add"
    assert tool_span.attributes["description"] is not None
    assert tool_span.attributes["parameters"] is not None


@pytest.mark.parametrize("is_async", [True, False])
def test_trace_chat_engine(multi_index, is_async):
    chat_engine = multi_index.as_chat_engine()

    response = asyncio.run(chat_engine.achat("Hello")) if is_async else chat_engine.chat("Hello")
    assert response.response == '[{"role": "user", "content": "Hello"}]'

    # Since chat engine is a complex agent-based system, it is challenging to strictly
    # validate the trace structure and attributes. The detailed validation is done in
    # other tests for individual components.
    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK
