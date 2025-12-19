from unittest.mock import patch

from haystack import Document, Pipeline, component
from haystack.components.rankers import LostInTheMiddleRanker
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces


@component
class Add:
    def run(self, a: int, b: int):
        return {"sum": a + b}


@component
class Multiply:
    def run(self, value: int, factor: int):
        return {"product": value * factor}


def test_haystack_autolog_single_trace():
    mlflow.haystack.autolog()

    pipe = Pipeline()
    pipe.add_component("adder", Add())
    pipe.run({"adder": {"a": 1, "b": 2}})

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert spans[0].span_type == SpanType.CHAIN
    assert spans[0].name == "haystack.pipeline.run"
    assert spans[0].inputs == {"adder": {"a": 1, "b": 2}}
    assert spans[0].outputs == {"adder": {"sum": 3}}
    assert spans[1].span_type == SpanType.TOOL
    assert spans[1].name == "Add"
    assert spans[1].inputs == {"a": 1, "b": 2}
    assert spans[1].outputs == {"sum": 3}

    mlflow.haystack.autolog(disable=True)
    pipe.run({"adder": {"a": 3, "b": 4}})
    assert len(get_traces()) == 1


def test_pipeline_with_multiple_components_single_trace():
    mlflow.haystack.autolog()

    pipe = Pipeline()
    pipe.add_component("adder", Add())
    pipe.add_component("multiplier", Multiply())

    pipe.run({"adder": {"a": 1, "b": 2}, "multiplier": {"value": 3, "factor": 4}})

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert spans[0].span_type == SpanType.CHAIN
    assert spans[0].name == "haystack.pipeline.run"
    assert spans[1].span_type == SpanType.TOOL
    assert spans[2].span_type == SpanType.TOOL
    assert spans[1].name == "Add"
    assert spans[2].name == "Multiply"
    assert spans[1].inputs == {"a": 1, "b": 2}
    assert spans[1].outputs == {"sum": 3}
    assert spans[2].inputs == {"value": 3, "factor": 4}
    assert spans[2].outputs == {"product": 12}

    mlflow.haystack.autolog(disable=True)
    pipe.run({"adder": {"a": 1, "b": 2}, "multiplier": {"value": 3, "factor": 4}})

    traces = get_traces()
    assert len(traces) == 1


def test_token_usage_parsed_for_llm_component():
    mlflow.haystack.autolog()

    @component
    class MyLLM:
        def run(self, prompt: str):
            return {}

    pipe = Pipeline()
    pipe.add_component("my_llm", MyLLM())

    output = {
        "replies": [
            {
                "content": [{"text": "hi"}],
                "meta": {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}},
            }
        ]
    }

    with patch.object(MyLLM, "run", return_value=output):
        pipe.run({"my_llm": {"prompt": "hello"}})

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[1]
    assert span.span_type == SpanType.LLM
    assert span.name == "MyLLM"
    assert span.attributes[SpanAttributeKey.CHAT_USAGE] == {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
    }

    mlflow.haystack.autolog(disable=True)

    traces = get_traces()
    with patch.object(MyLLM, "run", return_value=output):
        pipe.run({"my_llm": {"prompt": "hello"}})

    assert len(traces) == 1


def test_autolog_disable():
    mlflow.haystack.autolog()

    pipe1 = Pipeline()
    pipe1.add_component("adder", Add())
    pipe1.run({"adder": {"a": 1, "b": 2}})
    assert len(get_traces()) == 1

    mlflow.haystack.autolog(disable=True)
    pipe2 = Pipeline()
    pipe2.add_component("adder", Add())
    pipe2.run({"adder": {"a": 2, "b": 3}})
    assert len(get_traces()) == 1


def test_in_memory_retriever_component_traced():
    mlflow.set_experiment("haystack_retriever")
    mlflow.haystack.autolog()

    store = InMemoryDocumentStore()
    store.write_documents([Document(content="foo")])
    pipe = Pipeline()
    pipe.add_component("retriever", InMemoryBM25Retriever(document_store=store))
    pipe.run({"retriever": {"query": "foo"}})

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[1]
    assert span.span_type == SpanType.RETRIEVER
    assert span.name == "InMemoryBM25Retriever"
    assert span.outputs["documents"][0]["content"] == "foo"


def test_multiple_components_in_pipeline_reranker():
    mlflow.haystack.autolog()

    pipe = Pipeline()
    store = InMemoryDocumentStore()
    store.write_documents([Document(content="foo")])

    pipe.add_component("retriever", InMemoryBM25Retriever(document_store=store))
    pipe.add_component("reranker", LostInTheMiddleRanker())

    pipe.connect("retriever.documents", "reranker.documents")

    pipe.run({"retriever": {"query": "foo"}})

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert spans[0].span_type == SpanType.CHAIN
    assert spans[0].name == "haystack.pipeline.run"
    assert spans[1].name == "InMemoryBM25Retriever"
    assert spans[2].name == "LostInTheMiddleRanker"
    assert spans[1].span_type == SpanType.RETRIEVER
    assert spans[2].span_type == SpanType.RERANKER
    assert spans[1].inputs["query"] == "foo"
    assert spans[2].inputs["documents"][0]["content"] == "foo"

    mlflow.haystack.autolog(disable=True)
    pipe.run({"retriever": {"query": "foo"}})
    assert len(get_traces()) == 1
