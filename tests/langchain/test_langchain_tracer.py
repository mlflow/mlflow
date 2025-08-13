import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import langchain
import pydantic
import pytest
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.llms.openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from packaging.version import Version

import mlflow
from mlflow.entities import Document as MlflowDocument
from mlflow.entities import Trace
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatus, SpanStatusCode
from mlflow.exceptions import MlflowException
from mlflow.langchain.langchain_tracer import MlflowLangchainTracer
from mlflow.langchain.model import _LangChainModelWrapper
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.provider import trace_disabled
from mlflow.utils import IS_PYDANTIC_V2_OR_NEWER

from tests.tracing.helper import get_traces

# The mock OpenAI endpoint simply echos the prompt back as the completion.
# So the expected output will be the prompt itself.
TEST_CONTENT = "What is MLflow?"


def create_openai_llmchain():
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is {product}?",
    )
    return LLMChain(llm=llm, prompt=prompt)


def create_retriever():
    loader = TextLoader("tests/scoring/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = FakeEmbeddings(size=5)
    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever()


def _validate_trace_json_serialization(trace):
    trace_dict = trace.to_dict()
    trace_from_dict = Trace.from_dict(trace_dict)
    trace_json = trace.to_json()
    trace_from_json = Trace.from_json(trace_json)
    for loaded_trace in [trace_from_dict, trace_from_json]:
        assert trace.info == loaded_trace.info
        assert trace.data.request == loaded_trace.data.request
        assert trace.data.response == loaded_trace.data.response
        assert len(trace.data.spans) == len(loaded_trace.data.spans)
        for i in range(len(trace.data.spans)):
            for attr in [
                "name",
                "request_id",
                "span_id",
                "start_time_ns",
                "end_time_ns",
                "parent_id",
                "status",
                "inputs",
                "outputs",
                "_trace_id",
                "attributes",
                "events",
            ]:
                assert getattr(trace.data.spans[i], attr) == getattr(
                    loaded_trace.data.spans[i], attr
                )


def test_llm_success():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    callback.on_llm_start(
        {},
        ["test prompt"],
        run_id=run_id,
        name="test_llm",
    )

    callback.on_llm_new_token("test", run_id=run_id)

    callback.on_llm_end(LLMResult(generations=[[{"text": "generated text"}]]), run_id=run_id)
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 1
    llm_span = trace.data.spans[0]

    assert llm_span.name == "test_llm"

    assert llm_span.span_type == "LLM"
    assert llm_span.start_time_ns is not None
    assert llm_span.end_time_ns is not None
    assert llm_span.status == SpanStatus(SpanStatusCode.OK)
    assert llm_span.inputs == ["test prompt"]
    assert llm_span.outputs["generations"][0][0]["text"] == "generated text"
    assert llm_span.events[0].name == "new_token"

    _validate_trace_json_serialization(trace)


def test_llm_error():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    callback.on_llm_start(
        {},
        ["test prompt"],
        run_id=run_id,
        name="test_llm",
    )
    mock_error = Exception("mock exception")
    callback.on_llm_error(error=mock_error, run_id=run_id)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    error_event = SpanEvent.from_exception(mock_error)
    assert len(trace.data.spans) == 1
    llm_span = trace.data.spans[0]
    assert llm_span.status.status_code == SpanStatusCode.ERROR
    assert llm_span.status.description == str(mock_error)
    assert llm_span.inputs == ["test prompt"]
    assert llm_span.outputs is None
    # timestamp is auto-generated when converting the error to event
    assert llm_span.events[0].name == error_event.name
    assert llm_span.events[0].attributes == error_event.attributes

    _validate_trace_json_serialization(trace)


def test_llm_internal_exception():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    callback.on_llm_start(
        {},
        ["test prompt"],
        run_id=run_id,
        name="test_llm",
    )
    try:
        with pytest.raises(
            Exception,
            match="Span for run_id dummy not found.",
        ):
            callback.on_llm_end(LLMResult(generations=[[{"text": "generated"}]]), run_id="dummy")
    finally:
        callback.flush()


def test_chat_model():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    input_messages = [SystemMessage("system prompt"), HumanMessage("test prompt")]
    callback.on_chat_model_start(
        {},
        [input_messages],
        run_id=run_id,
        name="test_chat_model",
    )
    callback.on_llm_end(
        LLMResult(generations=[[{"text": "generated text"}]]),
        run_id=run_id,
    )

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 1
    chat_model_span = trace.data.spans[0]
    assert chat_model_span.name == "test_chat_model"
    assert chat_model_span.span_type == "CHAT_MODEL"
    assert chat_model_span.status.status_code == SpanStatusCode.OK
    assert chat_model_span.inputs == [[msg.dict() for msg in input_messages]]
    assert chat_model_span.outputs["generations"][0][0]["text"] == "generated text"


def test_chat_model_with_tool():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    input_messages = [HumanMessage("test prompt")]
    # OpenAI tool format
    tool_definition = {
        "type": "function",
        "function": {
            "name": "GetWeather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "properties": {
                    "location": {
                        "description": "The city and state, e.g. San Francisco, CA",
                        "type": "string",
                    }
                },
                "required": ["location"],
                "type": "object",
            },
        },
    }
    callback.on_chat_model_start(
        {},
        [input_messages],
        run_id=run_id,
        name="test_chat_model",
        invocation_params={"tools": [tool_definition]},
    )
    callback.on_llm_end(
        LLMResult(generations=[[{"text": "generated text"}]]),
        run_id=run_id,
    )

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 1
    chat_model_span = trace.data.spans[0]
    assert chat_model_span.status.status_code == SpanStatusCode.OK
    assert chat_model_span.get_attribute(SpanAttributeKey.CHAT_TOOLS) == [tool_definition]


def test_chat_model_with_non_openai_tool():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    input_messages = [HumanMessage("test prompt")]
    # Anthropic tool format
    tool_definition = {
        "name": "get_weather",
        "description": "Get the weather for a location.",
        "input_schema": {
            "properties": {
                "location": {
                    "description": "The city and state, e.g. San Francisco, CA",
                    "type": "string",
                }
            },
            "required": ["location"],
            "type": "object",
        },
    }
    callback.on_chat_model_start(
        {},
        [input_messages],
        run_id=run_id,
        name="test_chat_model",
        invocation_params={"tools": [tool_definition]},
    )
    callback.on_llm_end(
        LLMResult(generations=[[{"text": "generated text"}]]),
        run_id=run_id,
    )

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 1
    chat_model_span = trace.data.spans[0]
    assert chat_model_span.status.status_code == SpanStatusCode.OK
    assert chat_model_span.get_attribute(SpanAttributeKey.CHAT_TOOLS) == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location.",
            },
        }
    ]


def test_retriever_success():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    callback.on_retriever_start(
        {},
        query="test query",
        run_id=run_id,
        name="test_retriever",
    )

    documents = [
        Document(
            page_content="document content 1",
            metadata={"chunk_id": "1", "doc_uri": "uri1"},
        ),
        Document(
            page_content="document content 2",
            metadata={"chunk_id": "2", "doc_uri": "uri2"},
        ),
    ]
    callback.on_retriever_end(documents, run_id=run_id)
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 1
    retriever_span = trace.data.spans[0]

    assert retriever_span.name == "test_retriever"
    assert retriever_span.span_type == "RETRIEVER"
    assert retriever_span.inputs == "test query"
    assert retriever_span.outputs == [
        MlflowDocument.from_langchain_document(doc).to_dict() for doc in documents
    ]
    assert retriever_span.start_time_ns is not None
    assert retriever_span.end_time_ns is not None
    assert retriever_span.status.status_code == SpanStatusCode.OK

    _validate_trace_json_serialization(trace)


def test_retriever_error():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    callback.on_retriever_start(
        {},
        query="test query",
        run_id=run_id,
        name="test_retriever",
    )
    mock_error = Exception("mock exception")
    callback.on_retriever_error(error=mock_error, run_id=run_id)
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 1
    retriever_span = trace.data.spans[0]
    assert retriever_span.inputs == "test query"
    assert retriever_span.outputs is None
    error_event = SpanEvent.from_exception(mock_error)
    assert retriever_span.status.status_code == SpanStatusCode.ERROR
    assert retriever_span.events[0].name == error_event.name
    assert retriever_span.events[0].attributes == error_event.attributes

    _validate_trace_json_serialization(trace)


def test_retriever_internal_exception():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    callback.on_retriever_start(
        {},
        query="test query",
        run_id=run_id,
        name="test_retriever",
    )

    try:
        with pytest.raises(
            Exception,
            match="Span for run_id dummy not found.",
        ):
            callback.on_retriever_end(
                [
                    Document(
                        page_content="document content 1",
                        metadata={"chunk_id": "1", "doc_uri": "uri1"},
                    )
                ],
                run_id="dummy",
            )
    finally:
        callback.flush()


def test_multiple_components():
    callback = MlflowLangchainTracer()
    chain_run_id = str(uuid.uuid4())
    callback.on_chain_start(
        {},
        inputs={"input": "test input"},
        run_id=chain_run_id,
        name="test_chain",
    )
    for i in range(2):
        llm_run_id = str(uuid.uuid4())
        retriever_run_id = str(uuid.uuid4())
        callback.on_llm_start(
            {},
            [f"test prompt {i}"],
            run_id=llm_run_id,
            name="test_llm",
            parent_run_id=chain_run_id,
        )
        callback.on_retriever_start(
            {},
            query=f"test query {i}",
            run_id=retriever_run_id,
            name="test_retriever",
            parent_run_id=llm_run_id,
        )
        callback.on_retriever_end(
            [
                Document(
                    page_content=f"document content {i}",
                    metadata={
                        "chunk_id": str(i),
                        "doc_uri": f"https://mock_uri.com/{i}",
                    },
                )
            ],
            run_id=retriever_run_id,
        )
        callback.on_llm_end(
            LLMResult(generations=[[{"text": f"generated text {i}"}]]),
            run_id=llm_run_id,
        )
    callback.on_chain_end(
        outputs={"output": "test output"},
        run_id=chain_run_id,
    )
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 5
    chain_span = trace.data.spans[0]
    assert chain_span.start_time_ns is not None
    assert chain_span.end_time_ns is not None
    assert chain_span.name == "test_chain"
    assert chain_span.span_type == "CHAIN"
    assert chain_span.parent_id is None
    assert chain_span.status.status_code == SpanStatusCode.OK
    assert chain_span.inputs == {"input": "test input"}
    assert chain_span.outputs == {"output": "test output"}
    for i in range(2):
        llm_span = trace.data.spans[1 + i * 2]
        assert llm_span.inputs == [f"test prompt {i}"]
        assert llm_span.outputs["generations"][0][0]["text"] == f"generated text {i}"
        retriever_span = trace.data.spans[2 + i * 2]
        assert retriever_span.inputs == f"test query {i}"
        assert (
            retriever_span.outputs[0]
            == MlflowDocument(
                page_content=f"document content {i}",
                metadata={
                    "chunk_id": str(i),
                    "doc_uri": f"https://mock_uri.com/{i}",
                },
            ).to_dict()
        )

    _validate_trace_json_serialization(trace)


def test_tool_success():
    callback = MlflowLangchainTracer()
    prompt = SystemMessagePromptTemplate.from_template("You are a nice assistant.") + "{question}"
    llm = OpenAI(temperature=0.9)

    chain = prompt | llm | StrOutputParser()
    chain_tool = tool("chain_tool", chain)

    tool_input = {"question": "What up"}
    response = chain_tool.invoke(tool_input, config={"callbacks": [callback]})

    # str output is converted to _ChatResponse
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    spans = trace.data.spans
    assert len(spans) == 5

    # Tool
    tool_span = spans[0]
    assert tool_span.span_type == "TOOL"
    assert tool_span.inputs == tool_input
    assert tool_span.outputs is not None
    tool_span_id = tool_span.span_id

    # RunnableSequence
    runnable_sequence_span = spans[1]
    assert runnable_sequence_span.parent_id == tool_span_id
    assert runnable_sequence_span.span_type == "CHAIN"
    assert runnable_sequence_span.inputs == tool_input
    assert runnable_sequence_span.outputs is not None

    # PromptTemplate
    prompt_template_span = spans[2]
    assert prompt_template_span.span_type == "CHAIN"
    # LLM
    llm_span = spans[3]
    assert llm_span.span_type == "LLM"
    # StrOutputParser
    output_parser_span = spans[4]
    assert output_parser_span.span_type == "CHAIN"
    assert output_parser_span.outputs == response

    _validate_trace_json_serialization(trace)


def test_tracer_thread_safe():
    tracer = MlflowLangchainTracer()

    def worker_function(worker_id):
        chain_run_id = str(uuid.uuid4())
        tracer.on_chain_start(
            {}, {"input": "test input"}, run_id=chain_run_id, name=f"chain_{worker_id}"
        )
        # wait for a random time (0.5 ~ 1s) to simulate real-world scenario
        time.sleep(random.random() / 2 + 0.5)
        tracer.on_chain_end({"output": "test output"}, run_id=chain_run_id)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker_function, i) for i in range(10)]
        for future in futures:
            future.result()

    traces = get_traces()
    assert len(traces) == 10
    assert all(len(trace.data.spans) == 1 for trace in traces)


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.1.0"),
    reason="ChatPromptTemplate expecting dict input",
)
def test_tracer_does_not_add_spans_to_trace_after_root_run_has_finished():
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.chat_models.base import SimpleChatModel
    from langchain.schema.messages import BaseMessage

    class FakeChatModel(SimpleChatModel):
        """Fake Chat Model wrapper for testing purposes."""

        def _call(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> str:
            return TEST_CONTENT

        @property
        def _llm_type(self) -> str:
            return "fake chat model"

    run_id_for_on_chain_end = None

    class ExceptionCatchingTracer(MlflowLangchainTracer):
        def on_chain_end(self, outputs, *, run_id, inputs=None, **kwargs):
            nonlocal run_id_for_on_chain_end
            run_id_for_on_chain_end = run_id
            super().on_chain_end(outputs, run_id=run_id, inputs=inputs, **kwargs)

    prompt = SystemMessagePromptTemplate.from_template("You are a nice assistant.") + "{question}"
    chain = prompt | FakeChatModel() | StrOutputParser()

    tracer = ExceptionCatchingTracer()

    chain.invoke(
        "What is MLflow?",
        config={"callbacks": [tracer]},
    )

    with pytest.raises(MlflowException, match="Span for run_id .* not found."):
        # After the chain is invoked, verify that the tracer no longer holds references to spans,
        # ensuring that the tracer does not add spans to the trace after the root run has finished
        tracer.on_chain_end({"output": "test output"}, run_id=run_id_for_on_chain_end, inputs=None)


def test_tracer_noop_when_tracing_disabled(monkeypatch):
    llm_chain = create_openai_llmchain()
    model = _LangChainModelWrapper(llm_chain)

    @trace_disabled
    def _predict():
        return model._predict_with_callbacks(
            ["MLflow"],
            callback_handlers=[MlflowLangchainTracer()],
            convert_chat_responses=True,
        )

    mock_logger = MagicMock()
    monkeypatch.setattr(mlflow.tracking.client, "_logger", mock_logger)

    response = _predict()
    assert response == [{"text": TEST_CONTENT}]
    assert get_traces() == []
    # No warning should be issued
    mock_logger.warning.assert_not_called()


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.1.0"),
    reason="ChatPromptTemplate expecting dict input",
)
def test_tracer_with_manual_traces():
    # Validate if the callback works properly when outer and inner spans
    # are created by fluent APIs.
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["color"],
        template="What is the complementary color of {color}?",
    )

    # Inner spans are created within RunnableLambda
    def foo(s: str):
        with mlflow.start_span(name="foo_inner") as span:
            span.set_inputs(s)
            s = s.replace("red", "blue")
            s = bar(s)
            span.set_outputs(s)
        return s

    @mlflow.trace
    def bar(s):
        return s.replace("blue", "green")

    chain = RunnableLambda(foo) | prompt | llm | StrOutputParser()

    @mlflow.trace(name="parent", span_type="SPECIAL")
    def run(message):
        return chain.invoke(message, config={"callbacks": [MlflowLangchainTracer()]})

    response = run("red")
    expected_response = "What is the complementary color of green?"
    assert response == expected_response

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    spans = trace.data.spans
    assert spans[0].name == "parent"
    assert spans[1].name == "RunnableSequence"
    assert spans[1].parent_id == spans[0].span_id
    assert spans[2].name == "foo"
    assert spans[2].parent_id == spans[1].span_id
    assert spans[3].name == "foo_inner"
    assert spans[3].parent_id == spans[2].span_id
    assert spans[4].name == "bar"
    assert spans[4].parent_id == spans[3].span_id
    assert spans[5].name == "PromptTemplate"
    assert spans[5].parent_id == spans[1].span_id


def test_serialize_invocation_params_success():
    class DummyModel(pydantic.BaseModel):
        field: str

    callback = MlflowLangchainTracer()
    attributes = {"invocation_params": {"response_format": DummyModel, "other_param": "preserved"}}
    result = callback._serialize_invocation_params(attributes)
    expected_schema = (
        DummyModel.model_json_schema() if IS_PYDANTIC_V2_OR_NEWER else DummyModel.schema()
    )
    assert "invocation_params" in result
    assert "response_format" in result["invocation_params"]
    assert result["invocation_params"]["response_format"] == expected_schema
    assert result["invocation_params"]["other_param"] == "preserved"


def test_serialize_invocation_params_failure():
    class FaultyModel(pydantic.BaseModel):
        field: str

        @classmethod
        def model_json_schema(cls):
            raise Exception("dummy failure")

    callback = MlflowLangchainTracer()
    attributes = {"invocation_params": {"response_format": FaultyModel, "other_param": "preserved"}}
    result = callback._serialize_invocation_params(attributes)
    assert result["invocation_params"]["response_format"] == FaultyModel
    assert result["invocation_params"]["other_param"] == "preserved"


def test_serialize_invocation_params_non_pydantic_response_format():
    callback = MlflowLangchainTracer()
    test_cases = ["string_value", {"dict_key": "value"}, 123, ["list", "of", "items"], None]

    for test_value in test_cases:
        attributes = {
            "invocation_params": {"response_format": test_value, "other_param": "preserved"}
        }
        result = callback._serialize_invocation_params(attributes)
        assert result["invocation_params"]["response_format"] == test_value
        assert result["invocation_params"]["other_param"] == "preserved"


def test_serialize_invocation_params_no_invocation_params():
    callback = MlflowLangchainTracer()
    attributes = {"other_key": "value"}
    result = callback._serialize_invocation_params(attributes)
    assert result == attributes


def test_serialize_invocation_params_none():
    callback = MlflowLangchainTracer()
    result = callback._serialize_invocation_params(None)
    assert result is None
