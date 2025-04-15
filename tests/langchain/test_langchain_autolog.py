import random
import time
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
from typing import Any, Optional
from unittest import mock

import langchain
import pytest
from langchain.chains.llm import LLMChain
from langchain_core.callbacks.base import (
    AsyncCallbackHandler,
    BaseCallbackHandler,
    BaseCallbackManager,
)
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.runnables.config import RunnableConfig

from mlflow.entities.trace import Trace

# NB: We run this test suite twice - once with langchain_community installed and once without.
try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain_community.document_loaders import TextLoader
    from langchain_community.llms import OpenAI
    from langchain_community.vectorstores import FAISS

    _LC_COMMUNITY_INSTALLED = True
except ImportError:
    from langchain_openai import ChatOpenAI, OpenAI

    _LC_COMMUNITY_INSTALLED = False

# langchain-text-splitters is moved to a separate package since LangChain v0.1.10
try:
    from langchain_text_splitters.character import CharacterTextSplitter
except ImportError:
    from langchain.text_splitter import CharacterTextSplitter

import json

from packaging.version import Version

import mlflow
from mlflow.entities.trace_status import TraceStatus
from mlflow.models.dependencies_schemas import DependenciesSchemasType, set_retriever_schema
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION_KEY, SpanAttributeKey, TraceMetadataKey

from tests.langchain.conftest import DeterministicDummyEmbeddings
from tests.tracing.conftest import async_logging_enabled
from tests.tracing.helper import get_traces, purge_traces, score_in_model_serving

MODEL_DIR = "model"
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


def create_openai_runnable(temperature=0.9):
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is {product}?",
    )
    return prompt | ChatOpenAI(temperature=temperature) | StrOutputParser()


@pytest.fixture
def model_infos():
    models = [create_openai_runnable(temperature=temperature / 10) for temperature in range(1, 5)]
    model_infos = []
    for model in models:
        with mlflow.start_run():
            model_infos.append(
                mlflow.langchain.log_model(model, "model", input_example={"product": "MLflow"})
            )
    return model_infos


def create_retriever(tmp_path):
    # Create the vector db, persist the db to a local fs folder
    loader = TextLoader("tests/langchain/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = DeterministicDummyEmbeddings(size=5)
    db = FAISS.from_documents(docs, embeddings)
    persist_dir = str(tmp_path / "faiss_index")
    db.save_local(persist_dir)
    query = "What did the president say about Ketanji Brown Jackson"
    return db.as_retriever(), query


def create_fake_chat_model():
    class FakeChatModel(SimpleChatModel):
        """Fake Chat Model wrapper for testing purposes."""

        def _call(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            return TEST_CONTENT

        @property
        def _llm_type(self) -> str:
            return "fake chat model"

    return FakeChatModel()


def create_runnable_sequence():
    prompt_with_history_str = """
    Here is a history between you and a human: {chat_history}

    Now, please answer this question: {question}
    """
    prompt_with_history = PromptTemplate(
        input_variables=["chat_history", "question"], template=prompt_with_history_str
    )

    def extract_question(input):
        return input[-1]["content"]

    def extract_history(input):
        return input[:-1]

    chat_model = create_fake_chat_model()
    chain_with_history = (
        {
            "question": itemgetter("messages") | RunnableLambda(extract_question),
            "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
        }
        | prompt_with_history
        | chat_model
        | StrOutputParser()
    )
    input_example = {"messages": [{"role": "user", "content": "Who owns MLflow?"}]}
    return chain_with_history, input_example


def test_autolog_record_exception(async_logging_enabled):
    def always_fail(input):
        raise Exception("Error!")

    model = RunnableLambda(always_fail)

    mlflow.langchain.autolog()

    with pytest.raises(Exception, match="Error!"):
        model.invoke("test")

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.status == "ERROR"
    assert len(trace.data.spans) == 1
    assert trace.data.spans[0].name == "always_fail"


def test_llmchain_autolog(async_logging_enabled):
    mlflow.langchain.autolog()
    question = "MLflow"
    model = create_openai_llmchain()
    assert model.invoke(question) == {"product": "MLflow", "text": TEST_CONTENT}
    model.invoke(question)
    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 2
    for trace in traces:
        spans = trace.data.spans
        assert len(spans) == 2  # chain + llm
        assert spans[0].name == "LLMChain"
        assert spans[0].span_type == "CHAIN"
        assert spans[0].inputs == {"product": "MLflow"}
        assert spans[0].outputs == {"text": TEST_CONTENT}
        assert spans[1].name == "OpenAI"
        assert spans[1].parent_id == spans[0].span_id
        assert spans[1].span_type == "LLM"
        assert spans[1].inputs == ["What is MLflow?"]
        assert spans[1].outputs["generations"][0][0]["text"] == TEST_CONTENT
        attrs = spans[1].attributes
        assert attrs["invocation_params"]["model_name"] == "gpt-3.5-turbo-instruct"
        assert attrs["invocation_params"]["temperature"] == 0.9
        assert attrs[SpanAttributeKey.CHAT_MESSAGES] == [
            {
                "role": "user",
                "content": "What is MLflow?",
            },
            {
                "role": "assistant",
                "content": "What is MLflow?",
            },
        ]


def test_llmchain_autolog_should_not_generate_trace_while_saving_models(tmp_path):
    mlflow.langchain.autolog()
    question = "MLflow"

    model = create_openai_llmchain()
    # Either save_model or log_model should not generate traces
    mlflow.langchain.save_model(model, path=tmp_path / "model", input_example=question)
    with mlflow.start_run():
        mlflow.langchain.log_model(model, "model", input_example=question)

    traces = get_traces()
    assert len(traces) == 0


def test_llmchain_autolog_no_optional_artifacts_by_default():
    mlflow.langchain.autolog()
    question = "MLflow"
    answer = {"product": "MLflow", "text": TEST_CONTENT}
    model = create_openai_llmchain()
    with mock.patch("mlflow.MlflowClient.create_run") as create_run_mock:
        assert model.invoke(question) == answer
        create_run_mock.assert_not_called()

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert len(spans) == 2


@pytest.mark.skipif(not _LC_COMMUNITY_INSTALLED, reason="This test requires langchain_community")
@pytest.mark.skipif(
    Version(langchain.__version__) >= Version("0.3.0"),
    reason="LLMChain saving does not work in LangChain v0.3.0",
)
def test_loaded_llmchain_autolog():
    mlflow.langchain.autolog()
    model = create_openai_llmchain()
    question = {"product": "MLflow"}
    with mlflow.start_run() as run:
        for _ in range(3):
            model.invoke(question)

    logged_models = mlflow.search_logged_models(
        filter_string=f"source_run_id='{run.info.run_id}'", output_format="list"
    )
    assert len(logged_models) == 1
    logged_model = logged_models[0]

    traces = get_traces()
    assert len(traces) == 3
    for i in range(3):
        assert (
            traces[i].data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID)
            == logged_model.model_id
        )

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(model, "model", input_example=question)
    assert model_info.model_id != logged_model.model_id
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    loaded_model.invoke(question)
    traces = get_traces()
    assert len(traces) == 4
    assert traces[0].data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID) == model_info.model_id


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.2.0"),
    reason="Callback does not pass all messages in older versions",
)
def test_chat_model_autolog():
    mlflow.langchain.autolog()
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the weather in San Francisco?"),
        AIMessage(
            content="foo",
            tool_calls=[{"name": "GetWeather", "args": {"location": "San Francisco"}, "id": "123"}],
        ),
        ToolMessage(content="Weather in San Francisco is 70F.", tool_call_id="123"),
    ]
    response = model.invoke(messages)

    traces = get_traces()
    assert len(traces) == 1
    assert len(traces[0].data.spans) == 1

    span = traces[0].data.spans[0]
    assert span.name == "ChatOpenAI"
    assert span.span_type == "CHAT_MODEL"
    # LangChain uses pydantic V1 until LangChain v0.3.0
    if Version(langchain.__version__) >= Version("0.3.0"):
        assert span.inputs == [[msg.model_dump() for msg in messages]]
    else:
        assert span.inputs == [[msg.dict() for msg in messages]]
    assert span.outputs["generations"][0][0]["message"]["content"] == response.content
    assert span.get_attribute("invocation_params")["model"] == "gpt-4o-mini"
    assert span.get_attribute("invocation_params")["temperature"] == 0.9
    assert span.get_attribute(SpanAttributeKey.CHAT_MESSAGES) == [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "What is the weather in San Francisco?",
        },
        {
            "role": "assistant",
            "content": "foo",
            "tool_calls": [
                {
                    "function": {
                        "arguments": '{"location": "San Francisco"}',
                        "name": "GetWeather",
                    },
                    "id": "123",
                    "type": "function",
                }
            ],
        },
        {
            "role": "tool",
            "content": "Weather in San Francisco is 70F.",
            "tool_call_id": "123",
        },
        {
            "role": "assistant",
            "content": response.content,
        },
    ]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.3.0"),
    reason="langchain-openai.ChatOpenAI requires LangChain >= 0.3.0",
)
def test_chat_model_bind_tool_autolog():
    # Community version of ChatOpenAI does not support bind_tools
    from langchain_openai import ChatOpenAI

    mlflow.langchain.autolog()

    @langchain.tools.tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"Weather in {location} is 70F."

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
    model_with_tools = model.bind_tools([get_weather])
    model_with_tools.invoke("What is the weather in San Francisco?")

    traces = get_traces()
    assert len(traces) == 1
    assert len(traces[0].data.spans) == 1

    span = traces[0].data.spans[0]
    assert span.name == "ChatOpenAI"
    assert span.get_attribute(SpanAttributeKey.CHAT_MESSAGES) == [
        {
            "role": "user",
            "content": "What is the weather in San Francisco?",
        },
        {
            "content": '[{"role": "user", "content": "What is the weather in San Francisco?"}]',
            "role": "assistant",
        },
    ]
    assert span.get_attribute(SpanAttributeKey.CHAT_TOOLS) == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location.",
                "parameters": {
                    "properties": {
                        "location": {
                            "type": "string",
                        }
                    },
                    "required": ["location"],
                    "type": "object",
                },
            },
        }
    ]


@pytest.mark.skipif(not _LC_COMMUNITY_INSTALLED, reason="This test requires langchain_community")
@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.2.0"),
    reason="ToolCall message is not available in older versions",
)
def test_agent_autolog(async_logging_enabled):
    mlflow.langchain.autolog()

    # Load the agent definition (with OpenAI mock) from the sample script
    from tests.langchain.sample_code.openai_agent import create_openai_agent

    model = create_openai_agent()
    input = {"input": "The result of 2 * 3 is 6."}
    expected_output = {"output": "The result of 2 * 3 is 6."}

    assert model(input, return_only_outputs=True) == expected_output

    model = create_openai_agent()

    # ensure invoke is patched
    assert model.invoke(input, return_only_outputs=True) == expected_output

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 2
    for trace in traces:
        spans = [(s.name, s.span_type) for s in trace.data.spans]
        assert len(spans) == 16
        assert trace.data.spans[0].inputs == input
        assert trace.data.spans[0].outputs == expected_output


def test_runnable_sequence_autolog(async_logging_enabled):
    mlflow.langchain.autolog()
    chain, input_example = create_runnable_sequence()
    assert chain.invoke(input_example) == TEST_CONTENT

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    for trace in traces:
        spans = {(s.name, s.span_type) for s in trace.data.spans}
        # Since the chain includes parallel execution, the order of some
        # spans is not deterministic.
        assert spans == {
            ("RunnableSequence_1", "CHAIN"),
            ("RunnableParallel<question,chat_history>", "CHAIN"),
            ("RunnableSequence_2", "CHAIN"),
            ("RunnableLambda_1", "CHAIN"),
            ("extract_question", "CHAIN"),
            ("RunnableSequence_3", "CHAIN"),
            ("RunnableLambda_2", "CHAIN"),
            ("extract_history", "CHAIN"),
            ("PromptTemplate", "CHAIN"),
            ("FakeChatModel", "CHAT_MODEL"),
            ("StrOutputParser", "CHAIN"),
        }


@pytest.mark.skipif(not _LC_COMMUNITY_INSTALLED, reason="This test requires langchain_community")
def test_retriever_autolog(tmp_path, async_logging_enabled):
    mlflow.langchain.autolog()
    model, query = create_retriever(tmp_path)
    model.invoke(query)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "VectorStoreRetriever"
    assert spans[0].span_type == "RETRIEVER"
    assert spans[0].inputs == query
    assert spans[0].outputs[0]["metadata"] == {"source": "tests/langchain/state_of_the_union.txt"}


class CustomCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.logs = []

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        self.logs.append("chain_start")

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        self.logs.append("chain_end")


class AsyncCustomCallbackHandler(AsyncCallbackHandler):
    def __init__(self):
        self.logs = []

    async def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        self.logs.append("chain_start")

    async def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        self.logs.append("chain_end")


_CONFIG_PATTERNS = [
    # Config with no user callbacks
    RunnableConfig(max_concurrency=1),
    RunnableConfig(callbacks=None),
    # With user callbacks
    RunnableConfig(callbacks=[CustomCallbackHandler()]),
    RunnableConfig(callbacks=BaseCallbackManager([CustomCallbackHandler()])),
]

_ASYNC_CONFIG_PATTERNS = [
    RunnableConfig(callbacks=[AsyncCustomCallbackHandler()]),
    RunnableConfig(callbacks=BaseCallbackManager([AsyncCustomCallbackHandler()])),
]


def _reset_callback_handlers(handlers):
    if handlers:
        for handler in handlers:
            handler.logs = []


def _extract_callback_handlers(config) -> Optional[list[BaseCallbackHandler]]:
    if isinstance(config, list):
        callbacks = []
        for c in config:
            callbacks_in_c = _extract_callback_handlers(c)
            if callbacks_in_c:
                callbacks.extend(callbacks_in_c)
        return callbacks
    # RunnableConfig is also a dict
    elif isinstance(config, dict) and "callbacks" in config:
        callbacks = config["callbacks"]
        if isinstance(callbacks, BaseCallbackManager):
            return callbacks.handlers
        else:
            return callbacks
    else:
        return None


@pytest.mark.parametrize("invoke_arg", ["args", "kwargs", None])
@pytest.mark.parametrize("config", _CONFIG_PATTERNS)
def test_langchain_autolog_callback_injection_in_invoke(invoke_arg, config, async_logging_enabled):
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    original_handlers = _extract_callback_handlers(config)
    _reset_callback_handlers(original_handlers)

    input = {"product": "MLflow"}
    if invoke_arg == "args":
        model.invoke(input, config)
    elif invoke_arg == "kwargs":
        model.invoke(input, config=config)
    elif invoke_arg is None:
        model.invoke(input)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].data.spans[0].name == "RunnableSequence"
    assert traces[0].data.spans[0].inputs == input
    assert traces[0].data.spans[0].outputs == '[{"role": "user", "content": "What is MLflow?"}]'
    # Original callback should not be mutated
    handlers = _extract_callback_handlers(config)
    assert handlers == original_handlers

    # The original callback is called by the chain
    if handlers and invoke_arg:
        # NB: Langchain has a bug that the callback is called different times when
        # passed by a list or a callback manager. As a workaround we only check
        # the content of the events not the count.
        # https://github.com/langchain-ai/langchain/issues/24642
        assert set(handlers[0].logs) == {"chain_start", "chain_end"}


@pytest.mark.parametrize("invoke_arg", ["args", "kwargs", None])
@pytest.mark.parametrize("config", _CONFIG_PATTERNS + _ASYNC_CONFIG_PATTERNS)
@pytest.mark.asyncio
async def test_langchain_autolog_callback_injection_in_ainvoke(
    invoke_arg, config, async_logging_enabled
):
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    original_handlers = _extract_callback_handlers(config)
    _reset_callback_handlers(original_handlers)

    input = {"product": "MLflow"}
    if invoke_arg == "args":
        await model.ainvoke(input, config)
    elif invoke_arg == "kwargs":
        await model.ainvoke(input, config=config)
    elif invoke_arg is None:
        await model.ainvoke(input)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].data.spans[0].name == "RunnableSequence"
    assert traces[0].data.spans[0].inputs == input
    assert traces[0].data.spans[0].outputs == '[{"role": "user", "content": "What is MLflow?"}]'

    # Original callback should not be mutated
    handlers = _extract_callback_handlers(config)
    assert handlers == original_handlers

    # The original callback is called by the chain
    if handlers and invoke_arg:
        # NB: Langchain has a bug that the callback is called different times when
        # passed by a list or a callback manager. As a workaround we only check
        # the content of the events not the count.
        # https://github.com/langchain-ai/langchain/issues/24642
        assert set(handlers[0].logs) == {"chain_start", "chain_end"}


@pytest.mark.parametrize("invoke_arg", ["args", "kwargs"])
@pytest.mark.parametrize(
    "config",
    _CONFIG_PATTERNS
    # list of configs are also supported for batch call
    + [[config, config] for config in _CONFIG_PATTERNS],
)
def test_langchain_autolog_callback_injection_in_batch(invoke_arg, config, async_logging_enabled):
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    original_handlers = _extract_callback_handlers(config)
    _reset_callback_handlers(original_handlers)

    input = {"product": "MLflow"}
    if invoke_arg == "args":
        model.batch([input] * 2, config)
    elif invoke_arg == "kwargs":
        model.batch([input] * 2, config=config)
    elif invoke_arg is None:
        model.batch([input] * 2)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 2
    for trace in traces:
        assert trace.info.status == "OK"
        assert trace.data.spans[0].name == "RunnableSequence"
        assert trace.data.spans[0].inputs == input
        assert trace.data.spans[0].outputs == '[{"role": "user", "content": "What is MLflow?"}]'

    # Original callback should not be mutated
    handlers = _extract_callback_handlers(config)
    assert handlers == original_handlers

    # The original callback is called by the chain
    if handlers and invoke_arg:
        for handler in handlers:
            assert set(handler.logs) == {"chain_start", "chain_end"}


def test_tracing_source_run_in_batch():
    mlflow.langchain.autolog()

    model = create_openai_llmchain()
    input = {"product": "MLflow"}
    with mlflow.start_run() as run:
        model.batch([input] * 2)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run.info.run_id


@pytest.mark.skipif(not _LC_COMMUNITY_INSTALLED, reason="This test requires langchain_community")
def test_tracing_source_run_in_pyfunc_model_predict():
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    input = {"product": "MLflow"}
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(model, "model")

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    with mlflow.start_run() as run:
        pyfunc_model.predict([input] * 2)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run.info.run_id


@pytest.mark.parametrize("invoke_arg", ["args", "kwargs", None])
@pytest.mark.parametrize(
    "config",
    _CONFIG_PATTERNS
    + _ASYNC_CONFIG_PATTERNS
    # list of configs are also supported for batch call
    + [[config, config] for config in _CONFIG_PATTERNS + _ASYNC_CONFIG_PATTERNS],
)
@pytest.mark.asyncio
async def test_langchain_autolog_callback_injection_in_abatch(
    invoke_arg, config, async_logging_enabled
):
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    original_handlers = _extract_callback_handlers(config)
    _reset_callback_handlers(original_handlers)

    input = {"product": "MLflow"}
    if invoke_arg == "args":
        await model.abatch([input] * 2, config)
    elif invoke_arg == "kwargs":
        await model.abatch([input] * 2, config=config)
    elif invoke_arg is None:
        await model.abatch([input] * 2)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 2
    for trace in traces:
        assert trace.info.status == "OK"
        assert trace.data.spans[0].name == "RunnableSequence"
        assert trace.data.spans[0].inputs == input
        assert trace.data.spans[0].outputs == '[{"role": "user", "content": "What is MLflow?"}]'

    # Original callback should not be mutated
    handlers = _extract_callback_handlers(config)
    assert handlers == original_handlers

    # The original callback is called by the chain
    if handlers and invoke_arg:
        for handler in handlers:
            assert set(handler.logs) == {"chain_start", "chain_end"}


@pytest.mark.parametrize("invoke_arg", ["args", "kwargs", None])
@pytest.mark.parametrize("config", _CONFIG_PATTERNS)
def test_langchain_autolog_callback_injection_in_stream(invoke_arg, config, async_logging_enabled):
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    original_handlers = _extract_callback_handlers(config)
    _reset_callback_handlers(original_handlers)

    input = {"product": "MLflow"}
    if invoke_arg == "args":
        list(model.stream(input, config))
    elif invoke_arg == "kwargs":
        list(model.stream(input, config=config))
    elif invoke_arg is None:
        list(model.stream(input))

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].data.spans[0].name == "RunnableSequence"
    assert traces[0].data.spans[0].inputs == input
    assert traces[0].data.spans[0].outputs == "Hello world"

    # Original callback should not be mutated
    handlers = _extract_callback_handlers(config)
    assert handlers == original_handlers

    # The original callback is called by the chain
    if handlers and invoke_arg:
        assert set(handlers[0].logs) == {"chain_start", "chain_end"}


@pytest.mark.parametrize("invoke_arg", ["args", "kwargs", None])
@pytest.mark.parametrize("config", _CONFIG_PATTERNS + _ASYNC_CONFIG_PATTERNS)
@pytest.mark.asyncio
async def test_langchain_autolog_callback_injection_in_astream(
    invoke_arg, config, async_logging_enabled
):
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    original_handlers = _extract_callback_handlers(config)
    _reset_callback_handlers(original_handlers)
    input = {"product": "MLflow"}

    async def invoke_astream(model, config):
        if invoke_arg == "args":
            astream = model.astream(input, config)
        elif invoke_arg == "kwargs":
            astream = model.astream(input, config=config)
        elif invoke_arg is None:
            astream = model.astream(input)

        # Consume the stream
        async for _ in astream:
            pass

    await invoke_astream(model, config)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].data.spans[0].name == "RunnableSequence"
    assert traces[0].data.spans[0].inputs == input
    assert traces[0].data.spans[0].outputs == "Hello world"

    # Original callback should not be mutated
    handlers = _extract_callback_handlers(config)
    assert handlers == original_handlers

    # The original callback is called by the chain
    if handlers and invoke_arg:
        assert set(handlers[0].logs) == {"chain_start", "chain_end"}


@pytest.mark.skipif(not _LC_COMMUNITY_INSTALLED, reason="This test requires langchain_community")
def test_langchain_autolog_produces_expected_traces_with_streaming(tmp_path, async_logging_enabled):
    mlflow.langchain.autolog()
    retriever, _ = create_retriever(tmp_path)
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question based on the context: {context}\nQuestion: {question}"
    )
    chat_model = create_fake_chat_model()
    retrieval_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )
    question = "What is a good name for a company that makes MLflow?"
    list(retrieval_chain.stream(question))
    retrieval_chain.invoke(question)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 2
    stream_trace = traces[0]
    invoke_trace = traces[1]

    assert stream_trace.info.status == invoke_trace.info.status == TraceStatus.OK
    assert stream_trace.data.request == invoke_trace.data.request
    assert stream_trace.data.response == invoke_trace.data.response
    assert len(stream_trace.data.spans) == len(invoke_trace.data.spans)


def test_langchain_autolog_tracing_thread_safe(async_logging_enabled):
    mlflow.langchain.autolog()

    model = create_openai_runnable()

    def _invoke():
        # Add random sleep to simulate real LLM prediction
        time.sleep(random.uniform(0.1, 0.5))

        model.invoke({"product": "MLflow"})

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_invoke) for _ in range(30)]
        _ = [f.result() for f in futures]

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 30
    for trace in traces:
        assert trace.info.status == "OK"
        assert len(trace.data.spans) == 4
        assert trace.data.spans[0].name == "RunnableSequence"


@pytest.mark.parametrize("log_traces", [True, False, None])
def test_langchain_tracer_injection_for_arbitrary_runnables(log_traces, async_logging_enabled):
    from langchain.schema.runnable import RouterRunnable, RunnableLambda

    should_log_traces = log_traces is not False

    if log_traces is not None:
        mlflow.langchain.autolog(log_traces=log_traces)
    else:
        mlflow.langchain.autolog()

    add = RunnableLambda(func=lambda x: x + 1)
    square = RunnableLambda(func=lambda x: x**2)
    model = RouterRunnable(runnables={"add": add, "square": square})

    model.invoke({"key": "square", "input": 3})

    if async_logging_enabled and should_log_traces:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    if should_log_traces:
        assert len(traces) == 1
        assert traces[0].data.spans[0].span_type == "CHAIN"
    else:
        assert len(traces) == 0


def test_langchain_autolog_extra_model_classes_warning():
    class NotARunnable:
        def __init__(self, x):
            self.x = x

    with mock.patch("mlflow.langchain.logger.warning") as mock_warning:
        mlflow.langchain.autolog(extra_model_classes=[NotARunnable])
        mock_warning.assert_called_once_with(
            "Unsupported classes found in extra_model_classes: ['NotARunnable']. "
            "Only subclasses of Runnable are supported."
        )
        mock_warning.reset_mock()

        mlflow.langchain.autolog(extra_model_classes=[Runnable])
        mock_warning.assert_not_called()


@pytest.mark.skip(reason="This test is not thread safe, please run locally")
def test_set_retriever_schema_work_for_langchain_model():
    set_retriever_schema(
        primary_key="primary-key",
        text_column="text-column",
        doc_uri="doc-uri",
        other_columns=["column1", "column2"],
    )

    model = create_openai_runnable()
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(model, "model", input_example="MLflow")

    mlflow.langchain.autolog()

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    pyfunc_model.predict("MLflow")

    traces = get_traces()
    assert len(traces) == 1
    assert DependenciesSchemasType.RETRIEVERS.value in traces[0].info.tags

    purge_traces()

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    list(pyfunc_model.predict_stream("MLflow"))

    traces = get_traces()
    assert len(traces) == 1
    assert DependenciesSchemasType.RETRIEVERS.value in traces[0].info.tags


def test_langchain_auto_tracing_work_when_langchain_parent_package_not_installed():
    original_import = __import__

    def _mock_import(name, *args):
        if name.startswith("langchain."):
            raise ImportError("No module named 'langchain'")
        return original_import(name, *args)

    with mock.patch("builtins.__import__", side_effect=_mock_import):
        mlflow.langchain.autolog()

        chain, input_example = create_runnable_sequence()
        assert chain.invoke(input_example) == TEST_CONTENT
        assert chain.invoke(input_example) == TEST_CONTENT

        if async_logging_enabled:
            mlflow.flush_trace_async_logging(terminate=True)

        traces = get_traces()
        assert len(traces) == 2
        assert all(len(trace.data.spans) == 11 for trace in traces)


def test_langchain_auto_tracing_in_serving_runnable():
    mlflow.langchain.autolog()

    chain = create_openai_runnable()

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain,
            "model",
            input_example={"product": "MLflow"},
        )

    expected_output = '[{"role": "user", "content": "What is MLflow?"}]'

    request_id, predictions, trace = score_in_model_serving(
        model_info.model_uri,
        [{"product": "MLflow"}],
    )

    assert predictions == [expected_output]
    trace = Trace.from_dict(trace)
    assert trace.info.request_id == request_id
    assert trace.info.request_metadata[TRACE_SCHEMA_VERSION_KEY] == "2"
    spans = trace.data.spans
    assert len(spans) == 4

    root_span = spans[0]
    assert root_span.start_time_ns // 1_000_000 == trace.info.timestamp_ms
    # there might be slight difference when we truncate nano seconds to milliseconds
    assert (
        root_span.end_time_ns // 1_000_000
        - (trace.info.timestamp_ms + trace.info.execution_time_ms)
    ) <= 1
    assert root_span.inputs == {"product": "MLflow"}
    assert root_span.outputs == expected_output
    assert root_span.span_type == "CHAIN"

    root_span_id = root_span.span_id
    child_span = spans[2]
    assert child_span.parent_id == root_span_id
    assert child_span.inputs[0][0]["content"] == "What is MLflow?"
    assert child_span.outputs["generations"][0][0]["text"] == expected_output
    assert child_span.span_type == "CHAT_MODEL"


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.2.0"),
    reason="ToolCall message is not available in older versions",
)
def test_langchain_auto_tracing_in_serving_agent():
    mlflow.langchain.autolog()

    input_example = {"input": "What is 2 * 3?"}
    expected_output = {"output": "The result of 2 * 3 is 6."}

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            "tests/langchain/sample_code/openai_agent.py",
            "langchain_model",
            input_example=input_example,
        )

    request_id, response, trace_dict = score_in_model_serving(
        model_info.model_uri,
        input_example,
    )

    trace = Trace.from_dict(trace_dict)
    assert trace.info.request_id == request_id
    assert trace.info.status == "OK"

    spans = trace.data.spans
    assert len(spans) == 16

    root_span = spans[0]
    assert root_span.name == "AgentExecutor"
    assert root_span.span_type == "CHAIN"
    assert root_span.inputs == input_example
    assert root_span.outputs == expected_output
    assert root_span.start_time_ns // 1_000_000 == trace.info.timestamp_ms
    assert (
        root_span.end_time_ns // 1_000_000
        - (trace.info.timestamp_ms + trace.info.execution_time_ms)
    ) <= 1


def test_langchain_tracing_multi_threads():
    mlflow.langchain.autolog()

    temperatures = [(t + 1) / 10 for t in range(4)]
    models = [create_openai_runnable(temperature=t) for t in temperatures]

    with ThreadPoolExecutor(max_workers=len(temperatures)) as executor:
        futures = [executor.submit(models[i].invoke, {"product": "MLflow"}) for i in range(4)]
        for f in futures:
            f.result()

    traces = get_traces()
    assert len(traces) == 4
    assert (
        sorted(
            trace.data.spans[2].get_attribute("invocation_params")["temperature"]
            for trace in traces
        )
        == temperatures
    )


def test_autolog_traces_linked_to_models():
    # log two models + load model
    mlflow.langchain.autolog()
    messages = {"product": "MLflow"}
    model1 = create_openai_runnable(temperature=0.9)
    model2 = create_openai_runnable(temperature=0.1)

    with mlflow.start_run():
        mlflow.langchain.log_model(model2, "model2")
    logged_model2 = mlflow.last_logged_model()

    with mlflow.start_run():
        mlflow.langchain.log_model(model1, "model1")
    logged_model1 = mlflow.last_logged_model()
    assert logged_model1.model_id != logged_model2.model_id

    # traces generated after model logging should be linked to the logged model
    model2.invoke(messages)
    model1.invoke(messages)

    # traces generated on loaded model should be linked to the logged model
    model3 = mlflow.langchain.load_model(logged_model1.model_uri)
    model3.invoke(messages)

    traces = get_traces()
    assert len(traces) == 3
    assert (
        traces[0].data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID) == logged_model1.model_id
    )
    assert traces[1].data.spans[2].get_attribute("invocation_params")["temperature"] == 0.9
    assert (
        traces[1].data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID) == logged_model1.model_id
    )
    span2 = traces[2].data.spans[2]
    assert span2.get_attribute("invocation_params")["temperature"] == 0.1
    assert span2.get_attribute(SpanAttributeKey.MODEL_ID) == logged_model2.model_id


@pytest.mark.asyncio
async def test_autolog_traces_linked_to_loaded_models(model_infos):
    mlflow.langchain.autolog()

    for model_info in model_infos:
        loaded_model = mlflow.langchain.load_model(model_info.model_uri)
        await loaded_model.ainvoke(
            {"product": f"{loaded_model.steps[1].temperature}_{model_info.model_id}"}
        )

    traces = get_traces()
    assert len(traces) == len(model_infos)
    for trace in traces:
        temp = trace.data.spans[2].get_attribute("invocation_params")["temperature"]
        logged_temp, logged_model_id = json.loads(trace.data.request)["product"].split(
            "_", maxsplit=1
        )
        assert logged_model_id is not None
        assert str(temp) == logged_temp
        assert trace.data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID) == logged_model_id


@pytest.mark.parametrize("func", ["invoke", "batch", "stream"])
def test_autolog_traces_linked_to_models_inference(model_infos, func):
    mlflow.langchain.autolog()

    for model_info in model_infos:
        loaded_model = mlflow.langchain.load_model(model_info.model_uri)
        msg = {"product": f"{loaded_model.steps[1].temperature}_{model_info.model_id}"}
        if func == "invoke":
            loaded_model.invoke(msg)
        elif func == "batch":
            loaded_model.batch([msg])
        elif func == "stream":
            list(loaded_model.stream(msg))

    traces = get_traces()
    assert len(traces) == len(model_infos)
    for trace in traces:
        temp = trace.data.spans[2].get_attribute("invocation_params")["temperature"]
        logged_temp, logged_model_id = json.loads(trace.data.request)["product"].split(
            "_", maxsplit=1
        )
        assert logged_model_id is not None
        assert str(temp) == logged_temp
        assert trace.data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID) == logged_model_id


def test_autolog_traces_linked_to_models_multi_threading(model_infos):
    mlflow.langchain.autolog()

    # This needs to be outside of _invoke because load_model disables tracing
    # and it impacts tracing behavior in multi-threads
    model_and_model_ids = [
        (mlflow.langchain.load_model(model_info.model_uri), model_info.model_id)
        for model_info in model_infos
    ]

    def _invoke(loaded_model, model_id):
        loaded_model.invoke({"product": f"{loaded_model.steps[1].temperature}_{model_id}"})

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_invoke, loaded_model, model_id)
            for loaded_model, model_id in model_and_model_ids
        ]
        for future in futures:
            future.result()

    traces = get_traces()
    assert len(traces) == len(model_infos)
    for trace in traces:
        temp = trace.data.spans[2].get_attribute("invocation_params")["temperature"]
        logged_temp, logged_model_id = json.loads(trace.data.request)["product"].split(
            "_", maxsplit=1
        )
        assert logged_model_id is not None
        assert str(temp) == logged_temp
        assert trace.data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID) == logged_model_id


def test_new_model_logged_after_loaded():
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(model, "model", input_example={"product": "MLflow"})

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    loaded_model.invoke({"product": "MLflow"})
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID) == model_info.model_id

    with mlflow.start_run():
        new_model_info = mlflow.langchain.log_model(
            model, "model", input_example={"product": "MLflow"}
        )
    assert new_model_info.model_id != model_info.model_id

    loaded_model.invoke({"product": "MLflow"})
    traces = get_traces()
    assert len(traces) == 2
    assert traces[0].data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID) == model_info.model_id

    new_loaded_model = mlflow.langchain.load_model(new_model_info.model_uri)
    new_loaded_model.invoke({"product": "MLflow"})
    traces = get_traces()
    assert len(traces) == 3
    assert (
        traces[0].data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID) == new_model_info.model_id
    )


def test_log_model_multiple_times_different_model_id():
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    with mlflow.start_run():
        model_info1 = mlflow.langchain.log_model(
            model, "model", input_example={"product": "MLflow"}
        )
        model_info2 = mlflow.langchain.log_model(
            model, "model", input_example={"product": "MLflow"}
        )
    assert model_info1.model_id != model_info2.model_id

    model.invoke({"product": "MLflow"})
    traces = get_traces()
    assert len(traces) == 1
    # traces link to the latest model_id
    assert traces[0].data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID) == model_info2.model_id


@pytest.mark.parametrize("log_models", [True, False])
def test_autolog_log_models_and_link_traces_invoke(log_models):
    mlflow.langchain.autolog(log_models=log_models)
    chain, input_example = create_runnable_sequence()
    with mlflow.start_run() as run:
        for _ in range(3):
            chain.invoke(input_example)

    logged_models = mlflow.search_logged_models(
        filter_string=f"source_run_id='{run.info.run_id}'", output_format="list"
    )
    traces = get_traces()
    assert len(traces) == 3
    if log_models:
        assert len(logged_models) == 1
        logged_model_id = logged_models[0].model_id
        for i in range(3):
            assert (
                traces[i].data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID) == logged_model_id
            )
    else:
        assert len(logged_models) == 0
        logged_model_id = None
        for i in range(3):
            assert SpanAttributeKey.MODEL_ID not in traces[i].data.spans[0].attributes

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(chain, "model", input_example=input_example)

    assert model_info.model_id != logged_model_id
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    loaded_model.invoke(input_example)
    traces = get_traces()
    assert len(traces) == 4
    assert traces[0].data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID) == model_info.model_id


@pytest.mark.asyncio
@pytest.mark.parametrize("log_models", [True, False])
async def test_autolog_log_models_and_link_traces_ainvoke(log_models):
    mlflow.langchain.autolog(log_models=log_models)
    chain, input_example = create_runnable_sequence()
    with mlflow.start_run() as run:
        for _ in range(3):
            await chain.ainvoke(input_example)

    logged_models = mlflow.search_logged_models(
        filter_string=f"source_run_id='{run.info.run_id}'", output_format="list"
    )
    traces = get_traces()
    assert len(traces) == 3
    if log_models:
        assert len(logged_models) == 1
        logged_model_id = logged_models[0].model_id
        for i in range(3):
            assert (
                traces[i].data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID) == logged_model_id
            )
    else:
        assert len(logged_models) == 0
        logged_model_id = None
        for i in range(3):
            assert SpanAttributeKey.MODEL_ID not in traces[i].data.spans[0].attributes

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(chain, "model", input_example=input_example)

    assert model_info.model_id != logged_model_id
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    await loaded_model.ainvoke(input_example)
    traces = get_traces()
    assert len(traces) == 4
    assert traces[0].data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID) == model_info.model_id
