import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
from typing import Any
from unittest import mock

import langchain_core
import pytest
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.router import RouterRunnable
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_text_splitters.character import CharacterTextSplitter
from packaging import version

import mlflow
from mlflow.entities.span import SpanType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_status import TraceStatus
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION_KEY, SpanAttributeKey, TraceMetadataKey

from tests.langchain.conftest import DeterministicDummyEmbeddings
from tests.tracing.conftest import async_logging_enabled
from tests.tracing.helper import (
    get_traces,
    purge_traces,
    score_in_model_serving,
    skip_when_testing_trace_sdk,
)

MODEL_DIR = "model"
# The mock OpenAI endpoint simply echos the prompt back as the completion.
# So the expected output will be the prompt itself.
TEST_CONTENT = "What is MLflow?"

_SIMPLE_MODEL_CODE_PATH = "tests/langchain/sample_code/simple_runnable.py"

IS_LANGCHAIN_v1 = version.parse(langchain_core.__version__).major >= 1


def create_openai_runnable(temperature=0.9):
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is {product}?",
    )
    llm = ChatOpenAI(temperature=temperature, stream_usage=True)
    return prompt | llm | StrOutputParser()


@pytest.fixture
def model_info():
    with mlflow.start_run():
        return mlflow.langchain.log_model(_SIMPLE_MODEL_CODE_PATH, pip_requirements=["mlflow"])


@pytest.fixture
def model_infos():
    model_infos = []
    for _ in range(3):
        with mlflow.start_run():
            info = mlflow.langchain.log_model(_SIMPLE_MODEL_CODE_PATH, pip_requirements=["mlflow"])
            model_infos.append(info)
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
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
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
    for msg, expected in zip(span.inputs[0], messages, strict=True):
        assert msg["type"] == expected.type
        assert msg["content"] == expected.content
    assert span.outputs["generations"][0][0]["message"]["content"] == response.content
    assert span.get_attribute("invocation_params")["model"] == "gpt-4o-mini"
    assert span.get_attribute("invocation_params")["temperature"] == 0.9
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "langchain"
    assert span.model_name == "gpt-4o-mini"


def test_chat_model_bind_tool_autolog():
    mlflow.langchain.autolog()

    @tool
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
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "langchain"
    assert span.model_name == "gpt-4o-mini"


@pytest.mark.skipif(not IS_LANGCHAIN_v1, reason="create_agent is not supported in langchain v0")
@skip_when_testing_trace_sdk
def test_agent_autolog(async_logging_enabled):
    mlflow.langchain.autolog()

    # Load the agent definition (with OpenAI mock) from the sample script
    from langchain.agents import create_agent

    from tests.langchain.sample_code.openai_agent import FakeOpenAI, add, multiply

    model = create_agent(FakeOpenAI(), [add, multiply], system_prompt="You are a helpful assistant")
    prompt = "What is 2 * 3?"
    expected_output = "The result of 2 * 3 is 6."

    result = model.invoke({"messages": [HumanMessage(content=prompt)]})
    assert result["messages"][-1].content == expected_output

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    assert len(traces[0].data.spans) == 7
    spans = traces[0].data.spans
    assert spans[0].name == "LangGraph"
    assert spans[0].span_type == SpanType.CHAIN
    assert spans[0].inputs["messages"][0]["content"] == prompt
    assert spans[0].outputs["messages"][-1]["content"] == expected_output
    llm_spans = [s for s in spans if s.span_type == SpanType.CHAT_MODEL]
    assert len(llm_spans) == 2
    assert all(s.name == "FakeOpenAI" for s in llm_spans)
    tool_spans = [s for s in traces[0].data.spans if s.span_type == SpanType.TOOL]
    assert len(tool_spans) == 1
    assert tool_spans[0].name == "multiply"
    assert tool_spans[0].inputs["a"] == 2
    assert tool_spans[0].inputs["b"] == 3
    assert tool_spans[0].outputs["content"] == "6"


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
            ("RunnableSequence", "CHAIN"),
            ("RunnableParallel<question,chat_history>", "CHAIN"),
            ("RunnableSequence", "CHAIN"),
            ("RunnableLambda", "CHAIN"),
            ("extract_question", "CHAIN"),
            ("RunnableSequence", "CHAIN"),
            ("RunnableLambda", "CHAIN"),
            ("extract_history", "CHAIN"),
            ("PromptTemplate", "CHAIN"),
            ("FakeChatModel", "CHAT_MODEL"),
            ("StrOutputParser", "CHAIN"),
        }


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


def _extract_callback_handlers(config) -> list[BaseCallbackHandler] | None:
    if isinstance(config, list):
        callbacks = []
        for c in config:
            if callbacks_in_c := _extract_callback_handlers(c):
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


@skip_when_testing_trace_sdk
def test_tracing_source_run_in_batch():
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    input = {"product": "MLflow"}
    with mlflow.start_run() as run:
        model.batch([input] * 2)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run.info.run_id


@skip_when_testing_trace_sdk
def test_tracing_source_run_in_pyfunc_model_predict(model_info):
    mlflow.langchain.autolog()

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    with mlflow.start_run() as run:
        pyfunc_model.predict([{"product": "MLflow"}] * 2)

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


@pytest.mark.asyncio
async def test_langchain_autolog_token_usage(mock_litellm_cost):
    mlflow.langchain.autolog()

    model = create_openai_runnable()

    def _validate_token_counts(trace):
        actual = trace.info.token_usage
        assert actual == {"input_tokens": 9, "output_tokens": 12, "total_tokens": 21}

    def _validate_model_name(trace):
        # Find the ChatOpenAI span
        chat_model_span = next(s for s in trace.data.spans if s.name == "ChatOpenAI")
        assert chat_model_span.model_name == "gpt-3.5-turbo"

    def _validate_cost(trace):
        # Find the ChatOpenAI span
        chat_model_span = next(s for s in trace.data.spans if s.name == "ChatOpenAI")
        assert chat_model_span.llm_cost == {
            "input_cost": 9.0,
            "output_cost": 24.0,
            "total_cost": 33.0,
        }

    # Normal invoke
    model.invoke({"product": "MLflow"})
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    _validate_token_counts(trace)
    _validate_model_name(trace)
    _validate_cost(trace)

    # Invoke with streaming
    list(model.stream({"product": "MLflow"}))
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    _validate_token_counts(trace)
    _validate_model_name(trace)
    _validate_cost(trace)

    # Async invoke
    await model.ainvoke({"product": "MLflow"})
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    _validate_token_counts(trace)
    _validate_model_name(trace)
    _validate_cost(trace)

    # When both OpenAI and LangChain autologging is enabled,
    # no duplicated token usage should be logged
    mlflow.openai.autolog()

    model.invoke({"product": "MLflow"})
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    _validate_token_counts(trace)
    _validate_model_name(trace)
    _validate_cost(trace)


@pytest.mark.parametrize("log_traces", [True, False, None])
def test_langchain_tracer_injection_for_arbitrary_runnables(log_traces, async_logging_enabled):
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


@skip_when_testing_trace_sdk
@pytest.mark.skip(reason="This test is not thread safe, please run locally")
def test_set_retriever_schema_work_for_langchain_model(model_info):
    from mlflow.models.dependencies_schemas import DependenciesSchemasType, set_retriever_schema

    set_retriever_schema(
        primary_key="primary-key",
        text_column="text-column",
        doc_uri="doc-uri",
        other_columns=["column1", "column2"],
    )

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
        # Allow langchain.globals and its dependencies for langchain-core 0.3.76 compatibility
        allowed_langchain_modules = {
            "langchain.globals",
            "langchain._api",
            "langchain._api.interactive_env",
        }
        if name.startswith("langchain.") and name not in allowed_langchain_modules:
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


@skip_when_testing_trace_sdk
def test_langchain_auto_tracing_in_serving_runnable(model_info):
    mlflow.langchain.autolog()

    expected_output = '[{"role": "user", "content": "What is MLflow?"}]'
    databricks_request_id, predictions, trace = score_in_model_serving(
        model_info.model_uri,
        [{"product": "MLflow"}],
    )

    assert predictions == [expected_output]
    trace = Trace.from_dict(trace)
    assert trace.info.trace_id.startswith("tr-")
    assert trace.info.client_request_id == databricks_request_id
    assert trace.info.request_metadata[TRACE_SCHEMA_VERSION_KEY] == "3"
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


@pytest.mark.skipif(not IS_LANGCHAIN_v1, reason="create_agent is not supported in langchain v0")
@skip_when_testing_trace_sdk
def test_langchain_auto_tracing_in_serving_agent():
    mlflow.langchain.autolog()

    input_example = {"input": "What is 2 * 3?"}

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            "tests/langchain/sample_code/openai_agent.py",
            name="langchain_model",
            input_example=input_example,
        )

    databricks_request_id, response, trace_dict = score_in_model_serving(
        model_info.model_uri,
        input_example,
    )

    trace = Trace.from_dict(trace_dict)
    assert trace.info.trace_id.startswith("tr-")
    assert trace.info.client_request_id == databricks_request_id
    assert trace.info.status == "OK"

    spans = trace.data.spans
    assert len(spans) == 7

    root_span = spans[0]
    assert root_span.name == "LangGraph"
    assert root_span.span_type == SpanType.CHAIN
    assert root_span.inputs["input"] == "What is 2 * 3?"
    assert root_span.outputs["messages"][-1]["content"] == "The result of 2 * 3 is 6."
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


@skip_when_testing_trace_sdk
@pytest.mark.parametrize("func", ["invoke", "batch", "stream"])
def test_autolog_link_traces_to_loaded_model(model_infos, func):
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
        assert trace.info.request_metadata[TraceMetadataKey.MODEL_ID] == logged_model_id


@skip_when_testing_trace_sdk
@pytest.mark.parametrize("func", ["ainvoke", "abatch", "astream"])
@pytest.mark.asyncio
async def test_autolog_link_traces_to_loaded_model_async(model_infos, func):
    mlflow.langchain.autolog()

    for model_info in model_infos:
        loaded_model = mlflow.langchain.load_model(model_info.model_uri)
        msg = {"product": f"{loaded_model.steps[1].temperature}_{model_info.model_id}"}
        if func == "ainvoke":
            await loaded_model.ainvoke(msg)
        elif func == "abatch":
            await loaded_model.abatch([msg])
        elif func == "astream":
            async for chunk in loaded_model.astream(msg):
                pass

    traces = get_traces()
    assert len(traces) == len(model_infos)
    for trace in traces:
        temp = trace.data.spans[2].get_attribute("invocation_params")["temperature"]
        logged_temp, logged_model_id = json.loads(trace.data.request)["product"].split(
            "_", maxsplit=1
        )
        assert logged_model_id is not None
        assert str(temp) == logged_temp
        assert trace.info.request_metadata[TraceMetadataKey.MODEL_ID] == logged_model_id


@skip_when_testing_trace_sdk
def test_autolog_link_traces_to_loaded_model_pyfunc(model_infos):
    mlflow.langchain.autolog()

    for model_info in model_infos:
        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        loaded_model.predict({"product": model_info.model_id})

    traces = get_traces()
    assert len(traces) == len(model_infos)
    for trace in traces:
        logged_model_id = json.loads(trace.data.request)["product"]
        assert logged_model_id is not None
        assert trace.info.request_metadata[TraceMetadataKey.MODEL_ID] == logged_model_id


@skip_when_testing_trace_sdk
def test_autolog_link_traces_to_active_model(model_infos):
    model = mlflow.create_external_model(name="test_model")
    mlflow.set_active_model(model_id=model.model_id)
    mlflow.langchain.autolog()

    for model_info in model_infos:
        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        loaded_model.predict({"product": model_info.model_id})

    traces = get_traces()
    assert len(traces) == len(model_infos)
    for trace in traces:
        logged_model_id = json.loads(trace.data.request)["product"]
        assert logged_model_id is not None
        assert trace.info.request_metadata[TraceMetadataKey.MODEL_ID] == model.model_id
        assert model.model_id != logged_model_id


@skip_when_testing_trace_sdk
def test_model_loading_set_active_model_id_without_fetching_logged_model(model_info):
    mlflow.langchain.autolog()

    with mock.patch("mlflow.get_logged_model", side_effect=Exception("get_logged_model failed")):
        loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    loaded_model.invoke({"product": "MLflow"})

    traces = get_traces()
    assert len(traces) == 1
    model_id = traces[0].info.request_metadata[TraceMetadataKey.MODEL_ID]
    assert model_id == model_info.model_id


@skip_when_testing_trace_sdk
@pytest.mark.parametrize("log_traces", [True, False])
def test_langchain_tracing_evaluate(log_traces):
    from mlflow.genai import scorer

    if log_traces:
        mlflow.langchain.autolog()
        mlflow.openai.autolog()  # Our chain contains OpenAI call as well

    chain = create_openai_runnable()

    data = [
        {
            "inputs": {"product": "MLflow"},
            "expectations": {"expected_response": "MLflow is an open-source platform."},
        },
        {
            "inputs": {"product": "Spark"},
            "expectations": {"expected_response": "Spark is a unified analytics engine."},
        },
    ]

    def predict_fn(product: str) -> str:
        return chain.invoke({"product": product})

    @scorer
    def exact_match(outputs: str, expectations: dict[str, str]) -> bool:
        return outputs == expectations["expected_response"]

    result = mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=data,
        scorers=[exact_match],
    )
    assert result.metrics["exact_match/mean"] == 0.0
    assert result.result_df is not None

    # Traces should be enabled automatically
    assert len(get_traces()) == 2
    for trace in get_traces():
        assert len(trace.data.spans) == 5
        assert trace.data.spans[0].name == "RunnableSequence"
        assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == result.run_id
        assert len(trace.info.assessments) == 2


@pytest.mark.asyncio
async def test_autolog_run_tracer_inline_with_manual_traces_async():
    mlflow.langchain.autolog(run_tracer_inline=True)

    prompt = PromptTemplate(
        input_variables=["color"],
        template="What is the complementary color of {color}?",
    )
    llm = ChatOpenAI()

    @mlflow.trace
    def manual_transform(s: str):
        return s.replace("red", "blue")

    chain = RunnableLambda(manual_transform) | prompt | llm | StrOutputParser()

    @mlflow.trace(name="parent")
    async def run(message):
        return await chain.ainvoke(message)

    response = await run("red")
    expected_response = '[{"role": "user", "content": "What is the complementary color of blue?"}]'
    assert response == expected_response

    traces = get_traces()
    assert len(traces) == 1

    trace = traces[0]
    spans = trace.data.spans
    assert spans[0].name == "parent"
    assert spans[1].name == "RunnableSequence"
    assert spans[1].parent_id == spans[0].span_id
    assert spans[2].name == "manual_transform"
    assert spans[2].parent_id == spans[1].span_id
    # Find and verify ChatOpenAI span has model name
    chat_model_span = next(s for s in spans if s.name == "ChatOpenAI")
    assert chat_model_span.model_name == "gpt-3.5-turbo"
