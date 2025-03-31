import asyncio
from unittest import mock

import importlib_metadata
import pytest
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers.base import BaseEventHandler
from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler
from llama_index.llms.openai import OpenAI
from packaging.version import Version

import mlflow
from mlflow.models.model import _MODEL_TRACKER
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces

llama_core_version = Version(importlib_metadata.version("llama-index-core"))


def test_autolog_enable_tracing(multi_index):
    mlflow.llama_index.autolog()

    query_engine = multi_index.as_query_engine()

    query_engine.query("Hello")
    query_engine.query("Hola")

    traces = get_traces()
    assert len(traces) == 2

    # Enabling autolog multiple times should not create duplicate spans
    mlflow.llama_index.autolog()
    mlflow.llama_index.autolog()

    chat_engine = multi_index.as_chat_engine()
    chat_engine.chat("Hello again!")

    assert len(get_traces()) == 3

    mlflow.llama_index.autolog(disable=True)
    query_engine.query("Hello again!")

    traces = get_traces()
    assert len(get_traces()) == 3


def test_autolog_preserve_user_provided_handlers():
    user_span_handler = mock.MagicMock(spec=BaseSpanHandler)
    user_event_handler = mock.MagicMock(spec=BaseEventHandler)

    dsp = get_dispatcher()
    dsp.add_span_handler(user_span_handler)
    dsp.add_event_handler(user_event_handler)

    mlflow.llama_index.autolog()

    llm = OpenAI()
    llm.complete("Hello")

    assert user_span_handler in dsp.span_handlers
    assert user_event_handler in dsp.event_handlers
    user_span_handler.span_enter.assert_called_once()
    user_span_handler.span_exit.assert_called_once()
    assert user_event_handler.handle.call_count == 2  # LLM start + end

    traces = get_traces()
    assert len(traces) == 1

    user_span_handler.reset_mock()
    user_event_handler.reset_mock()

    mlflow.llama_index.autolog(disable=True)

    assert user_span_handler in dsp.span_handlers
    assert user_event_handler in dsp.event_handlers

    llm.complete("Hello, again!")

    user_span_handler.span_enter.assert_called_once()
    user_span_handler.span_exit.assert_called_once()
    assert user_event_handler.handle.call_count == 2

    traces = get_traces()
    assert len(traces) == 1


def test_autolog_should_not_generate_traces_during_logging_loading(single_index):
    mlflow.llama_index.autolog()

    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            single_index, "model", input_example="Hello", engine_type="query"
        )
    loaded = mlflow.pyfunc.load_model(model_info.model_uri)

    assert len(get_traces()) == 0

    loaded.predict("Hello")
    assert len(get_traces()) == 1


@pytest.mark.parametrize(
    ("code_path", "engine_type", "engine_method", "input_arg"),
    [
        (
            "tests/llama_index/sample_code/query_engine_with_reranker.py",
            "query",
            "query",
            "str_or_query_bundle",
        ),
        ("tests/llama_index/sample_code/basic_chat_engine.py", "chat", "chat", "message"),
        (
            "tests/llama_index/sample_code/basic_retriever.py",
            "retriever",
            "retrieve",
            "str_or_query_bundle",
        ),
    ],
)
def test_autolog_link_traces_to_logged_model_engine(
    code_path, engine_type, engine_method, input_arg
):
    model_infos = []
    for i in range(5):
        with mlflow.start_run():
            model_infos.append(
                mlflow.llama_index.log_model(
                    code_path, f"model_{i}", input_example="Hello", engine_type=engine_type
                )
            )

    mlflow.llama_index.autolog()
    for model_info in model_infos:
        model = mlflow.llama_index.load_model(model_info.model_uri)
        getattr(model, engine_method)(f"Hello {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == 5
    for trace in traces:
        span = trace.data.spans[0]
        model_id = span.get_attribute(SpanAttributeKey.MODEL_ID)
        assert model_id is not None
        assert span.inputs[input_arg] == f"Hello {model_id}"


@pytest.mark.parametrize("is_stream", [False, True])
def test_autolog_link_traces_to_logged_model_index_query(single_index, is_stream):
    model_infos = []
    for i in range(5):
        with mlflow.start_run():
            model_infos.append(
                mlflow.llama_index.log_model(
                    single_index, f"model_{i}", input_example="Hello", engine_type="query"
                )
            )

    mlflow.llama_index.autolog()
    for model_info in model_infos:
        model = mlflow.llama_index.load_model(model_info.model_uri)
        engine = model.as_query_engine(streaming=is_stream)
        response = engine.query(f"Hello {model_info.model_id}")
        if is_stream:
            response = "".join(response.response_gen)

    traces = get_traces()
    assert len(traces) == 5
    for trace in traces:
        span = trace.data.spans[0]
        model_id = span.get_attribute(SpanAttributeKey.MODEL_ID)
        assert span.inputs["str_or_query_bundle"] == f"Hello {model_id}"


@pytest.mark.asyncio
async def test_autolog_link_traces_to_logged_model_index_query_async(single_index):
    model_infos = []
    for i in range(5):
        with mlflow.start_run():
            model_infos.append(
                mlflow.llama_index.log_model(
                    single_index, f"model_{i}", input_example="Hello", engine_type="query"
                )
            )

    mlflow.llama_index.autolog()
    for model_info in model_infos:
        model = mlflow.llama_index.load_model(model_info.model_uri)
        engine = model.as_query_engine()
        await engine.aquery(f"Hello {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == 5
    for trace in traces:
        span = trace.data.spans[0]
        model_id = span.get_attribute(SpanAttributeKey.MODEL_ID)
        assert span.inputs["str_or_query_bundle"] == f"Hello {model_id}"


@pytest.mark.parametrize(
    "chat_mode",
    [
        ChatMode.BEST,
        ChatMode.CONTEXT,
        ChatMode.CONDENSE_QUESTION,
        ChatMode.CONDENSE_PLUS_CONTEXT,
        ChatMode.SIMPLE,
        ChatMode.REACT,
        ChatMode.OPENAI,
    ],
)
def test_autolog_link_traces_to_logged_model_index_chat(single_index, chat_mode):
    model_infos = []
    for i in range(5):
        with mlflow.start_run():
            model_infos.append(
                mlflow.llama_index.log_model(
                    single_index, f"model_{i}", input_example="Hello", engine_type="chat"
                )
            )

    mlflow.llama_index.autolog()
    for model_info in model_infos:
        model = mlflow.llama_index.load_model(model_info.model_uri)
        engine = model.as_chat_engine(chat_mode=chat_mode)
        engine.chat(f"Hello {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == 5
    for trace in traces:
        span = trace.data.spans[0]
        model_id = span.get_attribute(SpanAttributeKey.MODEL_ID)
        assert span.inputs["message"] == f"Hello {model_id}"


def test_autolog_link_traces_to_logged_model_index_retriever(single_index):
    model_infos = []
    for i in range(5):
        with mlflow.start_run():
            model_infos.append(
                mlflow.llama_index.log_model(
                    single_index, f"model_{i}", input_example="Hello", engine_type="retriever"
                )
            )

    mlflow.llama_index.autolog()
    for model_info in model_infos:
        model = mlflow.llama_index.load_model(model_info.model_uri)
        engine = model.as_retriever()
        engine.retrieve(f"Hello {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == 5
    for trace in traces:
        span = trace.data.spans[0]
        model_id = span.get_attribute(SpanAttributeKey.MODEL_ID)
        assert span.inputs["str_or_query_bundle"] == f"Hello {model_id}"


@pytest.mark.skipif(
    llama_core_version < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
@pytest.mark.asyncio
async def test_autolog_link_traces_to_logged_model_workflow():
    mlflow.llama_index.autolog()
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            "tests/llama_index/sample_code/simple_workflow.py",
            "model",
            input_example={"topic": "Hello"},
        )
    loaded_workflow = mlflow.llama_index.load_model(model_info.model_uri)
    await loaded_workflow.run(topic=f"Hello {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    model_id = span.get_attribute(SpanAttributeKey.MODEL_ID)
    assert model_id is not None
    assert span.inputs["kwargs"]["topic"] == f"Hello {model_id}"


def test_autolog_link_traces_to_original_model_after_logging(single_index):
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            single_index, "model", input_example="Hello", engine_type="query"
        )

    mlflow.llama_index.autolog()
    engine = single_index.as_query_engine()
    engine.query(f"Hello {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    model_id = span.get_attribute(SpanAttributeKey.MODEL_ID)
    assert model_id is not None
    assert span.inputs["str_or_query_bundle"] == f"Hello {model_id}"


@pytest.mark.parametrize("is_async", [False, True])
def test_autolog_create_logged_model_and_link_traces_index(single_index, is_async):
    mlflow.llama_index.autolog()

    with mlflow.start_run() as run:
        engine = single_index.as_query_engine()
        for _ in range(5):
            if is_async:
                asyncio.run(engine.aquery("Hello"))
            else:
                engine.query("Hello")
    logged_models = mlflow.search_logged_models(
        filter_string=f"source_run_id='{run.info.run_id}'", output_format="list"
    )
    assert len(logged_models) == 1
    logged_model = logged_models[0]
    traces = get_traces()
    assert len(traces) == 5
    for i in range(5):
        assert (
            traces[i].data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID)
            == logged_model.model_id
        )

    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            single_index, "model", input_example="Hello", engine_type="query"
        )
    loaded_index = mlflow.llama_index.load_model(model_info.model_uri)
    if is_async:
        asyncio.run(loaded_index.as_query_engine().aquery("Hello"))
        asyncio.run(loaded_index.as_chat_engine().achat("Hello"))
        asyncio.run(loaded_index.as_retriever().aretrieve("Hello"))
    else:
        loaded_index.as_query_engine().query("Hello")
        loaded_index.as_chat_engine().chat("Hello")
        loaded_index.as_retriever().retrieve("Hello")
    traces = get_traces()
    assert len(traces) == 8
    for i in range(3):
        assert (
            traces[i].data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID) == model_info.model_id
        )


@pytest.mark.parametrize("is_async", [True, False])
def test_autolog_create_logged_model_and_link_traces_engine(single_index, is_async):
    engine = single_index.as_query_engine()
    mlflow.llama_index.autolog()

    with mlflow.start_run() as run:
        for _ in range(5):
            if is_async:
                asyncio.run(engine.aquery("Hello"))
            else:
                engine.query("Hello")
    logged_models = mlflow.search_logged_models(
        filter_string=f"source_run_id='{run.info.run_id}'", output_format="list"
    )
    assert len(logged_models) == 1
    logged_model = logged_models[0]
    traces = get_traces()
    assert len(traces) == 5
    for i in range(5):
        assert (
            traces[i].data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID)
            == logged_model.model_id
        )
    # This is required because settings contains OpenAIEmbedding, it might introduce
    # some side effect on tracing when two tests run together
    mlflow.llama_index.autolog(disable=True)


@pytest.mark.skipif(
    llama_core_version < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
def test_autolog_create_logged_model_and_link_traces_workflow():
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            "tests/llama_index/sample_code/simple_workflow.py",
            "model",
            input_example={"topic": "Hello"},
        )
    workflow = mlflow.llama_index.load_model(model_info.model_uri)
    # clear here to mimic the model is not loaded, only for testing
    _MODEL_TRACKER.clear()

    # This is needed since pytest.asyncio doesn't function well
    async def run_workflow(topic):
        await workflow.run(topic=topic)

    mlflow.llama_index.autolog()

    with mlflow.start_run() as run:
        for i in range(5):
            asyncio.run(run_workflow("Hello"))
            traces = get_traces()
            assert len(traces) == i + 1
    logged_models = mlflow.search_logged_models(
        filter_string=f"source_run_id='{run.info.run_id}'", output_format="list"
    )
    assert len(logged_models) == 1
    logged_model = logged_models[0]
    traces = get_traces()
    assert len(traces) == 5
    for i in range(5):
        assert (
            traces[i].data.spans[0].get_attribute(SpanAttributeKey.MODEL_ID)
            == logged_model.model_id
        )
