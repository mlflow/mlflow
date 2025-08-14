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
from mlflow.tracing.constant import TraceMetadataKey

from tests.tracing.helper import get_traces, skip_when_testing_trace_sdk

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


@skip_when_testing_trace_sdk
def test_autolog_should_not_generate_traces_during_logging_loading(single_index):
    mlflow.llama_index.autolog()

    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            single_index,
            name="model",
            pip_requirements=["mlflow"],
            engine_type="query",
        )
    loaded = mlflow.pyfunc.load_model(model_info.model_uri)

    assert len(get_traces()) == 0

    loaded.predict("Hello")
    assert len(get_traces()) == 1


@skip_when_testing_trace_sdk
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
def test_autolog_link_traces_to_loaded_model_engine(
    code_path, engine_type, engine_method, input_arg
):
    model_infos = []
    for i in range(3):
        with mlflow.start_run():
            model_infos.append(
                mlflow.llama_index.log_model(
                    code_path,
                    name=f"model_{i}",
                    pip_requirements=["mlflow"],
                    engine_type=engine_type,
                )
            )

    mlflow.llama_index.autolog()
    for model_info in model_infos:
        model = mlflow.llama_index.load_model(model_info.model_uri)
        getattr(model, engine_method)(f"Hello {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == 3
    for trace in traces:
        span = trace.data.spans[0]
        model_id = trace.info.request_metadata[TraceMetadataKey.MODEL_ID]
        assert model_id is not None
        assert span.inputs[input_arg] == f"Hello {model_id}"


@skip_when_testing_trace_sdk
@pytest.mark.parametrize("is_stream", [False, True])
def test_autolog_link_traces_to_loaded_model_index_query(single_index, is_stream):
    model_infos = []
    for i in range(3):
        with mlflow.start_run():
            model_infos.append(
                mlflow.llama_index.log_model(
                    single_index,
                    name=f"model_{i}",
                    pip_requirements=["mlflow"],
                    engine_type="query",
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
    assert len(traces) == 3
    for trace in traces:
        span = trace.data.spans[0]
        model_id = trace.info.request_metadata[TraceMetadataKey.MODEL_ID]
        assert span.inputs["str_or_query_bundle"] == f"Hello {model_id}"


@skip_when_testing_trace_sdk
@pytest.mark.asyncio
async def test_autolog_link_traces_to_loaded_model_index_query_async(single_index):
    model_infos = []
    for i in range(3):
        with mlflow.start_run():
            model_infos.append(
                mlflow.llama_index.log_model(
                    single_index,
                    name=f"model_{i}",
                    pip_requirements=["mlflow"],
                    engine_type="query",
                )
            )

    mlflow.llama_index.autolog()
    for model_info in model_infos:
        model = mlflow.llama_index.load_model(model_info.model_uri)
        engine = model.as_query_engine()
        await engine.aquery(f"Hello {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == 3
    for trace in traces:
        span = trace.data.spans[0]
        model_id = trace.info.request_metadata[TraceMetadataKey.MODEL_ID]
        assert span.inputs["str_or_query_bundle"] == f"Hello {model_id}"


@skip_when_testing_trace_sdk
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
def test_autolog_link_traces_to_loaded_model_index_chat(single_index, chat_mode):
    if llama_core_version >= Version("0.13.0") and chat_mode in [ChatMode.OPENAI, ChatMode.REACT]:
        pytest.skip("OpenAI and React chat modes are removed in 0.13.0")

    model_infos = []
    for i in range(3):
        with mlflow.start_run():
            model_infos.append(
                mlflow.llama_index.log_model(
                    single_index, name=f"model_{i}", pip_requirements=["mlflow"], engine_type="chat"
                )
            )

    mlflow.llama_index.autolog()
    for model_info in model_infos:
        model = mlflow.llama_index.load_model(model_info.model_uri)
        engine = model.as_chat_engine(chat_mode=chat_mode)
        engine.chat(f"Hello {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == 3
    for trace in traces:
        span = trace.data.spans[0]
        model_id = trace.info.request_metadata[TraceMetadataKey.MODEL_ID]
        assert span.inputs["message"] == f"Hello {model_id}"


@skip_when_testing_trace_sdk
def test_autolog_link_traces_to_loaded_model_index_retriever(single_index):
    model_infos = []
    for i in range(3):
        with mlflow.start_run():
            model_infos.append(
                mlflow.llama_index.log_model(
                    single_index,
                    name=f"model_{i}",
                    pip_requirements=["mlflow"],
                    engine_type="retriever",
                )
            )

    mlflow.llama_index.autolog()
    for model_info in model_infos:
        model = mlflow.llama_index.load_model(model_info.model_uri)
        engine = model.as_retriever()
        engine.retrieve(f"Hello {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == 3
    for trace in traces:
        span = trace.data.spans[0]
        model_id = trace.info.request_metadata[TraceMetadataKey.MODEL_ID]
        assert span.inputs["str_or_query_bundle"] == f"Hello {model_id}"


@skip_when_testing_trace_sdk
@pytest.mark.skipif(
    llama_core_version < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
@pytest.mark.asyncio
async def test_autolog_link_traces_to_loaded_model_workflow():
    mlflow.llama_index.autolog()
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            "tests/llama_index/sample_code/simple_workflow.py",
            name="model",
            pip_requirements=["mlflow"],
        )
    loaded_workflow = mlflow.llama_index.load_model(model_info.model_uri)
    await loaded_workflow.run(topic=f"Hello {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    model_id = traces[0].info.request_metadata[TraceMetadataKey.MODEL_ID]
    assert model_id is not None
    assert span.inputs["kwargs"]["topic"] == f"Hello {model_id}"


@skip_when_testing_trace_sdk
@pytest.mark.skipif(
    llama_core_version < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
def test_autolog_link_traces_to_loaded_model_workflow_pyfunc():
    mlflow.llama_index.autolog()
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            "tests/llama_index/sample_code/simple_workflow.py",
            name="model",
            pip_requirements=["mlflow"],
        )
    loaded_workflow = mlflow.pyfunc.load_model(model_info.model_uri)
    loaded_workflow.predict({"topic": f"Hello {model_info.model_id}"})

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    model_id = traces[0].info.request_metadata[TraceMetadataKey.MODEL_ID]
    assert model_id is not None
    assert span.inputs["kwargs"]["topic"] == f"Hello {model_id}"


@skip_when_testing_trace_sdk
@pytest.mark.skipif(
    llama_core_version < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
def test_autolog_link_traces_to_active_model():
    model = mlflow.create_external_model(name="test_model")
    mlflow.set_active_model(model_id=model.model_id)
    mlflow.llama_index.autolog()
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            "tests/llama_index/sample_code/simple_workflow.py",
            name="model",
            pip_requirements=["mlflow"],
        )
    loaded_workflow = mlflow.pyfunc.load_model(model_info.model_uri)
    loaded_workflow.predict({"topic": f"Hello {model_info.model_id}"})

    traces = get_traces()
    assert len(traces) == 1
    model_id = traces[0].info.request_metadata[TraceMetadataKey.MODEL_ID]
    assert model_id == model.model_id
    assert model_id != model_info.model_id


@skip_when_testing_trace_sdk
@pytest.mark.skipif(
    llama_core_version < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
def test_model_loading_set_active_model_id_without_fetching_logged_model():
    mlflow.llama_index.autolog()
    model_info = mlflow.llama_index.log_model(
        "tests/llama_index/sample_code/simple_workflow.py",
        name="model",
        pip_requirements=["mlflow"],
    )
    with mock.patch("mlflow.get_logged_model", side_effect=Exception("get_logged_model failed")):
        loaded_workflow = mlflow.pyfunc.load_model(model_info.model_uri)
    loaded_workflow.predict({"topic": f"Hello {model_info.model_id}"})

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    model_id = traces[0].info.request_metadata[TraceMetadataKey.MODEL_ID]
    assert model_id is not None
    assert span.inputs["kwargs"]["topic"] == f"Hello {model_id}"
