import asyncio
import concurrent.futures
import contextvars
import gc
import multiprocessing
import sys
import threading
import time
from unittest import mock

import llama_index.core
import numpy as np
import pandas as pd
import pytest
from llama_index.core import QueryBundle
from llama_index.core.llms import ChatMessage
from packaging.version import Version

import mlflow
from mlflow.llama_index.pyfunc_wrapper import (
    _CHAT_MESSAGE_HISTORY_PARAMETER_NAME,
    CHAT_ENGINE_NAME,
    QUERY_ENGINE_NAME,
    RETRIEVER_ENGINE_NAME,
    create_pyfunc_wrapper,
)


################## Inferece Input #################
def test_format_predict_input_str_chat(single_index):
    wrapped_model = create_pyfunc_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input("string")
    assert formatted_data == "string"


def test_format_predict_input_dict_chat(single_index):
    wrapped_model = create_pyfunc_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input({"query": "string"})
    assert isinstance(formatted_data, dict)


def test_format_predict_input_message_history_chat(single_index):
    payload = {
        "message": "string",
        _CHAT_MESSAGE_HISTORY_PARAMETER_NAME: [{"role": "user", "content": "hi"}] * 3,
    }
    wrapped_model = create_pyfunc_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(payload)

    assert isinstance(formatted_data, dict)
    assert formatted_data["message"] == payload["message"]
    assert isinstance(formatted_data[_CHAT_MESSAGE_HISTORY_PARAMETER_NAME], list)
    assert all(
        isinstance(x, ChatMessage) for x in formatted_data[_CHAT_MESSAGE_HISTORY_PARAMETER_NAME]
    )


@pytest.mark.parametrize(
    "data",
    [
        [
            {
                "query": "string",
                _CHAT_MESSAGE_HISTORY_PARAMETER_NAME: [{"role": "user", "content": "hi"}] * 3,
            }
        ]
        * 3,
        pd.DataFrame(
            [
                {
                    "query": "string",
                    _CHAT_MESSAGE_HISTORY_PARAMETER_NAME: [{"role": "user", "content": "hi"}] * 3,
                }
            ]
            * 3
        ),
    ],
)
def test_format_predict_input_message_history_chat_iterable(single_index, data):
    wrapped_model = create_pyfunc_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(data)

    if isinstance(data, pd.DataFrame):
        data = data.to_dict("records")

    assert isinstance(formatted_data, list)
    assert formatted_data[0]["query"] == data[0]["query"]
    assert isinstance(formatted_data[0][_CHAT_MESSAGE_HISTORY_PARAMETER_NAME], list)
    assert all(
        isinstance(x, ChatMessage) for x in formatted_data[0][_CHAT_MESSAGE_HISTORY_PARAMETER_NAME]
    )


def test_format_predict_input_message_history_chat_invalid_type(single_index):
    payload = {
        "message": "string",
        _CHAT_MESSAGE_HISTORY_PARAMETER_NAME: ["invalid history string", "user: hi"],
    }
    wrapped_model = create_pyfunc_wrapper(single_index, CHAT_ENGINE_NAME)
    with pytest.raises(ValueError, match="It must be a list of dicts"):
        _ = wrapped_model._format_predict_input(payload)


@pytest.mark.parametrize(
    "data",
    [
        "string",
        ["string"],  # iterables of length 1 should be treated non-iterables
        {"query_str": "string"},
        {"query_str": "string", "custom_embedding_strs": ["string"], "embedding": [1.0]},
        pd.DataFrame({
            "query_str": ["string"],
            "custom_embedding_strs": [["string"]],
            "embedding": [[1.0]],
        }),
    ],
)
def test_format_predict_input_no_iterable_query(single_index, data):
    wrapped_model = create_pyfunc_wrapper(single_index, QUERY_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(data)
    assert isinstance(formatted_data, QueryBundle)


@pytest.mark.parametrize(
    "data",
    [
        ["string", "string"],
        [{"query_str": "string"}] * 4,
        [{"query_str": "string", "custom_embedding_strs": ["string"], "embedding": [1.0]}] * 4,
        [
            pd.DataFrame({
                "query_str": ["string"],
                "custom_embedding_strs": [["string"]],
                "embedding": [[1.0]],
            })
        ]
        * 2,
    ],
)
def test_format_predict_input_iterable_query(single_index, data):
    wrapped_model = create_pyfunc_wrapper(single_index, QUERY_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(data)

    assert isinstance(formatted_data, list)
    assert all(isinstance(x, QueryBundle) for x in formatted_data)


@pytest.mark.parametrize(
    "data",
    [
        "string",
        ["string"],  # iterables of length 1 should be treated non-iterables
        {"query_str": "string"},
        {"query_str": "string", "custom_embedding_strs": ["string"], "embedding": [1.0]},
        pd.DataFrame({
            "query_str": ["string"],
            "custom_embedding_strs": [["string"]],
            "embedding": [[1.0]],
        }),
    ],
)
def test_format_predict_input_no_iterable_retriever(single_index, data):
    wrapped_model = create_pyfunc_wrapper(single_index, RETRIEVER_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(data)
    assert isinstance(formatted_data, QueryBundle)


@pytest.mark.parametrize(
    "data",
    [
        ["string", "string"],
        [{"query_str": "string"}] * 4,
        [{"query_str": "string", "custom_embedding_strs": ["string"], "embedding": [1.0]}] * 4,
        [
            pd.DataFrame({
                "query_str": ["string"],
                "custom_embedding_strs": [["string"]],
                "embedding": [[1.0]],
            })
        ]
        * 2,
    ],
)
def test_format_predict_input_iterable_retriever(single_index, data):
    wrapped_model = create_pyfunc_wrapper(single_index, RETRIEVER_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(data)
    assert isinstance(formatted_data, list)
    assert all(isinstance(x, QueryBundle) for x in formatted_data)


@pytest.mark.parametrize(
    "engine_type",
    ["query", "retriever"],
)
def test_format_predict_input_correct(single_index, engine_type):
    wrapped_model = create_pyfunc_wrapper(single_index, engine_type)

    assert isinstance(
        wrapped_model._format_predict_input(pd.DataFrame({"query_str": ["hi"]})), QueryBundle
    )
    assert isinstance(wrapped_model._format_predict_input(np.array(["hi"])), QueryBundle)
    assert isinstance(wrapped_model._format_predict_input({"query_str": ["hi"]}), QueryBundle)
    assert isinstance(wrapped_model._format_predict_input({"query_str": "hi"}), QueryBundle)
    assert isinstance(wrapped_model._format_predict_input(["hi"]), QueryBundle)
    assert isinstance(wrapped_model._format_predict_input("hi"), QueryBundle)


@pytest.mark.parametrize(
    "engine_type",
    ["query", "retriever"],
)
def test_format_predict_input_correct_schema_complex(single_index, engine_type):
    wrapped_model = create_pyfunc_wrapper(single_index, engine_type)

    payload = {
        "query_str": "hi",
        "image_path": "some/path",
        "custom_embedding_strs": [["a"]],
        "embedding": [[1.0]],
    }
    assert isinstance(wrapped_model._format_predict_input(pd.DataFrame(payload)), QueryBundle)
    payload.update({
        "custom_embedding_strs": ["a"],
        "embedding": [1.0],
    })
    assert isinstance(wrapped_model._format_predict_input(payload), QueryBundle)


@pytest.mark.parametrize(
    ("engine_type", "input"),
    [
        ("query", {"query_str": "hello!"}),
        ("retriever", {"query_str": "hello!"}),
    ],
)
def test_spark_udf_retriever_and_query_engine(model_path, spark, single_index, engine_type, input):
    mlflow.llama_index.save_model(
        llama_index_model=single_index,
        engine_type=engine_type,
        path=model_path,
        input_example=input,
    )
    udf = mlflow.pyfunc.spark_udf(spark, model_path, result_type="string")
    df = spark.createDataFrame([{"query_str": "hi"}])
    df = df.withColumn("predictions", udf())
    pdf = df.toPandas()
    assert len(pdf["predictions"].tolist()) == 1
    assert isinstance(pdf["predictions"].tolist()[0], str)


def test_spark_udf_chat(model_path, spark, single_index):
    engine_type = "chat"
    input = pd.DataFrame({
        "message": ["string"],
        _CHAT_MESSAGE_HISTORY_PARAMETER_NAME: [[{"role": "user", "content": "string"}]],
    })
    mlflow.llama_index.save_model(
        llama_index_model=single_index,
        engine_type=engine_type,
        path=model_path,
        input_example=input,
    )
    udf = mlflow.pyfunc.spark_udf(spark, model_path, result_type="string")
    df = spark.createDataFrame(input)
    df = df.withColumn("predictions", udf())
    pdf = df.toPandas()
    assert len(pdf["predictions"].tolist()) == 1
    assert isinstance(pdf["predictions"].tolist()[0], str)


@pytest.mark.skipif(
    Version(llama_index.core.__version__) < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
@pytest.mark.asyncio
async def test_wrap_workflow():
    from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

    class MyWorkflow(Workflow):
        @step
        async def my_step(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result=f"Hi, {ev.name}!")

    w = MyWorkflow(timeout=10, verbose=False)
    wrapper = create_pyfunc_wrapper(w)
    assert wrapper.get_raw_model() == w

    result = wrapper.predict({"name": "Alice"})
    assert result == "Hi, Alice!"

    results = wrapper.predict([
        {"name": "Bob"},
        {"name": "Charlie"},
    ])
    assert results == ["Hi, Bob!", "Hi, Charlie!"]

    results = wrapper.predict(pd.DataFrame({"name": ["David"]}))
    assert results == "Hi, David!"

    results = wrapper.predict(pd.DataFrame({"name": ["Eve", "Frank"]}))
    assert results == ["Hi, Eve!", "Hi, Frank!"]


@pytest.mark.skipif(
    Version(llama_index.core.__version__) < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
@pytest.mark.asyncio
async def test_wrap_workflow_raise_exception():
    from llama_index.core.workflow import (
        StartEvent,
        StopEvent,
        Workflow,
        WorkflowRuntimeError,
        step,
    )

    class MyWorkflow(Workflow):
        @step
        async def my_step(self, ev: StartEvent) -> StopEvent:
            raise ValueError("Expected error")

    w = MyWorkflow(timeout=10, verbose=False)
    wrapper = create_pyfunc_wrapper(w)

    with pytest.raises(
        (
            ValueError,  # llama_index < 0.12.1
            WorkflowRuntimeError,  # llama_index >= 0.12.1
        ),
        match="Expected error",
    ):
        wrapper.predict({"name": "Alice"})


def _create_workflow_with_loop_bound_state():
    from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

    class MyWorkflow(Workflow):
        # Mimics an async client (e.g. AsyncOpenAI) cached on the workflow, whose state is
        # bound to the event loop that first used it.
        event = None

        @step
        async def my_step(self, ev: StartEvent) -> StopEvent:
            if self.event is None:
                self.event = asyncio.Event()
            self.event.clear()
            asyncio.get_running_loop().call_soon(self.event.set)
            await self.event.wait()
            return StopEvent(result=f"Hi, {ev.name}!")

    return MyWorkflow(timeout=10, verbose=False)


@pytest.mark.skipif(
    Version(llama_index.core.__version__) < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
def test_wrap_workflow_repeated_predict_with_loop_bound_state():
    wrapper = create_pyfunc_wrapper(_create_workflow_with_loop_bound_state())

    assert wrapper.predict({"name": "Alice"}) == "Hi, Alice!"
    assert wrapper.predict({"name": "Bob"}) == "Hi, Bob!"
    assert wrapper.predict([{"name": "Charlie"}, {"name": "David"}]) == [
        "Hi, Charlie!",
        "Hi, David!",
    ]


@pytest.mark.skipif(
    Version(llama_index.core.__version__) < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
@pytest.mark.asyncio
async def test_wrap_workflow_repeated_predict_with_loop_bound_state_in_running_loop():
    wrapper = create_pyfunc_wrapper(_create_workflow_with_loop_bound_state())

    assert wrapper.predict({"name": "Alice"}) == "Hi, Alice!"
    assert wrapper.predict({"name": "Bob"}) == "Hi, Bob!"


@pytest.mark.skipif(
    Version(llama_index.core.__version__) < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
def test_wrap_workflow_propagates_context():
    from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

    var = contextvars.ContextVar("var", default=None)

    class MyWorkflow(Workflow):
        @step
        async def my_step(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result=var.get())

    wrapper = create_pyfunc_wrapper(MyWorkflow(timeout=10, verbose=False))

    token = var.set("value")
    try:
        assert wrapper.predict({"name": "Alice"}) == "value"
    finally:
        var.reset(token)


@pytest.mark.skipif(
    Version(llama_index.core.__version__) < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
def test_wrap_workflow_stops_event_loop_on_garbage_collection():
    wrapper = create_pyfunc_wrapper(_create_workflow_with_loop_bound_state())
    assert wrapper.predict({"name": "Alice"}) == "Hi, Alice!"
    loop = wrapper._event_loop
    assert loop.is_running()

    del wrapper
    gc.collect()

    deadline = time.monotonic() + 5
    while not loop.is_closed() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert loop.is_closed()


@pytest.mark.skipif(
    Version(llama_index.core.__version__) < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
def test_wrap_workflow_cancels_task_when_wait_is_interrupted():
    wrapper = create_pyfunc_wrapper(_create_workflow_with_loop_bound_state())
    started = threading.Event()
    cancelled = threading.Event()

    async def never_finishes():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def interrupted_result(self, timeout=None):
        started.wait(5)
        raise KeyboardInterrupt("interrupted")

    with (
        mock.patch.object(concurrent.futures.Future, "result", interrupted_result),
        pytest.raises(KeyboardInterrupt, match="interrupted"),
    ):
        wrapper._wait_async_task(never_finishes())

    assert cancelled.wait(5)
    # The shared loop is still usable after the interrupted call.
    assert wrapper.predict({"name": "Alice"}) == "Hi, Alice!"


@pytest.mark.skipif(
    Version(llama_index.core.__version__) < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
def test_wrap_workflow_cancels_other_items_when_one_fails():
    from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

    cancelled = threading.Event()

    class MyWorkflow(Workflow):
        @step
        async def my_step(self, ev: StartEvent) -> StopEvent:
            if ev.name == "fail":
                raise ValueError("Expected error")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    wrapper = create_pyfunc_wrapper(MyWorkflow(timeout=10, verbose=False))

    with pytest.raises(Exception, match="Expected error"):
        wrapper.predict([{"name": "wait"}, {"name": "fail"}])

    assert cancelled.wait(5)


@pytest.mark.skipif(
    Version(llama_index.core.__version__) < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
def test_wrap_workflow_cancels_steps_when_wait_is_interrupted():
    from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

    started = threading.Event()
    cancelled = threading.Event()

    class MyWorkflow(Workflow):
        @step
        async def my_step(self, ev: StartEvent) -> StopEvent:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    wrapper = create_pyfunc_wrapper(MyWorkflow(timeout=10, verbose=False))

    def interrupted_result(self, timeout=None):
        started.wait(5)
        raise KeyboardInterrupt("interrupted")

    with (
        mock.patch.object(concurrent.futures.Future, "result", interrupted_result),
        pytest.raises(KeyboardInterrupt, match="interrupted"),
    ):
        wrapper.predict({"name": "Alice"})

    assert cancelled.wait(5)


def _predict_in_child(wrapper, queue):
    queue.put(wrapper.predict({"name": "Bob"}))


@pytest.mark.skipif(
    Version(llama_index.core.__version__) < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
@pytest.mark.skipif(sys.platform == "win32", reason="fork is not available on Windows")
def test_wrap_workflow_predict_after_fork():
    from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

    class MyWorkflow(Workflow):
        @step
        async def my_step(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result=f"Hi, {ev.name}!")

    wrapper = create_pyfunc_wrapper(MyWorkflow(timeout=10, verbose=False))
    assert wrapper.predict({"name": "Alice"}) == "Hi, Alice!"

    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(target=_predict_in_child, args=(wrapper, queue))
    process.start()
    try:
        assert queue.get(timeout=30) == "Hi, Bob!"
    finally:
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
    assert process.exitcode == 0
    assert wrapper.predict({"name": "Charlie"}) == "Hi, Charlie!"
