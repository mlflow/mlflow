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
        pd.DataFrame(
            {"query_str": ["string"], "custom_embedding_strs": [["string"]], "embedding": [[1.0]]}
        ),
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
            pd.DataFrame(
                {
                    "query_str": ["string"],
                    "custom_embedding_strs": [["string"]],
                    "embedding": [[1.0]],
                }
            )
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
        pd.DataFrame(
            {"query_str": ["string"], "custom_embedding_strs": [["string"]], "embedding": [[1.0]]}
        ),
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
            pd.DataFrame(
                {
                    "query_str": ["string"],
                    "custom_embedding_strs": [["string"]],
                    "embedding": [[1.0]],
                }
            )
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
    payload.update(
        {
            "custom_embedding_strs": ["a"],
            "embedding": [1.0],
        }
    )
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
    input = pd.DataFrame(
        {
            "message": ["string"],
            _CHAT_MESSAGE_HISTORY_PARAMETER_NAME: [[{"role": "user", "content": "string"}]],
        }
    )
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

    results = wrapper.predict(
        [
            {"name": "Bob"},
            {"name": "Charlie"},
        ]
    )
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
    from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

    class MyWorkflow(Workflow):
        @step
        async def my_step(self, ev: StartEvent) -> StopEvent:
            raise ValueError("Expected error")

    w = MyWorkflow(timeout=10, verbose=False)
    wrapper = create_pyfunc_wrapper(w)

    with pytest.raises(ValueError, match="Expected error"):
        wrapper.predict({"name": "Alice"})
