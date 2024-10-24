import json

import langchain
import pytest
from packaging.version import Version

import mlflow
from mlflow.entities.span import SpanType
from mlflow.entities.span_status import SpanStatusCode

from tests.tracing.helper import get_traces


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.2.0"),
    reason="Agent behavior is not stable across minor versions",
)
def test_langgraph_save_as_code():
    input_example = {"messages": [{"role": "user", "content": "what is the weather in sf?"}]}

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            lc_model="tests/langchain/sample_code/langgraph.py",
            artifact_path="langgraph",
            input_example=input_example,
        )

    # (role, content)
    expected_messages = [
        ("human", "what is the weather in sf?"),
        ("agent", ""),  # tool message does not have content
        ("tools", "It's always sunny in sf"),
        ("agent", "The weather in San Francisco is always sunny!"),
    ]

    loaded_graph = mlflow.langchain.load_model(model_info.model_uri)
    response = loaded_graph.invoke(input_example)
    messages = response["messages"]
    assert len(messages) == 4
    for msg, (role, expected_content) in zip(messages, expected_messages):
        assert msg.content == expected_content

    # Need to reload to reset the iterator in FakeOpenAI
    loaded_graph = mlflow.langchain.load_model(model_info.model_uri)
    response = loaded_graph.stream(input_example)
    # .stream() response does not includes the first Human message
    for chunk, (role, expected_content) in zip(response, expected_messages[1:]):
        assert chunk[role]["messages"][0].content == expected_content

    loaded_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)
    response = loaded_pyfunc.predict(input_example)[0]
    messages = response["messages"]
    assert len(messages) == 4
    for msg, (role, expected_content) in zip(messages, expected_messages):
        assert msg["content"] == expected_content
    # response should be json serializable
    assert json.dumps(response) is not None

    loaded_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)
    response = loaded_pyfunc.predict_stream(input_example)
    for chunk, (role, expected_content) in zip(response, expected_messages[1:]):
        assert chunk[role]["messages"][0]["content"] == expected_content


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.2.0"),
    reason="Agent behavior is not stable across minor versions",
)
def test_langgraph_tracing():
    mlflow.langchain.autolog()

    input_example = {"messages": [{"role": "user", "content": "what is the weather in sf?"}]}

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            lc_model="tests/langchain/sample_code/langgraph.py",
            artifact_path="langgraph",
            input_example=input_example,
        )

    loaded_graph = mlflow.langchain.load_model(model_info.model_uri)

    # No trace should be created for the first call
    assert mlflow.get_last_active_trace() is None

    loaded_graph.invoke(input_example)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].data.spans[0].name == "LangGraph"
    assert traces[0].data.spans[0].inputs == input_example

    # (type, content)
    expected_messages = [
        ("human", "what is the weather in sf?"),
        ("ai", ""),  # tool message does not have content
        ("tool", "It's always sunny in sf"),
        ("ai", "The weather in San Francisco is always sunny!"),
    ]

    messages = traces[0].data.spans[0].outputs["messages"]
    assert len(messages) == 4
    for msg, (type, expected_content) in zip(messages, expected_messages):
        assert msg["type"] == type
        assert msg["content"] == expected_content

    # Validate tool span
    tool_span = next(span for span in traces[0].data.spans if span.span_type == SpanType.TOOL)
    assert tool_span.name == "get_weather"
    assert tool_span.inputs == {"city": "sf"}
    assert tool_span.outputs["content"] == "It's always sunny in sf"
    assert tool_span.outputs["status"] == "success"
    assert tool_span.status.status_code == SpanStatusCode.OK
