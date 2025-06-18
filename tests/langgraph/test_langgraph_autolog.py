import json

import mlflow
from mlflow.entities.span import SpanType
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.constant import TokenUsageKey, TraceMetadataKey

from tests.tracing.helper import get_traces, skip_when_testing_trace_sdk


@skip_when_testing_trace_sdk
def test_langgraph_save_as_code():
    input_example = {"messages": [{"role": "user", "content": "what is the weather in sf?"}]}

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            "tests/langgraph/sample_code/langgraph_prebuilt.py",
            name="langgraph",
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


@skip_when_testing_trace_sdk
def test_langgraph_tracing_prebuilt():
    mlflow.langchain.autolog()

    input_example = {"messages": [{"role": "user", "content": "what is the weather in sf?"}]}

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            "tests/langgraph/sample_code/langgraph_prebuilt.py",
            name="langgraph",
            input_example=input_example,
        )

    loaded_graph = mlflow.langchain.load_model(model_info.model_uri)

    # No trace should be created for the first call
    assert mlflow.get_trace(mlflow.get_last_active_trace_id()) is None

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

    # Validate token usage
    token_usage = json.loads(traces[0].info.trace_metadata[TraceMetadataKey.TOKEN_USAGE])
    assert token_usage == {
        TokenUsageKey.INPUT_TOKENS: 15,
        TokenUsageKey.OUTPUT_TOKENS: 30,
        TokenUsageKey.TOTAL_TOKENS: 45,
    }


@skip_when_testing_trace_sdk
def test_langgraph_tracing_diy_graph():
    mlflow.langchain.autolog()

    input_example = {"messages": [{"role": "user", "content": "hi"}]}

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            "tests/langgraph/sample_code/langgraph_diy.py",
            name="langgraph",
        )

    loaded_graph = mlflow.langchain.load_model(model_info.model_uri)
    loaded_graph.invoke(input_example)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].data.spans[0].name == "LangGraph"
    assert traces[0].data.spans[0].inputs == input_example

    chat_spans = [span for span in traces[0].data.spans if span.name.startswith("ChatOpenAI")]
    assert len(chat_spans) == 3


@skip_when_testing_trace_sdk
def test_langgraph_tracing_with_custom_span():
    mlflow.langchain.autolog()

    input_example = {"messages": [{"role": "user", "content": "what is the weather in sf?"}]}

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            "tests/langgraph/sample_code/langgraph_with_custom_span.py",
            name="langgraph",
            input_example=input_example,
        )

    loaded_graph = mlflow.langchain.load_model(model_info.model_uri)

    # No trace should be created for the first call
    assert mlflow.get_trace(mlflow.get_last_active_trace_id()) is None

    loaded_graph.invoke(input_example)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].data.spans[0].name == "LangGraph"
    assert traces[0].data.spans[0].inputs == input_example

    spans = traces[0].data.spans

    # Validate chat model spans
    chat_spans = [s for s in spans if s.span_type == SpanType.CHAT_MODEL]
    assert len(chat_spans) == 3

    # Validate tool span
    tool_span = next(s for s in spans if s.span_type == SpanType.TOOL)
    assert tool_span.name == "get_weather"
    assert tool_span.inputs == {"city": "sf"}
    assert tool_span.outputs["content"] == "It's always sunny in sf"
    assert tool_span.outputs["status"] == "success"
    assert tool_span.status.status_code == SpanStatusCode.OK

    # Validate inner span
    inner_span = next(s for s in spans if s.name == "get_weather_inner")
    assert inner_span.parent_id == tool_span.span_id
    assert inner_span.inputs == "sf"
    assert inner_span.outputs == "It's always sunny in sf"

    inner_runnable_span = next(s for s in spans if s.parent_id == inner_span.span_id)
    assert inner_runnable_span.name == "RunnableSequence_2"


@skip_when_testing_trace_sdk
def test_langgraph_chat_agent_trace():
    input_example = {"messages": [{"role": "user", "content": "hi"}]}

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="agent",
            python_model="tests/langgraph/sample_code/langgraph_chat_agent.py",
        )
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    # No trace should be created for loading it in
    assert mlflow.get_trace(mlflow.get_last_active_trace_id()) is None

    loaded_model.predict(input_example)
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].info.request_metadata[TraceMetadataKey.MODEL_ID] == model_info.model_id
    assert traces[0].data.spans[0].name == "LangGraph"
    assert traces[0].data.spans[0].inputs == input_example

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    list(loaded_model.predict_stream(input_example))
    traces = get_traces()
    assert len(traces) == 2
    assert traces[0].info.status == "OK"
    assert traces[0].info.request_metadata[TraceMetadataKey.MODEL_ID] == model_info.model_id
    assert traces[0].data.spans[0].name == "LangGraph"
    assert traces[0].data.spans[0].inputs == input_example


@skip_when_testing_trace_sdk
def test_langgraph_autolog_with_update_current_span():
    model_info = mlflow.langchain.log_model(
        lc_model="tests/langgraph/sample_code/langgraph_with_autolog.py",
        input_example={"status": "done"},
    )
    assert model_info.signature is not None
    assert model_info.signature.inputs is not None
    assert model_info.signature.outputs is not None
