import json
from dataclasses import asdict

import mlflow
from mlflow.telemetry.client import get_telemetry_client
from mlflow.telemetry.parser import LogModelParams, ModelType
from mlflow.types.schema import Object, ParamSchema, ParamSpec, Property


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


def test_langgraph_model_invoke_with_dictionary_params(monkeypatch):
    input_example = {"messages": [{"role": "user", "content": "What's the weather in nyc?"}]}
    params = {"config": {"configurable": {"thread_id": "1"}}}

    monkeypatch.setenv("MLFLOW_CONVERT_MESSAGES_DICT_FOR_LANGCHAIN", "false")
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            "tests/langgraph/sample_code/langgraph_prebuilt.py",
            name="model",
            input_example=(input_example, params),
        )
    assert model_info.signature.params == ParamSchema(
        [
            ParamSpec(
                "config",
                Object([Property("configurable", Object([Property("thread_id", "string")]))]),
                params["config"],
            )
        ]
    )
    langchain_model = mlflow.langchain.load_model(model_info.model_uri)
    result = langchain_model.invoke(input_example, **params)
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert len(pyfunc_model.predict(input_example, params)[0]["messages"]) == len(
        result["messages"]
    )


def test_log_model_sends_telemetry_record(mock_requests):
    """Test that log_model sends telemetry records."""
    mlflow.langchain.log_model(
        "tests/langgraph/sample_code/langgraph_prebuilt.py",
        name="langgraph",
        input_example={"messages": [{"role": "user", "content": "what is the weather in sf?"}]},
    )
    # Wait for telemetry to be sent
    get_telemetry_client().flush()

    # Check that telemetry record was sent
    assert len(mock_requests) == 1
    record = mock_requests[0]
    data = json.loads(record["data"])
    assert data["api_module"] == mlflow.langchain.log_model.__module__
    assert data["api_name"] == "log_model"
    assert data["params"] == asdict(
        LogModelParams(
            flavor="langchain",
            model=ModelType.MODEL_PATH,
            is_pip_requirements_set=False,
            is_extra_pip_requirements_set=False,
            is_code_paths_set=False,
            is_params_set=False,
            is_metadata_set=False,
        )
    )
    assert data["status"] == "success"
