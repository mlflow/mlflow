import json

import mlflow
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
