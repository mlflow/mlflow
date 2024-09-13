import json

import langchain
import pytest
from packaging.version import Version

import mlflow


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
