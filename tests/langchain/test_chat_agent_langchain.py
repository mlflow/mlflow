import langchain
import pytest
from packaging.version import Version

if Version(langchain.__version__) < Version("0.2.0"):
    pytest.skip("Tests require langchain version 0.2.0 or higher", allow_module_level=True)

import mlflow


def test_langchain_chat_agent_save_as_code():
    # (role, content)
    expected_messages = [
        ("assistant", "1"),
        ("assistant", "2"),
        ("assistant", "3"),
    ]

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="agent",
            python_model="tests/langchain/sample_code/langchain_chat_agent.py",
        )
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    response = loaded_model.predict({"messages": [{"role": "user", "content": "hi"}]})
    messages = response["messages"]
    assert len(messages) == 1
    for msg, (role, expected_content) in zip(messages, expected_messages[:1]):
        assert msg["role"] == role
        assert msg["content"] == expected_content

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    responses = loaded_model.predict_stream({"messages": [{"role": "user", "content": "hi"}]})
    for response, (role, expected_content) in zip(responses, expected_messages):
        assert response["delta"]["role"] == role
        assert response["delta"]["content"] == expected_content
