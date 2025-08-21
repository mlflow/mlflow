# Test integration with the `databricks-langchain` package.
from typing import Generator
from unittest import mock

import langchain
import pytest
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from openai.types.chat.chat_completion import ChatCompletion
from packaging.version import Version

import mlflow

_MOCK_CHAT_RESPONSE = {
    "id": "chatcmpl_id",
    "object": "chat.completion",
    "created": 1721875529,
    "model": "meta-llama-3.1-70b-instruct-072424",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "What is MLflow?",
            },
            "finish_reason": "stop",
            "logprobs": None,
        }
    ],
    "usage": {"prompt_tokens": 30, "completion_tokens": 36, "total_tokens": 66},
}


@pytest.fixture(autouse=True)
def mock_client(monkeypatch) -> Generator:
    # In databricks-langchain <= 0.7.0, ChatDatabricks uses MLflow deployment client
    deploy_client = mock.MagicMock()
    deploy_client.predict.return_value = _MOCK_CHAT_RESPONSE
    # For newer version, ChatDatabricks uses workspace OpenAI client
    openai_client = mock.MagicMock()
    openai_client.chat.completions.create.return_value = ChatCompletion.validate(
        _MOCK_CHAT_RESPONSE
    )

    with (
        mock.patch("mlflow.deployments.get_deploy_client", return_value=deploy_client),
        mock.patch(
            "databricks_langchain.chat_models.get_openai_client", return_value=openai_client
        ),
    ):
        yield


@pytest.fixture
def model_path(tmp_path):
    return tmp_path / "model"


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.3.0"),
    reason="databricks-langchain requires langchain >= 0.3.0",
)
def test_save_and_load_chat_databricks(model_path):
    from databricks_langchain import ChatDatabricks

    llm = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct")
    prompt = PromptTemplate.from_template("What is {product}?")
    chain = prompt | llm | StrOutputParser()

    mlflow.langchain.save_model(chain, path=model_path)

    loaded_model = mlflow.langchain.load_model(model_path)
    assert loaded_model == chain

    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_path)
    prediction = loaded_pyfunc_model.predict([{"product": "MLflow"}])
    assert prediction == ["What is MLflow?"]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.3.0"),
    reason="databricks-langchain requires langchain >= 0.3.0",
)
def test_save_and_load_chat_databricks_legacy(model_path):
    # Test saving and loading the community version of ChatDatabricks
    from langchain.chat_models import ChatDatabricks

    llm = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct")
    prompt = PromptTemplate.from_template("What is {product}?")
    chain = prompt | llm | StrOutputParser()

    mlflow.langchain.save_model(chain, path=model_path)

    loaded_model = mlflow.langchain.load_model(model_path)
    assert loaded_model == chain

    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_path)
    prediction = loaded_pyfunc_model.predict([{"product": "MLflow"}])
    assert prediction == ["What is MLflow?"]
