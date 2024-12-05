# Test integration with the `langchain-databricks` package.
from typing import Generator
from unittest import mock

import langchain
import pytest
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
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
                "role": "user",
                "content": "What is MLflow?",
            },
            "finish_reason": "stop",
            "logprobs": None,
        }
    ],
    "usage": {"prompt_tokens": 30, "completion_tokens": 36, "total_tokens": 66},
}


@pytest.fixture(autouse=True)
def mock_client() -> Generator:
    client = mock.MagicMock()
    client.predict.return_value = _MOCK_CHAT_RESPONSE
    with mock.patch("mlflow.deployments.get_deploy_client", return_value=client):
        yield


@pytest.fixture
def model_path(tmp_path):
    return tmp_path / "model"


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.2.0"),
    reason="langchain-databricks requires langchain >= 0.2.0",
)
def test_save_and_load_chat_databricks(model_path):
    from langchain_databricks import ChatDatabricks

    llm = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct")
    prompt = PromptTemplate.from_template("What is {product}?")
    chain = prompt | llm | StrOutputParser()

    mlflow.langchain.save_model(chain, path=model_path)

    with model_path.joinpath("requirements.txt").open() as f:
        reqs = {req.split("==")[0] for req in f.read().split("\n")}
    assert "langchain-databricks" in reqs

    loaded_model = mlflow.langchain.load_model(model_path)
    assert loaded_model == chain

    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_path)
    prediction = loaded_pyfunc_model.predict([{"product": "MLflow"}])
    assert prediction == ["What is MLflow?"]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.2.0"),
    reason="langchain-databricks requires langchain >= 0.2.0",
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
