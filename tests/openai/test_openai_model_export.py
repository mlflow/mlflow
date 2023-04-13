import json
from unittest import mock
from contextlib import contextmanager

from pyspark.sql import SparkSession
import openai
import openai.error
import pytest
import pandas as pd

import mlflow


TEST_CONTENT = "test"


class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self.content = json.dumps(json_data).encode()
        self.headers = {"Content-Type": "application/json"}


def _mock_chat_completion_json():
    # https://platform.openai.com/docs/api-reference/chat/create
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": TEST_CONTENT},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }


def _mock_chat_completion_response():
    return MockResponse(200, _mock_chat_completion_json())


def _mock_models_retrieve_json():
    # https://platform.openai.com/docs/api-reference/models/retrieve
    return {"id": "gpt-3.5-turbo", "object": "model", "owned_by": "openai", "permission": []}


def _mock_models_retrieve_response():
    return MockResponse(200, _mock_models_retrieve_json())


@contextmanager
def _mock_request(**kwargs):
    with mock.patch("requests.Session.request", **kwargs) as m:
        yield m


class MockAsyncResponse:
    def __init__(self, status, json_data):
        self.status = status
        self._json = json_data
        self.headers = {"Content-Type": "application/json"}

    async def read(self):
        return json.dumps(self._json).encode()

    def __await__(self):
        yield
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self


@contextmanager
def _mock_async_request():
    with mock.patch(
        "aiohttp.ClientSession.request",
        return_value=MockAsyncResponse(200, _mock_chat_completion_json()),
    ) as m:
        yield m


@pytest.fixture(scope="module")
def spark():
    with SparkSession.builder.master("local[*]").getOrCreate() as s:
        yield s


def test_log_model():
    with mlflow.start_run():
        model_info = mlflow.openai.log_model(
            model="gpt-3.5-turbo",
            task="chat.completions",
            artifact_path="model",
            temperature=0.9,
            messages=[{"role": "system", "content": "You are an MLflow expert."}],
        )

    loaded_model = mlflow.openai.load_model(model_info.model_uri)
    assert loaded_model["model"] == "gpt-3.5-turbo"
    assert loaded_model["task"] == "chat.completions"
    assert loaded_model["temperature"] == 0.9
    assert loaded_model["messages"] == [{"role": "system", "content": "You are an MLflow expert."}]
    with _mock_request(return_value=_mock_chat_completion_response()) as mock:
        completion = openai.ChatCompletion.create(
            model=loaded_model["model"],
            messages=loaded_model["messages"],
            temperature=loaded_model["temperature"],
        )
        assert completion.choices[0].message.content == TEST_CONTENT
        mock.assert_called_once()


def test_task_argument_accepts_class(tmp_path):
    mlflow.openai.save_model(model="gpt-3.5-turbo", task=openai.ChatCompletion, path=tmp_path)
    loaded_model = mlflow.openai.load_model(tmp_path)
    assert loaded_model["task"] == "chat.completions"


def test_model_argument_accepts_retrieved_model(tmp_path):
    with _mock_request(return_value=_mock_models_retrieve_response()) as mock:
        model = openai.Model.retrieve("gpt-3.5-turbo")
        mock.assert_called_once()
    mlflow.openai.save_model(model=model, task=openai.ChatCompletion, path=tmp_path)
    loaded_model = mlflow.openai.load_model(tmp_path)
    assert loaded_model["model"] == "gpt-3.5-turbo"


def test_signature_is_automatically_created_for_chat_completion(tmp_path):
    mlflow.openai.save_model(model="gpt-3.5-turbo", task="chat.completions", path=tmp_path)
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"name": "role", "type": "string"},
        {"name": "content", "type": "string"},
    ]


def test_pyfunc_flavor_is_only_added_for_chat_completion(tmp_path):
    mlflow.openai.save_model(model="gpt-3.5-turbo", task="embeddings", path=tmp_path)
    model = mlflow.models.Model(tmp_path)
    assert "pyfunc" not in model.flavors


def test_save_model_with_secret_scope(tmp_path, monkeypatch):
    scope = "test"
    monkeypatch.setenv("MLFLOW_OPENAI_SECRET_SCOPE", scope)
    with mock.patch("mlflow.openai.is_in_databricks_runtime", return_value=True):
        mlflow.openai.save_model(model="gpt-3.5-turbo", task="chat.completions", path=tmp_path)
    with tmp_path.joinpath("openai.json").open() as f:
        creds = json.load(f)
        assert creds == {
            "OPENAI_API_TYPE": f"{scope}:openai_api_type",
            "OPENAI_API_KEY": f"{scope}:openai_api_key",
            "OPENAI_API_KEY_PATH": f"{scope}:openai_api_key_path",
            "OPENAI_API_BASE": f"{scope}:openai_api_base",
            "OPENAI_ORGANIZATION": f"{scope}:openai_organization",
        }


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame({"role": ["user"], "content": ["What is MLflow?"]}),
        [{"role": "user", "content": "What is MLflow?"}],
    ],
)
def test_pyfunc_predict(tmp_path, data):
    mlflow.openai.save_model(
        model="gpt-3.5-turbo",
        task="chat.completions",
        path=tmp_path,
        messages=[{"system": "user", "content": "You're an MLflow maintainer."}],
    )
    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    assert loaded_model.predict(data) == [TEST_CONTENT]


def test_spark_udf(tmp_path, spark):
    mlflow.openai.save_model(model="gpt-3.5-turbo", task="chat.completions", path=tmp_path)
    udf = mlflow.pyfunc.spark_udf(spark, tmp_path, result_type="string")
    df = spark.createDataFrame(
        [
            ("user", "What is MLflow?"),
            ("user", "What is Spark?"),
        ],
        ["role", "content"],
    )
    df = df.withColumn("answer", udf())
    pdf = df.toPandas()
    assert pdf["answer"].tolist() == [TEST_CONTENT, TEST_CONTENT]


class ChatCompletionModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What is MLflow?"}],
        )
        return completion.choices[0].message.content


@pytest.mark.parametrize(
    "error",
    [
        openai.error.RateLimitError(message="RateLimitError", code=403),
        openai.error.Timeout(message="Timeout", code=408),
        openai.error.ServiceUnavailableError(message="ServiceUnavailable", code=503),
        openai.error.APIConnectionError(message="APIConnectionError", code=500),
        openai.error.APIError(message="APIError", code=500),
    ],
)
def test_auto_request_retry(tmp_path, error):
    mlflow.pyfunc.save_model(tmp_path, python_model=ChatCompletionModel())
    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    resp = _mock_chat_completion_response()
    with _mock_request(side_effect=[error, resp]) as mock_request:
        text = loaded_model.predict(None)
        assert text == TEST_CONTENT
        assert mock_request.call_count == 2


def test_auto_request_retry_exceeds_maximum_attempts(tmp_path):
    mlflow.pyfunc.save_model(tmp_path, python_model=ChatCompletionModel())
    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    with pytest.raises(openai.error.RateLimitError, match="RateLimitError"):
        with _mock_request(
            side_effect=openai.error.RateLimitError(message="RateLimitError", code=403)
        ) as mock_request:
            loaded_model.predict(None)

    assert mock_request.call_count == 5


def test_auto_request_retry_is_disabled_when_env_var_is_false(tmp_path, monkeypatch):
    mlflow.pyfunc.save_model(tmp_path, python_model=ChatCompletionModel())
    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    with pytest.raises(openai.error.RateLimitError, match="RateLimitError"):
        with _mock_request(
            side_effect=openai.error.RateLimitError(message="RateLimitError", code=403)
        ) as mock_request:
            monkeypatch.setenv("MLFLOW_OPENAI_RETRIES_ENABLED", "false")
            loaded_model.predict(None)

    assert mock_request.call_count == 1
