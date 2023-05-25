import yaml
import json
import importlib
from unittest import mock

from pyspark.sql import SparkSession
import openai
import openai.error
import pytest
import pandas as pd

import mlflow
from mlflow.openai.utils import (
    _mock_request,
    _mock_chat_completion_response,
    _mock_models_retrieve_response,
)


@pytest.fixture(scope="module")
def spark():
    with SparkSession.builder.master("local[*]").getOrCreate() as s:
        yield s


@pytest.fixture(autouse=True)
def set_envs(monkeypatch):
    monkeypatch.setenvs(
        {
            "MLFLOW_OPENAI_TESTING": "true",
            "OPENAI_API_KEY": "test",
        }
    )
    importlib.reload(openai)


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
    with _mock_request(return_value=_mock_chat_completion_response(content="foo")) as mock:
        completion = openai.ChatCompletion.create(
            model=loaded_model["model"],
            messages=loaded_model["messages"],
            temperature=loaded_model["temperature"],
        )
        assert completion.choices[0].message.content == "foo"
        mock.assert_called_once()


def test_single_variable(tmp_path):
    mlflow.openai.save_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        path=tmp_path,
        messages=[{"role": "user", "content": "{x}"}],
    )

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame(
        {
            "x": [
                "a",
                "b",
            ]
        }
    )
    expected_output = [
        [{"content": "a", "role": "user"}],
        [{"content": "b", "role": "user"}],
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

    data = [
        {"x": "a"},
        {"x": "b"},
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

    data = [
        "a",
        "b",
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output


def test_multiple_variables(tmp_path):
    mlflow.openai.save_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        path=tmp_path,
        messages=[{"role": "user", "content": "{x} {y}"}],
    )
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"name": "x", "type": "string"},
        {"name": "y", "type": "string"},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string"},
    ]

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame(
        {
            "x": [
                "a",
                "b",
            ],
            "y": [
                "c",
                "d",
            ],
        }
    )
    expected_output = [
        [{"content": "a c", "role": "user"}],
        [{"content": "b d", "role": "user"}],
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

    data = [
        {"x": "a", "y": "c"},
        {"x": "b", "y": "d"},
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output


def test_multiple_prompts(tmp_path):
    mlflow.openai.save_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        path=tmp_path,
        messages=[
            {"role": "user", "content": "{x}"},
            {"role": "user", "content": "{y}"},
        ],
    )
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"name": "x", "type": "string"},
        {"name": "y", "type": "string"},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string"},
    ]

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame(
        {
            "x": [
                "a",
                "b",
            ],
            "y": [
                "c",
                "d",
            ],
        }
    )
    expected_output = [
        [{"content": "a", "role": "user"}, {"content": "c", "role": "user"}],
        [{"content": "b", "role": "user"}, {"content": "d", "role": "user"}],
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

    data = [
        {"x": "a", "y": "c"},
        {"x": "b", "y": "d"},
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output


def test_no_variables(tmp_path):
    mlflow.openai.save_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        path=tmp_path,
        messages=[{"role": "user", "content": "a"}],
    )
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"type": "string"},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string"},
    ]

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame(
        {
            "content": ["b", "c"],
        }
    )
    expected_output = [
        [{"content": "a", "role": "user"}, {"content": "b", "role": "user"}],
        [{"content": "a", "role": "user"}, {"content": "c", "role": "user"}],
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

    data = [
        {"content": "b"},
        {"content": "c"},
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

    data = [
        "b",
        "c",
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output


def test_no_messages(tmp_path):
    mlflow.openai.save_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        path=tmp_path,
    )
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"type": "string"},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string"},
    ]

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame(
        {
            "content": ["b", "c"],
        }
    )
    expected_output = [
        [{"content": "b", "role": "user"}],
        [{"content": "c", "role": "user"}],
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

    data = [
        {"content": "b"},
        {"content": "c"},
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

    data = [
        "b",
        "c",
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output


@pytest.mark.parametrize(
    "messages",
    [
        ["a", "b"],
        [{"k": "v"}],
    ],
)
def test_invalid_messages(tmp_path, messages):
    with pytest.raises(
        mlflow.MlflowException,
        match="it must be a list of dictionaries with keys 'role' and 'content'",
    ):
        mlflow.openai.save_model(
            model="gpt-3.5-turbo",
            task=openai.ChatCompletion,
            path=tmp_path,
            messages=messages,
        )


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


def test_pyfunc_flavor_is_only_added_for_chat_completion(tmp_path):
    mlflow.openai.save_model(model="gpt-3.5-turbo", task="embeddings", path=tmp_path)
    model = mlflow.models.Model(tmp_path)
    assert "pyfunc" not in model.flavors


def test_save_model_with_secret_scope(tmp_path, monkeypatch):
    scope = "test"
    monkeypatch.setenv("MLFLOW_OPENAI_SECRET_SCOPE", scope)
    with mock.patch("mlflow.openai.is_in_databricks_runtime", return_value=True), mock.patch(
        "mlflow.openai.check_databricks_secret_scope_access"
    ):
        mlflow.openai.save_model(model="gpt-3.5-turbo", task="chat.completions", path=tmp_path)
    with tmp_path.joinpath("openai.yaml").open() as f:
        creds = yaml.safe_load(f)
        assert creds == {
            "OPENAI_API_TYPE": f"{scope}:openai_api_type",
            "OPENAI_API_KEY": f"{scope}:openai_api_key",
            "OPENAI_API_KEY_PATH": f"{scope}:openai_api_key_path",
            "OPENAI_API_BASE": f"{scope}:openai_api_base",
            "OPENAI_ORGANIZATION": f"{scope}:openai_organization",
        }


def test_spark_udf(tmp_path, spark):
    mlflow.openai.save_model(
        model="gpt-3.5-turbo",
        task="chat.completions",
        path=tmp_path,
        messages=[
            {"role": "user", "content": "{x} {y}"},
        ],
    )
    udf = mlflow.pyfunc.spark_udf(spark, tmp_path, result_type="string")
    df = spark.createDataFrame(
        [
            ("a", "b"),
            ("c", "d"),
        ],
        ["x", "y"],
    )
    df = df.withColumn("z", udf())
    pdf = df.toPandas()
    assert list(map(json.loads, pdf["z"])) == [
        [{"content": "a b", "role": "user"}],
        [{"content": "c d", "role": "user"}],
    ]


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
    resp = _mock_chat_completion_response(content="test")
    with _mock_request(side_effect=[error, resp]) as mock_request:
        text = loaded_model.predict(None)
        assert text == "test"
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
