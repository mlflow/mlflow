import importlib
import json
from unittest import mock

import numpy as np
import openai
import pandas as pd
import pytest
import requests
import yaml
from pyspark.sql import SparkSession

import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import load_serving_example
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema, TensorSpec

from tests.helper_functions import pyfunc_serve_and_score_model
from tests.openai.conftest import is_v1


@pytest.fixture(scope="module")
def spark():
    with SparkSession.builder.master("local[*]").getOrCreate() as s:
        yield s


def chat_completions():
    return openai.chat.completions if is_v1 else openai.ChatCompletion


def completions():
    return openai.completions if is_v1 else openai.Completion


def embeddings():
    return openai.embeddings if is_v1 else openai.Embedding


@pytest.fixture(autouse=True)
def set_envs(monkeypatch, mock_openai):
    monkeypatch.setenvs(
        {
            "MLFLOW_TESTING": "true",
            "OPENAI_API_KEY": "test",
            "OPENAI_API_BASE": mock_openai,
        }
    )
    if is_v1:
        openai.base_url = mock_openai
    else:
        importlib.reload(openai)


def test_log_model():
    with mlflow.start_run():
        model_info = mlflow.openai.log_model(
            model="gpt-4o-mini",
            task="chat.completions",
            artifact_path="model",
            temperature=0.9,
            messages=[{"role": "system", "content": "You are an MLflow expert."}],
        )

    loaded_model = mlflow.openai.load_model(model_info.model_uri)
    assert loaded_model["model"] == "gpt-4o-mini"
    assert loaded_model["task"] == "chat.completions"
    assert loaded_model["temperature"] == 0.9
    assert loaded_model["messages"] == [{"role": "system", "content": "You are an MLflow expert."}]


def test_chat_single_variable(tmp_path):
    mlflow.openai.save_model(
        model="gpt-4o-mini",
        task=chat_completions(),
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


def test_completion_single_variable(tmp_path):
    mlflow.openai.save_model(
        model="text-davinci-003",
        task=completions(),
        path=tmp_path,
        prompt="Say {text}",
    )

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame(
        {
            "x": [
                "this is a test",
                "this is another test",
            ]
        }
    )
    expected_output = ["Say this is a test", "Say this is another test"]
    assert model.predict(data) == expected_output

    data = [
        {"x": "this is a test"},
        {"x": "this is another test"},
    ]
    assert model.predict(data) == expected_output

    data = [
        "this is a test",
        "this is another test",
    ]
    assert model.predict(data) == expected_output


def test_chat_multiple_variables(tmp_path):
    mlflow.openai.save_model(
        model="gpt-4o-mini",
        task=chat_completions(),
        path=tmp_path,
        messages=[{"role": "user", "content": "{x} {y}"}],
    )
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"name": "x", "type": "string", "required": True},
        {"name": "y", "type": "string", "required": True},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string", "required": True},
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


def test_chat_role_content(tmp_path):
    mlflow.openai.save_model(
        model="gpt-4o-mini",
        task=chat_completions(),
        path=tmp_path,
        messages=[{"role": "{role}", "content": "{content}"}],
    )
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"name": "content", "type": "string", "required": True},
        {"name": "role", "type": "string", "required": True},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string", "required": True},
    ]

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame(
        {
            "role": [
                "system",
                "user",
            ],
            "content": [
                "c",
                "d",
            ],
        }
    )
    expected_output = [
        [{"content": "c", "role": "system"}],
        [{"content": "d", "role": "user"}],
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output


def test_completion_multiple_variables(tmp_path):
    mlflow.openai.save_model(
        model="text-davinci-003",
        task=completions(),
        path=tmp_path,
        prompt="Say {x} and {y}",
    )
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"name": "x", "type": "string", "required": True},
        {"name": "y", "type": "string", "required": True},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string", "required": True},
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
    expected_output = ["Say a and c", "Say b and d"]
    assert model.predict(data) == expected_output

    data = [
        {"x": "a", "y": "c"},
        {"x": "b", "y": "d"},
    ]
    assert model.predict(data) == expected_output


def test_chat_multiple_messages(tmp_path):
    mlflow.openai.save_model(
        model="gpt-4o-mini",
        task=chat_completions(),
        path=tmp_path,
        messages=[
            {"role": "user", "content": "{x}"},
            {"role": "user", "content": "{y}"},
        ],
    )
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"name": "x", "type": "string", "required": True},
        {"name": "y", "type": "string", "required": True},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string", "required": True},
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


def test_chat_no_variables(tmp_path):
    mlflow.openai.save_model(
        model="gpt-4o-mini",
        task=chat_completions(),
        path=tmp_path,
        messages=[{"role": "user", "content": "a"}],
    )
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"type": "string", "required": True},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string", "required": True},
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


def test_completion_no_variable(tmp_path):
    mlflow.openai.save_model(
        model="text-davinci-003",
        task=completions(),
        path=tmp_path,
    )

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame(
        {
            "x": [
                "this is a test",
                "this is another test",
            ]
        }
    )
    expected_output = ["this is a test", "this is another test"]
    assert model.predict(data) == expected_output

    data = [
        {"x": "this is a test"},
        {"x": "this is another test"},
    ]
    assert model.predict(data) == expected_output

    data = [
        "this is a test",
        "this is another test",
    ]
    assert model.predict(data) == expected_output


def test_chat_no_messages(tmp_path):
    mlflow.openai.save_model(
        model="gpt-4o-mini",
        task=chat_completions(),
        path=tmp_path,
    )
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"type": "string", "required": True},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string", "required": True},
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
            model="gpt-4o-mini",
            task=chat_completions(),
            path=tmp_path,
            messages=messages,
        )


def test_task_argument_accepts_class(tmp_path):
    mlflow.openai.save_model(model="gpt-4o-mini", task=chat_completions(), path=tmp_path)
    loaded_model = mlflow.openai.load_model(tmp_path)
    assert loaded_model["task"] == "chat.completions"


@pytest.mark.skipif(is_v1, reason="Requires OpenAI SDK v0")
def test_model_argument_accepts_retrieved_model(tmp_path):
    model = openai.Model.retrieve("gpt-4o-mini")
    mlflow.openai.save_model(model=model, task=chat_completions(), path=tmp_path)
    loaded_model = mlflow.openai.load_model(tmp_path)
    assert loaded_model["model"] == "gpt-4o-mini"


def test_save_model_with_secret_scope(tmp_path, monkeypatch):
    scope = "test"
    monkeypatch.setenv("MLFLOW_OPENAI_SECRET_SCOPE", scope)
    with mock.patch("mlflow.openai.is_in_databricks_runtime", return_value=True), mock.patch(
        "mlflow.openai.check_databricks_secret_scope_access"
    ):
        with pytest.warns(FutureWarning, match="MLFLOW_OPENAI_SECRET_SCOPE.+deprecated"):
            mlflow.openai.save_model(model="gpt-4o-mini", task="chat.completions", path=tmp_path)
    with tmp_path.joinpath("openai.yaml").open() as f:
        creds = yaml.safe_load(f)
        assert creds == {
            "OPENAI_API_TYPE": f"{scope}:openai_api_type",
            "OPENAI_API_KEY": f"{scope}:openai_api_key",
            "OPENAI_API_KEY_PATH": f"{scope}:openai_api_key_path",
            "OPENAI_API_BASE": f"{scope}:openai_api_base",
            "OPENAI_BASE_URL": f"{scope}:openai_base_url",
            "OPENAI_ORGANIZATION": f"{scope}:openai_organization",
            "OPENAI_API_VERSION": f"{scope}:openai_api_version",
            "OPENAI_DEPLOYMENT_NAME": f"{scope}:openai_deployment_name",
            "OPENAI_ENGINE": f"{scope}:openai_engine",
        }


def test_spark_udf_chat(tmp_path, spark):
    mlflow.openai.save_model(
        model="gpt-4o-mini",
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
    def predict(self, context, model_input, params=None):
        completion = chat_completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is MLflow?"}],
        )
        return completion.choices[0].message.content


def test_embeddings(tmp_path):
    mlflow.openai.save_model(
        model="text-embedding-ada-002",
        task=embeddings(),
        path=tmp_path,
    )

    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [{"type": "string", "required": True}]
    assert model.signature.outputs.to_dict() == [
        {"type": "tensor", "tensor-spec": {"dtype": "float64", "shape": (-1,)}}
    ]

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame({"text": ["a", "b"]})
    preds = model.predict(data)
    assert list(map(len, preds)) == [1536, 1536]

    data = pd.DataFrame({"text": ["a"] * 100})
    preds = model.predict(data)
    assert list(map(len, preds)) == [1536] * 100


def test_embeddings_batch_size_azure(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_TYPE", "azure")
    monkeypatch.setenv("OPENAI_ENGINE", "test_engine")
    mlflow.openai.save_model(
        model="text-embedding-ada-002",
        task=embeddings(),
        path=tmp_path,
    )
    model = mlflow.pyfunc.load_model(tmp_path)

    assert model._model_impl.api_config.batch_size == 16


def test_embeddings_pyfunc_server_and_score():
    df = pd.DataFrame({"text": ["a", "b"]})
    with mlflow.start_run():
        model_info = mlflow.openai.log_model(
            model="text-embedding-ada-002",
            task=embeddings(),
            artifact_path="model",
            input_example=df,
        )
    inference_payload = load_serving_example(model_info.model_uri)
    resp = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    expected = mlflow.pyfunc.load_model(model_info.model_uri).predict(df)
    actual = pd.DataFrame(data=json.loads(resp.content.decode("utf-8")))
    pd.testing.assert_frame_equal(actual, pd.DataFrame({"predictions": expected}))


def test_spark_udf_embeddings(tmp_path, spark):
    mlflow.openai.save_model(
        model="text-embedding-ada-002",
        task=embeddings(),
        path=tmp_path,
    )
    udf = mlflow.pyfunc.spark_udf(spark, tmp_path, result_type="array<double>")
    df = spark.createDataFrame(
        [
            ("a",),
            ("b",),
        ],
        ["x"],
    )
    df = df.withColumn("z", udf("x")).toPandas()
    assert list(map(len, df["z"])) == [1536, 1536]


def test_inference_params(tmp_path):
    mlflow.openai.save_model(
        model="text-embedding-ada-002",
        task=embeddings(),
        path=tmp_path,
        signature=ModelSignature(
            inputs=Schema([ColSpec(type="string", name=None)]),
            outputs=Schema([TensorSpec(type=np.dtype("float64"), shape=(-1,))]),
            params=ParamSchema([ParamSpec(name="batch_size", dtype="long", default=16)]),
        ),
    )

    model_info = mlflow.models.Model.load(tmp_path)
    assert (
        len([p for p in model_info.signature.params if p.name == "batch_size" and p.default == 16])
        == 1
    )

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame({"text": ["a", "b"]})
    preds = model.predict(data, params={"batch_size": 5})
    assert list(map(len, preds)) == [1536, 1536]


def test_inference_params_overlap(tmp_path):
    with pytest.raises(mlflow.MlflowException, match=r"any of \['prefix'\] as parameters"):
        mlflow.openai.save_model(
            model="text-davinci-003",
            task=completions(),
            path=tmp_path,
            prefix="Classify the following text's sentiment:",
            signature=ModelSignature(
                inputs=Schema([ColSpec(type="string", name=None)]),
                outputs=Schema([ColSpec(type="string", name=None)]),
                params=ParamSchema([ParamSpec(name="prefix", default=None, dtype="string")]),
            ),
        )


def test_engine_and_deployment_id_for_azure_openai(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_TYPE", "azure")
    mlflow.openai.save_model(
        model="text-embedding-ada-002",
        task=embeddings(),
        path=tmp_path,
    )
    with pytest.raises(
        MlflowException, match=r"Either engine or deployment_id must be set for Azure OpenAI API"
    ):
        mlflow.pyfunc.load_model(tmp_path)


@pytest.mark.parametrize(
    ("api_type", "auth_headers"),
    [
        ("azure", {"api-key": "test"}),
        ("azure_ad", {"Authorization": "Bearer test"}),
        ("azuread", {"Authorization": "Bearer test"}),
        ("openai", {"Authorization": "Bearer test"}),
    ],
)
def test_openai_request_auth_headers(api_type, auth_headers, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_TYPE", api_type)
    if "azure" in api_type:
        monkeypatch.setenv("OPENAI_DEPLOYMENT_NAME", "test")
    mlflow.openai.save_model(
        model="gpt-4o",
        task="chat.completions",
        path=tmp_path,
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    with mock.patch("requests.Session.request") as mock_request:
        model.predict("What is the meaning of life?")
        mock_request.assert_called_once_with(
            method="post",
            url=mock.ANY,
            data=mock.ANY,
            json=mock.ANY,
            timeout=mock.ANY,
            headers=auth_headers,
        )


@pytest.mark.parametrize("env_name", ["OPENAI_API_BASE", "OPENAI_BASE_URL"])
def test_openai_base_url(env_name, tmp_path, monkeypatch, mock_openai):
    base = mock_openai.rstrip("/")
    monkeypatch.setenv(env_name, base + "/")
    mlflow.openai.save_model(model="gpt-4o", task="chat.completions", path=tmp_path)
    model = mlflow.pyfunc.load_model(tmp_path)

    with mock.patch("requests.post", side_effect=requests.post) as mock_request:
        resp = model.predict("test")
        mock_request.assert_called_once()
        url = mock_request.call_args.kwargs["url"]
        assert url == f"{base}/chat/completions"
        assert resp == ['[{"role": "user", "content": "test"}]']
