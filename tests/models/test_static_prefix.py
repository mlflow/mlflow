"""
Tests for static_prefix functionality in model serving.
"""

import json
from unittest import mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.pyfunc.scoring_server import init


class SimpleTestModel(PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


def test_invocations_with_static_prefix(tmp_path, monkeypatch):
    model_path = str(tmp_path / "model")
    mlflow.pyfunc.save_model(path=model_path, python_model=SimpleTestModel())

    monkeypatch.setenv("MLFLOW_STATIC_PREFIX", "/api/v1")
    model = mlflow.pyfunc.load_model(model_path)
    app = init(model)
    client = TestClient(app)

    test_df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    data = json.dumps({"dataframe_split": test_df.to_dict(orient="split")})

    response = client.post(
        "/api/v1/invocations",
        content=data,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result


def test_ping_with_static_prefix(tmp_path, monkeypatch):
    model_path = str(tmp_path / "model")
    mlflow.pyfunc.save_model(path=model_path, python_model=SimpleTestModel())

    monkeypatch.setenv("MLFLOW_STATIC_PREFIX", "/api/v1")
    model = mlflow.pyfunc.load_model(model_path)
    app = init(model)
    client = TestClient(app)

    response = client.get("/api/v1/ping")
    assert response.status_code == 200


def test_version_with_static_prefix(tmp_path, monkeypatch):
    model_path = str(tmp_path / "model")
    mlflow.pyfunc.save_model(path=model_path, python_model=SimpleTestModel())

    monkeypatch.setenv("MLFLOW_STATIC_PREFIX", "/api/v1")
    model = mlflow.pyfunc.load_model(model_path)
    app = init(model)
    client = TestClient(app)

    response = client.get("/api/v1/version")
    assert response.status_code == 200
    assert response.text


def test_invocations_without_static_prefix(tmp_path, monkeypatch):
    model_path = str(tmp_path / "model")
    mlflow.pyfunc.save_model(path=model_path, python_model=SimpleTestModel())

    monkeypatch.delenv("MLFLOW_STATIC_PREFIX", raising=False)
    model = mlflow.pyfunc.load_model(model_path)
    app = init(model)
    client = TestClient(app)

    test_df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    data = json.dumps({"dataframe_split": test_df.to_dict(orient="split")})

    response = client.post(
        "/invocations",
        content=data,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result


@pytest.mark.parametrize(
    ("prefix", "endpoint", "expected_path"),
    [
        ("/api/v1", "ping", "/api/v1/ping"),
        ("/api/v1", "invocations", "/api/v1/invocations"),
        ("/models", "ping", "/models/ping"),
        ("/ml/v2", "invocations", "/ml/v2/invocations"),
    ],
)
def test_static_prefix_with_various_prefixes(
    tmp_path, prefix, endpoint, expected_path, monkeypatch
):
    model_path = str(tmp_path / "model")
    mlflow.pyfunc.save_model(path=model_path, python_model=SimpleTestModel())

    monkeypatch.setenv("MLFLOW_STATIC_PREFIX", prefix)
    model = mlflow.pyfunc.load_model(model_path)
    app = init(model)
    client = TestClient(app)

    if endpoint == "invocations":
        test_df = pd.DataFrame({"x": [1, 2, 3]})
        data = json.dumps({"dataframe_split": test_df.to_dict(orient="split")})
        response = client.post(
            expected_path,
            content=data,
            headers={"Content-Type": "application/json"},
        )
    else:
        response = client.get(expected_path)

    assert response.status_code == 200


def test_non_prefixed_paths_return_404_when_prefix_set(tmp_path, monkeypatch):
    model_path = str(tmp_path / "model")
    mlflow.pyfunc.save_model(path=model_path, python_model=SimpleTestModel())

    monkeypatch.setenv("MLFLOW_STATIC_PREFIX", "/api/v1")
    model = mlflow.pyfunc.load_model(model_path)
    app = init(model)
    client = TestClient(app)

    # Non-prefixed paths should return 404
    assert client.get("/ping").status_code == 404
    assert client.get("/version").status_code == 404


def test_static_prefix_works_with_nginx_enabled(monkeypatch):
    from mlflow.models import Model
    from mlflow.models.container import _serve_pyfunc
    from mlflow.utils import env_manager as em

    model_dict = {
        "flavors": {
            "python_function": {
                "loader_module": "mlflow.pyfunc.model",
            }
        }
    }
    model = Model.from_dict(model_dict)

    monkeypatch.setenv("MLFLOW_STATIC_PREFIX", "/api/v1")
    monkeypatch.delenv("ENABLE_MLSERVER", raising=False)

    with (
        mock.patch("mlflow.models.container.Popen") as mock_popen,
        mock.patch("mlflow.pyfunc.scoring_server.get_cmd") as mock_get_cmd,
        mock.patch("mlflow.models.container._await_subprocess_exit_any"),
        mock.patch("mlflow.models.container._sigterm_handler"),
        mock.patch("mlflow.models.container.check_call"),
    ):
        mock_popen.return_value.pid = 12345
        mock_get_cmd.return_value = ("python -m mlflow.pyfunc", {})
        _serve_pyfunc(model, env_manager=em.LOCAL)


def test_static_prefix_works_with_nginx_disabled(monkeypatch):
    from mlflow.models import Model
    from mlflow.models.container import _serve_pyfunc
    from mlflow.utils import env_manager as em

    model_dict = {
        "flavors": {
            "python_function": {
                "loader_module": "mlflow.pyfunc.model",
            }
        }
    }
    model = Model.from_dict(model_dict)

    monkeypatch.setenv("MLFLOW_STATIC_PREFIX", "/api/v1")
    monkeypatch.setenv("DISABLE_NGINX", "true")
    monkeypatch.delenv("ENABLE_MLSERVER", raising=False)

    with (
        mock.patch("mlflow.models.container.Popen") as mock_popen,
        mock.patch("mlflow.pyfunc.scoring_server.get_cmd") as mock_get_cmd,
        mock.patch("mlflow.models.container._await_subprocess_exit_any"),
        mock.patch("mlflow.models.container._sigterm_handler"),
    ):
        mock_popen.return_value.pid = 12345
        mock_get_cmd.return_value = ("python -m mlflow.pyfunc", {})
        _serve_pyfunc(model, env_manager=em.LOCAL)


def test_static_prefix_not_supported_with_mlserver(monkeypatch):
    from mlflow.models import Model
    from mlflow.models.container import _serve_pyfunc
    from mlflow.utils import env_manager as em

    model_dict = {
        "flavors": {
            "python_function": {
                "loader_module": "mlflow.pyfunc.model",
            }
        }
    }
    model = Model.from_dict(model_dict)

    monkeypatch.setenv("MLFLOW_STATIC_PREFIX", "/api/v1")
    monkeypatch.setenv("ENABLE_MLSERVER", "true")

    with (
        pytest.raises(
            ValueError,
            match="--static-prefix.*is not supported with ENABLE_MLSERVER=true",
        ),
        mock.patch("mlflow.models.container.Popen"),
    ):
        _serve_pyfunc(model, env_manager=em.LOCAL)
