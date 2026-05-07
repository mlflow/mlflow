import os
from typing import Any
from unittest import mock

import pytest

from mlflow.pyfunc.backend import PyFuncBackend
from mlflow.pyfunc.mlserver import (
    MLServerDefaultModelName,
    MLServerMLflowRuntime,
    get_cmd,
    warn_mlserver_deprecated,
)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (
            {"port": 5000, "host": "0.0.0.0", "nworkers": 4},
            {
                "MLSERVER_HTTP_PORT": "5000",
                "MLSERVER_HOST": "0.0.0.0",
                "MLSERVER_PARALLEL_WORKERS": "4",
                "MLSERVER_MODEL_NAME": MLServerDefaultModelName,
            },
        ),
        (
            {"host": "0.0.0.0", "nworkers": 4},
            {
                "MLSERVER_HOST": "0.0.0.0",
                "MLSERVER_PARALLEL_WORKERS": "4",
                "MLSERVER_MODEL_NAME": MLServerDefaultModelName,
            },
        ),
        (
            {"port": 5000, "nworkers": 4},
            {
                "MLSERVER_HTTP_PORT": "5000",
                "MLSERVER_PARALLEL_WORKERS": "4",
                "MLSERVER_MODEL_NAME": MLServerDefaultModelName,
            },
        ),
        (
            {"port": 5000},
            {
                "MLSERVER_HTTP_PORT": "5000",
                "MLSERVER_MODEL_NAME": MLServerDefaultModelName,
            },
        ),
        (
            {"model_name": "mymodel", "model_version": "12"},
            {"MLSERVER_MODEL_NAME": "mymodel", "MLSERVER_MODEL_VERSION": "12"},
        ),
        ({}, {"MLSERVER_MODEL_NAME": MLServerDefaultModelName}),
    ],
)
def test_get_cmd(params: dict[str, Any], expected: dict[str, Any]):
    model_uri = "/foo/bar"
    cmd, cmd_env = get_cmd(model_uri=model_uri, **params)

    assert cmd == f"mlserver start {model_uri}"

    assert cmd_env == {
        "MLSERVER_MODEL_URI": model_uri,
        "MLSERVER_MODEL_IMPLEMENTATION": MLServerMLflowRuntime,
        **expected,
        **os.environ.copy(),
    }


def test_warn_mlserver_deprecated_message():
    with pytest.warns(FutureWarning, match=r"MLServer integration .* deprecated .* MLflow 3\.13"):
        warn_mlserver_deprecated()


@pytest.mark.parametrize("enable_mlserver", [True, False])
def test_serve_emits_deprecation_warning_only_when_enabled(enable_mlserver):
    backend = PyFuncBackend(config={}, env_manager="local")
    with (
        mock.patch("mlflow.pyfunc.mlserver.warn_mlserver_deprecated") as mock_warn,
        mock.patch(
            "mlflow.pyfunc.backend._download_artifact_from_uri",
            side_effect=RuntimeError("stop"),
        ),
        pytest.raises(RuntimeError, match="stop"),
    ):
        backend.serve(
            model_uri="dummy",
            port=1234,
            host="localhost",
            timeout=None,
            enable_mlserver=enable_mlserver,
        )

    if enable_mlserver:
        mock_warn.assert_called_once()
    else:
        mock_warn.assert_not_called()


@pytest.mark.parametrize("enable_mlserver", [True, False])
def test_generate_dockerfile_emits_deprecation_warning_only_when_enabled(tmp_path, enable_mlserver):
    backend = PyFuncBackend(config={}, env_manager="virtualenv")
    with (
        mock.patch("mlflow.pyfunc.mlserver.warn_mlserver_deprecated") as mock_warn,
        mock.patch("mlflow.pyfunc.backend.docker_utils.generate_dockerfile"),
    ):
        backend.generate_dockerfile(
            model_uri=None,
            output_dir=str(tmp_path),
            enable_mlserver=enable_mlserver,
        )

    if enable_mlserver:
        mock_warn.assert_called_once()
    else:
        mock_warn.assert_not_called()
