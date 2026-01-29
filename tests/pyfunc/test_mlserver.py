import os
from typing import Any

import pytest

from mlflow.pyfunc.mlserver import MLServerDefaultModelName, MLServerMLflowRuntime, get_cmd


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


def test_get_cmd_respects_env_var_model_implementation(monkeypatch):
    """Test that MLSERVER_MODEL_IMPLEMENTATION env var takes precedence over the default."""
    custom_implementation = "custom.runtime.CustomRuntime"
    monkeypatch.setenv("MLSERVER_MODEL_IMPLEMENTATION", custom_implementation)

    model_uri = "/foo/bar"
    _, cmd_env = get_cmd(model_uri=model_uri)

    assert cmd_env["MLSERVER_MODEL_IMPLEMENTATION"] == custom_implementation
