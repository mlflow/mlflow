import pytest
import os

from mlflow.pyfunc.mlserver import get_cmd, MLServerMLflowRuntime, MLServerDefaultModelName


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (
            {"port": 5000, "host": "0.0.0.0", "nworkers": 4},
            {
                "MLSERVER_HTTP_PORT": "5000",
                "MLSERVER_HOST": "0.0.0.0",
                "MLSERVER_MODEL_PARALLEL_WORKERS": "4",
            },
        ),
        (
            {"host": "0.0.0.0", "nworkers": 4},
            {"MLSERVER_HOST": "0.0.0.0", "MLSERVER_MODEL_PARALLEL_WORKERS": "4"},
        ),
        (
            {"port": 5000, "nworkers": 4},
            {"MLSERVER_HTTP_PORT": "5000", "MLSERVER_MODEL_PARALLEL_WORKERS": "4"},
        ),
        ({"port": 5000}, {"MLSERVER_HTTP_PORT": "5000"}),
        ({}, {}),
    ],
)
def test_get_cmd(params: dict, expected: dict):
    model_uri = "/foo/bar"
    cmd, cmd_env = get_cmd(model_uri=model_uri, **params)

    assert cmd == f"mlserver start {model_uri}"

    assert cmd_env == {
        "MLSERVER_MODEL_URI": model_uri,
        "MLSERVER_MODEL_IMPLEMENTATION": MLServerMLflowRuntime,
        "MLSERVER_MODEL_NAME": MLServerDefaultModelName,
        **expected,
        **os.environ.copy(),
    }


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"port": 5000, "timeout": 70}, "Timeout is not yet supported in MLServer."),
    ],
)
def test_raise_error_mlserver_timeout(params: dict, expected: str):
    model_uri = "/foo/bar"

    with pytest.raises(Exception, match=expected):
        get_cmd(model_uri=model_uri, **params)
