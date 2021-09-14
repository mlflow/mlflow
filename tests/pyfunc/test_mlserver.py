import pytest
import os

from mlflow.pyfunc.mlserver import get_cmd, MLServerMLflowRuntime


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"port": 5000, "host": "0.0.0.0", "nworkers": 4},
            {
                "MLSERVER_HTTP_PORT": "5000",
                "MLSERVER_HOST": "0.0.0.0",
                "MLSERVER_MODEL_PARALLEL_WORKERS": "4",
                "MLSERVER_MODEL_IMPLEMENTATION": MLServerMLflowRuntime,
            },
        ),
        (
            {"host": "0.0.0.0", "nworkers": 4},
            {
                "MLSERVER_HOST": "0.0.0.0",
                "MLSERVER_MODEL_PARALLEL_WORKERS": "4",
                "MLSERVER_MODEL_IMPLEMENTATION": MLServerMLflowRuntime,
            },
        ),
        (
            {"port": 5000, "nworkers": 4},
            {
                "MLSERVER_HTTP_PORT": "5000",
                "MLSERVER_MODEL_PARALLEL_WORKERS": "4",
                "MLSERVER_MODEL_IMPLEMENTATION": MLServerMLflowRuntime,
            },
        ),
        (
            {"port": 5000},
            {
                "MLSERVER_HTTP_PORT": "5000",
                "MLSERVER_MODEL_IMPLEMENTATION": MLServerMLflowRuntime,
            },
        ),
        (
            {},
            {
                "MLSERVER_MODEL_IMPLEMENTATION": MLServerMLflowRuntime,
            },
        ),
    ],
)
def test_get_cmd(params: dict, expected: dict):
    model_uri = "/foo/bar"
    cmd, cmd_env = get_cmd(model_uri=model_uri, **params)

    assert cmd == f"mlserver start {model_uri}"

    assert cmd_env == {"MLSERVER_MODEL_URI": model_uri, **expected, **os.environ.copy()}
