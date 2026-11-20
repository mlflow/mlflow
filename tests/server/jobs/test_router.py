import json
import os

import pytest

from mlflow.exceptions import MlflowException
from mlflow.server.jobs.executor_registry import shutdown_executor_registry
from mlflow.server.jobs.router import select_executor_backend

_CUSTOM_SCORER = {
    "name": "custom",
    "call_source": "return True",
    "call_signature": "(outputs)",
    "original_func_name": "custom",
}
_BUILTIN_SCORER = {"name": "safety", "builtin_scorer_class": "Safety"}


@pytest.fixture(autouse=True)
def _reset_executor_registry():
    yield
    shutdown_executor_registry()


def test_non_custom_routes_to_default(monkeypatch):
    monkeypatch.setenv("MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND", "local")
    monkeypatch.delenv("MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND", raising=False)
    assert select_executor_backend(is_custom_scorer=False) == "local"


def test_custom_with_override_routes_to_custom(monkeypatch):
    monkeypatch.setenv("MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND", "local")
    monkeypatch.setenv("MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND", "sandbox")
    assert select_executor_backend(is_custom_scorer=True) == "sandbox"


def test_custom_without_override_routes_to_default(monkeypatch):
    monkeypatch.setenv("MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND", "local")
    monkeypatch.delenv("MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND", raising=False)
    assert select_executor_backend(is_custom_scorer=True) == "local"


@pytest.fixture
def docker_custom_scorer_backend(monkeypatch):
    if os.name == "nt":
        pytest.skip("The docker job executor requires a POSIX host")
    monkeypatch.setenv("MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND", "local")
    monkeypatch.setenv("MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND", "docker")
    # Routing only asks the backend which jobs it supports, so it needs no Docker daemon.
    monkeypatch.setattr(
        "mlflow.server.jobs.docker_executor.DockerJobExecutor.check_requirements", lambda self: None
    )


@pytest.mark.usefixtures("docker_custom_scorer_backend")
def test_custom_scorer_job_supported_by_custom_backend_routes_to_it():
    params = {"serialized_scorer": json.dumps(_CUSTOM_SCORER)}
    backend = select_executor_backend(
        is_custom_scorer=True, job_name="invoke_scorer", params=params
    )
    assert backend == "docker"


@pytest.mark.usefixtures("docker_custom_scorer_backend")
@pytest.mark.parametrize(
    ("job_name", "params"),
    [
        # A job type the docker backend does not run.
        ("invoke_genai_evaluate", {"serialized_scorers": [json.dumps(_CUSTOM_SCORER)]}),
        # A supported job type, but the ensemble also needs network access.
        (
            "invoke_scorer",
            {
                "serialized_scorer": json.dumps({
                    "name": "mixed",
                    "ensemble_scorer_data": {
                        "ensemble_fn": "majority",
                        "scorers": [_CUSTOM_SCORER, _BUILTIN_SCORER],
                    },
                })
            },
        ),
    ],
)
def test_custom_scorer_job_unsupported_by_custom_backend_is_refused(job_name, params):
    # Running it on the default backend would run the custom scorer code on the server host.
    with pytest.raises(
        MlflowException, match="does not run code-based scorers anywhere else"
    ) as exc_info:
        select_executor_backend(is_custom_scorer=True, job_name=job_name, params=params)
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"


@pytest.mark.usefixtures("docker_custom_scorer_backend")
@pytest.mark.parametrize("job_name", ["run_online_trace_scorer", "run_online_session_scorer"])
def test_online_scoring_job_with_custom_scorers_runs_on_default_backend(job_name):
    # It runs its custom scorers in job containers of its own.
    params = {
        "online_scorers": [{"name": "custom", "serialized_scorer": json.dumps(_CUSTOM_SCORER)}]
    }

    backend = select_executor_backend(is_custom_scorer=True, job_name=job_name, params=params)

    assert backend == "local"
