import json

import pytest

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
    monkeypatch.setenv("MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND", "local")
    monkeypatch.setenv("MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND", "docker")


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
def test_custom_scorer_job_unsupported_by_custom_backend_stays_on_default(job_name, params):
    backend = select_executor_backend(is_custom_scorer=True, job_name=job_name, params=params)
    assert backend == "local"
