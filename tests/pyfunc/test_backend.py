from unittest import mock

import pytest

from mlflow.environment_variables import MLFLOW_ALLOW_MODEL_CODE_EXECUTION
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import ENV
from mlflow.pyfunc.backend import PyFuncBackend
from mlflow.utils import env_manager as em
from mlflow.utils.environment import Environment


@pytest.mark.parametrize("env_manager", [em.UV, em.VIRTUALENV, em.CONDA])
@pytest.mark.parametrize("entrypoint", ["prepare_env", "predict", "serve", "serve_stdin"])
def test_model_code_disabled_refuses_environment_creation(
    monkeypatch, tmp_path, env_manager, entrypoint
):
    monkeypatch.setenv(MLFLOW_ALLOW_MODEL_CODE_EXECUTION.name, "false")
    backend = PyFuncBackend({ENV: {"conda": "conda.yaml"}}, env_manager=env_manager)
    kwargs = {
        "predict": {"input_path": None, "output_path": None, "content_type": "json"},
        "serve": {"port": 5000, "host": "127.0.0.1", "timeout": 30},
    }.get(entrypoint, {})

    with (
        mock.patch(
            "mlflow.utils.model_utils._is_in_databricks_managed_environment", return_value=False
        ) as managed_environment,
        mock.patch(
            "mlflow.pyfunc.backend._download_artifact_from_uri", return_value=str(tmp_path)
        ) as download,
        mock.patch(
            "mlflow.pyfunc.backend.scoring_server.get_cmd", return_value=("serve", {})
        ) as get_command,
        mock.patch(
            "mlflow.pyfunc.backend._get_or_create_virtualenv",
            side_effect=AssertionError("Environment creation must not run"),
        ) as virtualenv,
        mock.patch(
            "mlflow.pyfunc.backend.get_or_create_conda_env",
            side_effect=AssertionError("Environment creation must not run"),
        ) as conda,
    ):
        with pytest.raises(MlflowException, match=MLFLOW_ALLOW_MODEL_CODE_EXECUTION.name):
            getattr(backend, entrypoint)("model", **kwargs)

    managed_environment.assert_called_once_with()
    if entrypoint == "prepare_env":
        download.assert_not_called()
    else:
        download.assert_called_once_with("model")
    if entrypoint == "serve":
        get_command.assert_called_once_with(str(tmp_path), 5000, "127.0.0.1", 30, 1)
    else:
        get_command.assert_not_called()
    virtualenv.assert_not_called()
    conda.assert_not_called()


@pytest.mark.parametrize("env_manager", [em.UV, em.VIRTUALENV, em.CONDA])
def test_model_code_disabled_refuses_cached_environment(monkeypatch, env_manager):
    backend = PyFuncBackend({ENV: {"conda": "conda.yaml"}}, env_manager=env_manager)
    backend._environment = Environment("activate")
    monkeypatch.setenv(MLFLOW_ALLOW_MODEL_CODE_EXECUTION.name, "false")

    with (
        mock.patch(
            "mlflow.utils.model_utils._is_in_databricks_managed_environment", return_value=False
        ) as managed_environment,
        mock.patch("mlflow.pyfunc.backend._download_artifact_from_uri") as download,
    ):
        with pytest.raises(MlflowException, match=MLFLOW_ALLOW_MODEL_CODE_EXECUTION.name):
            backend.prepare_env("model")

    managed_environment.assert_called_once_with()
    download.assert_not_called()


@pytest.mark.parametrize("env_manager", [em.UV, em.VIRTUALENV, em.CONDA])
@pytest.mark.parametrize(("allow_model_code", "managed"), [("true", False), ("false", True)])
def test_allowed_model_environment_preparation(
    monkeypatch, tmp_path, env_manager, allow_model_code, managed
):
    monkeypatch.setenv(MLFLOW_ALLOW_MODEL_CODE_EXECUTION.name, allow_model_code)
    backend = PyFuncBackend({ENV: {"conda": "conda.yaml"}}, env_manager=env_manager)
    with (
        mock.patch(
            "mlflow.utils.model_utils._is_in_databricks_managed_environment", return_value=managed
        ) as managed_environment,
        mock.patch(
            "mlflow.pyfunc.backend._download_artifact_from_uri", return_value=str(tmp_path)
        ) as download,
        mock.patch(
            "mlflow.pyfunc.backend._get_or_create_virtualenv", return_value="activate"
        ) as virtualenv,
        mock.patch(
            "mlflow.pyfunc.backend.get_or_create_conda_env", return_value=Environment("activate")
        ) as conda,
        mock.patch.object(Environment, "execute") as execute,
    ):
        environment = backend.prepare_env("model")

    if allow_model_code == "false":
        managed_environment.assert_called_once_with()
    else:
        managed_environment.assert_not_called()
    download.assert_called_once_with("model")
    if env_manager == em.CONDA:
        conda.assert_called_once()
        virtualenv.assert_not_called()
    else:
        virtualenv.assert_called_once()
        conda.assert_not_called()
    execute.assert_called_once_with('python -c ""')
    assert backend._environment is environment


@pytest.mark.parametrize(("allow_model_code", "managed"), [("true", False), ("false", True)])
def test_allowed_cached_model_environment(monkeypatch, allow_model_code, managed):
    monkeypatch.setenv(MLFLOW_ALLOW_MODEL_CODE_EXECUTION.name, allow_model_code)
    backend = PyFuncBackend({}, env_manager=em.UV)
    backend._environment = Environment("activate")
    with (
        mock.patch(
            "mlflow.utils.model_utils._is_in_databricks_managed_environment", return_value=managed
        ) as managed_environment,
        mock.patch("mlflow.pyfunc.backend._download_artifact_from_uri") as download,
    ):
        assert backend.prepare_env("model") is backend._environment

    if allow_model_code == "false":
        managed_environment.assert_called_once_with()
    else:
        managed_environment.assert_not_called()
    download.assert_not_called()


@pytest.mark.parametrize("entrypoint", ["generate_dockerfile", "build_image"])
@pytest.mark.parametrize("env_manager", [em.LOCAL, em.VIRTUALENV, em.CONDA])
def test_model_code_disabled_refuses_docker_preparation(
    monkeypatch, tmp_path, entrypoint, env_manager
):
    monkeypatch.setenv(MLFLOW_ALLOW_MODEL_CODE_EXECUTION.name, "false")
    backend = PyFuncBackend({ENV: {"conda": "conda.yaml"}}, env_manager=env_manager)
    kwargs = (
        {"output_dir": str(tmp_path / "context")}
        if entrypoint == "generate_dockerfile"
        else {"image_name": "test-model"}
    )
    with (
        mock.patch(
            "mlflow.utils.model_utils._is_in_databricks_managed_environment", return_value=False
        ),
        mock.patch("mlflow.pyfunc.backend._download_artifact_from_uri") as download,
        mock.patch("mlflow.pyfunc.backend.docker_utils.generate_dockerfile") as generate,
        mock.patch("mlflow.pyfunc.backend.docker_utils.build_image_from_context") as build,
    ):
        with pytest.raises(MlflowException, match=MLFLOW_ALLOW_MODEL_CODE_EXECUTION.name):
            getattr(backend, entrypoint)("model", **kwargs)
    download.assert_not_called()
    generate.assert_not_called()
    build.assert_not_called()
    if entrypoint == "generate_dockerfile":
        assert not (tmp_path / "context").exists()


@pytest.mark.parametrize(("allow_model_code", "managed"), [("true", False), ("false", True)])
def test_allowed_model_docker_preparation(monkeypatch, tmp_path, allow_model_code, managed):
    monkeypatch.setenv(MLFLOW_ALLOW_MODEL_CODE_EXECUTION.name, allow_model_code)
    backend = PyFuncBackend({}, env_manager=em.LOCAL)
    with (
        mock.patch(
            "mlflow.utils.model_utils._is_in_databricks_managed_environment", return_value=managed
        ),
        mock.patch("mlflow.pyfunc.backend._download_artifact_from_uri", return_value=str(tmp_path)),
        mock.patch.object(backend, "_model_installation_steps", return_value="install-model"),
        mock.patch("mlflow.pyfunc.backend.docker_utils.generate_dockerfile") as generate,
    ):
        backend.generate_dockerfile("model", str(tmp_path / "context"), base_image="python:3.12")
    assert generate.call_args.kwargs["model_install_steps"] == "install-model"


def test_model_code_disabled_allows_dockerfile_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv(MLFLOW_ALLOW_MODEL_CODE_EXECUTION.name, "false")
    backend = PyFuncBackend({}, env_manager=em.VIRTUALENV)
    with mock.patch("mlflow.pyfunc.backend.docker_utils.generate_dockerfile") as generate:
        backend.generate_dockerfile(None, str(tmp_path / "context"))
    assert generate.call_args.kwargs["model_install_steps"] == ""
