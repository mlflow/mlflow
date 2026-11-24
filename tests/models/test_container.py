import os
import sys
from unittest import mock

import pytest
import yaml

from mlflow.environment_variables import MLFLOW_ALLOW_MODEL_CODE_EXECUTION
from mlflow.exceptions import MlflowException
from mlflow.models.container import _install_model_dependencies_to_env
from mlflow.utils import env_manager as em


@pytest.mark.parametrize("env_manager", [em.LOCAL, em.CONDA, em.VIRTUALENV])
def test_model_code_disabled_refuses_container_dependencies(monkeypatch, tmp_path, env_manager):
    monkeypatch.setenv(MLFLOW_ALLOW_MODEL_CODE_EXECUTION.name, "false")
    with (
        mock.patch(
            "mlflow.utils.model_utils._is_in_databricks_managed_environment", return_value=False
        ) as managed_environment,
        mock.patch("mlflow.models.container.Model.load") as load,
        mock.patch(
            "mlflow.models.container.Popen",
            side_effect=AssertionError("Dependency installation must not run"),
        ) as popen,
        mock.patch(
            "mlflow.models.container._get_or_create_virtualenv",
            side_effect=AssertionError("Environment creation must not run"),
        ) as virtualenv,
    ):
        with pytest.raises(MlflowException, match=MLFLOW_ALLOW_MODEL_CODE_EXECUTION.name):
            _install_model_dependencies_to_env(str(tmp_path), env_manager)

    managed_environment.assert_called_once_with()
    load.assert_not_called()
    popen.assert_not_called()
    virtualenv.assert_not_called()


@pytest.mark.parametrize(("allow_model_code", "managed"), [("true", False), ("false", True)])
def test_allowed_container_dependencies(monkeypatch, tmp_path, allow_model_code, managed):
    monkeypatch.setenv(MLFLOW_ALLOW_MODEL_CODE_EXECUTION.name, allow_model_code)
    _create_model_artifact(tmp_path, dependencies=["example-package==1.0"])
    with (
        mock.patch(
            "mlflow.utils.model_utils._is_in_databricks_managed_environment", return_value=managed
        ) as managed_environment,
        mock.patch(
            "mlflow.models.container.Popen", **{"return_value.wait.return_value": 0}
        ) as popen,
    ):
        assert _install_model_dependencies_to_env(str(tmp_path), em.LOCAL) == []

    if allow_model_code == "false":
        managed_environment.assert_called_once_with()
    else:
        managed_environment.assert_not_called()
    popen.assert_called_once_with([sys.executable, "-m", "pip", "install", "example-package==1.0"])
    popen.return_value.wait.assert_called_once_with()


def _create_model_artifact(model_path, dependencies, build_dependencies=None):
    """Helper to create a minimal model artifact for testing."""
    with open(os.path.join(model_path, "MLmodel"), "w") as f:
        yaml.dump(
            {
                "flavors": {
                    "python_function": {
                        "env": {"virtualenv": "python_env.yaml"},
                        "loader_module": "mlflow.pyfunc.model",
                    }
                }
            },
            f,
        )

    with open(os.path.join(model_path, "requirements.txt"), "w") as f:
        f.write("")

    with open(os.path.join(model_path, "python_env.yaml"), "w") as f:
        yaml.dump(
            {
                "python": "3.12",
                "build_dependencies": build_dependencies or [],
                "dependencies": dependencies,
            },
            f,
        )


def test_command_injection_via_semicolon_blocked(tmp_path):
    model_path = str(tmp_path)
    _create_model_artifact(
        model_path,
        dependencies=["numpy; echo INJECTED > /tmp/test_injection_semicolon.txt; #"],
    )

    evidence_file = "/tmp/test_injection_semicolon.txt"
    if os.path.exists(evidence_file):
        os.remove(evidence_file)

    with pytest.raises(Exception, match="Failed to install model dependencies"):
        _install_model_dependencies_to_env(model_path, env_manager=em.LOCAL)

    assert not os.path.exists(evidence_file), "Command injection via semicolon succeeded!"


def test_command_injection_via_pipe_blocked(tmp_path):
    model_path = str(tmp_path)
    _create_model_artifact(
        model_path,
        dependencies=["numpy | echo INJECTED > /tmp/test_injection_pipe.txt"],
    )

    evidence_file = "/tmp/test_injection_pipe.txt"
    if os.path.exists(evidence_file):
        os.remove(evidence_file)

    with pytest.raises(Exception, match="Failed to install model dependencies"):
        _install_model_dependencies_to_env(model_path, env_manager=em.LOCAL)

    assert not os.path.exists(evidence_file), "Command injection via pipe succeeded!"


def test_command_injection_via_backticks_blocked(tmp_path):
    model_path = str(tmp_path)
    _create_model_artifact(
        model_path,
        dependencies=["`echo INJECTED > /tmp/test_injection_backtick.txt`"],
    )

    evidence_file = "/tmp/test_injection_backtick.txt"
    if os.path.exists(evidence_file):
        os.remove(evidence_file)

    with pytest.raises(Exception, match="Failed to install model dependencies"):
        _install_model_dependencies_to_env(model_path, env_manager=em.LOCAL)

    assert not os.path.exists(evidence_file), "Command injection via backticks succeeded!"


def test_command_injection_via_dollar_parens_blocked(tmp_path):
    model_path = str(tmp_path)
    _create_model_artifact(
        model_path,
        dependencies=["$(echo INJECTED > /tmp/test_injection_dollar.txt)"],
    )

    evidence_file = "/tmp/test_injection_dollar.txt"
    if os.path.exists(evidence_file):
        os.remove(evidence_file)

    with pytest.raises(Exception, match="Failed to install model dependencies"):
        _install_model_dependencies_to_env(model_path, env_manager=em.LOCAL)

    assert not os.path.exists(evidence_file), "Command injection via $() succeeded!"


def test_command_injection_via_ampersand_blocked(tmp_path):
    model_path = str(tmp_path)
    _create_model_artifact(
        model_path,
        dependencies=["numpy && echo INJECTED > /tmp/test_injection_ampersand.txt"],
    )

    evidence_file = "/tmp/test_injection_ampersand.txt"
    if os.path.exists(evidence_file):
        os.remove(evidence_file)

    with pytest.raises(Exception, match="Failed to install model dependencies"):
        _install_model_dependencies_to_env(model_path, env_manager=em.LOCAL)

    assert not os.path.exists(evidence_file), "Command injection via && succeeded!"


def test_legitimate_package_install(tmp_path):
    model_path = str(tmp_path)
    _create_model_artifact(
        model_path,
        dependencies=["pip"],
        build_dependencies=[],
    )

    result = _install_model_dependencies_to_env(model_path, env_manager=em.LOCAL)
    assert result == []


def test_requirements_file_reference(tmp_path):
    model_path = str(tmp_path)
    _create_model_artifact(
        model_path,
        dependencies=["-r requirements.txt"],
        build_dependencies=["pip"],
    )

    with open(os.path.join(model_path, "requirements.txt"), "w") as f:
        f.write("# empty requirements\n")

    result = _install_model_dependencies_to_env(model_path, env_manager=em.LOCAL)
    assert result == []


def test_requirements_path_replacement(tmp_path):
    model_path = str(tmp_path)
    _create_model_artifact(
        model_path,
        dependencies=["-r requirements.txt"],
    )

    with open(os.path.join(model_path, "requirements.txt"), "w") as f:
        f.write("six\n")

    with mock.patch("mlflow.models.container.Popen") as mock_popen:
        mock_popen.return_value.wait.return_value = 0

        _install_model_dependencies_to_env(model_path, env_manager=em.LOCAL)

        call_args = mock_popen.call_args[0][0]
        assert isinstance(call_args, list), "Should use list args, not shell string"

        assert "-r" in call_args
        req_index = call_args.index("-r")
        req_path = call_args[req_index + 1]
        assert req_path == os.path.join(model_path, "requirements.txt")


def test_no_shell_execution(tmp_path):
    model_path = str(tmp_path)
    _create_model_artifact(
        model_path,
        dependencies=["pip"],
    )

    with mock.patch("mlflow.models.container.Popen") as mock_popen:
        mock_popen.return_value.wait.return_value = 0

        _install_model_dependencies_to_env(model_path, env_manager=em.LOCAL)

        call_args = mock_popen.call_args
        assert isinstance(call_args[0][0], list)
        assert call_args[1].get("shell") is not True


def test_build_dependencies_processed(tmp_path):
    model_path = str(tmp_path)
    _create_model_artifact(
        model_path,
        dependencies=["pip"],
        build_dependencies=["setuptools", "wheel"],
    )

    with mock.patch("mlflow.models.container.Popen") as mock_popen:
        mock_popen.return_value.wait.return_value = 0

        _install_model_dependencies_to_env(model_path, env_manager=em.LOCAL)

        call_args = mock_popen.call_args[0][0]
        assert "setuptools" in call_args
        assert "wheel" in call_args
        assert "pip" in call_args


def test_package_name_with_requirements_substring_not_modified(tmp_path):
    model_path = str(tmp_path)
    _create_model_artifact(
        model_path,
        dependencies=["my-requirements.txt-parser", "requirements.txt-tools"],
    )

    with mock.patch("mlflow.models.container.Popen") as mock_popen:
        mock_popen.return_value.wait.return_value = 0

        _install_model_dependencies_to_env(model_path, env_manager=em.LOCAL)

        call_args = mock_popen.call_args[0][0]
        assert "my-requirements.txt-parser" in call_args
        assert "requirements.txt-tools" in call_args
        assert not any(model_path in arg for arg in call_args if "parser" in arg or "tools" in arg)


@pytest.mark.parametrize("env_manager", [em.UV, "unknown"])
def test_unsupported_env_manager_raises(tmp_path, env_manager):
    model_path = str(tmp_path)
    _create_model_artifact(model_path, dependencies=["pip"])

    with pytest.raises(MlflowException, match=r"Building a model container supports one of"):
        _install_model_dependencies_to_env(model_path, env_manager=env_manager)
