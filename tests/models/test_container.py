"""
Tests for mlflow.models.container module.

Includes security tests for command injection prevention.
"""

import os
from unittest import mock

import pytest
import yaml

from mlflow.models.container import _install_model_dependencies_to_env
from mlflow.utils import env_manager as em


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
