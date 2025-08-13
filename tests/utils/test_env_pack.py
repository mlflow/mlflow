import subprocess
import sys
import tarfile
import venv
from pathlib import Path
from unittest import mock

import pytest
import yaml

from mlflow.utils import env_pack
from mlflow.utils.databricks_utils import DatabricksRuntimeVersion


@pytest.fixture
def mock_dbr_version():
    with mock.patch.object(
        DatabricksRuntimeVersion,
        "parse",
        return_value=DatabricksRuntimeVersion(
            is_client_image=True,
            major=2,
            minor=0,
        ),
    ):
        yield


def test_tar_function_path_handling(tmp_path):
    """Test that _tar function correctly handles Path objects."""
    # Create test files
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    (root_dir / "test.txt").write_text("test content")
    (root_dir / "__pycache__").mkdir()
    (root_dir / "__pycache__" / "test.pyc").write_text("bytecode")
    (root_dir / "wheels_info.json").write_text("{}")

    # Create tar file
    tar_path = tmp_path / "test.tar"
    env_pack._tar(root_dir, tar_path)

    # Verify tar contents
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        names = {m.name for m in members}

        assert names == {".", "./test.txt"}


def test_pack_env_for_databricks_model_serving_pip_requirements(tmp_path, mock_dbr_version):
    """Test that pack_env_for_databricks_model_serving correctly handles pip requirements
    installation.
    """
    # Mock download_artifacts to return a path
    mock_artifacts_dir = tmp_path / "artifacts"
    mock_artifacts_dir.mkdir()
    (mock_artifacts_dir / "requirements.txt").write_text("numpy==1.21.0")

    # Create MLmodel file with correct runtime version
    mlmodel_path = mock_artifacts_dir / "MLmodel"
    mlmodel_path.write_text(
        yaml.dump(
            {
                "databricks_runtime": "client.2.0",
                "flavors": {"python_function": {"model_path": "model.pkl"}},
            }
        )
    )

    # Create a mock environment directory
    mock_env_dir = tmp_path / "mock_env"
    venv.create(mock_env_dir, with_pip=True)

    with (
        mock.patch(
            "mlflow.utils.env_pack.download_artifacts",
            return_value=str(mock_artifacts_dir),
        ),
        mock.patch("subprocess.run") as mock_run,
        mock.patch("sys.prefix", str(mock_env_dir)),
    ):
        # Mock subprocess.run to simulate successful pip install
        mock_run.return_value = mock.Mock(returncode=0)
        with env_pack.pack_env_for_databricks_model_serving(
            "models:/test-model/1", enforce_pip_requirements=True
        ) as artifacts_dir:
            # Verify artifacts directory exists and contains expected files
            artifacts_path = Path(artifacts_dir)
            assert artifacts_path.exists()
            assert (artifacts_path / env_pack._ARTIFACT_PATH).exists()
            assert (artifacts_path / env_pack._ARTIFACT_PATH / env_pack._MODEL_VERSION_TAR).exists()
            assert (
                artifacts_path / env_pack._ARTIFACT_PATH / env_pack._MODEL_ENVIRONMENT_TAR
            ).exists()

            # Verify the environment tar contains our mock files
            env_tar_path = (
                artifacts_path / env_pack._ARTIFACT_PATH / env_pack._MODEL_ENVIRONMENT_TAR
            )
            with tarfile.open(env_tar_path, "r:tar") as tar:
                members = tar.getmembers()
                member_names = {m.name for m in members}

                # Check for pip in site-packages based on platform
                if sys.platform == "win32":
                    expected_pip_path = "./Lib/site-packages/pip"
                else:
                    expected_pip_path = (
                        f"./lib/python{sys.version_info.major}.{sys.version_info.minor}"
                        "/site-packages/pip"
                    )

                assert expected_pip_path in member_names

            # Verify subprocess.run was called with correct arguments
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            assert args[0] == [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(mock_artifacts_dir / "requirements.txt"),
            ]
            assert kwargs["check"] is True
            assert kwargs["stdout"] == subprocess.PIPE
            assert kwargs["stderr"] == subprocess.STDOUT
            assert kwargs["text"] is True


def test_pack_env_for_databricks_model_serving_pip_requirements_error(tmp_path, mock_dbr_version):
    """Test that pack_env_for_databricks_model_serving correctly handles pip install errors."""
    # Mock download_artifacts to return a path
    mock_artifacts_dir = tmp_path / "artifacts"
    mock_artifacts_dir.mkdir()
    (mock_artifacts_dir / "requirements.txt").write_text("invalid-package==1.0.0")

    # Create MLmodel file with correct runtime version
    mlmodel_path = mock_artifacts_dir / "MLmodel"
    mlmodel_path.write_text(
        yaml.dump(
            {
                "databricks_runtime": "client.2.0",
                "flavors": {"python_function": {"model_path": "model.pkl"}},
            }
        )
    )

    with (
        mock.patch(
            "mlflow.utils.env_pack.download_artifacts",
            return_value=str(mock_artifacts_dir),
        ),
        mock.patch("subprocess.run") as mock_run,
        mock.patch("mlflow.utils.env_pack.eprint") as mock_eprint,
    ):
        mock_run.return_value = mock.Mock(
            returncode=1,
            stdout="ERROR: Could not find a version that satisfies the requirement invalid-package",
        )
        mock_run.side_effect = subprocess.CalledProcessError(1, "pip install", "Error message")

        with pytest.raises(
            subprocess.CalledProcessError,
            match="Command 'pip install' returned non-zero exit status 1.",
        ):
            with env_pack.pack_env_for_databricks_model_serving(
                "models:/test/1", enforce_pip_requirements=True
            ):
                pass

        # Verify error messages were printed
        mock_eprint.assert_any_call("Error installing requirements:")
        mock_eprint.assert_any_call("Error message")


def test_pack_env_for_databricks_model_serving_unsupported_version():
    """Test that pack_env_for_databricks_model_serving raises error for non-client image."""
    with mock.patch.object(
        DatabricksRuntimeVersion,
        "parse",
        return_value=DatabricksRuntimeVersion(
            is_client_image=False,  # Not a client image
            major=13,
            minor=0,
        ),
    ):
        with pytest.raises(ValueError, match="Serverless environment is required"):
            with env_pack.pack_env_for_databricks_model_serving("models:/test/1"):
                pass


def test_pack_env_for_databricks_model_serving_runtime_version_check(tmp_path, monkeypatch):
    """Test that pack_env_for_databricks_model_serving correctly checks runtime version
    compatibility.
    """
    # Mock download_artifacts to return a path
    mock_artifacts_dir = tmp_path / "artifacts"
    mock_artifacts_dir.mkdir()

    # Create MLmodel file with different runtime version
    mlmodel_path = mock_artifacts_dir / "MLmodel"
    mlmodel_path.write_text(
        yaml.dump(
            {
                "databricks_runtime": "client.3.0",  # Different major version
                "flavors": {"python_function": {"model_path": "model.pkl"}},
            }
        )
    )

    # Set current runtime to client.2.0
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "client.2.0")

    with mock.patch(
        "mlflow.utils.env_pack.download_artifacts", return_value=str(mock_artifacts_dir)
    ):
        with pytest.raises(ValueError, match="Runtime version mismatch"):
            with env_pack.pack_env_for_databricks_model_serving("models:/test-model/1"):
                pass

    # Test that same major version works
    mlmodel_path.write_text(
        yaml.dump(
            {
                "databricks_runtime": "client.2.1",  # Same major version
                "flavors": {"python_function": {"model_path": "model.pkl"}},
            }
        )
    )

    # Create a mock environment directory
    mock_env_dir = tmp_path / "mock_env"
    mock_env_dir.mkdir()

    with (
        mock.patch(
            "mlflow.utils.env_pack.download_artifacts", return_value=str(mock_artifacts_dir)
        ),
        mock.patch("sys.prefix", str(mock_env_dir)),
    ):
        with env_pack.pack_env_for_databricks_model_serving(
            "models:/test-model/1"
        ) as artifacts_dir:
            assert Path(artifacts_dir).exists()


def test_pack_env_for_databricks_model_serving_missing_runtime_version(tmp_path, mock_dbr_version):
    """Test that pack_env_for_databricks_model_serving requires databricks_runtime field."""
    # Mock download_artifacts to return a path
    mock_artifacts_dir = tmp_path / "artifacts"
    mock_artifacts_dir.mkdir()

    # Create MLmodel file without databricks_runtime field
    mlmodel_path = mock_artifacts_dir / "MLmodel"
    mlmodel_path.write_text(
        yaml.dump(
            {
                "flavors": {"python_function": {"model_path": "model.pkl"}},
            }
        )
    )

    with mock.patch(
        "mlflow.utils.env_pack.download_artifacts", return_value=str(mock_artifacts_dir)
    ):
        with pytest.raises(
            ValueError, match="Model must have been created in a Databricks runtime environment"
        ):
            with env_pack.pack_env_for_databricks_model_serving("models:/test-model/1"):
                pass
