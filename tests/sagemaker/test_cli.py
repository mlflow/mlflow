import os
import shutil
from pathlib import Path
from unittest import mock

import docker
import pytest
from click.testing import CliRunner

import mlflow
from mlflow.models.docker_utils import build_image_from_context
from mlflow.sagemaker.cli import build_and_push_container

from tests.pyfunc.docker.test_docker import assert_dockerfiles_equal

_MLFLOW_ROOT = Path(mlflow.__file__).parent.parent
_RESOURCE_DIR = os.path.join(_MLFLOW_ROOT, "tests", "resources", "dockerfile")
_TEST_IMAGE_NAME = "test-sagemaker-image"

_docker_client = docker.from_env()


@pytest.mark.parametrize(
    ("env_manager", "install_java"), [("conda", None), ("virtualenv", None), ("virtualenv", False)]
)
def test_build_and_push_container(tmp_path, env_manager, install_java):
    dst_dir = tmp_path / "context"

    # Copy the context dir to a temp dir so we can verify the generated Dockerfile
    def _build_image_with_copy(context_dir, image_name, network=None):
        shutil.copytree(context_dir, dst_dir)
        for _ in range(3):
            try:
                # Docker image build is unstable on GitHub Actions, retry up to 3 times
                build_image_from_context(context_dir, image_name)
                break
            except RuntimeError:
                pass
        else:
            raise RuntimeError("Docker image build failed.")

    with mock.patch(
        "mlflow.models.docker_utils.build_image_from_context", side_effect=_build_image_with_copy
    ):
        res = CliRunner().invoke(
            build_and_push_container,
            [
                "--no-push",
                "--mlflow-home",
                ".",
                "--env-manager",
                env_manager,
                "--container",
                _TEST_IMAGE_NAME,
            ]
            + ([f"--install-java={install_java}"] if install_java is not None else []),
            catch_exceptions=False,
        )
        assert res.exit_code == 0

    actual = dst_dir / "Dockerfile"
    expected = (
        Path(_RESOURCE_DIR)
        / f"Dockerfile_sagemaker_{env_manager}{'_no_java' if install_java is False else ''}"
    )
    assert_dockerfiles_equal(actual, expected)

    # Clean up generated image
    _docker_client.images.remove(_TEST_IMAGE_NAME, force=True)


def test_build_and_push_container_network_option():
    with mock.patch("mlflow.models.docker_utils.build_image_from_context") as mock_build:
        res = CliRunner().invoke(
            build_and_push_container,
            [
                "--no-push",
                "--mlflow-home",
                ".",
                "--network",
                "sagemaker",
                "--container",
                _TEST_IMAGE_NAME,
            ],
            catch_exceptions=False,
        )
        assert res.exit_code == 0
        mock_build.assert_called_once()
        assert mock_build.call_args[1]["network"] == "sagemaker"


def test_build_image_from_context_fallback_to_cli_when_docker_sdk_fails():
    with (
        mock.patch("mlflow.models.docker_utils.Popen") as mock_popen,
        mock.patch("docker.from_env", side_effect=Exception("connection refused")),
        mock.patch(
            "subprocess.run",
            return_value=mock.Mock(stdout="25.0.14\n", returncode=0),
        ),
    ):
        mock_popen.return_value.wait.return_value = 0
        build_image_from_context("/tmp/ctx", "test-image", network="sagemaker")

        cmd = mock_popen.call_args[0][0]
        assert "--platform" in cmd
        assert "linux/amd64" in cmd
        assert "--network" in cmd
        assert "sagemaker" in cmd


def test_build_image_from_context_no_platform_flag_when_version_detection_fails():
    with (
        mock.patch("mlflow.models.docker_utils.Popen") as mock_popen,
        mock.patch("docker.from_env", side_effect=Exception("connection refused")),
        mock.patch(
            "subprocess.run",
            side_effect=Exception("docker not found"),
        ),
    ):
        mock_popen.return_value.wait.return_value = 0
        build_image_from_context("/tmp/ctx", "test-image")

        cmd = mock_popen.call_args[0][0]
        assert "--platform" not in cmd


def test_build_image_from_context_no_platform_flag_when_cli_returns_malformed_output():
    with (
        mock.patch("mlflow.models.docker_utils.Popen") as mock_popen,
        mock.patch("docker.from_env", side_effect=Exception("connection refused")),
        mock.patch(
            "subprocess.run",
            return_value=mock.Mock(stdout="unknown\n", returncode=0),
        ),
    ):
        mock_popen.return_value.wait.return_value = 0
        build_image_from_context("/tmp/ctx", "test-image")

        cmd = mock_popen.call_args[0][0]
        assert "--platform" not in cmd


def test_build_image_from_context_no_network_flag_when_not_specified():
    with (
        mock.patch("mlflow.models.docker_utils.Popen") as mock_popen,
        mock.patch("docker.from_env", side_effect=Exception("connection refused")),
        mock.patch(
            "subprocess.run",
            return_value=mock.Mock(stdout="25.0.14\n", returncode=0),
        ),
    ):
        mock_popen.return_value.wait.return_value = 0
        build_image_from_context("/tmp/ctx", "test-image")

        cmd = mock_popen.call_args[0][0]
        assert "--network" not in cmd
