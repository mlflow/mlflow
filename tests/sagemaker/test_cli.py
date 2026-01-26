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


@pytest.fixture(scope="module", autouse=True)
def cleanup_base_image():
    """Clean up Docker images pulled during tests (only in CI environments)."""
    if not os.environ.get("CI"):
        yield
        return

    client = docker.from_env()
    images = {img.id for img in client.images.list()}
    yield
    for img in client.images.list():
        if img.id not in images:
            try:
                client.images.remove(img.id, force=True)
            except Exception:
                # Ignore errors during cleanup
                pass


@pytest.mark.parametrize(
    ("env_manager", "install_java"), [("conda", None), ("virtualenv", None), ("virtualenv", False)]
)
def test_build_and_push_container(tmp_path, env_manager, install_java):
    dst_dir = tmp_path / "context"

    # Copy the context dir to a temp dir so we can verify the generated Dockerfile
    def _build_image_with_copy(context_dir, image_name):
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
