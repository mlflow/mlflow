import difflib
import os
import pytest
import shutil
from click.testing import CliRunner
from unittest import mock

from mlflow.models.docker_utils import _build_image_from_context
from mlflow.sagemaker.cli import build_and_push_container


_MLFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RESOURCE_DIR = os.path.join(_MLFLOW_ROOT, "tests", "resources", "dockerfile")


@pytest.mark.parametrize("env_manager", ["conda", "virtualenv"])
def test_build_and_push_container(tmp_path, env_manager):
    dst_dir = tmp_path / "context"

    # Mock build method to copy the context dir to a temp dir so we can verify the generated Dockerfile
    def _build_image_with_copy(context_dir, image_name):
        shutil.copytree(context_dir, dst_dir)
        _build_image_from_context(context_dir, image_name)

    with mock.patch("mlflow.models.docker_utils._build_image_from_context",
                    side_effect=_build_image_with_copy):
        res = CliRunner().invoke(
            build_and_push_container,
            [
                "--no-push",
                "--mlflow-home",
                ".",
                "--env-manager",
                env_manager,
            ],
            catch_exceptions=False,
        )
        assert res.exit_code == 0

        with open(os.path.join(dst_dir, "Dockerfile"), "r") as f:
            actual_dockerfile = f.read()

        with open(os.path.join(_RESOURCE_DIR, f"Dockerfile_sagemaker_{env_manager}"), "r") as f:
            expected_dockerfile = f.read()

        assert actual_dockerfile == expected_dockerfile
