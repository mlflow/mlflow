import difflib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest import mock

import pytest
import sklearn
import sklearn.neighbors

import mlflow
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.utils.env_manager import CONDA, VIRTUALENV
from mlflow.version import VERSION

_MLFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RESOURCE_DIR = os.path.join(_MLFLOW_ROOT, "tests", "resources", "dockerfile")


def assert_dockerfiles_equal(actual_dockerfile_path: Path, expected_dockerfile_path: Path):
    actual_dockerfile = actual_dockerfile_path.read_text()
    expected_dockerfile = expected_dockerfile_path.read_text().replace("$VERSION", VERSION)

    assert (
        actual_dockerfile == expected_dockerfile
    ), "Generated Dockerfile does not match expected one. Diff:\n" + "\n".join(
        difflib.unified_diff(expected_dockerfile.splitlines(), actual_dockerfile.splitlines())
    )


def log_model():
    with mlflow.start_run():
        knn_model = sklearn.neighbors.KNeighborsClassifier()
        model_info = mlflow.sklearn.log_model(
            knn_model,
            "model",
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"scikit-learn=={sklearn.__version__}",
            ],  # Skip requirements inference for speed up
        )
        return model_info.model_uri


@dataclass
class Param:
    expected_dockerfile: str
    env_manager: str = VIRTUALENV
    mlflow_home: Optional[str] = None
    install_mlflow: bool = False
    enable_mlserver: bool = False
    # If True, image is built with --model-uri param
    specify_model_uri: bool = True


_TEST_PARAMS = [
    Param(
        expected_dockerfile="Dockerfile_default",
    ),
    Param(
        env_manager=CONDA,
        expected_dockerfile="Dockerfile_conda",
    ),
    Param(
        mlflow_home=".",
        expected_dockerfile="Dockerfile_with_mlflow_home",
    ),
    Param(
        install_mlflow=True,
        expected_dockerfile="Dockerfile_install_mlflow",
    ),
    Param(
        mlflow_home=_MLFLOW_ROOT,
        install_mlflow=True,
        expected_dockerfile="Dockerfile_install_mlflow_from_mlflow_home",
    ),
    Param(
        enable_mlserver=True,
        expected_dockerfile="Dockerfile_enable_mlserver",
    ),
    Param(specify_model_uri=False, expected_dockerfile="Dockerfile_no_model_uri"),
]


@pytest.mark.parametrize("params", _TEST_PARAMS)
def test_generate_dockerfile(tmp_path, params):
    model_uri = log_model() if params.specify_model_uri else None

    backend = get_flavor_backend(model_uri, docker_build=True, env_manager=params.env_manager)

    backend.generate_dockerfile(
        model_uri=model_uri,
        output_dir=tmp_path,
        mlflow_home=params.mlflow_home,
        install_mlflow=params.install_mlflow,
        enable_mlserver=params.enable_mlserver,
    )

    actual = tmp_path / "Dockerfile"
    expected = Path(_RESOURCE_DIR) / params.expected_dockerfile
    assert_dockerfiles_equal(actual, expected)


@pytest.mark.parametrize("params", _TEST_PARAMS)
def test_build_image(tmp_path, params):
    model_uri = log_model() if params.specify_model_uri else None

    backend = get_flavor_backend(model_uri, docker_build=True, env_manager=params.env_manager)

    # Copy the context dir to a temp dir so we can verify the generated Dockerfile
    dst_dir = tmp_path / "context"
    with mock.patch(
        "mlflow.models.docker_utils.build_image_from_context",
        side_effect=lambda context_dir, image_name: shutil.copytree(context_dir, dst_dir),
    ):
        backend.build_image(
            model_uri=model_uri,
            image_name="test_image",
            mlflow_home=params.mlflow_home,
            install_mlflow=params.install_mlflow,
            enable_mlserver=params.enable_mlserver,
        )

    actual = dst_dir / "Dockerfile"
    expected = Path(_RESOURCE_DIR) / params.expected_dockerfile
    assert_dockerfiles_equal(actual, expected)
