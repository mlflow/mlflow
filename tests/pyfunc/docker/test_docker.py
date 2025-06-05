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
from mlflow.models import Model
from mlflow.models.docker_utils import build_image_from_context
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.env_manager import CONDA, LOCAL, VIRTUALENV
from mlflow.version import VERSION

from tests.pyfunc.docker.conftest import RESOURCE_DIR, get_released_mlflow_version


def assert_dockerfiles_equal(actual_dockerfile_path: Path, expected_dockerfile_path: Path):
    actual_dockerfile = (
        actual_dockerfile_path.read_text()
        .replace(VERSION, get_released_mlflow_version())
        # TODO: Remove this once https://github.com/pandas-dev/pandas/issues/61564 is resolved
        .replace(" pandas!=2.3.0", "")
    )
    expected_dockerfile = (
        expected_dockerfile_path.read_text()
        .replace("${{ MLFLOW_VERSION }}", get_released_mlflow_version())
        .replace("${{ PYTHON_VERSION }}", PYTHON_VERSION)
    )
    assert actual_dockerfile == expected_dockerfile, (
        "Generated Dockerfile does not match expected one. Diff:\n"
        + "\n".join(
            difflib.unified_diff(expected_dockerfile.splitlines(), actual_dockerfile.splitlines())
        )
    )


def save_model(tmp_path):
    knn_model = sklearn.neighbors.KNeighborsClassifier()
    model_path = os.path.join(tmp_path, "model")
    mlflow.sklearn.save_model(
        knn_model,
        path=model_path,
        pip_requirements=[
            f"mlflow=={get_released_mlflow_version()}",
            f"scikit-learn=={sklearn.__version__}",
            # TODO: Remove this once https://github.com/pandas-dev/pandas/issues/61564 is resolved
            "pandas!=2.3.0",
        ],  # Skip requirements inference for speed up
    )
    return model_path


def add_spark_flavor_to_model(model_path):
    model_config_path = os.path.join(model_path, "MLmodel")
    model = Model.load(model_config_path)
    model.add_flavor("spark", spark_version="3.5.0")
    model.save(model_config_path)


@dataclass
class Param:
    expected_dockerfile: str
    env_manager: Optional[str] = None
    mlflow_home: Optional[str] = None
    install_mlflow: bool = False
    enable_mlserver: bool = False
    # If True, image is built with --model-uri param
    specify_model_uri: bool = True


@pytest.mark.parametrize(
    "params",
    [
        Param(expected_dockerfile="Dockerfile_default"),
        Param(expected_dockerfile="Dockerfile_default", env_manager=LOCAL),
        Param(expected_dockerfile="Dockerfile_java_flavor", env_manager=VIRTUALENV),
        Param(expected_dockerfile="Dockerfile_conda", env_manager=CONDA),
        Param(install_mlflow=True, expected_dockerfile="Dockerfile_install_mlflow"),
        Param(enable_mlserver=True, expected_dockerfile="Dockerfile_enable_mlserver"),
        Param(mlflow_home=".", expected_dockerfile="Dockerfile_with_mlflow_home"),
        Param(specify_model_uri=False, expected_dockerfile="Dockerfile_no_model_uri"),
    ],
)
def test_build_image(tmp_path, params):
    model_uri = save_model(tmp_path) if params.specify_model_uri else None

    backend = get_flavor_backend(model_uri, docker_build=True, env_manager=params.env_manager)

    # Copy the context dir to a temp dir so we can verify the generated Dockerfile
    def _build_image_with_copy(context_dir, image_name):
        # Replace mlflow dev version in Dockerfile with the latest released one
        dockerfile = Path(context_dir) / "Dockerfile"
        content = dockerfile.read_text()
        content = content.replace(
            f"pip install mlflow=={VERSION}",
            # TODO: Remove ` pandas!=2.3.0` once https://github.com/pandas-dev/pandas/issues/61564
            # is resolved
            f"pip install mlflow=={get_released_mlflow_version()} pandas!=2.3.0",
        )
        dockerfile.write_text(content)

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

    dst_dir = tmp_path / "context"
    with mock.patch(
        "mlflow.models.docker_utils.build_image_from_context",
        side_effect=_build_image_with_copy,
    ):
        backend.build_image(
            model_uri=model_uri,
            image_name="test_image",
            mlflow_home=params.mlflow_home,
            install_mlflow=params.install_mlflow,
            enable_mlserver=params.enable_mlserver,
        )

    actual = dst_dir / "Dockerfile"
    expected = Path(RESOURCE_DIR) / params.expected_dockerfile
    assert_dockerfiles_equal(actual, expected)


def test_generate_dockerfile_for_java_flavor(tmp_path):
    model_path = save_model(tmp_path)
    add_spark_flavor_to_model(model_path)

    backend = get_flavor_backend(model_path, docker_build=True, env_manager=None)

    backend.generate_dockerfile(
        model_uri=model_path,
        output_dir=tmp_path,
    )

    actual = tmp_path / "Dockerfile"
    expected = Path(RESOURCE_DIR) / "Dockerfile_java_flavor"
    assert_dockerfiles_equal(actual, expected)


def test_generate_dockerfile_for_custom_image(tmp_path):
    model_path = save_model(tmp_path)
    add_spark_flavor_to_model(model_path)

    backend = get_flavor_backend(model_path, docker_build=True, env_manager=None)

    backend.generate_dockerfile(
        base_image="quay.io/jupyter/scipy-notebook:latest",
        model_uri=model_path,
        output_dir=tmp_path,
    )

    actual = tmp_path / "Dockerfile"
    expected = Path(RESOURCE_DIR) / "Dockerfile_custom_scipy"
    assert_dockerfiles_equal(actual, expected)
