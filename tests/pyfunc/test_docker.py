import os
import mlflow
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.utils.file_utils import TempDir
import pytest
import shutil
import sklearn
import sklearn.neighbors
from unittest import mock

from mlflow.utils.env_manager import CONDA, VIRTUALENV

_MLFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RESOURCE_DIR = os.path.join(_MLFLOW_ROOT, "tests", "resources", "dockerfile")


def log_model():
    with mlflow.start_run() as run:
        knn_model = sklearn.neighbors.KNeighborsClassifier()
        mlflow.sklearn.log_model(
            knn_model,
            f"model",
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"scikit-learn=={sklearn.__version__}"
            ] # Skip requirements inference for speed up
        )
        model_uri = f"runs:/{run.info.run_id}/model"
    return model_uri


class TestParam:
    def __init__(self,
                 expected_dockerfile,
                 env_manager=VIRTUALENV,
                 mlflow_home=None,
                 install_mlflow=False,
                 enable_mlserver=False,
                 specify_model_uri=True):
        self.env_manager = env_manager
        self.mlflow_home = mlflow_home
        self.install_mlflow = install_mlflow
        self.enable_mlserver = enable_mlserver
        self.specify_model_uri = specify_model_uri
        self.expected_dockerfile = expected_dockerfile

_TEST_PARAMS = [
    TestParam(
        expected_dockerfile="Dockerfile_default",
    ),
    TestParam(
        env_manager=CONDA,
        expected_dockerfile="Dockerfile_conda",
    ),
    TestParam(
        mlflow_home=".",
        expected_dockerfile="Dockerfile_with_mlflow_home",
    ),
    TestParam(
        install_mlflow=True,
        expected_dockerfile="Dockerfile_install_mlflow",
    ),
    TestParam(
        mlflow_home=_MLFLOW_ROOT,
        install_mlflow=True,
        expected_dockerfile="Dockerfile_install_mlflow_from_mlflow_home",
    ),
    TestParam(
        enable_mlserver=True,
        expected_dockerfile="Dockerfile_enable_mlserver",
    ),
    TestParam(
        specify_model_uri=False,
        expected_dockerfile="Dockerfile_no_model_uri"
    ),
]


@pytest.mark.parametrize("params", _TEST_PARAMS)
def test_generate_dockerfile(tmp_path, params):
    model_uri = log_model() if params.specify_model_uri else None

    backend = get_flavor_backend(model_uri, docker_build=True, env_manager=params.env_manager)

    backend.generate_dockerfile(
        model_uri=model_uri,
        output_path=tmp_path,
        mlflow_home=params.mlflow_home,
        install_mlflow=params.install_mlflow,
        enable_mlserver=params.enable_mlserver,
    )

    with open(os.path.join(tmp_path, "Dockerfile"), "r") as f:
        actual_dockerfile = f.read()

    with open(os.path.join(_RESOURCE_DIR, params.expected_dockerfile), "r") as f:
        expected_dockerfile = f.read()

    assert actual_dockerfile == expected_dockerfile


@pytest.mark.parametrize("params", _TEST_PARAMS)
def test_build_image(tmp_path, params):
    model_uri = log_model() if params.specify_model_uri else None

    backend = get_flavor_backend(model_uri, docker_build=True, env_manager=params.env_manager)

    # Mock build method to just copy the context dir to a temp dir so we can verify the generated Dockerfile
    dst_dir = tmp_path / "context"
    with mock.patch("mlflow.models.docker_utils._build_image_from_context",
                    side_effect = lambda context_dir, image_name: shutil.copytree(context_dir, dst_dir)):
        backend.build_image(
            model_uri=model_uri,
            image_name="test_image",
            mlflow_home=params.mlflow_home,
            install_mlflow=params.install_mlflow,
            enable_mlserver=params.enable_mlserver,
        )

    with open(os.path.join(dst_dir, "Dockerfile"), "r") as f:
        actual_dockerfile = f.read()

    with open(os.path.join(_RESOURCE_DIR, params.expected_dockerfile), "r") as f:
        expected_dockerfile = f.read()

    assert actual_dockerfile == expected_dockerfile