import logging
import os
import subprocess
from functools import lru_cache

import pytest
import requests
from packaging.version import Version

import docker
import mlflow

TEST_IMAGE_NAME = "test_image"
MLFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESOURCE_DIR = os.path.join(MLFLOW_ROOT, "tests", "resources", "dockerfile")

docker_client = docker.from_env()

_logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def clean_up_docker():
    yield

    # Get all containers using the test image
    containers = docker_client.containers.list(filters={"ancestor": TEST_IMAGE_NAME})
    for container in containers:
        container.remove(force=True)

    # Clean up the image
    try:
        docker_client.images.remove(TEST_IMAGE_NAME, force=True)
    except docker.errors.ImageNotFound:
        pass

    # Clean up the build cache and volumes
    try:
        subprocess.run(["docker", "builder", "prune", "-a", "-f"], check=True)
    except subprocess.CalledProcessError as e:
        _logger.warning("Failed to clean up docker system: %s", e)


@lru_cache(maxsize=1)
def get_released_mlflow_version():
    url = "https://pypi.org/pypi/mlflow/json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    versions = [
        v for v in map(Version, data["releases"]) if not (v.is_devrelease or v.is_prerelease)
    ]
    return str(sorted(versions, reverse=True)[0])


def save_model_with_latest_mlflow_version(flavor, **kwargs):
    """
    Save a model with overriding MLflow version from dev version to the latest released version.
    By default a model is saved with the dev version of MLflow, which is not available on PyPI.
    Usually we can be workaround this by adding --serve-wheel flag that starts local PyPI server,
    however, this doesn't work when installing dependencies inside Docker container. Hence, this
    function uses `extra_pip_requirements` to save the model with the latest released MLflow.
    """
    latest_mlflow_version = get_released_mlflow_version()
    if flavor == "langchain":
        kwargs["pip_requirements"] = [f"mlflow[gateway]=={latest_mlflow_version}", "langchain"]
    else:
        kwargs["extra_pip_requirements"] = [f"mlflow=={latest_mlflow_version}"]
    flavor_module = getattr(mlflow, flavor)
    flavor_module.save_model(**kwargs)
