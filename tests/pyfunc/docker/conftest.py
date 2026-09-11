import logging
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import docker
import pytest
import requests
from packaging.version import Version

import mlflow
from mlflow.models.docker_utils import UBUNTU_BASE_IMAGE

TEST_IMAGE_NAME = "test_image"
MLFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESOURCE_DIR = os.path.join(MLFLOW_ROOT, "tests", "resources", "dockerfile")

docker_client = docker.from_env()

_logger = logging.getLogger(__name__)

# GitHub Actions runners run in Azure, where `archive.ubuntu.com` is reachable but regularly
# throttles to a crawl mid-download, while the runner image's own apt is pointed at the
# in-datacenter `azure.archive.ubuntu.com`. A `docker build` starts from a stock base image and
# inherits none of that, so rewrite its apt sources to use the same mirror the host already uses.
_APT_MIRROR_STEP = r"""RUN sed -i \
    -e 's|http://archive.ubuntu.com|http://azure.archive.ubuntu.com|g' \
    -e 's|http://security.ubuntu.com|http://azure.archive.ubuntu.com|g' \
    /etc/apt/sources.list.d/ubuntu.sources
"""


def use_azure_apt_mirror(context_dir):
    """
    Rewrite the apt mirror in a generated build context's Dockerfile, in place.

    No-op off GitHub Actions and for non-Ubuntu base images. Call this after the context has
    been copied for Dockerfile comparison, so that the copy under assertion stays untouched.
    """
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return

    dockerfile = Path(context_dir) / "Dockerfile"
    lines = dockerfile.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(f"FROM {UBUNTU_BASE_IMAGE}"):
            lines.insert(i + 1, "\n" + _APT_MIRROR_STEP)
            dockerfile.write_text("".join(lines))
            return


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
        subprocess.check_call(["docker", "builder", "prune", "-a", "-f"])
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
    return str(max(versions))


def save_model_with_latest_mlflow_version(flavor, extra_pip_requirements=None, **kwargs):
    """
    Save a model with overriding MLflow version from dev version to the latest released version.
    By default a model is saved with the dev version of MLflow, which is not available on PyPI.
    Usually we can be workaround this by adding --serve-wheel flag that starts local PyPI server,
    however, this doesn't work when installing dependencies inside Docker container. Hence, this
    function uses `extra_pip_requirements` to save the model with the latest released MLflow.
    """
    latest_mlflow_version = get_released_mlflow_version()
    if flavor == "langchain":
        kwargs["pip_requirements"] = [
            f"mlflow[gateway]=={latest_mlflow_version}",
            "langchain<1.1.0",
        ]
    else:
        extra_pip_requirements = extra_pip_requirements or []
        extra_pip_requirements.append(f"mlflow=={latest_mlflow_version}")
        if flavor == "lightgbm":
            # Adding pyarrow < 18 to prevent pip installation resolution conflicts.
            extra_pip_requirements.append("pyarrow<18")
        kwargs["extra_pip_requirements"] = extra_pip_requirements
    flavor_module = getattr(mlflow, flavor)
    flavor_module.save_model(**kwargs)
