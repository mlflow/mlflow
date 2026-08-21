import os
import shutil
import subprocess

import pytest


@pytest.fixture(scope="package", autouse=False)
def build_mlflow_image():
    subprocess.check_call([
        "docker",
        "build",
        "-t",
        "mlflow-integration-test",
        "-f",
        "docker/Dockerfile.full.dev",
        ".",
    ])
    yield
    # Clean up the image after all tests complete
    subprocess.check_call(["docker", "image", "rm", "-f", "mlflow-integration-test"])


@pytest.fixture(scope="module", autouse=True)
def _configure_testcontainers() -> None:
    """
    Check for podman binary and configure podman socket if available. If you are
    running podman on linux without the podman machine, you will need to manually
    open a socket by running ``podman system service --time=0`` prior to starting
    the tests.
    """

    podman = shutil.which("podman")
    if podman is None:
        yield
        return

    result = subprocess.run(
        [podman, "machine", "inspect", "--format", "{{.ConnectionInfo.PodmanSocket.Path}}"],
        capture_output=True,
        text=True,
        check=False,
    )

    podman_socket = None
    if result.returncode == 0:
        podman_socket = result.stdout.strip()

    XDG_RUNTIME_DIR = os.environ.get("XDG_RUNTIME_DIR")
    if podman_socket is None and XDG_RUNTIME_DIR is not None:
        podman_socket = f"{XDG_RUNTIME_DIR}/podman/podman.sock"

    assert podman_socket is not None
    assert podman_socket != ""

    ENV_DOCKER_HOST = "DOCKER_HOST"
    ENV_TESTCONTAINERS_RYUK_DISABLED = "TESTCONTAINERS_RYUK_DISABLED"

    docker_host = os.environ.get(ENV_DOCKER_HOST)
    ryuk_disabled = os.environ.get(ENV_TESTCONTAINERS_RYUK_DISABLED)

    if docker_host is None:
        os.environ[ENV_DOCKER_HOST] = f"unix://{podman_socket}"

    if ryuk_disabled is None:
        os.environ[ENV_TESTCONTAINERS_RYUK_DISABLED] = "true"

    yield

    if docker_host is None:
        _ = os.environ.pop(ENV_DOCKER_HOST, None)

    if ryuk_disabled is None:
        _ = os.environ.pop(ENV_TESTCONTAINERS_RYUK_DISABLED, None)
