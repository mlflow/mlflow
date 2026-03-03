import subprocess

import pytest


@pytest.fixture(scope="package", autouse=True)
def build_mlflow_image():
    subprocess.check_call(
        [
            "docker",
            "build",
            "-t",
            "mlflow-integration-test",
            "-f",
            "docker/Dockerfile.full.dev",
            ".",
        ]
    )
    yield
    # Clean up the image after all tests complete
    subprocess.check_call(["docker", "image", "rm", "-f", "mlflow-integration-test"])
