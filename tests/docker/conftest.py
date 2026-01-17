import subprocess

import pytest


@pytest.fixture(scope="session", autouse=True)
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
    subprocess.run(
        ["docker", "image", "rm", "mlflow-integration-test"],
        check=False,
    )
