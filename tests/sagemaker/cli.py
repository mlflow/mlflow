import subprocess

import pytest


@pytest.mark.parametrize("env_manager", ["conda", "virtualenv"])
def test_build_docker(env_manager):
    subprocess.run(
        [
            "mlflow",
            "sagemaker",
            "build-and-push-container",
            "--no-push",
            "--mlflow-home",
            ".",
            "--env-manager",
            env_manager,
        ],
        check=True,
    )
