import pytest
from click.testing import CliRunner

from mlflow.sagemaker.cli import build_and_push_container


@pytest.mark.parametrize("env_manager", ["conda", "virtualenv"])
def test_build_and_push_container(env_manager):
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
