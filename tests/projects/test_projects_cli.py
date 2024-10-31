import hashlib
import json
import logging
import os
import shutil
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow import MlflowClient, cli
from mlflow.utils import process

from tests.integration.utils import invoke_cli_runner
from tests.projects.utils import (
    GIT_PROJECT_URI,
    SSH_PROJECT_URI,
    TEST_DOCKER_PROJECT_DIR,
    TEST_PROJECT_DIR,
    docker_example_base_image,  # noqa: F401
)

_logger = logging.getLogger(__name__)

skip_if_skinny = pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="MLflow skinny does not have dependencies to run this test",
)


@pytest.mark.parametrize("name", ["friend", "friend=you", "='friend'"])
def test_run_local_params(name):
    excitement_arg = 2
    invoke_cli_runner(
        cli.run,
        [
            TEST_PROJECT_DIR,
            "-e",
            "greeter",
            "-P",
            "greeting=hi",
            "-P",
            f"name={name}",
            "-P",
            f"excitement={excitement_arg}",
        ],
    )


@skip_if_skinny
def test_run_local_with_docker_args(docker_example_base_image):
    # Verify that Docker project execution is successful when Docker flag and string
    # commandline arguments are supplied (`tty` and `name`, respectively)
    invoke_cli_runner(cli.run, [TEST_DOCKER_PROJECT_DIR, "-A", "tty", "-A", "name=mycontainer"])


@pytest.mark.parametrize("experiment_name", [b"test-experiment".decode("utf-8"), "test-experiment"])
def test_run_local_experiment_specification(experiment_name):
    invoke_cli_runner(
        cli.run,
        [
            TEST_PROJECT_DIR,
            "-e",
            "greeter",
            "-P",
            "name=test",
            "--experiment-name",
            experiment_name,
        ],
    )

    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    invoke_cli_runner(
        cli.run,
        [TEST_PROJECT_DIR, "-e", "greeter", "-P", "name=test", "--experiment-id", experiment_id],
    )


@pytest.fixture(scope="module", autouse=True)
def clean_mlruns_dir():
    yield
    dir_path = os.path.join(TEST_PROJECT_DIR, "mlruns")
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


@skip_if_skinny
def test_run_local_conda_env():
    with open(os.path.join(TEST_PROJECT_DIR, "conda.yaml")) as handle:
        conda_env_contents = handle.read()
    expected_env_name = "mlflow-{}".format(
        hashlib.sha1(conda_env_contents.encode("utf-8"), usedforsecurity=False).hexdigest()
    )
    try:
        process._exec_cmd(cmd=["conda", "env", "remove", "--name", expected_env_name])
    except process.ShellCommandException:
        _logger.error(
            "Unable to remove conda environment %s. The environment may not have been present, "
            "continuing with running the test.",
            expected_env_name,
        )
    invoke_cli_runner(
        cli.run,
        [TEST_PROJECT_DIR, "-e", "check_conda_env", "-P", f"conda_env_name={expected_env_name}"],
    )


@skip_if_skinny
def test_run_git_https():
    # Invoke command twice to ensure we set Git state in an isolated manner (e.g. don't attempt to
    # create a git repo in the same directory twice, etc)
    assert GIT_PROJECT_URI.startswith("https")
    invoke_cli_runner(cli.run, [GIT_PROJECT_URI, "--env-manager", "local", "-P", "alpha=0.5"])
    invoke_cli_runner(cli.run, [GIT_PROJECT_URI, "--env-manager", "local", "-P", "alpha=0.5"])


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="SSH keys are unavailable in GitHub Actions"
)
def test_run_git_ssh():
    # Note: this test requires SSH authentication to GitHub, and so is disabled in GitHub Actions,
    # where SSH keys are unavailable. However it should be run locally whenever logic related to
    # running Git projects is modified.
    assert SSH_PROJECT_URI.startswith("git@")
    invoke_cli_runner(cli.run, [SSH_PROJECT_URI, "--env-manager", "local", "-P", "alpha=0.5"])
    invoke_cli_runner(cli.run, [SSH_PROJECT_URI, "--env-manager", "local", "-P", "alpha=0.5"])


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="SSH keys are unavailable in GitHub Actions"
)
def test_run_git_ssh_from_release_version():
    # Note: this test requires SSH authentication to GitHub, and so is disabled in GitHub Actions,
    # where SSH keys are unavailable. However it should be run locally whenever logic related to
    # running Git projects is modified.
    assert SSH_PROJECT_URI.startswith("git@")
    invoke_cli_runner(
        cli.run, [SSH_PROJECT_URI, "--no-conda", "-P", "alpha=0.5", "-v", "version_testing"]
    )
    invoke_cli_runner(
        cli.run, [SSH_PROJECT_URI, "--no-conda", "-P", "alpha=0.5", "-v", "version_testing"]
    )


@pytest.mark.notrackingurimock
def test_run_databricks_cluster_spec(tmp_path):
    cluster_spec = {
        "spark_version": "5.0.x-scala2.11",
        "num_workers": 2,
        "node_type_id": "i3.xlarge",
    }
    cluster_spec_path = tmp_path.joinpath("cluster-spec.json")
    with open(cluster_spec_path, "w") as handle:
        json.dump(cluster_spec, handle)

    with mock.patch("mlflow.projects._run") as run_mock:
        for cluster_spec_arg in [json.dumps(cluster_spec), cluster_spec_path]:
            invoke_cli_runner(
                cli.run,
                [
                    TEST_PROJECT_DIR,
                    "-b",
                    "databricks",
                    "--backend-config",
                    cluster_spec_arg,
                    "-e",
                    "greeter",
                    "-P",
                    "name=hi",
                ],
                env={"MLFLOW_TRACKING_URI": "databricks://profile"},
            )
            assert run_mock.call_count == 1
            _, run_kwargs = run_mock.call_args_list[0]
            assert run_kwargs["backend_config"] == cluster_spec
            run_mock.reset_mock()
        res = CliRunner().invoke(
            cli.run,
            [
                TEST_PROJECT_DIR,
                "-m",
                "databricks",
                "--cluster-spec",
                json.dumps(cluster_spec) + "JUNK",
                "-e",
                "greeter",
                "-P",
                "name=hi",
            ],
            env={"MLFLOW_TRACKING_URI": "databricks://profile"},
        )
        assert res.exit_code != 0


def test_mlflow_run():
    with mock.patch("mlflow.cli.projects") as mock_projects:
        result = CliRunner().invoke(cli.run)
        mock_projects.run.assert_not_called()
        assert "Missing argument 'URI'" in result.output

    with mock.patch("mlflow.cli.projects") as mock_projects:
        CliRunner().invoke(cli.run, ["project_uri"])
        mock_projects.run.assert_called_once()

    with mock.patch("mlflow.cli.projects") as mock_projects:
        CliRunner().invoke(cli.run, ["--experiment-id", "5", "project_uri"])
        mock_projects.run.assert_called_once()

    with mock.patch("mlflow.cli.projects") as mock_projects:
        CliRunner().invoke(cli.run, ["--experiment-name", "random name", "project_uri"])
        mock_projects.run.assert_called_once()

    with mock.patch("mlflow.cli.projects") as mock_projects:
        result = CliRunner().invoke(
            cli.run, ["--experiment-id", "51", "--experiment-name", "name blah", "uri"]
        )
        mock_projects.run.assert_not_called()
        assert "Specify only one of 'experiment-name' or 'experiment-id' options." in result.output
