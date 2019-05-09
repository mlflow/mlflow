import json
import hashlib
import mock
import os
import logging

from click.testing import CliRunner
import pytest

from mlflow import cli
from mlflow.utils import process
from tests.integration.utils import invoke_cli_runner
from tests.projects.utils import TEST_PROJECT_DIR, GIT_PROJECT_URI, SSH_PROJECT_URI, \
    TEST_NO_SPEC_PROJECT_DIR
from tests.projects.utils import tracking_uri_mock  # pylint: disable=unused-import

_logger = logging.getLogger(__name__)


@pytest.mark.large
def test_run_local_params(tracking_uri_mock):  # pylint: disable=unused-argument
    excitement_arg = 2
    name = "friend"
    invoke_cli_runner(cli.run, [TEST_PROJECT_DIR, "-e", "greeter", "-P",
                                "greeting=hi", "-P", "name=%s" % name,
                                "-P", "excitement=%s" % excitement_arg])


@pytest.mark.large
def test_run_local_conda_env(tracking_uri_mock):  # pylint: disable=unused-argument
    with open(os.path.join(TEST_PROJECT_DIR, "conda.yaml"), "r") as handle:
        conda_env_contents = handle.read()
    expected_env_name = "mlflow-%s" % hashlib.sha1(conda_env_contents.encode("utf-8")).hexdigest()
    try:
        process.exec_cmd(cmd=["conda", "env", "remove", "--name", expected_env_name])
    except process.ShellCommandException:
        _logger.error(
            "Unable to remove conda environment %s. The environment may not have been present, "
            "continuing with running the test.", expected_env_name)
    invoke_cli_runner(cli.run, [TEST_PROJECT_DIR, "-e", "check_conda_env", "-P",
                                "conda_env_name=%s" % expected_env_name])


@pytest.mark.large
def test_run_local_no_spec(tracking_uri_mock):  # pylint: disable=unused-argument
    # Run an example project that doesn't contain an MLproject file
    expected_env_name = "mlflow-%s" % hashlib.sha1("".encode("utf-8")).hexdigest()
    invoke_cli_runner(cli.run, [TEST_NO_SPEC_PROJECT_DIR, "-e", "check_conda_env.py", "-P",
                                "conda-env-name=%s" % expected_env_name])


@pytest.mark.large
def test_run_git_https(tracking_uri_mock):  # pylint: disable=unused-argument
    # Invoke command twice to ensure we set Git state in an isolated manner (e.g. don't attempt to
    # create a git repo in the same directory twice, etc)
    assert GIT_PROJECT_URI.startswith("https")
    invoke_cli_runner(cli.run, [GIT_PROJECT_URI, "--no-conda", "-P", "alpha=0.5"])
    invoke_cli_runner(cli.run, [GIT_PROJECT_URI, "--no-conda", "-P", "alpha=0.5"])


@pytest.mark.large
@pytest.mark.requires_ssh
def test_run_git_ssh(tracking_uri_mock):  # pylint: disable=unused-argument
    # Note: this test requires SSH authentication to GitHub, and so is disabled in Travis, where SSH
    # keys are unavailable. However it should be run locally whenever logic related to running
    # Git projects is modified.
    assert SSH_PROJECT_URI.startswith("git@")
    invoke_cli_runner(cli.run, [SSH_PROJECT_URI, "--no-conda", "-P", "alpha=0.5"])
    invoke_cli_runner(cli.run, [SSH_PROJECT_URI, "--no-conda", "-P", "alpha=0.5"])


def test_run_databricks_cluster_spec(tmpdir):
    cluster_spec = {
        "spark_version": "5.0.x-scala2.11",
        "num_workers": 2,
        "node_type_id": "i3.xlarge",
    }
    cluster_spec_path = str(tmpdir.join("cluster-spec.json"))
    with open(cluster_spec_path, "w") as handle:
        json.dump(cluster_spec, handle)

    with mock.patch("mlflow.projects._run") as run_mock:
        for cluster_spec_arg in [json.dumps(cluster_spec), cluster_spec_path]:
            invoke_cli_runner(
                cli.run, [TEST_PROJECT_DIR, "-b", "databricks", "--backend-config",
                          cluster_spec_arg, "-e", "greeter", "-P", "name=hi"],
                env={'MLFLOW_TRACKING_URI': 'databricks://profile'})
            assert run_mock.call_count == 1
            _, run_kwargs = run_mock.call_args_list[0]
            assert run_kwargs["backend_config"] == cluster_spec
            run_mock.reset_mock()
        res = CliRunner().invoke(
            cli.run, [TEST_PROJECT_DIR, "-m", "databricks", "--cluster-spec",
                      json.dumps(cluster_spec) + "JUNK", "-e", "greeter", "-P", "name=hi"],
            env={'MLFLOW_TRACKING_URI': 'databricks://profile'})
        assert res.exit_code != 0
