import pytest

from mlflow import cli
from tests.integration.utils import invoke_cli_runner, update_temp_env
from tests.projects.utils import TEST_PROJECT_DIR, GIT_PROJECT_URI, SSH_PROJECT_URI
from tests.projects.utils import tracking_uri_mock  # pylint: disable=unused-import


@pytest.mark.large
def test_run_local(tracking_uri_mock):  # pylint: disable=unused-argument
    excitement_arg = 2
    name = "friend"
    invoke_cli_runner(cli.run, [TEST_PROJECT_DIR, "-e", "greeter", "-P",
                                "greeting=hi", "-P", "name=%s" % name,
                                "-P", "excitement=%s" % excitement_arg])


@pytest.mark.large
def test_run_git_https(tracking_uri_mock):  # pylint: disable=unused-argument
    # Invoke command twice to ensure we set Git state in an isolated manner (e.g. don't attempt to
    # create a git repo in the same directory twice, etc)
    assert GIT_PROJECT_URI.startswith("https")
    invoke_cli_runner(cli.run, [GIT_PROJECT_URI, "-P", "alpha=0.5"])
    invoke_cli_runner(cli.run, [GIT_PROJECT_URI, "-P", "alpha=0.5"])


@pytest.mark.large
def test_run_git_ssh(tracking_uri_mock):  # pylint: disable=unused-argument
    assert SSH_PROJECT_URI.startswith("git@")
    # Disable host-key checking so the test can run without prompting for approval
    with update_temp_env({"GIT_SSH_COMMAND": "ssh -o StrictHostKeyChecking=no"}):
        invoke_cli_runner(cli.run, [SSH_PROJECT_URI, "-P", "alpha=0.5"])
        invoke_cli_runner(cli.run, [SSH_PROJECT_URI, "-P", "alpha=0.5"])
