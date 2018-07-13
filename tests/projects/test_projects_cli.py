import pytest

import mlflow
from mlflow import cli
from mlflow.utils.file_utils import TempDir
from tests.integration.utils import invoke_cli_runner, update_temp_env
from tests.projects.utils import TEST_PROJECT_DIR, GIT_PROJECT_URI


def _assert_succeeded(run_output):
    assert "=== Run succeeded ===" in run_output


@pytest.mark.large
def test_run_local():
    with TempDir() as tmp:
        with update_temp_env({mlflow.tracking._TRACKING_URI_ENV_VAR: tmp.path()}):
            excitement_arg = 2
            res = invoke_cli_runner(cli.run, [TEST_PROJECT_DIR, "-e", "greeter", "-P",
                                              "greeting=hi", "--no-conda", "-P", "name=friend",
                                              "-P", "excitement=%s" % excitement_arg])
            _assert_succeeded(res.output)


@pytest.mark.large
def test_run_git():
    with TempDir() as tmp:
        with update_temp_env({mlflow.tracking._TRACKING_URI_ENV_VAR: tmp.path()}):
            res = invoke_cli_runner(cli.run, [GIT_PROJECT_URI, "--no-conda", "-P", "alpha=0.5"])
            assert "python train.py 0.5 0.1" in res.output
            _assert_succeeded(res.output)

GIT_SUBDIR_URI = "https://github.com/juntai-zheng/mlflow-git-features.git"

def test_git_subdirectories():
    with TempDir() as tmp:
        with update_temp_env({mlflow.tracking._TRACKING_URI_ENV_VAR: tmp.path()}):
            res = invoke_cli_runner(cli.run, [GIT_SUBDIR_URI + "#example/", "-e", "main", "--no-conda", "-P", "alpha=0.5"])
            assert "python train.py 0.5 0.1" in res.output
            _assert_succeeded(res.output)


def test_missing_subdir():
    with TempDir() as tmp:
        with update_temp_env({mlflow.tracking._TRACKING_URI_ENV_VAR: tmp.path()}):
            res = None
            try:
                res = invoke_cli_runner(cli.run, [GIT_PROJECT_URI + "#fake", "--no-conda", "-P", "alpha=0.5"])
            except AssertionError as e:
                print(e)
                assert res == None

def test_restrict_periods():
    with TempDir() as tmp:
        with update_temp_env({mlflow.tracking._TRACKING_URI_ENV_VAR: tmp.path()}):
            res = None
            try:
                res = invoke_cli_runner(cli.run, [GIT_PROJECT_URI + "#..", "--no-conda", "-P", "alpha=0.5"])
            except AssertionError as e:
                assert e.args[0] == "Got non-zero exit code -1. Output is: "
                assert res == None
                