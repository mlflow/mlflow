
import os
import os.path
import shutil
from mlflow import cli
from mlflow.utils import process
from mlflow.utils.file_utils import path_to_local_file_uri
from tests.integration.utils import invoke_cli_runner
import pytest

from tests.projects.utils import tracking_uri_mock

EXAMPLES_DIR = 'examples'


@pytest.mark.large
@pytest.mark.parametrize("directory, params", [
    ('sklearn_logistic_regression', []),
    ('sklearn_elasticnet_wine', ['-P', 'alpha=0.5']),
    (os.path.join("sklearn_elasticnet_diabetes", "linux"), []),
    ('h2o', []),
    ('prophet', []),
])
def test_mlflow_run_example(tracking_uri_mock, directory, params):
    cli_run_list = [os.path.join(EXAMPLES_DIR, directory)] + params
    invoke_cli_runner(cli.run, cli_run_list)


@pytest.mark.large
@pytest.mark.parametrize("directory, command", [
    ('quickstart', ['python', 'mlflow_tracking.py']),
])
def test_command_example(tmpdir, directory, command):
    os.environ['MLFLOW_TRACKING_URI'] = path_to_local_file_uri(str(tmpdir.join("mlruns")))
    cwd_dir = os.path.join(EXAMPLES_DIR, directory)

    try:
        process.exec_cmd(command,
                         cwd=cwd_dir)
    finally:
        shutil.rmtree(str(tmpdir))
        del os.environ['MLFLOW_TRACKING_URI']
