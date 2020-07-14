
import os
import os.path
import re
import shutil

from mlflow import cli
from mlflow.utils import process
from mlflow.utils.file_utils import path_to_local_file_uri
from tests.integration.utils import invoke_cli_runner
import pytest

EXAMPLES_DIR = 'examples'


def is_conda_yaml(path):
    return bool(re.search('conda.ya?ml$', path))


def find_conda_yaml(directory):
    conda_yaml = list(filter(is_conda_yaml, os.listdir(directory)))[0]
    return os.path.join(directory, conda_yaml)


def replace_mlflow_with_dev_version(yml_path, mlflow_dir):
    with open(yml_path, 'r') as f:
        old_src = f.read()
        new_src = re.sub(r"- mlflow.+\n", "- {}\n".format(mlflow_dir), old_src)

    with open(yml_path, 'w') as f:
        f.write(new_src)


@pytest.mark.large
@pytest.mark.parametrize("directory, params", [
    ('h2o', []),
    ('hyperparam', ['-e', 'train']),
    ('hyperparam', ['-e', 'random']),
    ('hyperparam', ['-e', 'gpyopt']),
    ('hyperparam', ['-e', 'hyperopt', '-P', 'epochs=1']),
    ('lightgbm', ['-P', 'learning_rate=0.1', '-P', 'colsample_bytree=0.8', '-P', 'subsample=0.9']),
    ('prophet', []),
    ('pytorch', ['-P', 'epochs=2']),
    ('sklearn_logistic_regression', []),
    ('sklearn_elasticnet_wine', ['-P', 'alpha=0.5']),
    (os.path.join('sklearn_elasticnet_diabetes', 'linux'), []),
    ('spacy', []),
    (os.path.join('tensorflow', 'tf1'), ['-P', 'steps=10']),
    ('xgboost', ['-P', 'learning_rate=0.3', '-P', 'colsample_bytree=0.8', '-P', 'subsample=0.9'])
])
def test_mlflow_run_example(directory, params, tmpdir):
    example_dir = os.path.join(EXAMPLES_DIR, directory)
    tmpdir = os.path.join(tmpdir.strpath, directory)

    shutil.copytree(example_dir, tmpdir)
    conda_yml_path = find_conda_yaml(tmpdir)
    replace_mlflow_with_dev_version(conda_yml_path, os.path.abspath('.'))

    cli_run_list = [tmpdir] + params
    invoke_cli_runner(cli.run, cli_run_list)


@pytest.mark.large
@pytest.mark.parametrize("directory, command", [
    ('docker', ['docker', 'build', '-t', 'mlflow-docker-example', '-f', 'Dockerfile', '.']),
    ('gluon', ['python', 'train.py']),
    ('keras', ['python', 'train.py']),
    ('lightgbm', ['python', 'train.py', '--learning-rate', '0.2', '--colsample-bytree', '0.8',
                  '--subsample', '0.9']),
    ('quickstart', ['python', 'mlflow_tracking.py']),
    ('remote_store', ['python', 'remote_server.py']),
    ('xgboost', ['python', 'train.py', '--learning-rate', '0.2', '--colsample-bytree', '0.8',
                 '--subsample', '0.9'])
])
def test_command_example(directory, command):
    cwd_dir = os.path.join(EXAMPLES_DIR, directory)
    process.exec_cmd(command,
                     cwd=cwd_dir)
