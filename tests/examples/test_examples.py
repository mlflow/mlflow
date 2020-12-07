import os
import os.path
import re
import shutil

import mlflow
from mlflow import cli
from mlflow.utils import process
from tests.integration.utils import invoke_cli_runner
import pytest
import json
import hashlib

EXAMPLES_DIR = "examples"


def hash_conda_env(conda_env_path):
    # use the same hashing logic as `_get_conda_env_name` in mlflow/utils/conda.py
    return hashlib.sha1(open(conda_env_path).read().encode("utf-8")).hexdigest()


def get_conda_envs():
    stdout = process.exec_cmd(["conda", "env", "list", "--json"])[1]
    return [os.path.basename(env) for env in json.loads(stdout)["envs"]]


def is_mlflow_conda_env(env_name):
    return re.search(r"^mlflow-\w{40}$", env_name) is not None


def remove_conda_env(env_name):
    process.exec_cmd(["conda", "remove", "--name", env_name, "--yes", "--all"])


def get_free_disk_space():
    # https://stackoverflow.com/a/48929832/6943581
    return shutil.disk_usage("/")[-1] / (2 ** 30)


def is_conda_yaml(path):
    return bool(re.search("conda.ya?ml$", path))


def find_conda_yaml(directory):
    conda_yaml = list(filter(is_conda_yaml, os.listdir(directory)))[0]
    return os.path.join(directory, conda_yaml)


def replace_mlflow_with_dev_version(yml_path):
    with open(yml_path, "r") as f:
        old_src = f.read()
        mlflow_dir = os.path.dirname(mlflow.__path__[0])
        new_src = re.sub(r"- mlflow.*\n", "- {}\n".format(mlflow_dir), old_src)

    with open(yml_path, "w") as f:
        f.write(new_src)


@pytest.fixture(scope="function", autouse=True)
def report_free_disk_space(capsys):
    yield

    with capsys.disabled():
        print(" | Free disk space: {:.1f} GiB".format(get_free_disk_space()), end="")


@pytest.mark.large
@pytest.mark.parametrize(
    "directory, params",
    [
        ("h2o", []),
        ("hyperparam", ["-e", "train", "-P", "epochs=1"]),
        ("hyperparam", ["-e", "random", "-P", "epochs=1"]),
        ("hyperparam", ["-e", "gpyopt", "-P", "epochs=1"]),
        ("hyperparam", ["-e", "hyperopt", "-P", "epochs=1"]),
        (
            "lightgbm",
            ["-P", "learning_rate=0.1", "-P", "colsample_bytree=0.8", "-P", "subsample=0.9"],
        ),
        ("statsmodels", ["-P", "inverse_method=qr"]),
        ("prophet", []),
        ("pytorch", ["-P", "epochs=2"]),
        ("sklearn_logistic_regression", []),
        ("sklearn_elasticnet_wine", ["-P", "alpha=0.5"]),
        (os.path.join("sklearn_elasticnet_diabetes", "linux"), []),
        ("spacy", []),
        (os.path.join("tensorflow", "tf1"), ["-P", "steps=10"]),
        (
            "xgboost",
            ["-P", "learning_rate=0.3", "-P", "colsample_bytree=0.8", "-P", "subsample=0.9"],
        ),
        ("fastai", ["-P", "lr=0.02", "-P", "epochs=3"]),
        (os.path.join("pytorch", "MNIST/example1"), ["-P", "max_epochs=1"]),
        (os.path.join("pytorch", "MNIST/example2"), ["-P", "max_epochs=1"]),
        (
            os.path.join("pytorch", "BertNewsClassification"),
            ["-P", "max_epochs=1", "-P", "num_samples=100"],
        ),
    ],
)
def test_mlflow_run_example(directory, params, tmpdir):
    example_dir = os.path.join(EXAMPLES_DIR, directory)
    tmp_example_dir = os.path.join(tmpdir.strpath, directory)
    shutil.copytree(example_dir, tmp_example_dir)
    conda_yml_path = find_conda_yaml(tmp_example_dir)
    replace_mlflow_with_dev_version(conda_yml_path)

    # remove old conda environments to free disk space
    envs = list(filter(is_mlflow_conda_env, get_conda_envs()))
    current_env_name = "mlflow-" + hash_conda_env(conda_yml_path)
    envs_to_remove = list(filter(lambda e: e != current_env_name, envs))
    for env in envs_to_remove:
        remove_conda_env(env)

    cli_run_list = [tmp_example_dir] + params
    invoke_cli_runner(cli.run, cli_run_list)


@pytest.mark.large
@pytest.mark.parametrize(
    "directory, command",
    [
        ("docker", ["docker", "build", "-t", "mlflow-docker-example", "-f", "Dockerfile", "."]),
        ("gluon", ["python", "train.py"]),
        ("keras", ["python", "train.py"]),
        (
            "lightgbm",
            [
                "python",
                "train.py",
                "--learning-rate",
                "0.2",
                "--colsample-bytree",
                "0.8",
                "--subsample",
                "0.9",
            ],
        ),
        ("statsmodels", ["python", "train.py", "--inverse-method", "qr"]),
        ("quickstart", ["python", "mlflow_tracking.py"]),
        ("remote_store", ["python", "remote_server.py"]),
        (
            "xgboost",
            [
                "python",
                "train.py",
                "--learning-rate",
                "0.2",
                "--colsample-bytree",
                "0.8",
                "--subsample",
                "0.9",
            ],
        ),
        ("sklearn_autolog", ["python", "linear_regression.py"]),
        ("sklearn_autolog", ["python", "pipeline.py"]),
        ("sklearn_autolog", ["python", "grid_search_cv.py"]),
        ("shap", ["python", "regression.py"]),
        ("shap", ["python", "binary_classification.py"]),
        ("shap", ["python", "multiclass_classification.py"]),
    ],
)
def test_command_example(directory, command):
    cwd_dir = os.path.join(EXAMPLES_DIR, directory)
    process.exec_cmd(command, cwd=cwd_dir)
