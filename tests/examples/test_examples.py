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
    stdout = process._exec_cmd(["conda", "env", "list", "--json"]).stdout
    return [os.path.basename(env) for env in json.loads(stdout)["envs"]]


def is_mlflow_conda_env(env_name):
    return re.search(r"^mlflow-\w{40}$", env_name) is not None


def remove_conda_env(env_name):
    process._exec_cmd(["conda", "remove", "--name", env_name, "--yes", "--all"])


def get_free_disk_space():
    # https://stackoverflow.com/a/48929832/6943581
    return shutil.disk_usage("/")[-1] / (2**30)


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
def clean_envs_and_cache():
    yield

    if get_free_disk_space() < 7.0:  # unit: GiB
        process._exec_cmd(["./dev/remove-conda-envs.sh"])


@pytest.fixture(scope="function", autouse=True)
def report_free_disk_space(capsys):
    yield

    with capsys.disabled():
        # pylint: disable-next=print-function
        print(" | Free disk space: {:.1f} GiB".format(get_free_disk_space()), end="")


@pytest.mark.notrackingurimock
@pytest.mark.parametrize(
    "directory, params",
    [
        ("h2o", []),
        ("hyperparam", ["-e", "train", "-P", "epochs=1"]),
        ("hyperparam", ["-e", "random", "-P", "epochs=1"]),
        ("hyperparam", ["-e", "hyperopt", "-P", "epochs=1"]),
        (
            os.path.join("lightgbm", "lightgbm_native"),
            ["-P", "learning_rate=0.1", "-P", "colsample_bytree=0.8", "-P", "subsample=0.9"],
        ),
        (os.path.join("lightgbm", "lightgbm_sklearn"), []),
        ("statsmodels", ["-P", "inverse_method=qr"]),
        ("pytorch", ["-P", "epochs=2"]),
        ("sklearn_logistic_regression", []),
        ("sklearn_elasticnet_wine", ["-P", "alpha=0.5"]),
        (os.path.join("sklearn_elasticnet_diabetes", "linux"), []),
        ("spacy", []),
        (
            os.path.join("xgboost", "xgboost_native"),
            ["-P", "learning_rate=0.3", "-P", "colsample_bytree=0.8", "-P", "subsample=0.9"],
        ),
        (os.path.join("xgboost", "xgboost_sklearn"), []),
        ("fastai", ["-P", "lr=0.02", "-P", "epochs=3"]),
        (os.path.join("pytorch", "MNIST"), ["-P", "max_epochs=1"]),
        (
            os.path.join("pytorch", "BertNewsClassification"),
            ["-P", "max_epochs=1", "-P", "num_samples=100", "-P", "dataset=20newsgroups"],
        ),
        (
            os.path.join("pytorch", "AxHyperOptimizationPTL"),
            ["-P", "max_epochs=10", "-P", "total_trials=1"],
        ),
        (
            os.path.join("pytorch", "IterativePruning"),
            ["-P", "max_epochs=1", "-P", "total_trials=1"],
        ),
        (os.path.join("pytorch", "CaptumExample"), ["-P", "max_epochs=50"]),
        ("supply_chain_security", []),
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


@pytest.mark.notrackingurimock
@pytest.mark.parametrize(
    "directory, command",
    [
        ("docker", ["docker", "build", "-t", "mlflow-docker-example", "-f", "Dockerfile", "."]),
        ("gluon", ["python", "train.py"]),
        ("keras", ["python", "train.py"]),
        (
            os.path.join("lightgbm", "lightgbm_native"),
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
        (os.path.join("lightgbm", "lightgbm_sklearn"), ["python", "train.py"]),
        ("statsmodels", ["python", "train.py", "--inverse-method", "qr"]),
        ("quickstart", ["python", "mlflow_tracking.py"]),
        ("remote_store", ["python", "remote_server.py"]),
        (
            os.path.join("xgboost", "xgboost_native"),
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
        (os.path.join("xgboost", "xgboost_sklearn"), ["python", "train.py"]),
        ("catboost", ["python", "train.py"]),
        ("prophet", ["python", "train.py"]),
        ("sklearn_autolog", ["python", "linear_regression.py"]),
        ("sklearn_autolog", ["python", "pipeline.py"]),
        ("sklearn_autolog", ["python", "grid_search_cv.py"]),
        ("pyspark_ml_autologging", ["python", "logistic_regression.py"]),
        ("pyspark_ml_autologging", ["python", "one_vs_rest.py"]),
        ("pyspark_ml_autologging", ["python", "pipeline.py"]),
        ("shap", ["python", "regression.py"]),
        ("shap", ["python", "binary_classification.py"]),
        ("shap", ["python", "multiclass_classification.py"]),
        ("shap", ["python", "explainer_logging.py"]),
        ("ray_serve", ["python", "train_model.py"]),
        ("pip_requirements", ["python", "pip_requirements.py"]),
        ("fastai", ["python", "train.py", "--lr", "0.02", "--epochs", "3"]),
        ("pmdarima", ["python", "train.py"]),
        ("evaluation", ["python", "evaluate_on_binary_classifier.py"]),
        ("evaluation", ["python", "evaluate_on_multiclass_classifier.py"]),
        ("evaluation", ["python", "evaluate_on_regressor.py"]),
        ("evaluation", ["python", "evaluate_with_custom_metrics.py"]),
        ("evaluation", ["python", "evaluate_with_custom_metrics_comprehensive.py"]),
        ("evaluation", ["python", "evaluate_with_model_validation.py"]),
        ("diviner", ["python", "train.py"]),
        ("spark_udf", ["python", "spark_udf_datetime.py"]),
        ("pyfunc", ["python", "train.py"]),
    ],
)
def test_command_example(directory, command):
    cwd_dir = os.path.join(EXAMPLES_DIR, directory)
    process._exec_cmd(command, cwd=cwd_dir)
