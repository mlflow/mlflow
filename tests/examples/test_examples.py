import re
import shutil
from pathlib import Path

import mlflow
from mlflow import cli
from mlflow.utils import process
from mlflow.utils.virtualenv import _get_mlflow_virtualenv_root
import pytest
from tests.integration.utils import invoke_cli_runner

EXAMPLES_DIR = "examples"


def get_free_disk_space_in_GiB():
    # https://stackoverflow.com/a/48929832/6943581
    return shutil.disk_usage("/")[-1] / (2**30)


def find_python_env_yaml(directory: Path) -> Path:
    return next(filter(lambda p: p.name == "python_env.yaml", Path(directory).iterdir()))


def replace_mlflow_with_dev_version(yml_path: Path) -> None:
    old_src = yml_path.read_text()
    mlflow_dir = Path(mlflow.__path__[0]).parent
    new_src = re.sub(r"- mlflow.*\n", f"- {mlflow_dir}\n", old_src)
    yml_path.write_text(new_src)


@pytest.fixture(scope="function", autouse=True)
def report_free_disk_space(capsys):
    yield

    with capsys.disabled():
        # pylint: disable-next=print-function
        print(" | Free disk space: {:.1f} GiB".format(get_free_disk_space_in_GiB()), end="")


@pytest.fixture(scope="function", autouse=True)
def clean_up_mlflow_virtual_environments():
    yield

    for path in Path(_get_mlflow_virtualenv_root()).iterdir():
        if path.is_dir():
            shutil.rmtree(path)


@pytest.mark.notrackingurimock
@pytest.mark.parametrize(
    ("directory", "params"),
    [
        ("h2o", []),
        ("hyperparam", ["-e", "train", "-P", "epochs=1"]),
        ("hyperparam", ["-e", "random", "-P", "epochs=1"]),
        ("hyperparam", ["-e", "hyperopt", "-P", "epochs=1"]),
        (
            "lightgbm/lightgbm_native",
            ["-P", "learning_rate=0.1", "-P", "colsample_bytree=0.8", "-P", "subsample=0.9"],
        ),
        ("lightgbm/lightgbm_sklearn", []),
        ("statsmodels", ["-P", "inverse_method=qr"]),
        ("pytorch", ["-P", "epochs=2"]),
        ("sklearn_logistic_regression", []),
        ("sklearn_elasticnet_wine", ["-P", "alpha=0.5"]),
        ("sklearn_elasticnet_diabetes/linux", []),
        ("spacy", []),
        (
            "xgboost/xgboost_native",
            ["-P", "learning_rate=0.3", "-P", "colsample_bytree=0.8", "-P", "subsample=0.9"],
        ),
        ("xgboost/xgboost_sklearn", []),
        ("fastai", ["-P", "lr=0.02", "-P", "epochs=3"]),
        ("pytorch/MNIST", ["-P", "max_epochs=1"]),
        (
            "pytorch/BertNewsClassification",
            ["-P", "max_epochs=1", "-P", "num_samples=100", "-P", "dataset=20newsgroups"],
        ),
        (
            "pytorch/AxHyperOptimizationPTL",
            ["-P", "max_epochs=10", "-P", "total_trials=1"],
        ),
        (
            "pytorch/IterativePruning",
            ["-P", "max_epochs=1", "-P", "total_trials=1"],
        ),
        ("pytorch/CaptumExample", ["-P", "max_epochs=50"]),
        ("supply_chain_security", []),
    ],
)
def test_mlflow_run_example(directory, params, tmp_path):
    mlflow.set_tracking_uri(tmp_path.joinpath("mlruns").as_uri())
    example_dir = Path(EXAMPLES_DIR, directory)
    tmp_example_dir = tmp_path.joinpath(example_dir)
    shutil.copytree(example_dir, tmp_example_dir)
    python_env_path = find_python_env_yaml(tmp_example_dir)
    replace_mlflow_with_dev_version(python_env_path)
    cli_run_list = [tmp_example_dir] + params
    invoke_cli_runner(cli.run, list(map(str, cli_run_list)))


@pytest.mark.notrackingurimock
@pytest.mark.parametrize(
    ("directory", "command"),
    [
        ("docker", ["docker", "build", "-t", "mlflow-docker-example", "-f", "Dockerfile", "."]),
        ("gluon", ["python", "train.py"]),
        ("keras", ["python", "train.py"]),
        (
            "lightgbm/lightgbm_native",
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
        ("lightgbm/lightgbm_sklearn", ["python", "train.py"]),
        ("statsmodels", ["python", "train.py", "--inverse-method", "qr"]),
        ("quickstart", ["python", "mlflow_tracking.py"]),
        ("remote_store", ["python", "remote_server.py"]),
        (
            "xgboost/xgboost_native",
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
        ("xgboost/xgboost_sklearn", ["python", "train.py"]),
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
        ("sktime", ["python", "train.py"]),
    ],
)
def test_command_example(directory, command):
    cwd_dir = Path(EXAMPLES_DIR, directory)
    import os

    assert os.environ.get("MLFLOW_HOME") is not None
    process._exec_cmd(command, cwd=cwd_dir, env=os.environ)
