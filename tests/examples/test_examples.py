import os
import re
import shutil
import sys
from pathlib import Path

import pytest

import mlflow
from mlflow import cli
from mlflow.utils import process
from mlflow.utils.virtualenv import _get_mlflow_virtualenv_root

from tests.helper_functions import clear_hub_cache, flaky
from tests.integration.utils import invoke_cli_runner

EXAMPLES_DIR = "examples"


def find_python_env_yaml(directory: Path) -> Path:
    return next(filter(lambda p: p.name == "python_env.yaml", Path(directory).iterdir()))


def replace_mlflow_with_dev_version(yml_path: Path) -> None:
    old_src = yml_path.read_text()
    mlflow_dir = Path(mlflow.__path__[0]).parent
    new_src = re.sub(r"- mlflow.*\n", f"- {mlflow_dir}\n", old_src)
    yml_path.write_text(new_src)


@pytest.fixture(autouse=True)
def clean_up_mlflow_virtual_environments():
    yield

    for path in Path(_get_mlflow_virtualenv_root()).iterdir():
        if path.is_dir():
            shutil.rmtree(path)


@pytest.mark.notrackingurimock
@flaky()
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
        ("tensorflow", []),
        ("sktime", []),
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
        ("keras", [sys.executable, "train.py"]),
        (
            "lightgbm/lightgbm_native",
            [
                sys.executable,
                "train.py",
                "--learning-rate",
                "0.2",
                "--colsample-bytree",
                "0.8",
                "--subsample",
                "0.9",
            ],
        ),
        ("lightgbm/lightgbm_sklearn", [sys.executable, "train.py"]),
        ("statsmodels", [sys.executable, "train.py", "--inverse-method", "qr"]),
        ("quickstart", [sys.executable, "mlflow_tracking.py"]),
        ("remote_store", [sys.executable, "remote_server.py"]),
        (
            "xgboost/xgboost_native",
            [
                sys.executable,
                "train.py",
                "--learning-rate",
                "0.2",
                "--colsample-bytree",
                "0.8",
                "--subsample",
                "0.9",
            ],
        ),
        ("xgboost/xgboost_sklearn", [sys.executable, "train.py"]),
        ("catboost", [sys.executable, "train.py"]),
        ("prophet", [sys.executable, "train.py"]),
        ("sklearn_autolog", [sys.executable, "linear_regression.py"]),
        ("sklearn_autolog", [sys.executable, "pipeline.py"]),
        ("sklearn_autolog", [sys.executable, "grid_search_cv.py"]),
        ("pyspark_ml_autologging", [sys.executable, "logistic_regression.py"]),
        ("pyspark_ml_autologging", [sys.executable, "one_vs_rest.py"]),
        ("pyspark_ml_autologging", [sys.executable, "pipeline.py"]),
        ("shap", [sys.executable, "regression.py"]),
        ("shap", [sys.executable, "binary_classification.py"]),
        ("shap", [sys.executable, "multiclass_classification.py"]),
        ("shap", [sys.executable, "explainer_logging.py"]),
        ("ray_serve", [sys.executable, "train_model.py"]),
        ("pip_requirements", [sys.executable, "pip_requirements.py"]),
        ("fastai", [sys.executable, "train.py", "--lr", "0.02", "--epochs", "3"]),
        ("pmdarima", [sys.executable, "train.py"]),
        ("evaluation", [sys.executable, "evaluate_on_binary_classifier.py"]),
        ("evaluation", [sys.executable, "evaluate_on_multiclass_classifier.py"]),
        ("evaluation", [sys.executable, "evaluate_on_regressor.py"]),
        ("evaluation", [sys.executable, "evaluate_with_custom_metrics.py"]),
        ("evaluation", [sys.executable, "evaluate_with_custom_metrics_comprehensive.py"]),
        ("evaluation", [sys.executable, "evaluate_with_model_validation.py"]),
        ("diviner", [sys.executable, "train.py"]),
        ("spark_udf", [sys.executable, "spark_udf_datetime.py"]),
        ("pyfunc", [sys.executable, "train.py"]),
        ("tensorflow", [sys.executable, "train.py"]),
        ("transformers", [sys.executable, "conversational.py"]),
        ("transformers", [sys.executable, "load_components.py"]),
        ("transformers", [sys.executable, "simple.py"]),
        ("transformers", [sys.executable, "sentence_transformer.py"]),
        ("transformers", [sys.executable, "whisper.py"]),
        ("sentence_transformers", [sys.executable, "simple.py"]),
        ("tracing", [sys.executable, "fluent.py"]),
        ("tracing", [sys.executable, "client.py"]),
        ("tracing", [sys.executable, "multithreading.py"]),
    ],
)
def test_command_example(directory, command):
    cwd_dir = Path(EXAMPLES_DIR, directory)
    assert os.environ.get("MLFLOW_HOME") is not None
    if directory == "transformers":
        # NB: Clearing the huggingface_hub cache is to lower the disk storage pressure for CI
        clear_hub_cache()
    process._exec_cmd(command, cwd=cwd_dir, env=os.environ)
