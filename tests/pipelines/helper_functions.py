import mlflow
import os
import pathlib
import random
import shutil
import string
import sys
from typing import Generator

from contextlib import contextmanager
from mlflow.pipelines.utils.execution import _MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR
from mlflow.pipelines.steps.split import _OUTPUT_TEST_FILE_NAME, _OUTPUT_VALIDATION_FILE_NAME
from mlflow.pipelines.step import BaseStep
from mlflow.utils.file_utils import TempDir
from pathlib import Path
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

import pytest

PIPELINE_EXAMPLE_PATH_ENV_VAR_FOR_TESTS = "_PIPELINE_EXAMPLE_PATH"
PIPELINE_EXAMPLE_PATH_FROM_MLFLOW_ROOT = "examples/pipelines/sklearn_regression"

## Methods
def get_random_id(length=6):
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))


def setup_model_and_evaluate(tmp_pipeline_exec_path: Path):
    split_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "split", "outputs")
    split_step_output_dir.mkdir(parents=True)
    X, y = load_diabetes(as_frame=True, return_X_y=True)
    validation_df = X.assign(y=y).sample(n=50, random_state=9)
    validation_df.to_parquet(split_step_output_dir.joinpath(_OUTPUT_VALIDATION_FILE_NAME))
    test_df = X.assign(y=y).sample(n=100, random_state=42)
    test_df.to_parquet(split_step_output_dir.joinpath(_OUTPUT_TEST_FILE_NAME))

    run_id, model = train_and_log_model()
    train_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "train", "outputs")
    train_step_output_dir.mkdir(parents=True)
    train_step_output_dir.joinpath("run_id").write_text(run_id)
    output_model_path = train_step_output_dir.joinpath("model")
    if os.path.exists(output_model_path) and os.path.isdir(output_model_path):
        shutil.rmtree(output_model_path)
    mlflow.sklearn.save_model(model, output_model_path)

    evaluate_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)

    register_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "register", "outputs")
    register_step_output_dir.mkdir(parents=True)
    return evaluate_step_output_dir, register_step_output_dir


def train_and_log_model(is_dummy=False):
    mlflow.set_experiment("demo")
    with mlflow.start_run() as run:
        X, y = load_diabetes(as_frame=True, return_X_y=True)
        if is_dummy:
            model = DummyRegressor(strategy="constant", constant=42)
        else:
            model = LinearRegression()
        fitted_model = model.fit(X, y)
        mlflow.sklearn.log_model(fitted_model, artifact_path="train/model")
        return run.info.run_id, fitted_model


def train_log_and_register_model(model_name, is_dummy=False):
    run_id, _ = train_and_log_model(is_dummy)
    runs_uri = "runs:/{}/train/model".format(run_id)
    model_version = mlflow.register_model(runs_uri, model_name)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True,
    )
    return "models:/{model_name}/Production".format(model_name=model_name)


## Fixtures
@pytest.fixture
def enter_pipeline_example_directory():
    pipeline_example_path = os.environ.get(PIPELINE_EXAMPLE_PATH_ENV_VAR_FOR_TESTS)
    if pipeline_example_path is None:
        mlflow_repo_root_directory = pathlib.Path(mlflow.__file__).parent.parent
        pipeline_example_path = mlflow_repo_root_directory / PIPELINE_EXAMPLE_PATH_FROM_MLFLOW_ROOT

    with chdir(pipeline_example_path):
        yield pipeline_example_path


@pytest.fixture
def enter_test_pipeline_directory(enter_pipeline_example_directory):
    pipeline_example_root_path = enter_pipeline_example_directory

    with TempDir(chdr=True) as tmp:
        test_pipeline_path = tmp.path("test_pipeline")
        shutil.copytree(pipeline_example_root_path, test_pipeline_path)
        os.chdir(test_pipeline_path)
        yield os.getcwd()


@pytest.fixture
def tmp_pipeline_exec_path(monkeypatch, tmp_path) -> Path:
    path = tmp_path.joinpath("pipeline_execution")
    path.mkdir(parents=True)
    monkeypatch.setenv(_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR, str(path))
    yield path
    shutil.rmtree(path)


@pytest.fixture
def tmp_pipeline_root_path(tmp_path) -> Path:
    path = tmp_path.joinpath("pipeline_root")
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path)


@pytest.fixture
def clear_custom_metrics_module_cache():
    key = "steps.custom_metrics"
    if key in sys.modules:
        del sys.modules[key]


@contextmanager
def chdir(directory_path):
    og_dir = os.getcwd()
    try:
        os.chdir(directory_path)
        yield
    finally:
        os.chdir(og_dir)


class BaseStepImplemented(BaseStep):
    def _run(self, output_directory):
        pass

    def _inspect(self, output_directory):
        pass

    def clean(self):
        pass

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        pass

    @property
    def name(self):
        pass


def list_all_artifacts(
    tracking_uri: str, run_id: str, path: str = None
) -> Generator[str, None, None]:
    artifacts = mlflow.tracking.MlflowClient(tracking_uri).list_artifacts(run_id, path)
    for artifact in artifacts:
        if artifact.is_dir:
            yield from list_all_artifacts(tracking_uri, run_id, artifact.path)
        else:
            yield artifact.path
