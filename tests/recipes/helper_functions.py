import os
import random
import shutil
import string
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from sklearn.datasets import load_diabetes, load_iris
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import mlflow
from mlflow.recipes.step import BaseStep
from mlflow.recipes.steps.split import _OUTPUT_TEST_FILE_NAME, _OUTPUT_VALIDATION_FILE_NAME

RECIPE_EXAMPLE_PATH_ENV_VAR_FOR_TESTS = "_RECIPE_EXAMPLE_PATH"
RECIPE_EXAMPLE_PATH_FROM_MLFLOW_ROOT = "examples/recipes/regression"


## Methods
def get_random_id(length=6):
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))


def setup_model_and_evaluate(tmp_recipe_exec_path: Path):
    split_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "split", "outputs")
    split_step_output_dir.mkdir(parents=True)
    X, y = load_diabetes(as_frame=True, return_X_y=True)
    validation_df = X.assign(y=y).sample(n=50, random_state=9)
    validation_df.to_parquet(split_step_output_dir.joinpath(_OUTPUT_VALIDATION_FILE_NAME))
    test_df = X.assign(y=y).sample(n=100, random_state=42)
    test_df.to_parquet(split_step_output_dir.joinpath(_OUTPUT_TEST_FILE_NAME))

    run_id, model = train_and_log_model()
    train_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "train", "outputs")
    train_step_output_dir.mkdir(parents=True)
    train_step_output_dir.joinpath("run_id").write_text(run_id)
    output_model_path = train_step_output_dir.joinpath("sk_model")
    if os.path.exists(output_model_path) and os.path.isdir(output_model_path):
        shutil.rmtree(output_model_path)
    mlflow.sklearn.save_model(model, output_model_path)

    evaluate_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)

    register_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "register", "outputs")
    register_step_output_dir.mkdir(parents=True)
    return evaluate_step_output_dir, register_step_output_dir


def train_and_log_model(is_dummy=False):
    mlflow.set_experiment("demo")
    with mlflow.start_run() as run:
        X, y = load_diabetes(as_frame=True, return_X_y=True)
        model = DummyRegressor(strategy="constant", constant=42) if is_dummy else LinearRegression()
        fitted_model = model.fit(X, y)
        mlflow.sklearn.log_model(fitted_model, artifact_path="train/model")
        return run.info.run_id, fitted_model


def train_and_log_classification_model(is_dummy=False):
    mlflow.set_experiment("demo")
    with mlflow.start_run() as run:
        X, y = load_iris(as_frame=True, return_X_y=True)
        if is_dummy:
            model = DummyClassifier(strategy="constant", constant=42)
        else:
            model = LogisticRegression()
        fitted_model = model.fit(X, y)
        mlflow.sklearn.log_model(fitted_model, artifact_path="train/model")
        return run.info.run_id, fitted_model


def train_log_and_register_model(model_name, is_dummy=False):
    run_id, _ = train_and_log_model(is_dummy)
    runs_uri = f"runs:/{run_id}/train/model"
    mv = mlflow.register_model(runs_uri, model_name)
    return f"models:/{mv.name}/{mv.version}"


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
    def from_recipe_config(cls, recipe_config, recipe_root):
        pass

    @property
    def name(self):
        pass

    def _validate_and_apply_step_config(self):
        pass

    def step_class(self):
        pass


def list_all_artifacts(
    tracking_uri: str, run_id: str, path: Optional[str] = None
) -> Generator[str, None, None]:
    artifacts = mlflow.tracking.MlflowClient(tracking_uri).list_artifacts(run_id, path)
    for artifact in artifacts:
        if artifact.is_dir:
            yield from list_all_artifacts(tracking_uri, run_id, artifact.path)
        else:
            yield artifact.path
