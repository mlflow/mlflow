from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

def _ensure_databricks_stub():
    if "databricks" not in sys.modules:
        databricks_module = types.ModuleType("databricks")
        databricks_module.__spec__ = importlib.machinery.ModuleSpec(
            "databricks", loader=None
        )
        sys.modules["databricks"] = databricks_module
    if "databricks.agents" not in sys.modules:
        agents_module = types.ModuleType("databricks.agents")
        agents_module.__spec__ = importlib.machinery.ModuleSpec(
            "databricks.agents", loader=None
        )
        sys.modules["databricks.agents"] = agents_module
        sys.modules["databricks"].agents = agents_module


_ensure_databricks_stub()

import mlflow
from mlflow.models.evaluation.base import evaluate


@pytest.fixture(scope="module")
def logged_linear_regressor():
    data = load_diabetes()
    features = pd.DataFrame(data.data, columns=data.feature_names)
    targets = pd.Series(data.target, name="target")

    model = LinearRegression().fit(features, targets)
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(model, artifact_path="model")
    return model_info.model_uri, features.assign(target=targets)


def test_shap_summary_plot_uses_local_rng(logged_linear_regressor):
    try:
        shap = importlib.import_module("shap")
    except OSError as exc:
        pytest.skip(f"shap is not available in this environment: {exc}")

    if Version(shap.__version__) < Version("0.47.0"):
        pytest.skip("shap>=0.47.0 is required for RNG support in summary plots")

    model_uri, eval_df = logged_linear_regressor
    warning_pattern = r"The NumPy global RNG was seeded"
    rng_state = np.random.get_state()
    np.random.seed(7)

    def _run_eval(config):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=warning_pattern, category=FutureWarning)
            with mlflow.start_run():
                evaluate(
                    model_uri,
                    eval_df.copy(),
                    model_type="regressor",
                    targets="target",
                    evaluators="default",
                    evaluator_config=config,
                )

    try:
        _run_eval({"shap_plot_seed": 123})
        _run_eval({})
    finally:
        np.random.set_state(rng_state)
