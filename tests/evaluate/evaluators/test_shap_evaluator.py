from unittest import mock

import numpy as np
import pandas as pd

import mlflow
from mlflow.models.evaluation.evaluators.shap import _compute_df_mode_or_mean

from tests.evaluate.test_evaluation import (
    get_pipeline_model_dataset,
    get_run_data,
)


def _evaluate_explainer_with_exceptions(model_uri, dataset):
    with mlflow.start_run():
        mlflow.evaluate(
            model_uri,
            dataset._constructor_args["data"],
            model_type="classifier",
            targets=dataset._constructor_args["targets"],
            evaluators="shap",
            evaluator_config={
                "ignore_exceptions": False,
            },
        )


def test_default_explainer_pandas_df_str_cols(
    multiclass_logistic_regressor_model_uri, iris_pandas_df_dataset
):
    _evaluate_explainer_with_exceptions(
        multiclass_logistic_regressor_model_uri, iris_pandas_df_dataset
    )


def test_default_explainer_pandas_df_num_cols(
    multiclass_logistic_regressor_model_uri, iris_pandas_df_num_cols_dataset
):
    _evaluate_explainer_with_exceptions(
        multiclass_logistic_regressor_model_uri, iris_pandas_df_num_cols_dataset
    )


def test_pipeline_model_kernel_explainer_on_categorical_features(pipeline_model_uri):
    from mlflow.models.evaluation._shap_patch import _PatchedKernelExplainer

    data, target_col = get_pipeline_model_dataset()
    with mlflow.start_run() as run:
        mlflow.evaluate(
            pipeline_model_uri,
            data[0::3],
            model_type="classifier",
            targets=target_col,
            evaluators="shap",
            evaluator_config={"explainability_algorithm": "kernel"},
        )
    run_data = get_run_data(run.info.run_id)
    assert {
        "shap_beeswarm_plot.png",
        "shap_feature_importance_plot.png",
        "shap_summary_plot.png",
        "explainer",
    }.issubset(run_data.artifacts)

    explainer = mlflow.shap.load_explainer(f"runs:/{run.info.run_id}/explainer")
    assert isinstance(explainer, _PatchedKernelExplainer)


def test_compute_df_mode_or_mean():
    df = pd.DataFrame(
        {
            "a": [2.0, 2.0, 5.0],
            "b": [3, 3, 5],
            "c": [2.0, 2.0, 6.5],
            "d": [True, False, True],
            "e": ["abc", "b", "abc"],
            "f": [1.5, 2.5, np.nan],
            "g": ["ab", "ab", None],
            "h": pd.Series([2.0, 2.0, 6.5], dtype="category"),
        }
    )
    result = _compute_df_mode_or_mean(df)
    assert result == {
        "a": 2,
        "b": 3,
        "c": 3.5,
        "d": True,
        "e": "abc",
        "f": 2.0,
        "g": "ab",
        "h": 2.0,
    }

    # Test on dataframe that all columns are continuous.
    df2 = pd.DataFrame(
        {
            "c": [2.0, 2.0, 6.5],
            "f": [1.5, 2.5, np.nan],
        }
    )
    assert _compute_df_mode_or_mean(df2) == {"c": 3.5, "f": 2.0}

    # Test on dataframe that all columns are not continuous.
    df2 = pd.DataFrame(
        {
            "d": [True, False, True],
            "g": ["ab", "ab", None],
        }
    )
    assert _compute_df_mode_or_mean(df2) == {"d": True, "g": "ab"}


def test_xgboost_model_evaluate_work_with_shap_explainer():
    import shap
    import xgboost
    from sklearn.model_selection import train_test_split

    mlflow.xgboost.autolog(log_input_examples=True)
    X, y = shap.datasets.adult()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    xgb_model = xgboost.XGBClassifier()
    with mlflow.start_run():
        xgb_model.fit(X_train, y_train)

        eval_data = X_test
        eval_data["label"] = y_test

        model_uri = mlflow.get_artifact_uri("model")
        with mock.patch("mlflow.models.evaluation.evaluators.shap._logger.warning") as mock_warning:
            mlflow.evaluate(
                model_uri,
                eval_data,
                targets="label",
                model_type="classifier",
                evaluators=["default"],
            )
            assert not any(
                "Shap evaluation failed." in call_arg[0]
                for call_arg in mock_warning.call_args or []
                if isinstance(call_arg, tuple)
            )
