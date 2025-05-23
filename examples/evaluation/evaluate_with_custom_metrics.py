import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import infer_signature, make_metric

# loading the California housing dataset
cali_housing = fetch_california_housing(as_frame=True)

# split the dataset into train and test partitions
X_train, X_test, y_train, y_test = train_test_split(
    cali_housing.data, cali_housing.target, test_size=0.2, random_state=123
)

# train the model
lin_reg = LinearRegression().fit(X_train, y_train)

# Infer model signature
predictions = lin_reg.predict(X_train)
signature = infer_signature(X_train, predictions)

# creating the evaluation dataframe
eval_data = X_test.copy()
eval_data["target"] = y_test


def squared_diff_plus_one(eval_df, _builtin_metrics):
    """
    This example custom metric function creates a metric based on the ``prediction`` and
    ``target`` columns in ``eval_df`.
    """
    return np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 1) ** 2)


def sum_on_target_divided_by_two(_eval_df, builtin_metrics):
    """
    This example custom metric function creates a metric derived from existing metrics in
    ``builtin_metrics``.
    """
    return builtin_metrics["sum_on_target"] / 2


def prediction_target_scatter(eval_df, _builtin_metrics, artifacts_dir):
    """
    This example custom artifact generates and saves a scatter plot to ``artifacts_dir`` that
    visualizes the relationship between the predictions and targets for the given model to a
    file as an image artifact.
    """
    plt.scatter(eval_df["prediction"], eval_df["target"])
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title("Targets vs. Predictions")
    plot_path = os.path.join(artifacts_dir, "example_scatter_plot.png")
    plt.savefig(plot_path)
    return {"example_scatter_plot_artifact": plot_path}


with mlflow.start_run() as run:
    model_info = mlflow.sklearn.log_model(lin_reg, name="model", signature=signature)
    result = mlflow.evaluate(
        model=model_info.model_uri,
        data=eval_data,
        targets="target",
        model_type="regressor",
        evaluators=["default"],
        extra_metrics=[
            make_metric(
                eval_fn=squared_diff_plus_one,
                greater_is_better=False,
            ),
            make_metric(
                eval_fn=sum_on_target_divided_by_two,
                greater_is_better=True,
            ),
        ],
        custom_artifacts=[prediction_target_scatter],
    )

print(f"metrics:\n{result.metrics}")
print(f"artifacts:\n{result.artifacts}")
