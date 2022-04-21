from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
import mlflow
import os
import matplotlib.pyplot as plt

# loading the California housing dataset
cali_housing = fetch_california_housing(as_frame=True)

# split the dataset into train and test partitions
X_train, X_test, y_train, y_test = train_test_split(
    cali_housing.data, cali_housing.target, test_size=0.2, random_state=123
)

# train the model
lin_reg = LinearRegression().fit(X_train, y_train)

# creating the evaluation dataframe
eval_data = X_test.copy()
eval_data["target"] = y_test


def example_custom_metric_fn(eval_df, builtin_metrics, artifacts_dir):
    """
    This example custom metric function creates a metric based on the ``prediction`` and
    ``target`` columns in ``eval_df`` and a metric derived from existing metrics in
    ``builtin_metrics``. It also generates and saves a scatter plot to ``artifacts_dir`` that
    visualizes the relationship between the predictions and targets for the given model to a
    file as an image artifact.
    """
    metrics = {
        "squared_diff_plus_one": np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 1) ** 2),
        "sum_on_label_divided_by_two": builtin_metrics["sum_on_label"] / 2,
    }
    plt.scatter(eval_df["prediction"], eval_df["target"])
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title("Targets vs. Predictions")
    plot_path = os.path.join(artifacts_dir, "example_scatter_plot.png")
    plt.savefig(plot_path)
    artifacts = {"example_scatter_plot_artifact": plot_path}
    return metrics, artifacts


with mlflow.start_run() as run:
    mlflow.sklearn.log_model(lin_reg, "model")
    model_uri = mlflow.get_artifact_uri("model")
    result = mlflow.evaluate(
        model=model_uri,
        data=eval_data,
        targets="target",
        model_type="regressor",
        dataset_name="cali_housing",
        evaluators=["default"],
        custom_metrics=[example_custom_metric_fn],
    )

print(f"metrics:\n{result.metrics}")
print(f"artifacts:\n{result.artifacts}")
