# ruff: noqa
"""
python examples/demo.py
"""

import logging
import tempfile

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow


# Read the wine-quality csv file from the URL
csv_url = (
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
)
logger = logging.getLogger(__name__)
try:
    data = pd.read_csv(csv_url, sep=";")
except Exception as e:
    logger.exception(
        "Unable to download training & test CSV, check your internet connection. Error: %s", e
    )

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


alpha = 0.5
l1_ratio = 0.5

# Start a run to represent the training job
with mlflow.start_run() as training_run:
    # Load the training dataset with MLflow. We will link training metrics to this dataset.
    train_dataset: mlflow.data.pandas_dataset.PandasDataset = mlflow.data.from_pandas(
        train, name="train_dataset"
    )
    train_x = train_dataset.df.drop(["quality"], axis=1)
    train_y = train_dataset.df[["quality"]]

    # Fit a model to the training dataset
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    # Evaluate the model on the training dataset and log metrics
    predictions = lr.predict(train_x)
    (rmse, mae, r2) = eval_metrics(train_y, predictions)
    mlflow.log_metrics(
        metrics={
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
        },
        dataset=train_dataset,
    )

    # Log the model, specifying its ElasticNet parameters (alpha, l1_ratio)
    model = mlflow.sklearn.log_model(
        sk_model=lr,
        name="elasticnet",
        params={
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        },
    )

    # Fetch the model ID, and print the model
    model_id = model.model_id
    print("\n")
    print(model)
    print("\n")
    print(model_id)

# Start a run to represent the test dataset evaluation job
with mlflow.start_run() as evaluation_run:
    # Load the test dataset with MLflow. We will link test metrics to this dataset.
    test_dataset: mlflow.data.pandas_dataset.PandasDataset = mlflow.data.from_pandas(
        test, name="test_dataset"
    )
    test_x = test_dataset.df.drop(["quality"], axis=1)
    test_y = test_dataset.df[["quality"]]

    # Load the model
    model = mlflow.sklearn.load_model(f"models:/{model_id}")

    # Evaluate the model on the training dataset and log metrics
    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    mlflow.log_metrics(
        metrics={
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
        },
        dataset=test_dataset,
        # Specify the ID of the model logged above
        model_id=model_id,
    )

model = mlflow.get_logged_model(model_id)

training_run = mlflow.get_run(training_run.info.run_id)
print(training_run)
print("\n")
print(training_run.outputs)

evaluation_run = mlflow.get_run(evaluation_run.info.run_id)
print(evaluation_run)
print("\n")
print(evaluation_run.inputs)

print(f"models:/{model_id}")
mlflow.register_model(model_uri=f"models:/{model_id}", name="registered_elasticnet")
mlflow.MlflowClient().get_model_version("registered_elasticnet", 1)
