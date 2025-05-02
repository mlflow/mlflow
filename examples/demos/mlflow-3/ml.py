# MLflow 3 Traditional ML Example
# In this example, we will first run a model training job, which is tracked as
# an MLflow Run, to produce a trained model, which is tracked as an MLflow Logged Model.
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.entities import Dataset


# Helper function to compute metrics
def compute_metrics(actual, predicted):
    rmse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2


# Load Iris dataset and prepare the DataFrame
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["quality"] = (iris.target == 2).astype(int)  # Create a binary target for simplicity

# Split into training and testing datasets
train_df, test_df = train_test_split(iris_df, test_size=0.2, random_state=42)

# Start a run to represent the training job
with mlflow.start_run() as training_run:
    # Load the training dataset with MLflow. We will link training metrics to this dataset.
    train_dataset: Dataset = mlflow.data.from_pandas(train_df, name="train")
    train_x = train_dataset.df.drop(["quality"], axis=1)
    train_y = train_dataset.df[["quality"]]

    # Fit a model to the training dataset
    lr = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
    lr.fit(train_x, train_y)

    # Log the model, specifying its ElasticNet parameters (alpha, l1_ratio)
    # As a new feature, the LoggedModel entity is linked to its name and params
    logged_model = mlflow.sklearn.log_model(
        sk_model=lr,
        name="elasticnet",
        params={
            "alpha": 0.5,
            "l1_ratio": 0.5,
        },
        input_example=train_x,
    )

    # Inspect the LoggedModel and its properties
    print(logged_model.model_id, logged_model.params)
    # m-fa4e1bca8cb64971bce2322a8fd427d3, {'alpha': '0.5', 'l1_ratio': '0.5'}

    # Evaluate the model on the training dataset and log metrics
    # These metrics are now linked to the LoggedModel entity
    predictions = lr.predict(train_x)
    (rmse, mae, r2) = compute_metrics(train_y, predictions)
    mlflow.log_metrics(
        metrics={
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
        },
        model_id=logged_model.model_id,
        dataset=train_dataset,
    )

    # Inspect the LoggedModel, now with metrics
    logged_model = mlflow.get_logged_model(logged_model.model_id)
    print(logged_model.model_id, logged_model.metrics)
    # m-fa4e1bca8cb64971bce2322a8fd427d3, [<Metric: dataset_name='train', key='rmse', model_id='m-fa4e1bca8cb64971bce2322a8fd427d3, value=0.7538635773139717, ...>, ...]


# Some time later, when we get a new evaluation dataset based on the latest production data,
# we will run a new model evaluation job, which is tracked as a new MLflow Run,
# to measure the performance of the model on this new dataset.
# This example will produced two MLflow Runs (training_run and evaluation_run) and
# one MLflow Logged Model (elasticnet). From the resulting Logged Model,
# we can see all of the parameters and metadata. We can also see all of the metrics linked
# from the training and evaluation runs.

# Start a run to represent the test dataset evaluation job
with mlflow.start_run() as evaluation_run:
    # Load the test dataset with MLflow. We will link test metrics to this dataset.
    test_dataset: mlflow.entities.Dataset = mlflow.data.from_pandas(test_df, name="test")
    test_x = test_dataset.df.drop(["quality"], axis=1)
    test_y = test_dataset.df[["quality"]]

    # Load the model
    model = mlflow.sklearn.load_model(f"models:/{logged_model.model_id}")

    # Evaluate the model on the training dataset and log metrics, linking to model
    predictions = model.predict(test_x)
    (rmse, mae, r2) = compute_metrics(test_y, predictions)
    mlflow.log_metrics(
        metrics={
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
        },
        dataset=test_dataset,
        model_id=logged_model.model_id,
    )


print(mlflow.get_logged_model(logged_model.model_id).to_dictionary())

# Now register the model.
mlflow.register_model(logged_model.model_uri, name="my_ml_model")
