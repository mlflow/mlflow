from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

import mlflow

if __name__ == "__main__":
    # Enable auto-logging
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.sklearn.autolog()

    # Load data
    iris_dataset = load_iris()
    data = iris_dataset["data"]
    target = iris_dataset["target"]
    target_names = iris_dataset["target_names"]

    # Instantiate model
    model = GradientBoostingClassifier()

    # Split training and validation data
    data, target = shuffle(data, target)
    train_x = data[:100]
    train_y = target[:100]
    val_x = data[100:]
    val_y = target[100:]

    # Train and evaluate model
    model.fit(train_x, train_y)
    run_id = mlflow.last_active_run().info.run_id
    print("MSE:", mean_squared_error(model.predict(val_x), val_y))
    print("Target names: ", target_names)
    print(f"run_id: {run_id}")

    # Register the auto-logged model
    model_uri = f"runs:/{run_id}/model"
    registered_model_name = "RayMLflowIntegration"
    mv = mlflow.register_model(model_uri, registered_model_name)
    print(f"Name: {mv.name}")
    print(f"Version: {mv.version}")
