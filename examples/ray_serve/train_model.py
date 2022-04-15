import mlflow

from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle


if __name__ == "__main__":
    # Enable auto-logging
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.sklearn.autolog()

    # Load data
    iris_dataset = load_iris()
    data, target, target_names = (
        iris_dataset["data"],
        iris_dataset["target"],
        iris_dataset["target_names"],
    )

    # Instantiate model
    model = GradientBoostingClassifier()

    # Split training and validation data
    data, target = shuffle(data, target)
    train_x, train_y = data[:100], target[:100]
    val_x, val_y = data[100:], target[100:]

    # Train and evaluate model
    model.fit(train_x, train_y)
    run_id = mlflow.last_active_run().info.run_id
    print("MSE:", mean_squared_error(model.predict(val_x), val_y))
    print("Target names: ", target_names)
    print("run_id: {}".format(run_id))

    # Register the auto-logged model
    model_uri = "runs:/{}/model".format(run_id)
    registered_model_name = "RayMLflowIntegration"
    mv = mlflow.register_model(model_uri, registered_model_name)
    print("Name: {}".format(mv.name))
    print("Version: {}".format(mv.version))
