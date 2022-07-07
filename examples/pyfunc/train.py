from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn


class CustomPredict(mlflow.pyfunc.PythonModel):
    """Custom pyfunc class used to create customized mlflow models"""

    def load_context(self, context):

        self.model = mlflow.sklearn.load_model(context.artifacts["custom_model"])

    def predict(self, context, model_input):

        return self.model.predict(model_input)


X, y = load_iris(return_X_y=True, as_frame=True)
params = {"C": 1.0, "random_state": 42}

# create an mlflow client
client = MlflowClient()

model_name = "base_model"
custom_model_name = "custom_model"

with mlflow.start_run(run_name="test_pyfunc") as train_run:

    regression_model = LogisticRegression(**params).fit(X, y)

    model_info = mlflow.sklearn.log_model(
        sk_model=regression_model, artifact_path="model", registered_model_name=model_name
    )

    # start a child run to create custom imagine model
    with mlflow.start_run(run_name="test_custom_model", nested=True) as run:

        # log a custom model
        mlflow.pyfunc.log_model(
            artifact_path="",
            artifacts={custom_model_name: model_info.model_uri},
            python_model=CustomPredict(),
            registered_model_name=custom_model_name,
        )

        # load the latest model version
        mv = client.get_latest_versions(custom_model_name, ["None"])[0]

        # transition model to production
        client.transition_model_version_stage(
            name=custom_model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
