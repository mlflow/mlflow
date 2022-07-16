import os

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from custom_code import iris_classes

import mlflow
import mlflow.sklearn


class CustomPredict(mlflow.pyfunc.PythonModel):
    """Custom pyfunc class used to create customized mlflow models"""

    def load_context(self, context):

        self.model = mlflow.sklearn.load_model(context.artifacts["custom_model"])

    def predict(self, context, model_input):

        prediction = self.model.predict(model_input)
        return iris_classes(prediction)


X, y = load_iris(return_X_y=True, as_frame=True)
params = {"C": 1.0, "random_state": 42}

with mlflow.start_run(run_name="test_pyfunc"):

    regression_model = LogisticRegression(**params).fit(X, y)

    model_info = mlflow.sklearn.log_model(sk_model=regression_model, artifact_path="model")

    # start a child run to create custom imagine model
    with mlflow.start_run(run_name="test_custom_model", nested=True):

        # log a custom model
        mlflow.pyfunc.log_model(
            artifact_path="artifacts",
            code_path=[os.getcwd()],
            artifacts={"custom_model": model_info.model_uri},
            python_model=CustomPredict(),
        )
