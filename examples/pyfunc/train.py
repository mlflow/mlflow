import os
from typing import Any

from custom_code import iris_classes
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import mlflow
from mlflow.models import infer_signature


class CustomPredict(mlflow.pyfunc.PythonModel):
    """Custom pyfunc class used to create customized mlflow models"""

    def load_context(self, context):
        self.model = mlflow.sklearn.load_model(context.artifacts["custom_model"])

    def predict(self, context, model_input, params: dict[str, Any] | None = None):
        prediction = self.model.predict(model_input)
        return iris_classes(prediction)


X, y = load_iris(return_X_y=True, as_frame=True)
params = {"C": 1.0, "random_state": 42}
classifier = LogisticRegression(**params).fit(X, y)

predictions = classifier.predict(X)
signature = infer_signature(X, predictions)

with mlflow.start_run(run_name="test_pyfunc") as run:
    model_info = mlflow.sklearn.log_model(sk_model=classifier, name="model", signature=signature)

    # start a child run to create custom imagine model
    with mlflow.start_run(run_name="test_custom_model", nested=True):
        print(f"Pyfunc run ID: {run.info.run_id}")
        # log a custom model
        mlflow.pyfunc.log_model(
            name="artifacts",
            code_paths=[os.getcwd()],
            artifacts={"custom_model": model_info.model_uri},
            python_model=CustomPredict(),
            signature=signature,
        )
