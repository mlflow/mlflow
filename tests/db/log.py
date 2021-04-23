import os

import mlflow
from mlflow.tracking._tracking_service.utils import _TRACKING_URI_ENV_VAR

assert _TRACKING_URI_ENV_VAR in os.environ


class MockModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        pass


with mlflow.start_run() as run:
    print("Tracking URI:", mlflow.get_tracking_uri())
    mlflow.log_param("p", "param")
    mlflow.log_metric("m", 1.0)
    mlflow.set_tag("t", "tag")
    mlflow.pyfunc.log_model(
        artifact_path="model", python_model=MockModel(), registered_model_name="mock",
    )
    print(mlflow.get_run(run.info.run_id))
