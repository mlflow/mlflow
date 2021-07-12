import os

import mlflux
from mlflux.tracking._tracking_service.utils import _TRACKING_URI_ENV_VAR

assert _TRACKING_URI_ENV_VAR in os.environ


class MockModel(mlflux.pyfunc.PythonModel):
    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        pass


with mlflux.start_run() as run:
    print("Tracking URI:", mlflux.get_tracking_uri())
    mlflux.log_param("p", "param")
    mlflux.log_metric("m", 1.0)
    mlflux.set_tag("t", "tag")
    mlflux.pyfunc.log_model(
        artifact_path="model", python_model=MockModel(), registered_model_name="mock",
    )
    print(mlflux.get_run(run.info.run_id))
