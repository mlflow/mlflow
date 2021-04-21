import os
import mlflow
from mlflow.tracking._tracking_service.utils import _TRACKING_URI_ENV_VAR

assert _TRACKING_URI_ENV_VAR in os.environ

with mlflow.start_run():
    print("Tracking URI:", mlflow.get_tracking_uri())
    mlflow.log_param("p", "param")
    mlflow.log_metric("m", 1.0)
    mlflow.set_tag("t", "tag")
