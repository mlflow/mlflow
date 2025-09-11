from typing import Any
import json
from mlflow.server.handlers import _get_tracking_store
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.exceptions import MlflowException


_tracking_store = _get_tracking_store()
if not isinstance(_tracking_store, SqlAlchemyStore):
    raise MlflowException.invalid_parameter_value(
        "If enabling MLflow job execution, mlflow server must configure "
        "--backend-store-uri to a database URI."
    )


def submit_job(function, params: Any):
    from mlflow.server.job.job_runner import exec_job
    from mlflow.server.job import job_functions
    from mlflow.server.job.job_runner import exec_job

    assert function.__module__ == "mlflow.server.job.job_functions"
    serialized_params = json.dumps(params)
    func_name = function.__name__
    job_id = _tracking_store.create_job(func_name, serialized_params)

    # enqueue job
    exec_job(job_id, func_name, serialized_params)
