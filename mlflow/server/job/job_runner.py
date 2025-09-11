from typing import Any
import os
import sys
import threading
import time
import json
from huey import SqliteHuey
from huey.serializer import Serializer
import cloudpickle
from mlflow.server.job import job_functions
from mlflow.server.handlers import _get_tracking_store
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.exceptions import MlflowException


def _create_huey_instance():
    from mlflow.server import MLFLOW_HUEY_STORAGE_PATH

    class CloudPickleSerializer(Serializer):
        def serialize(self, data):
            return cloudpickle.dumps(data)

        def deserialize(self, data):
            return cloudpickle.loads(data)

    return SqliteHuey(
        filename=MLFLOW_HUEY_STORAGE_PATH,
        results=False,
        serializer=CloudPickleSerializer()
    )


huey = _create_huey_instance()


@huey.task()
def exec_job(job_id: str, function_name: str, serialized_params: str) -> None:
    tracking_store = _get_tracking_store()
    tracking_store.start_job(job_id)
    try:
        func = getattr(job_functions, function_name)
        params = json.loads(serialized_params)
        result = func(params)
        serialized_result = json.dumps(result)
        tracking_store.finish_job(job_id, serialized_result)
    except Exception as e:
        tracking_store.fail_job(job_id, repr(e))
