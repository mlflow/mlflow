"""
The ``mlflow.tracking`` module provides a Python CRUD interface to MLflow experiments
and runs. This is a lower level API that directly translates to MLflow
`REST API <../rest-api.html>`_ calls.
For a higher level API for managing an "active run", use the :py:mod:`mlflow` module.
"""

# Minimum APIs required for core tracing functionality of mlflow-tracing package.
from mlflow.tracking._tracking_service.utils import (
    _get_artifact_repo,
    _get_store,
    get_tracking_uri,
    is_tracking_uri_set,
    set_tracking_uri,
)
from mlflow.version import IS_TRACING_SDK_ONLY

__all__ = [
    "get_tracking_uri",
    "set_tracking_uri",
    "is_tracking_uri_set",
    "_get_artifact_repo",
    "_get_store",
]

# Importing the following APIs only if mlflow or mlflow-skinny is installed.
if not IS_TRACING_SDK_ONLY:
    from mlflow.tracking._model_registry.utils import (
        get_registry_uri,
        set_registry_uri,
    )
    from mlflow.tracking._tracking_service.utils import _get_artifact_repo
    from mlflow.tracking.client import MlflowClient

    __all__ += [
        "get_registry_uri",
        "set_registry_uri",
        "MlflowClient",
    ]
