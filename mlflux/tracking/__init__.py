"""
The ``mlflux.tracking`` module provides a Python CRUD interface to mlflux experiments
and runs. This is a lower level API that directly translates to mlflux
`REST API <../rest-api.html>`_ calls.
For a higher level API for managing an "active run", use the :py:mod:`mlflux` module.
"""

from mlflux.tracking.client import MlflowClient
from mlflux.tracking._tracking_service.utils import (
    set_tracking_uri,
    get_tracking_uri,
    is_tracking_uri_set,
    _get_store,
    _TRACKING_URI_ENV_VAR,
)
from mlflux.tracking._model_registry.utils import (
    set_registry_uri,
    get_registry_uri,
)
from mlflux.tracking.fluent import _EXPERIMENT_ID_ENV_VAR, _EXPERIMENT_NAME_ENV_VAR, _RUN_ID_ENV_VAR

__all__ = [
    "MlflowClient",
    "get_tracking_uri",
    "set_tracking_uri",
    "is_tracking_uri_set",
    "_get_store",
    "get_registry_uri",
    "set_registry_uri",
    "_EXPERIMENT_ID_ENV_VAR",
    "_EXPERIMENT_NAME_ENV_VAR",
    "_RUN_ID_ENV_VAR",
    "_TRACKING_URI_ENV_VAR",
]
