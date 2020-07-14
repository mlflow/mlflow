import os

from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.tracking._model_registry.registry import ModelRegistryStoreRegistry
from mlflow.tracking._tracking_service.utils import _TRACKING_USERNAME_ENV_VAR, \
    _TRACKING_PASSWORD_ENV_VAR, _TRACKING_TOKEN_ENV_VAR, _TRACKING_INSECURE_TLS_ENV_VAR, \
    _resolve_tracking_uri, get_tracking_uri
from mlflow.utils import rest_utils
from mlflow.utils.databricks_utils import get_databricks_host_creds


# NOTE: in contrast to tracking, we do not support the following ways to specify
# the model registry URI:
#  - via environment variables like MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, ...
# We do support specifying it
#  - via the ``model_registry_uri`` parameter when creating an ``MlflowClient`` or
#    ``ModelRegistryClient``.
#  - via a utility method ``mlflow.set_registry_uri``
#  - by not specifying anything: in this case we assume the model registry store URI is
#    the same as the tracking store URI. This means Tracking and Model Registry are
#    backed by the same backend DB/Rest server. However, note that we access them via
#    different ``Store`` classes (e.g. ``mlflow.store.tracking.SQLAlchemyStore`` &
#    ``mlflow.store.model_registry.SQLAlchemyStore``).
# This means the following combinations are not supported:
#  - Tracking RestStore & Model Registry RestStore that use different credentials.

_registry_uri = None


def set_registry_uri(uri):
    """
    Set the registry server URI. This method is especially useful if you have a registry server
    that's different from the tracking server.

    :param uri:

                - An empty string, or a local file path, prefixed with ``file:/``. Data is stored
                  locally at the provided file (or ``./mlruns`` if empty).
                - An HTTP URI like ``https://my-tracking-server:5000``.
                - A Databricks workspace, provided as the string "databricks" or, to use a
                  Databricks CLI
                  `profile <https://github.com/databricks/databricks-cli#installation>`_,
                  "databricks://<profileName>".
    """
    global _registry_uri
    _registry_uri = uri


def _get_registry_uri_from_context():
    global _registry_uri
    # in the future, REGISTRY_URI env var support can go here
    return _registry_uri


def get_registry_uri():
    """
    Get the current registry URI. If none has been specified, defaults to the tracking URI.

    :return: The registry URI.
    """
    return _get_registry_uri_from_context() or get_tracking_uri()


def _resolve_registry_uri(registry_uri=None, tracking_uri=None):
    return registry_uri or _get_registry_uri_from_context() or _resolve_tracking_uri(tracking_uri)


def _get_sqlalchemy_store(store_uri):
    from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore
    return SqlAlchemyStore(store_uri)


def _get_rest_store(store_uri, **_):
    def get_default_host_creds():
        return rest_utils.MlflowHostCreds(
            host=store_uri,
            username=os.environ.get(_TRACKING_USERNAME_ENV_VAR),
            password=os.environ.get(_TRACKING_PASSWORD_ENV_VAR),
            token=os.environ.get(_TRACKING_TOKEN_ENV_VAR),
            ignore_tls_verification=os.environ.get(_TRACKING_INSECURE_TLS_ENV_VAR) == 'true',
        )
    return RestStore(get_default_host_creds)


def _get_databricks_rest_store(store_uri, **_):
    return RestStore(lambda: get_databricks_host_creds(store_uri))


_model_registry_store_registry = ModelRegistryStoreRegistry()
_model_registry_store_registry.register('databricks', _get_databricks_rest_store)

for scheme in ['http', 'https']:
    _model_registry_store_registry.register(scheme, _get_rest_store)

for scheme in DATABASE_ENGINES:
    _model_registry_store_registry.register(scheme, _get_sqlalchemy_store)

_model_registry_store_registry.register_entrypoints()


def _get_store(store_uri=None):
    return _model_registry_store_registry.get_store(store_uri)
