import os
from functools import partial

from mlflow.environment_variables import MLFLOW_TRACKING_AWS_SIGV4
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.model_registry.file_store import FileStore
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.tracking._model_registry.registry import ModelRegistryStoreRegistry
from mlflow.tracking._tracking_service.utils import (
    _TRACKING_USERNAME_ENV_VAR,
    _TRACKING_PASSWORD_ENV_VAR,
    _TRACKING_TOKEN_ENV_VAR,
    _TRACKING_INSECURE_TLS_ENV_VAR,
    _TRACKING_CLIENT_CERT_PATH_ENV_VAR,
    _TRACKING_SERVER_CERT_PATH_ENV_VAR,
    _resolve_tracking_uri,
    get_tracking_uri,
)
from mlflow.utils import env, rest_utils
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME

_REGISTRY_URI_ENV_VAR = "MLFLOW_REGISTRY_URI"

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


def set_registry_uri(uri: str) -> None:
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

    .. code-block:: python
        :caption: Example

        import mflow

        # Set model registry uri, fetch the set uri, and compare
        # it with the tracking uri. They should be different
        mlflow.set_registry_uri("sqlite:////tmp/registry.db")
        mr_uri = mlflow.get_registry_uri()
        print("Current registry uri: {}".format(mr_uri))
        tracking_uri = mlflow.get_tracking_uri()
        print("Current tracking uri: {}".format(tracking_uri))

        # They should be different
        assert tracking_uri != mr_uri

    .. code-block:: text
        :caption: Output

        Current registry uri: sqlite:////tmp/registry.db
        Current tracking uri: file:///.../mlruns
    """
    global _registry_uri
    _registry_uri = uri


def _get_registry_uri_from_context():
    global _registry_uri
    # in the future, REGISTRY_URI env var support can go here
    if _registry_uri is not None:
        return _registry_uri
    elif env.get_env(_REGISTRY_URI_ENV_VAR) is not None:
        return env.get_env(_REGISTRY_URI_ENV_VAR)
    return _registry_uri


def get_registry_uri() -> str:
    """
    Get the current registry URI. If none has been specified, defaults to the tracking URI.

    :return: The registry URI.

    .. code-block:: python
        :caption: Example

        # Get the current model registry uri
        mr_uri = mlflow.get_registry_uri()
        print("Current model registry uri: {}".format(mr_uri))

        # Get the current tracking uri
        tracking_uri = mlflow.get_tracking_uri()
        print("Current tracking uri: {}".format(tracking_uri))

        # They should be the same
        assert mr_uri == tracking_uri

    .. code-block:: text
        :caption: Output

        Current model registry uri: file:///.../mlruns
        Current tracking uri: file:///.../mlruns
    """
    return _get_registry_uri_from_context() or get_tracking_uri()


def _resolve_registry_uri(registry_uri=None, tracking_uri=None):
    return registry_uri or _get_registry_uri_from_context() or _resolve_tracking_uri(tracking_uri)


def _get_sqlalchemy_store(store_uri):
    from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore

    return SqlAlchemyStore(store_uri)


def get_default_host_creds(store_uri):
    return rest_utils.MlflowHostCreds(
        host=store_uri,
        username=os.environ.get(_TRACKING_USERNAME_ENV_VAR),
        password=os.environ.get(_TRACKING_PASSWORD_ENV_VAR),
        token=os.environ.get(_TRACKING_TOKEN_ENV_VAR),
        aws_sigv4=MLFLOW_TRACKING_AWS_SIGV4.get(),
        ignore_tls_verification=os.environ.get(_TRACKING_INSECURE_TLS_ENV_VAR) == "true",
        client_cert_path=os.environ.get(_TRACKING_CLIENT_CERT_PATH_ENV_VAR),
        server_cert_path=os.environ.get(_TRACKING_SERVER_CERT_PATH_ENV_VAR),
    )


def _get_rest_store(store_uri, **_):
    return RestStore(partial(get_default_host_creds, store_uri))


def _get_databricks_rest_store(store_uri, **_):
    return RestStore(partial(get_databricks_host_creds, store_uri))


def _get_databricks_uc_rest_store(store_uri, **_):
    from mlflow.exceptions import MlflowException
    from mlflow.version import VERSION

    raise MlflowException(
        f"Detected Unity Catalog model registry URI '{store_uri}'. "
        f"However, the current version of the MLflow client ({VERSION}) does not support models "
        f"in the Unity Catalog. Please upgrade to the latest version of the MLflow Python client "
        f"to access models in the Unity Catalog, or specify a different registry URI via "
        f"mlflow.set_registry_uri()"
    )


# We define the global variable as `None` so that instantiating the store does not lead to circular
# dependency issues.
_model_registry_store_registry = None


def _get_file_store(store_uri, **_):
    return FileStore(store_uri)


def _get_store_registry():
    global _model_registry_store_registry
    if _model_registry_store_registry is not None:
        return _model_registry_store_registry

    _model_registry_store_registry = ModelRegistryStoreRegistry()
    _model_registry_store_registry.register("databricks", _get_databricks_rest_store)
    # Register a placeholder function that raises if users pass a registry URI with scheme
    # "databricks-uc"
    _model_registry_store_registry.register(
        _DATABRICKS_UNITY_CATALOG_SCHEME, _get_databricks_uc_rest_store
    )

    for scheme in ["http", "https"]:
        _model_registry_store_registry.register(scheme, _get_rest_store)

    for scheme in DATABASE_ENGINES:
        _model_registry_store_registry.register(scheme, _get_sqlalchemy_store)

    for scheme in ["", "file"]:
        _model_registry_store_registry.register(scheme, _get_file_store)

    _model_registry_store_registry.register_entrypoints()
    return _model_registry_store_registry


def _get_store(store_uri=None):
    return _get_store_registry().get_store(store_uri)
