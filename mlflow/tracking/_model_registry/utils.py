import importlib
from functools import partial

from mlflow.environment_variables import MLFLOW_REGISTRY_URI
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.model_registry.databricks_workspace_model_registry_rest_store import (
    DatabricksWorkspaceModelRegistryRestStore,
)
from mlflow.store.model_registry.file_store import FileStore
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.tracking._model_registry.registry import ModelRegistryStoreRegistry
from mlflow.tracking._tracking_service.utils import (
    _resolve_tracking_uri,
)
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.databricks_utils import (
    is_in_databricks_serverless_runtime,
    warn_on_deprecated_cross_workspace_registry_uri,
)
from mlflow.utils.uri import (
    _DATABRICKS_UNITY_CATALOG_SCHEME,
    _OSS_UNITY_CATALOG_SCHEME,
    construct_db_uc_uri_from_profile,
    get_db_info_from_uri,
    is_databricks_uri,
)

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
    """Set the registry server URI. This method is especially useful if you have a registry server
    that's different from the tracking server.

    Args:
        uri: An empty string, or a local file path, prefixed with ``file:/``. Data is stored
            locally at the provided file (or ``./mlruns`` if empty). An HTTP URI like
            ``https://my-tracking-server:5000`` or ``http://my-oss-uc-server:8080``. A Databricks
            workspace, provided as the string "databricks" or, to use a Databricks CLI
            `profile <https://github.com/databricks/databricks-cli#installation>`_,
            "databricks://<profileName>".

    .. code-block:: python
        :caption: Example

        import mflow

        # Set model registry uri, fetch the set uri, and compare
        # it with the tracking uri. They should be different
        mlflow.set_registry_uri("sqlite:////tmp/registry.db")
        mr_uri = mlflow.get_registry_uri()
        print(f"Current registry uri: {mr_uri}")
        tracking_uri = mlflow.get_tracking_uri()
        print(f"Current tracking uri: {tracking_uri}")

        # They should be different
        assert tracking_uri != mr_uri

    .. code-block:: text
        :caption: Output

        Current registry uri: sqlite:////tmp/registry.db
        Current tracking uri: file:///.../mlruns

    """
    global _registry_uri
    _registry_uri = uri
    if uri:
        # Set 'MLFLOW_REGISTRY_URI' environment variable
        # so that subprocess can inherit it.
        MLFLOW_REGISTRY_URI.set(_registry_uri)


def _get_registry_uri_from_spark_session():
    session = _get_active_spark_session()
    if session is None:
        return None

    if is_in_databricks_serverless_runtime():
        # Connected to Serverless
        return "databricks-uc"

    from pyspark.sql.utils import AnalysisException

    try:
        return session.conf.get("spark.mlflow.modelRegistryUri", None)
    except AnalysisException:
        # In serverless clusters, session.conf.get() is unsupported
        # and raises an AnalysisException. We may encounter this case
        # when DBConnect is used to connect to a serverless cluster,
        # in which case the prior `is_in_databricks_serverless_runtime()`
        # check will have returned false (as of 2025-06-07, it checks
        # an environment variable that isn't set by DBConnect)
        return None


def _get_registry_uri_from_context():
    if _registry_uri is not None:
        return _registry_uri
    elif (uri := MLFLOW_REGISTRY_URI.get()) or (uri := _get_registry_uri_from_spark_session()):
        return uri
    return _registry_uri


def _get_default_registry_uri_for_tracking_uri(tracking_uri: str | None) -> str | None:
    """
    Get the default registry URI for a given tracking URI.

    If the tracking URI starts with "databricks", returns "databricks-uc" with profile if present.
    Otherwise, returns the tracking URI itself.

    Args:
        tracking_uri: The tracking URI to get the default registry URI for

    Returns:
        The default registry URI
    """
    if tracking_uri is not None and is_databricks_uri(tracking_uri):
        # If the tracking URI is "databricks", we impute the registry URI as "databricks-uc"
        # corresponding to Databricks Unity Catalog Model Registry, which is the recommended
        # model registry offering on Databricks
        if tracking_uri == "databricks":
            return _DATABRICKS_UNITY_CATALOG_SCHEME
        else:
            # Extract profile from tracking URI and construct databricks-uc URI
            profile, key_prefix = get_db_info_from_uri(tracking_uri)
            if profile:
                # Reconstruct the profile string including key_prefix if present
                profile_string = f"{profile}:{key_prefix}" if key_prefix else profile
                return construct_db_uc_uri_from_profile(profile_string)
            else:
                return _DATABRICKS_UNITY_CATALOG_SCHEME

    # For non-databricks tracking URIs, use the tracking URI as the registry URI
    return tracking_uri


def get_registry_uri() -> str:
    """Get the current registry URI. If none has been specified, defaults to the tracking URI.

    Returns:
        The registry URI.

    .. code-block:: python

        # Get the current model registry uri
        mr_uri = mlflow.get_registry_uri()
        print(f"Current model registry uri: {mr_uri}")

        # Get the current tracking uri
        tracking_uri = mlflow.get_tracking_uri()
        print(f"Current tracking uri: {tracking_uri}")

        # They should be the same
        assert mr_uri == tracking_uri

    .. code-block:: text

        Current model registry uri: file:///.../mlruns
        Current tracking uri: file:///.../mlruns

    """
    return _resolve_registry_uri()


def _resolve_registry_uri(
    registry_uri: str | None = None, tracking_uri: str | None = None
) -> str | None:
    """
    Resolve the registry URI following the same logic as get_registry_uri().
    """
    return (
        registry_uri
        or _get_registry_uri_from_context()
        or _get_default_registry_uri_for_tracking_uri(_resolve_tracking_uri(tracking_uri))
    )


def _get_sqlalchemy_store(store_uri):
    from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore

    return SqlAlchemyStore(store_uri)


def _get_rest_store(store_uri, **_):
    return RestStore(partial(get_default_host_creds, store_uri))


def _get_databricks_rest_store(store_uri, tracking_uri, **_):
    warn_on_deprecated_cross_workspace_registry_uri(registry_uri=store_uri)
    return DatabricksWorkspaceModelRegistryRestStore(store_uri, tracking_uri)


# We define the global variable as `None` so that instantiating the store does not lead to circular
# dependency issues.
_model_registry_store_registry = None


def _get_file_store(store_uri, **_):
    return FileStore(store_uri)


def _get_store_registry():
    global _model_registry_store_registry
    from mlflow.store._unity_catalog.registry.rest_store import UcModelRegistryStore
    from mlflow.store._unity_catalog.registry.uc_oss_rest_store import UnityCatalogOssStore

    if _model_registry_store_registry is not None:
        return _model_registry_store_registry

    _model_registry_store_registry = ModelRegistryStoreRegistry()
    _model_registry_store_registry.register("databricks", _get_databricks_rest_store)
    # Register a placeholder function that raises if users pass a registry URI with scheme
    # "databricks-uc"
    _model_registry_store_registry.register(_DATABRICKS_UNITY_CATALOG_SCHEME, UcModelRegistryStore)
    _model_registry_store_registry.register(_OSS_UNITY_CATALOG_SCHEME, UnityCatalogOssStore)

    for scheme in ["http", "https"]:
        _model_registry_store_registry.register(scheme, _get_rest_store)

    if importlib.util.find_spec("sqlalchemy") is not None:
        for scheme in DATABASE_ENGINES:
            _model_registry_store_registry.register(scheme, _get_sqlalchemy_store)

    for scheme in ["", "file"]:
        _model_registry_store_registry.register(scheme, _get_file_store)

    _model_registry_store_registry.register_entrypoints()
    return _model_registry_store_registry


def _get_store(store_uri=None, tracking_uri=None):
    return _get_store_registry().get_store(store_uri, tracking_uri)
