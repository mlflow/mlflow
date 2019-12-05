import os

from mlflow.store.registry import StoreRegistry
from mlflow.store.tracking import DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.rest_store import RestStore, DatabricksRestStore
from mlflow.utils import rest_utils
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.uri import get_db_profile_from_uri


# Extra environment variables which take precedence for setting the basic/bearer
# auth on http requests.
_TRACKING_USERNAME_ENV_VAR = "MLFLOW_TRACKING_USERNAME"
_TRACKING_PASSWORD_ENV_VAR = "MLFLOW_TRACKING_PASSWORD"
_TRACKING_TOKEN_ENV_VAR = "MLFLOW_TRACKING_TOKEN"
_TRACKING_INSECURE_TLS_ENV_VAR = "MLFLOW_TRACKING_INSECURE_TLS"


class TrackingStoreRegistry(StoreRegistry):
    """Scheme-based registry for tracking store implementations

    This class allows the registration of a function or class to provide an
    implementation for a given scheme of `store_uri` through the `register`
    methods. Implementations declared though the entrypoints
    `mlflow.tracking_store` group can be automatically registered through the
    `register_entrypoints` method.

    When instantiating a store through the `get_store` method, the scheme of
    the store URI provided (or inferred from environment) will be used to
    select which implementation to instantiate, which will be called with same
    arguments passed to the `get_store` method.
    """

    def __init__(self):
        super(TrackingStoreRegistry, self).__init__("mlflow.tracking_store")

    def get_store(self, store_uri=None, artifact_uri=None):
        """Get a store from the registry based on the scheme of store_uri

        :param store_uri: The store URI. If None, it will be inferred from the environment. This URI
                          is used to select which tracking store implementation to instantiate and
                          is passed to the constructor of the implementation.
        :param artifact_uri: Artifact repository URI. Passed through to the tracking store
                             implementation.

        :return: An instance of `mlflow.store.tracking.AbstractStore` that fulfills the store URI
                 requirements.
        """
        from mlflow.tracking._tracking_service import utils
        store_uri = store_uri if store_uri is not None else utils.get_tracking_uri()
        builder = self.get_store_builder(store_uri)
        return builder(store_uri=store_uri, artifact_uri=artifact_uri)


def _get_file_store(store_uri, **_):
    return FileStore(store_uri, store_uri)


def _get_sqlalchemy_store(store_uri, artifact_uri):
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
    if artifact_uri is None:
        artifact_uri = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
    return SqlAlchemyStore(store_uri, artifact_uri)


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
    profile = get_db_profile_from_uri(store_uri)
    return DatabricksRestStore(lambda: get_databricks_host_creds(profile))


_tracking_store_registry = TrackingStoreRegistry()
_tracking_store_registry.register('', _get_file_store)
_tracking_store_registry.register('file', _get_file_store)
_tracking_store_registry.register('databricks', _get_databricks_rest_store)

for scheme in ['http', 'https']:
    _tracking_store_registry.register(scheme, _get_rest_store)

for scheme in DATABASE_ENGINES:
    _tracking_store_registry.register(scheme, _get_sqlalchemy_store)

_tracking_store_registry.register_entrypoints()


def register_tracking_store(uri_scheme, store_generator):
    _tracking_store_registry.register(uri_scheme, store_generator)


def get_tracking_store(store_uri=None, artifact_uri=None):
    return _tracking_store_registry.get_store(store_uri, artifact_uri)
