import os

from mlflow.store.registry import StoreRegistry
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.store.tracking.registry import _TRACKING_USERNAME_ENV_VAR, \
    _TRACKING_PASSWORD_ENV_VAR, _TRACKING_TOKEN_ENV_VAR, _TRACKING_INSECURE_TLS_ENV_VAR
from mlflow.utils import rest_utils, experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.uri import get_db_profile_from_uri


class ModelRegistryStoreRegistry(StoreRegistry):
    """Scheme-based registry for model registry store implementations

    This class allows the registration of a function or class to provide an
    implementation for a given scheme of `store_uri` through the `register`
    methods. Implementations declared though the entrypoints
    `mlflow.registry_store` group can be automatically registered through the
    `register_entrypoints` method.

    When instantiating a store through the `get_store` method, the scheme of
    the store URI provided (or inferred from environment) will be used to
    select which implementation to instantiate, which will be called with same
    arguments passed to the `get_store` method.
    """

    def __init__(self):
        super(ModelRegistryStoreRegistry, self).__init__("mlflow.model_registry_store")

    def get_store(self, store_uri=None):
        """Get a store from the registry based on the scheme of store_uri

        :param store_uri: The store URI. If None, it will be inferred from the environment. This URI
                          is used to select which tracking store implementation to instantiate and
                          is passed to the constructor of the implementation.

        :return: An instance of `mlflow.store.model_registry.AbstractStore` that fulfills the
                 store URI requirements.
        """
        from mlflow.tracking._tracking_service import utils
        store_uri = store_uri if store_uri is not None else utils.get_tracking_uri()
        builder = self.get_store_builder(store_uri)
        return builder(store_uri=store_uri)


# NOTE: in contrast to tracking, we do not support the following ways to specify
# the model registry URI:
#  - via a utility method like tracking.tracking.utils.set_tracking_uri
#  - via environment variables like MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, ...
# We do support specifying it
#  - via the ``model_registry_uri`` parameter when creating an ``MlflowClient`` or
#    ``ModelRegistryClient``.
#  - by not specifying anything: in this case we assume the model registry store URI is
#    the same as the tracking store URI. This means Tracking and Model Registry are
#    backed by the same backend DB/Rest server. However, note that we access them via
#    different ``Store`` classes (e.g. ``mlflow.store.tracking.SQLAlchemyStore`` &
#    ``mlflow.store.model_registry.SQLAlchemyStore``).
# This means the following combinations are not supported:
#  - Tracking RestStore & Model Registry RestStore that use different credentials.

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
    profile = get_db_profile_from_uri(store_uri)
    return RestStore(lambda: get_databricks_host_creds(profile))


_model_registry_store_registry = ModelRegistryStoreRegistry()
_model_registry_store_registry.register('databricks', _get_databricks_rest_store)

for scheme in ['http', 'https']:
    _model_registry_store_registry.register(scheme, _get_rest_store)

for scheme in DATABASE_ENGINES:
    _model_registry_store_registry.register(scheme, _get_sqlalchemy_store)

_model_registry_store_registry.register_entrypoints()


@experimental
def register_model_registry_store(uri_scheme, store_builder):
    """
    Register a Model Registry Store with the Model Registry Store Registry by associating a URI
    scheme with a function that constructs a Model Registry Store instance.

    :param uri_scheme: The URI scheme to associate with the specified Model Registry Store builder.
    :param store_builder: A function that builds a Model Registry Store. This function must accept
                          one parameter: ``store_uri`` and return a Model Registry Store instance
                          that is a subclass of
                          :py:class:`mlflow.store.model_registry.abstract_store.AbstractStore`.
    """
    _model_registry_store_registry.register(uri_scheme, store_builder)


@experimental
def get_model_registry_store(store_uri=None):
    """
    Get a Model Registry Store from the registry based on the scheme of the specified ``store_uri``.

    :param store_uri: The Model Registry Store URI. This URI is passed to a corresponding store
                      builder in order to obtain a Tracking Store instance.

    :return: An instance of `mlflow.store.model_registry.abstract_store.AbstractStore` corresponding
             to the specified ``store_uri``.
    """
    return _model_registry_store_registry.get_store(store_uri)
