import os

from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.tracking._model_registry.registry import ModelRegistryStoreRegistry
from mlflow.tracking._tracking_service.utils import _TRACKING_USERNAME_ENV_VAR, \
    _TRACKING_PASSWORD_ENV_VAR, _TRACKING_TOKEN_ENV_VAR, _TRACKING_INSECURE_TLS_ENV_VAR
from mlflow.utils import rest_utils
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.uri import get_db_profile_from_uri


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

# NOTE: SqlAlchemyStore code is commented out here - we can add it back once it's available

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


def _get_store(store_uri=None):
    return _model_registry_store_registry.get_store(store_uri)
