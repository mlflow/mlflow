from mlflow.store.model_registry.rest_store import RestStore
from mlflow.exceptions import MlflowException


def _raise_unsupported_method(method, message=None):
    messages = [
        f"Method '{method}' is unsupported for models in the Workspace Model Registry. "
        f"Use Models in Unity Catalog to access the latest features.",
    ]
    if message is not None:
        messages.append(message)
    messages.append("See the user guide for more information")
    raise MlflowException(" ".join(messages))


class DatabricksWorkspaceModelRegistryRestStore(RestStore):
    def set_registered_model_alias(self, name, alias, version):
        _raise_unsupported_method(method="set_registered_model_alias")

    def delete_registered_model_alias(self, name, alias):
        _raise_unsupported_method(method="delete_registered_model_alias")

    def get_model_version_by_alias(self, name, alias):
        _raise_unsupported_method(method="get_model_version_by_alias")
