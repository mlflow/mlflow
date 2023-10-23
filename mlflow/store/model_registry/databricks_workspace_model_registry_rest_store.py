from mlflow.exceptions import MlflowException
from mlflow.store.model_registry.rest_store import RestStore


def _raise_unsupported_method(method, message=None):
    messages = [
        f"Method '{method}' is unsupported for models in the Workspace Model Registry. "
        f"Upgrade to Models in Unity Catalog to access the latest features. You can configure "
        f"the MLflow Python client to access models in Unity Catalog by running "
        f"mlflow.set_registry_uri('databricks-uc') before accessing models.",
    ]
    if message is not None:
        messages.append(message)
    raise MlflowException(" ".join(messages))


class DatabricksWorkspaceModelRegistryRestStore(RestStore):
    def set_registered_model_alias(self, name, alias, version):
        _raise_unsupported_method(method="set_registered_model_alias")

    def delete_registered_model_alias(self, name, alias):
        _raise_unsupported_method(method="delete_registered_model_alias")

    def get_model_version_by_alias(self, name, alias):
        _raise_unsupported_method(
            method="get_model_version_by_alias",
            message="If attempting to load a model version by alias via a URI of the form "
            "'models:/model_name@alias_name', configure the MLflow client to target Unity Catalog "
            "and try again.",
        )

    def copy_model_version(self, src_mv, dst_name):
        """
        Copy a model version from one registered model to another as a new model version.

        :param src_mv: A :py:class:`mlflow.entities.model_registry.ModelVersion` object representing
                       the source model version.
        :param dst_name: the name of the registered model to copy the model version to. If a
                         registered model with this name does not exist, it will be created.
        :return: Single :py:class:`mlflow.entities.model_registry.ModelVersion` object representing
                 the cloned model version.
        """
        return self._copy_model_version_impl(src_mv, dst_name)

    def _await_model_version_creation(self, mv, await_creation_for):
        uc_hint = (
            " For faster model version creation, use Models in Unity Catalog "
            "(https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html)."
        )
        self._await_model_version_creation_impl(mv, await_creation_for, hint=uc_hint)
