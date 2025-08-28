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

    def _await_model_version_creation(self, mv, await_creation_for):
        uc_hint = (
            " For faster model version creation, use Models in Unity Catalog "
            "(https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html)."
        )
        self._await_model_version_creation_impl(mv, await_creation_for, hint=uc_hint)

    def copy_model_version(self, src_mv, dst_name):
        if len(dst_name.split(".")) == 3:
            import mlflow
            from mlflow.store._unity_catalog.registry.rest_store import (
                UcModelRegistryStore,
            )

            source_uri = f"models:/{src_mv.name}/{src_mv.version}"
            try:
                local_model_dir = mlflow.artifacts.download_artifacts(
                    artifact_uri=source_uri, tracking_uri="databricks"
                )
            except Exception as e:
                raise MlflowException(
                    f"Unable to download model {src_mv.name} version {src_mv.version} "
                    f"artifacts from Databricks workspace registry in order to migrate "
                    f"them to Unity Catalog. Please ensure the model version artifacts "
                    f"exist and that you can download them via "
                    f"mlflow.artifacts.download_artifacts()"
                ) from e
            uc_store = UcModelRegistryStore(
                store_uri="databricks-uc", tracking_uri="databricks"
            )
            bypass_signature_validation = (
                mlflow.environment_variables.MLFLOW_REGISTRY_MIGRATION_SKIP_SIGNATURE_VALIDATION.get()
            )
            return uc_store._create_model_version_with_optional_signature_validation(
                name=dst_name,
                source=source_uri,
                run_id=src_mv.run_id,
                local_model_path=local_model_dir,
                model_id=src_mv.model_id,
                bypass_signature_validation=bypass_signature_validation,
            )
        else:
            return super().copy_model_version(src_mv, dst_name)
