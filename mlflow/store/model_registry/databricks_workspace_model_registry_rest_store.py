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
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"copy_model_version called with src_mv={src_mv}, dst_name={dst_name}")
        
        logger.info(f"Checking if dst_name has 3 parts: {dst_name.split('.')}")
        if len(dst_name.split(".")) == 3:
            logger.info("dst_name has 3 parts, proceeding with Unity Catalog migration path")
            
            logger.info("Importing mlflow module")
            import mlflow
            
            logger.info("Importing UcModelRegistryStore from unity catalog registry")
            from mlflow.store._unity_catalog.registry.rest_store import UcModelRegistryStore
            
            logger.info("Starting try block for downloading artifacts")
            try:
                logger.info(f"Downloading artifacts for models:/{src_mv.name}/{src_mv.version}")
                local_model_dir = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"models:/{src_mv.name}/{src_mv.version}", tracking_uri=self.tracking_uri
                )
                logger.info(f"Successfully downloaded artifacts to: {local_model_dir}")
            except Exception as e:
                logger.info(f"Exception occurred during artifact download: {e}")
                logger.info("Raising MlflowException with detailed error message")
                raise MlflowException(
                    f"Unable to download model {src_mv.name} version {src_mv.version} "
                    f"artifacts from source artifact location '{src_mv.source}' in "
                    f"order to migrate them to Unity Catalog. Please ensure the source "
                    f"artifact location exists and that you can download from it via "
                    f"mlflow.artifacts.download_artifacts()"
                ) from e
            
            logger.info("Creating UcModelRegistryStore instance")
            uc_store = UcModelRegistryStore(
                store_uri="databricks-uc", tracking_uri=self.tracking_uri
            )
            logger.info("UcModelRegistryStore instance created successfully")
            
            logger.info("Getting bypass_signature_validation environment variable")
            bypass_signature_validation = (
                mlflow.environment_variables.MLFLOW_REGISTRY_MIGRATION_SKIP_SIGNATURE_VALIDATION.get()
            )
            logger.info(f"bypass_signature_validation value: {bypass_signature_validation}")
            
            logger.info("Calling _create_model_version_with_optional_signature_validation on uc_store")
            result = uc_store._create_model_version_with_optional_signature_validation(
                name=dst_name,
                source=src_mv.source,
                run_id=src_mv.run_id,
                local_model_path=local_model_dir,
                model_id=src_mv.model_id,
                bypass_signature_validation=bypass_signature_validation,
            )
            logger.info(f"_create_model_version_with_optional_signature_validation returned: {result}")
            logger.info("Returning result from Unity Catalog path")
            return result
        else:
            logger.info("dst_name does not have 3 parts, calling super().copy_model_version")
            result = super().copy_model_version(src_mv, dst_name)
            logger.info(f"super().copy_model_version returned: {result}")
            logger.info("Returning result from super() path")
            return result
