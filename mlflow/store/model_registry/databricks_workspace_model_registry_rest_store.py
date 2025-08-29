from functools import partial
import logging
import mlflow
from mlflow.environment_variables import (
    MLFLOW_SKIP_SIGNATURE_CHECK_FOR_MIGRATION_TO_DATABRICKS_UC_REGISTRY,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    RESOURCE_ALREADY_EXISTS,
    ErrorCode,
)
from mlflow.store._unity_catalog.registry.rest_store import UcModelRegistryStore
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.logging_utils import eprint
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME


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

_logger = logging.getLogger(__name__)
class DatabricksWorkspaceModelRegistryRestStore(RestStore):
    def __init__(self, store_uri, tracking_uri):
        super().__init__(get_host_creds=partial(get_databricks_host_creds, store_uri))
        self.tracking_uri = tracking_uri

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
        """
        Copy a model version from one registered model to another as a new model version.
        This method can be used within the Databricks workspace registry to copy model versions
        between registered models, or to migrate model versions from the Databricks workspace
        registry to Unity Catalog. During the migration, signature validation can be bypassed
        by setting the `MLFLOW_SKIP_SIGNATURE_CHECK_FOR_MIGRATION_TO_DATABRICKS_UC_REGISTRY`
        environment variable to `True`.

        Args:
            src_mv: A :py:class:`mlflow.entities.model_registry.ModelVersion` object representing
                the source model version.
            dst_name: The name of the registered model to copy the model version to. If a
                registered model with this name does not exist, it will be created.

        Returns:
            Single :py:class:`mlflow.entities.model_registry.ModelVersion` object representing
            the cloned model version.
        """
        _logger.info(f"Starting copy_model_version: src_mv.name={src_mv.name}, src_mv.version={src_mv.version}, dst_name={dst_name}")
        _logger.info(f"tracking_uri: {self.tracking_uri}")
        _logger.info(f"dst_name.count('.') = {dst_name.count('.')}")
        
        if dst_name.count(".") == 2:
            _logger.info(f"Detected Unity Catalog destination (dst_name.count('.') == 2), proceeding with UC migration")
            source_uri = f"models:/{src_mv.name}/{src_mv.version}"
            _logger.info(f"Source URI: {source_uri}")
            
            try:
                _logger.info(f"Downloading artifacts from {source_uri} using tracking_uri: {self.tracking_uri}")
                local_model_dir = mlflow.artifacts.download_artifacts(
                    artifact_uri=source_uri, tracking_uri=self.tracking_uri
                )
                _logger.info(f"Successfully downloaded artifacts to: {local_model_dir}")
            except Exception as e:
                _logger.error(f"Failed to download artifacts from {source_uri}: {e}")
                raise MlflowException(
                    f"Unable to download model {src_mv.name} version {src_mv.version} "
                    f"artifacts from Databricks workspace registry in order to migrate "
                    f"them to Unity Catalog. Please ensure the model version artifacts "
                    f"exist and that you can download them via "
                    f"mlflow.artifacts.download_artifacts()"
                ) from e
            
            _logger.info(f"Creating UC store with store_uri={_DATABRICKS_UNITY_CATALOG_SCHEME}, tracking_uri={self.tracking_uri}")
            uc_store = UcModelRegistryStore(
                store_uri=_DATABRICKS_UNITY_CATALOG_SCHEME,
                tracking_uri=self.tracking_uri,
            )
            
            try:
                _logger.info(f"Creating registered model: {dst_name}")
                create_model_response = uc_store.create_registered_model(dst_name)
                _logger.info(f"Successfully created registered model: {create_model_response.name}")
                eprint(f"Successfully registered model '{create_model_response.name}'.")
            except MlflowException as e:
                if e.error_code != ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                    _logger.error(f"Failed to create registered model '{dst_name}': {e}")
                    raise
                _logger.info(f"Registered model '{dst_name}' already exists, proceeding to create new version")
                eprint(
                    f"Registered model '{dst_name}' already exists."
                    f" Creating a new version of this model..."
                )
            
            env_var = MLFLOW_SKIP_SIGNATURE_CHECK_FOR_MIGRATION_TO_DATABRICKS_UC_REGISTRY
            bypass_signature_validation = env_var.get()
            _logger.info(f"Signature validation bypass setting: {bypass_signature_validation}")
            
            _logger.info(f"Creating model version with parameters: name={dst_name}, source={source_uri}, run_id={src_mv.run_id}, model_id={src_mv.model_id}")
            result = uc_store._create_model_version_with_optional_signature_validation(
                name=dst_name,
                source=source_uri,
                run_id=src_mv.run_id,
                local_model_path=local_model_dir,
                model_id=src_mv.model_id,
                bypass_signature_validation=bypass_signature_validation,
            )
            _logger.info(f"Successfully created model version: {result}")
            return result
        else:
            _logger.info(f"Using standard copy_model_version (dst_name.count('.') != 2)")
            return super().copy_model_version(src_mv, dst_name)
