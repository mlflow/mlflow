from typing import List, Optional

from mlflow.entities.model_registry import (
    ModelVersion,
    ModelVersionTag,
    RegisteredModel,
    RegisteredModelAlias,
    RegisteredModelTag,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_uc_registry_messages_pb2 import ModelVersion as ProtoModelVersion
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    ModelVersionStatus as ProtoModelVersionStatus,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    ModelVersionTag as ProtoModelVersionTag,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    RegisteredModel as ProtoRegisteredModel,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    RegisteredModelTag as ProtoRegisteredModelTag,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import TemporaryCredentials
from mlflow.store.artifact.artifact_repo import ArtifactRepository

_STRING_TO_STATUS = {k: ProtoModelVersionStatus.Value(k) for k in ProtoModelVersionStatus.keys()}
_STATUS_TO_STRING = {value: key for key, value in _STRING_TO_STATUS.items()}
_ACTIVE_CATALOG_QUERY = "SELECT current_catalog() AS catalog"
_ACTIVE_SCHEMA_QUERY = "SELECT current_database() AS schema"


def uc_model_version_status_to_string(status):
    return _STATUS_TO_STRING[status]


def model_version_from_uc_proto(uc_proto: ProtoModelVersion) -> ModelVersion:
    return ModelVersion(
        name=uc_proto.name,
        version=uc_proto.version,
        creation_timestamp=uc_proto.creation_timestamp,
        last_updated_timestamp=uc_proto.last_updated_timestamp,
        description=uc_proto.description,
        user_id=uc_proto.user_id,
        source=uc_proto.source,
        run_id=uc_proto.run_id,
        status=uc_model_version_status_to_string(uc_proto.status),
        status_message=uc_proto.status_message,
        aliases=[alias.alias for alias in (uc_proto.aliases or [])],
        tags=[ModelVersionTag(key=tag.key, value=tag.value) for tag in (uc_proto.tags or [])],
    )


def registered_model_from_uc_proto(uc_proto: ProtoRegisteredModel) -> RegisteredModel:
    return RegisteredModel(
        name=uc_proto.name,
        creation_timestamp=uc_proto.creation_timestamp,
        last_updated_timestamp=uc_proto.last_updated_timestamp,
        description=uc_proto.description,
        aliases=[
            RegisteredModelAlias(alias=alias.alias, version=alias.version)
            for alias in (uc_proto.aliases or [])
        ],
        tags=[RegisteredModelTag(key=tag.key, value=tag.value) for tag in (uc_proto.tags or [])],
    )


def uc_registered_model_tag_from_mlflow_tags(
    tags: Optional[List[RegisteredModelTag]],
) -> List[ProtoRegisteredModelTag]:
    if tags is None:
        return []
    return [ProtoRegisteredModelTag(key=t.key, value=t.value) for t in tags]


def uc_model_version_tag_from_mlflow_tags(
    tags: Optional[List[ModelVersionTag]],
) -> List[ProtoModelVersionTag]:
    if tags is None:
        return []
    return [ProtoModelVersionTag(key=t.key, value=t.value) for t in tags]


def get_artifact_repo_from_storage_info(
    storage_location: str, scoped_token: TemporaryCredentials
) -> ArtifactRepository:
    """
    Get an ArtifactRepository instance capable of reading/writing to a UC model version's
    file storage location

    Args:
        storage_location: Storage location of the model version
        scoped_token: Protobuf scoped token to use to authenticate to blob storage
    """
    try:
        return _get_artifact_repo_from_storage_info(
            storage_location=storage_location, scoped_token=scoped_token
        )
    except ImportError as e:
        raise MlflowException(
            "Unable to import necessary dependencies to access model version files in "
            "Unity Catalog. Please ensure you have the necessary dependencies installed, "
            "e.g. by running 'pip install mlflow[databricks]' or "
            "'pip install mlflow-skinny[databricks]'"
        ) from e


def _get_artifact_repo_from_storage_info(
    storage_location: str, scoped_token: TemporaryCredentials
) -> ArtifactRepository:
    credential_type = scoped_token.WhichOneof("credentials")
    if credential_type == "aws_temp_credentials":
        # Verify upfront that boto3 is importable
        import boto3  # noqa: F401

        from mlflow.store.artifact.optimized_s3_artifact_repo import OptimizedS3ArtifactRepository

        aws_creds = scoped_token.aws_temp_credentials
        return OptimizedS3ArtifactRepository(
            artifact_uri=storage_location,
            access_key_id=aws_creds.access_key_id,
            secret_access_key=aws_creds.secret_access_key,
            session_token=aws_creds.session_token,
        )
    elif credential_type == "azure_user_delegation_sas":
        from azure.core.credentials import AzureSasCredential

        from mlflow.store.artifact.azure_data_lake_artifact_repo import (
            AzureDataLakeArtifactRepository,
        )

        sas_token = scoped_token.azure_user_delegation_sas.sas_token
        return AzureDataLakeArtifactRepository(
            artifact_uri=storage_location, credential=AzureSasCredential(sas_token)
        )

    elif credential_type == "gcp_oauth_token":
        from google.cloud.storage import Client
        from google.oauth2.credentials import Credentials

        from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository

        credentials = Credentials(scoped_token.gcp_oauth_token.oauth_token)
        client = Client(project="mlflow", credentials=credentials)
        return GCSArtifactRepository(artifact_uri=storage_location, client=client)
    elif credential_type == "r2_temp_credentials":
        from mlflow.store.artifact.r2_artifact_repo import R2ArtifactRepository

        r2_creds = scoped_token.r2_temp_credentials
        return R2ArtifactRepository(
            artifact_uri=storage_location,
            access_key_id=r2_creds.access_key_id,
            secret_access_key=r2_creds.secret_access_key,
            session_token=r2_creds.session_token,
        )
    else:
        raise MlflowException(
            f"Got unexpected credential type {credential_type} when attempting to "
            "access model version files in Unity Catalog. Try upgrading to the latest "
            "version of the MLflow Python client."
        )


def get_full_name_from_sc(name, spark) -> str:
    """
    Constructs the full name of a registered model using the active catalog and schema in a spark
    session / context.

    Args:
        name: The model name provided by the user.
        spark: The active spark session.
    """
    num_levels = len(name.split("."))
    if num_levels >= 3 or spark is None:
        return name
    catalog = spark.sql(_ACTIVE_CATALOG_QUERY).collect()[0]["catalog"]
    # return the user provided name if the catalog is the hive metastore default
    if catalog in {"spark_catalog", "hive_metastore"}:
        return name
    if num_levels == 2:
        return f"{catalog}.{name}"
    schema = spark.sql(_ACTIVE_SCHEMA_QUERY).collect()[0]["schema"]
    return f"{catalog}.{schema}.{name}"
