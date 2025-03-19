import logging
from typing import Callable, Optional

from mlflow.entities.model_registry import (
    ModelVersion,
    ModelVersionTag,
    RegisteredModel,
    RegisteredModelAlias,
    RegisteredModelTag,
)
from mlflow.entities.model_registry.model_version_search import ModelVersionSearch
from mlflow.entities.model_registry.registered_model_search import RegisteredModelSearch
from mlflow.environment_variables import MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    EmitModelVersionLineageRequest,
    EmitModelVersionLineageResponse,
    IsDatabricksSdkModelsArtifactRepositoryEnabledRequest,
    IsDatabricksSdkModelsArtifactRepositoryEnabledResponse,
    ModelVersionLineageInfo,
    SseEncryptionAlgorithm,
    TemporaryCredentials,
)
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
from mlflow.protos.databricks_uc_registry_service_pb2 import UcModelRegistryService
from mlflow.protos.unity_catalog_oss_messages_pb2 import (
    TemporaryCredentials as TemporaryCredentialsOSS,
)
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    call_endpoint,
    extract_api_info_for_service,
)

_logger = logging.getLogger(__name__)
_METHOD_TO_INFO = extract_api_info_for_service(UcModelRegistryService, _REST_API_PATH_PREFIX)
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


def model_version_search_from_uc_proto(uc_proto: ProtoModelVersion) -> ModelVersionSearch:
    return ModelVersionSearch(
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
        aliases=[],
        tags=[],
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


def registered_model_search_from_uc_proto(uc_proto: ProtoRegisteredModel) -> RegisteredModelSearch:
    return RegisteredModelSearch(
        name=uc_proto.name,
        creation_timestamp=uc_proto.creation_timestamp,
        last_updated_timestamp=uc_proto.last_updated_timestamp,
        description=uc_proto.description,
        aliases=[],
        tags=[],
    )


def uc_registered_model_tag_from_mlflow_tags(
    tags: Optional[list[RegisteredModelTag]],
) -> list[ProtoRegisteredModelTag]:
    if tags is None:
        return []
    return [ProtoRegisteredModelTag(key=t.key, value=t.value) for t in tags]


def uc_model_version_tag_from_mlflow_tags(
    tags: Optional[list[ModelVersionTag]],
) -> list[ProtoModelVersionTag]:
    if tags is None:
        return []
    return [ProtoModelVersionTag(key=t.key, value=t.value) for t in tags]


def get_artifact_repo_from_storage_info(
    storage_location: str,
    scoped_token: TemporaryCredentials,
    base_credential_refresh_def: Callable[[], TemporaryCredentials],
    is_oss: bool = False,
) -> ArtifactRepository:
    """
    Get an ArtifactRepository instance capable of reading/writing to a UC model version's
    file storage location

    Args:
        storage_location: Storage location of the model version
        scoped_token: Protobuf scoped token to use to authenticate to blob storage
        base_credential_refresh_def: Function that returns temporary credentials for accessing blob
            storage. It is first used to determine the type of blob storage and to access it. It is
            then passed to the relevant ArtifactRepository implementation to refresh credentials as
            needed.
        is_oss: Whether the user is using the OSS version of Unity Catalog
    """
    try:
        if is_oss:
            return _get_artifact_repo_from_storage_info_oss(
                storage_location=storage_location,
                scoped_token=scoped_token,
                base_credential_refresh_def=base_credential_refresh_def,
            )
        else:
            return _get_artifact_repo_from_storage_info(
                storage_location=storage_location,
                scoped_token=scoped_token,
                base_credential_refresh_def=base_credential_refresh_def,
            )
    except ImportError as e:
        raise MlflowException(
            "Unable to import necessary dependencies to access model version files in "
            "Unity Catalog. Please ensure you have the necessary dependencies installed, "
            "e.g. by running 'pip install mlflow[databricks]' or "
            "'pip install mlflow-skinny[databricks]'"
        ) from e


def _get_artifact_repo_from_storage_info(
    storage_location: str,
    scoped_token: TemporaryCredentials,
    base_credential_refresh_def: Callable[[], TemporaryCredentials],
) -> ArtifactRepository:
    credential_type = scoped_token.WhichOneof("credentials")
    if credential_type == "aws_temp_credentials":
        # Verify upfront that boto3 is importable
        import boto3  # noqa: F401

        from mlflow.store.artifact.optimized_s3_artifact_repo import OptimizedS3ArtifactRepository

        aws_creds = scoped_token.aws_temp_credentials
        s3_upload_extra_args = _parse_aws_sse_credential(scoped_token)

        def aws_credential_refresh():
            new_scoped_token = base_credential_refresh_def()
            new_aws_creds = new_scoped_token.aws_temp_credentials
            new_s3_upload_extra_args = _parse_aws_sse_credential(new_scoped_token)
            return {
                "access_key_id": new_aws_creds.access_key_id,
                "secret_access_key": new_aws_creds.secret_access_key,
                "session_token": new_aws_creds.session_token,
                "s3_upload_extra_args": new_s3_upload_extra_args,
            }

        return OptimizedS3ArtifactRepository(
            artifact_uri=storage_location,
            access_key_id=aws_creds.access_key_id,
            secret_access_key=aws_creds.secret_access_key,
            session_token=aws_creds.session_token,
            credential_refresh_def=aws_credential_refresh,
            s3_upload_extra_args=s3_upload_extra_args,
        )
    elif credential_type == "azure_user_delegation_sas":
        from azure.core.credentials import AzureSasCredential

        from mlflow.store.artifact.azure_data_lake_artifact_repo import (
            AzureDataLakeArtifactRepository,
        )

        sas_token = scoped_token.azure_user_delegation_sas.sas_token

        def azure_credential_refresh():
            new_scoped_token = base_credential_refresh_def()
            new_sas_token = new_scoped_token.azure_user_delegation_sas.sas_token
            return {
                "credential": AzureSasCredential(new_sas_token),
            }

        return AzureDataLakeArtifactRepository(
            artifact_uri=storage_location,
            credential=AzureSasCredential(sas_token),
            credential_refresh_def=azure_credential_refresh,
        )

    elif credential_type == "gcp_oauth_token":
        from google.cloud.storage import Client
        from google.oauth2.credentials import Credentials

        from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository

        credentials = Credentials(scoped_token.gcp_oauth_token.oauth_token)

        def gcp_credential_refresh():
            new_scoped_token = base_credential_refresh_def()
            new_gcp_creds = new_scoped_token.gcp_oauth_token
            return {
                "oauth_token": new_gcp_creds.oauth_token,
            }

        client = Client(project="mlflow", credentials=credentials)
        return GCSArtifactRepository(
            artifact_uri=storage_location,
            client=client,
            credential_refresh_def=gcp_credential_refresh,
        )
    elif credential_type == "r2_temp_credentials":
        from mlflow.store.artifact.r2_artifact_repo import R2ArtifactRepository

        r2_creds = scoped_token.r2_temp_credentials

        def r2_credential_refresh():
            new_scoped_token = base_credential_refresh_def()
            new_r2_creds = new_scoped_token.r2_temp_credentials
            return {
                "access_key_id": new_r2_creds.access_key_id,
                "secret_access_key": new_r2_creds.secret_access_key,
                "session_token": new_r2_creds.session_token,
            }

        return R2ArtifactRepository(
            artifact_uri=storage_location,
            access_key_id=r2_creds.access_key_id,
            secret_access_key=r2_creds.secret_access_key,
            session_token=r2_creds.session_token,
            credential_refresh_def=r2_credential_refresh,
        )
    else:
        raise MlflowException(
            f"Got unexpected credential type {credential_type} when attempting to "
            "access model version files in Unity Catalog. Try upgrading to the latest "
            "version of the MLflow Python client."
        )


def _get_artifact_repo_from_storage_info_oss(
    storage_location: str,
    scoped_token: TemporaryCredentialsOSS,
    base_credential_refresh_def: Callable[[], TemporaryCredentialsOSS],
) -> ArtifactRepository:
    # OSS Temp Credential doesn't have a oneof credential field
    # So, we must check for the individual cloud credentials
    if len(scoped_token.aws_temp_credentials.access_key_id) > 0:
        # Verify upfront that boto3 is importable
        import boto3  # noqa: F401

        from mlflow.store.artifact.optimized_s3_artifact_repo import OptimizedS3ArtifactRepository

        aws_creds = scoped_token.aws_temp_credentials

        def aws_credential_refresh():
            new_scoped_token = base_credential_refresh_def()
            new_aws_creds = new_scoped_token.aws_temp_credentials
            return {
                "access_key_id": new_aws_creds.access_key_id,
                "secret_access_key": new_aws_creds.secret_access_key,
                "session_token": new_aws_creds.session_token,
            }

        return OptimizedS3ArtifactRepository(
            artifact_uri=storage_location,
            access_key_id=aws_creds.access_key_id,
            secret_access_key=aws_creds.secret_access_key,
            session_token=aws_creds.session_token,
            credential_refresh_def=aws_credential_refresh,
        )
    elif len(scoped_token.azure_user_delegation_sas.sas_token) > 0:
        from azure.core.credentials import AzureSasCredential

        from mlflow.store.artifact.azure_data_lake_artifact_repo import (
            AzureDataLakeArtifactRepository,
        )

        sas_token = scoped_token.azure_user_delegation_sas.sas_token

        def azure_credential_refresh():
            new_scoped_token = base_credential_refresh_def()
            new_sas_token = new_scoped_token.azure_user_delegation_sas.sas_token
            return {
                "credential": AzureSasCredential(new_sas_token),
            }

        return AzureDataLakeArtifactRepository(
            artifact_uri=storage_location,
            credential=AzureSasCredential(sas_token),
            credential_refresh_def=azure_credential_refresh,
        )

    elif len(scoped_token.gcp_oauth_token.oauth_token) > 0:
        from google.cloud.storage import Client
        from google.oauth2.credentials import Credentials

        from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository

        credentials = Credentials(scoped_token.gcp_oauth_token.oauth_token)
        client = Client(project="mlflow", credentials=credentials)
        return GCSArtifactRepository(artifact_uri=storage_location, client=client)
    else:
        raise MlflowException(
            "Got no credential type when attempting to "
            "access model version files in Unity Catalog. Try upgrading to the latest "
            "version of the MLflow Python client."
        )


def _parse_aws_sse_credential(scoped_token: TemporaryCredentials):
    encryption_details = scoped_token.encryption_details
    if not encryption_details:
        return {}

    if encryption_details.WhichOneof("encryption_details_type") != "sse_encryption_details":
        return {}

    sse_encryption_details = encryption_details.sse_encryption_details

    if sse_encryption_details.algorithm == SseEncryptionAlgorithm.AWS_SSE_S3:
        return {
            "ServerSideEncryption": "AES256",
        }
    if sse_encryption_details.algorithm == SseEncryptionAlgorithm.AWS_SSE_KMS:
        key_id = sse_encryption_details.aws_kms_key_arn.split("/")[-1]
        return {
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": key_id,
        }
    else:
        return {}


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


def is_databricks_sdk_models_artifact_repository_enabled(host_creds):
    # Return early if the environment variable is set to use the SDK models artifact repository
    if MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC.defined:
        return MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC.get()

    endpoint, method = _METHOD_TO_INFO[IsDatabricksSdkModelsArtifactRepositoryEnabledRequest]
    req_body = message_to_json(IsDatabricksSdkModelsArtifactRepositoryEnabledRequest())
    response_proto = IsDatabricksSdkModelsArtifactRepositoryEnabledResponse()

    try:
        resp = call_endpoint(
            host_creds=host_creds,
            endpoint=endpoint,
            method=method,
            json_body=req_body,
            response_proto=response_proto,
        )
        return resp.is_databricks_sdk_models_artifact_repository_enabled
    except Exception as e:
        _logger.warning(
            "Failed to confirm if DatabricksSDKModelsArtifactRepository should be used; "
            f"falling back to default. Error: {e}"
        )
    return False


def emit_model_version_lineage(host_creds, name, version, entities, direction):
    endpoint, method = _METHOD_TO_INFO[EmitModelVersionLineageRequest]

    req_body = message_to_json(
        EmitModelVersionLineageRequest(
            name=name,
            version=version,
            model_version_lineage_info=ModelVersionLineageInfo(
                entities=entities,
                direction=direction,
            ),
        )
    )
    response_proto = EmitModelVersionLineageResponse()
    try:
        call_endpoint(
            host_creds=host_creds,
            endpoint=endpoint,
            method=method,
            json_body=req_body,
            response_proto=response_proto,
        )
    except Exception as e:
        _logger.warning(f"Failed to emit best-effort model version lineage. Error: {e}")
