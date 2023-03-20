from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    ModelVersion as ProtoModelVersion,
    ModelVersionStatus as ProtoModelVersionStatus,
    RegisteredModel as ProtoRegisteredModel,
    TemporaryCredentials,
)
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository

_STRING_TO_STATUS = {k: ProtoModelVersionStatus.Value(k) for k in ProtoModelVersionStatus.keys()}
_STATUS_TO_STRING = {value: key for key, value in _STRING_TO_STATUS.items()}


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
    )


def registered_model_from_uc_proto(uc_proto: ProtoRegisteredModel) -> RegisteredModel:
    return RegisteredModel(
        name=uc_proto.name,
        creation_timestamp=uc_proto.creation_timestamp,
        last_updated_timestamp=uc_proto.last_updated_timestamp,
        description=uc_proto.description,
    )


def get_artifact_repo_from_storage_info(
    storage_location: str, scoped_token: TemporaryCredentials
) -> ArtifactRepository:
    """
    Get an ArtifactRepository instance capable of reading/writing to a UC model version's
    file storage location
    :param storage_location: Storage location of the model version
    :param scoped_token: Protobuf scoped token to use to authenticate to blob storage
    """
    credential_type = scoped_token.WhichOneof("credentials")
    if credential_type == "aws_temp_credentials":
        from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository

        aws_creds = scoped_token.aws_temp_credentials
        return S3ArtifactRepository(
            artifact_uri=storage_location,
            access_key_id=aws_creds.access_key_id,
            secret_access_key=aws_creds.secret_access_key,
            session_token=aws_creds.session_token,
        )
    elif credential_type == "azure_user_delegation_sas":
        from mlflow.store.artifact.azure_data_lake_artifact_repo import (
            AzureDataLakeArtifactRepository,
        )
        from azure.core.credentials import AzureSasCredential

        sas_token = scoped_token.azure_user_delegation_sas.sas_token
        return AzureDataLakeArtifactRepository(
            artifact_uri=storage_location, credential=AzureSasCredential(sas_token)
        )

    elif credential_type == "gcp_oauth_token":
        from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository
        from google.cloud.storage import Client
        from google.oauth2.credentials import Credentials

        credentials = Credentials(scoped_token.gcp_oauth_token.oauth_token)
        client = Client(project="mlflow", credentials=credentials)
        return GCSArtifactRepository(artifact_uri=storage_location, client=client)
    else:
        raise MlflowException(
            f"Got unexpected token type {credential_type} for Unity Catalog managed file access"
        )
