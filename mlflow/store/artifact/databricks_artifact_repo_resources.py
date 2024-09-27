from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional, Tuple

from mlflow.protos.databricks_artifacts_pb2 import (
    ArtifactCredentialInfo,
    DatabricksMlflowArtifactsService,
    ModelGetCredentialsForRead,
    ModelGetCredentialsForWrite,
)
from mlflow.protos.databricks_artifacts_pb2 import (
    GetCredentialsForRead as RunGetCredentialsForRead,
)
from mlflow.protos.databricks_artifacts_pb2 import (
    GetCredentialsForWrite as RunGetCredentialsForWrite,
)
from mlflow.protos.service_pb2 import GetLoggedModel, GetRun, MlflowService
from mlflow.utils.proto_json_utils import message_to_json


class _CredentialType(Enum):
    READ = 1
    WRITE = 2


class _Resource(ABC):
    def __init__(self, id: str, creds: Any, call_endpoint: Any):
        self.creds = creds
        self.id = id
        self.artifact_root = self.get_artifact_root()
        self.call_endpoint = call_endpoint

    @abstractmethod
    def get_credentials(
        self, cred_type: _CredentialType, paths: List[str], page_token: Optional[str] = None
    ) -> Tuple[List[ArtifactCredentialInfo], Optional[str]]:
        """
        Issue one or more requests for artifact credentials, providing read or write.
        """

    @abstractmethod
    def get_artifact_root(self) -> str:
        """
        Get the artifact root for this resource.
        """


class _Model(_Resource):
    def get_credentials(
        self, cred_type: _CredentialType, paths: List[str], page_token: Optional[str] = None
    ) -> Tuple[List[Any], Optional[str]]:
        api = (
            ModelGetCredentialsForWrite
            if cred_type == _CredentialType.WRITE
            else ModelGetCredentialsForRead
        )
        payload = api(model_id=self.id, paths=paths, page_token=page_token)
        response = self._call_endpoint(
            DatabricksMlflowArtifactsService, api, message_to_json(payload)
        )
        return response.credential_infos, response.next_page_token

    def get_artifact_root(self) -> str:
        json_body = message_to_json(GetLoggedModel(model_id=self.model_id))
        response = self._call_endpoint(MlflowService, GetLoggedModel, json_body)
        return response.model.info.artifact_uri


class _Run(_Resource):
    def get_credentials(
        self, cred_type: _CredentialType, paths: List[str], page_token: Optional[str] = None
    ) -> List[ArtifactCredentialInfo]:
        api = (
            RunGetCredentialsForWrite
            if cred_type == _CredentialType.WRITE
            else RunGetCredentialsForRead
        )
        json_body = api(run_id=self.id, path=paths, page_token=page_token)
        response = self._call_endpoint(
            DatabricksMlflowArtifactsService, api, message_to_json(json_body)
        )
        return response.credential_infos, response.next_page_token

    def get_artifact_root(self) -> str:
        json_body = message_to_json(GetRun(run_id=self.id))
        run_response = self._call_endpoint(MlflowService, GetRun, json_body)
        return run_response.run.info.artifact_uri
